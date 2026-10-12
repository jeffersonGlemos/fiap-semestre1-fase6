"""Cliente do PC (executado no terminal do VS Code): recebe as imagens por Wi-Fi e reconhece os objetos.

O modo é escolhido automaticamente a partir dos argumentos e da resposta do servidor:
    BASE  --modelo best.pt   a inferência roda AQUI, com o best.pt da Entrega 1 (YOLOv5 via torch.hub).
                             Fonte: stream MJPEG do Pi/ESP32-CAM, webcam local (0) ou arquivo.
    PLUS  sem --modelo       a fonte é o camera_server.py com Hailo (ou ONNX no Pi): o vídeo já chega
                             anotado e as detecções são lidas de <servidor>/deteccoes, em paralelo.
    SÓ VÍDEO                 sem --modelo e com servidor sem inferência: apenas exibe e mede a rede.

Na janela aparecem as caixas, o FPS e as latências de rede e de inferência. Detecções de cobra e
escorpião acendem uma faixa vermelha de alerta e geram uma linha em alertas.csv. Cada quadro gera
uma linha em frames.csv, que é analisado no notebook.
Teclas: q encerra; s salva o quadro atual em prints/.

Uso:
    python cliente_vscode.py --fonte http://192.168.0.42:8000/stream --modelo best.pt   # Pi, BASE
    python cliente_vscode.py --fonte http://192.168.0.50/stream --modelo best.pt        # ESP32-CAM
    python cliente_vscode.py --fonte 0 --modelo best.pt                                 # webcam do PC
    python cliente_vscode.py --fonte http://192.168.0.42:8000/stream                    # Pi + Hailo, PLUS
    python cliente_vscode.py --fonte demo.mp4 --modelo best.onnx --sem-janela --max-frames 100
"""

import argparse
import csv
import json
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from backends import CRITICAS, anotar

VERMELHO = (0, 0, 255)


class FonteMJPEG:
    """Lê um stream MJPEG por HTTP em uma thread separada e reconecta sozinha se a conexão cair.

    Optei por interpretar o multipart diretamente, em vez de usar cv2.VideoCapture, porque assim
    tenho acesso ao cabeçalho X-Timestamp de cada quadro. Ele registra o instante da captura no
    dispositivo e é a base para medir a latência fim a fim.
    """

    def __init__(self, url, timeout=5):
        self.url, self.timeout = url, timeout
        self.frame, self.ts_captura, self.novo = None, None, False
        self.lock, self.ativo = threading.Lock(), True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.ativo:
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                    print(f"Conectado a {self.url}")
                    self._consumir(resp)
            except Exception as erro:                        # queda do Wi-Fi, reinício do servidor etc.
                print(f"Stream indisponível ({erro.__class__.__name__}); nova tentativa em 2 s")
                time.sleep(2)

    def _consumir(self, resp):
        while self.ativo:
            cabecalho = {}
            linha = resp.readline()
            while linha and not linha.startswith(b"--"):     # avança até a fronteira da próxima parte
                linha = resp.readline()
            if not linha:
                raise ConnectionError("o stream foi encerrado")
            while (linha := resp.readline().strip()):        # cabeçalhos da parte, até a linha em branco
                chave, _, valor = linha.decode(errors="ignore").partition(":")
                cabecalho[chave.strip().lower()] = valor.strip()
            tamanho = int(cabecalho.get("content-length", 0))
            dados = resp.read(tamanho) if tamanho else self._ler_ate_eoi(resp)
            frame = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            ts = cabecalho.get("x-timestamp")
            with self.lock:
                self.frame, self.novo = frame, True
                # Só serve como relógio absoluto se estiver em epoch (o ESP32 sem NTP envia o tempo desde o boot)
                self.ts_captura = float(ts) if ts and float(ts) > 1e9 else None

    @staticmethod
    def _ler_ate_eoi(resp):
        """Alternativa para servidores que não enviam Content-Length: lê até o marcador de fim do JPEG (FFD9)."""
        dados = bytearray()
        while not dados.endswith(b"\xff\xd9"):
            dados += resp.read(1)
        return bytes(dados)

    def ler(self, espera=5.0):
        limite = time.time() + espera
        while time.time() < limite:
            with self.lock:
                if self.novo:
                    self.novo = False
                    return self.frame.copy(), self.ts_captura
            time.sleep(0.002)
        return None, None

    def fechar(self):
        self.ativo = False


class FonteLocal:
    """Webcam do PC (por índice) ou arquivo de vídeo, lidos pelo OpenCV."""

    def __init__(self, origem):
        self.cap = cv2.VideoCapture(int(origem) if origem.isdigit() else origem)
        if not self.cap.isOpened():
            raise SystemExit(f"Não foi possível abrir {origem!r}.")

    def ler(self, espera=None):
        ok, frame = self.cap.read()
        return (frame, time.time()) if ok else (None, None)

    def fechar(self):
        self.cap.release()


class MonitorRemoto:
    """Acompanha o servidor de câmera: mede o RTT da rede via /saude e, no modo PLUS, lê as detecções do Pi."""

    def __init__(self, base, intervalo=0.1):
        self.base, self.intervalo = base, intervalo
        self.ultimo = {"deteccoes": [], "latencia_inferencia_ms": None}
        self.rtt_ms, self.ativo = None, True
        self.info = self._tentar("/saude") or {}             # tanto o ESP32 quanto o camera_server respondem /saude
        self.infere = self.info.get("inferencia", "nenhuma") != "nenhuma"   # indica se o Pi está inferindo
        threading.Thread(target=self._loop, daemon=True).start()

    def _get(self, rota):
        with urllib.request.urlopen(self.base + rota, timeout=2) as resp:
            return json.loads(resp.read())

    def _tentar(self, rota):
        try:
            return self._get(rota)
        except Exception:                                    # rota inexistente ou servidor fora do ar
            return None

    def _loop(self):
        while self.ativo:
            t0 = time.perf_counter()
            if self._tentar("/saude") is not None:           # RTT = ida e volta de uma requisição curta
                self.rtt_ms = (time.perf_counter() - t0) * 1000
            if self.infere and (det := self._tentar("/deteccoes")):
                self.ultimo = det
            time.sleep(self.intervalo)


class ModeloLocal:
    """Modo BASE: carrega o best.pt (YOLOv5 via torch.hub) ou um .onnx (dispensa o PyTorch)."""

    def __init__(self, caminho, conf):
        self.onnx = caminho.suffix == ".onnx"
        if self.onnx:
            from backends.onnx_backend import OnnxBackend
            self.modelo = OnnxBackend(caminho, conf=conf)
        else:
            import torch                                     # import tardio: o modo PLUS não usa PyTorch
            self.modelo = torch.hub.load("ultralytics/yolov5", "custom", path=str(caminho),
                                         trust_repo=True)   # evita a confirmação interativa das versões novas do torch
            self.modelo.conf = conf
            self.nomes = self.modelo.names                   # nomes das classes gravados no best.pt

    def detect(self, frame):
        if self.onnx:
            return self.modelo.detect(frame)
        resultado = self.modelo(frame[:, :, ::-1])           # o hub espera RGB
        return [{"classe": self.nomes[int(c)], "conf": round(float(p), 4),
                 "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)]}
                for x1, y1, x2, y2, p, c in resultado.xyxy[0].tolist()]


def painel(frame, linhas, alerta):
    """Escreve as métricas no canto do quadro e, quando há alerta, desenha uma faixa vermelha no topo."""
    if alerta:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), VERMELHO, -1)
        cv2.putText(frame, f"ALERTA: {alerta}", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y = 70 if alerta else 26
    for texto in linhas:
        cv2.putText(frame, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)   # contorno preto para legibilidade
        cv2.putText(frame, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", required=True, help="URL do stream MJPEG, índice da webcam ou arquivo de vídeo")
    ap.add_argument("--modelo", type=Path, help="best.pt (ou .onnx): inferência no PC, modo BASE")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--logs", type=Path, default=Path("logs"), help="pasta onde ficam frames.csv e alertas.csv")
    ap.add_argument("--sem-janela", action="store_true", help="não abre a janela de vídeo (testes ou SSH)")
    ap.add_argument("--max-frames", type=int, default=0, help="encerra após N quadros (0 = sem limite)")
    args = ap.parse_args()

    remoto = args.fonte.startswith("http")
    fonte = FonteMJPEG(args.fonte) if remoto else FonteLocal(args.fonte)
    modelo = ModeloLocal(args.modelo, args.conf) if args.modelo else None
    base = args.fonte.rsplit("/", 1)[0] if remoto else None
    monitor = MonitorRemoto(base) if remoto else None
    plus = monitor if monitor and monitor.infere and not modelo else None
    modo = "BASE" if modelo else ("PLUS" if plus else "SÓ VÍDEO")
    print(f"Modo {modo} | fonte {args.fonte}")

    args.logs.mkdir(parents=True, exist_ok=True)
    if not args.sem_janela:
        Path("prints").mkdir(exist_ok=True)                  # pasta usada pela tecla s
    sessao = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_frames = open(args.logs / "frames.csv", "a", newline="")
    f_alertas = open(args.logs / "alertas.csv", "a", newline="")
    log_frames, log_alertas = csv.writer(f_frames), csv.writer(f_alertas)
    if f_frames.tell() == 0:                                 # o cabeçalho só é escrito em arquivo novo
        log_frames.writerow(["sessao", "timestamp", "frame", "modo", "fps", "latencia_fim_a_fim_ms",
                             "rtt_rede_ms", "latencia_inferencia_ms", "n_deteccoes", "classes", "alerta"])
    if f_alertas.tell() == 0:
        log_alertas.writerow(["sessao", "timestamp", "frame", "classe", "conf", "x1", "y1", "x2", "y2"])

    n, fps, t_fps, janela = 0, 0.0, time.time(), 0
    try:
        while True:
            frame, ts_captura = fonte.ler()
            if frame is None:
                if remoto:
                    continue                                 # a FonteMJPEG cuida da reconexão
                break                                        # fim do arquivo ou da webcam

            ms_inf = None
            if modelo:                                       # BASE: a inferência e o desenho acontecem aqui
                t0 = time.perf_counter()
                deteccoes = modelo.detect(frame)
                ms_inf = (time.perf_counter() - t0) * 1000
                anotar(frame, deteccoes)
            elif plus:                                       # PLUS: o quadro já chega anotado pelo Pi
                deteccoes = plus.ultimo.get("deteccoes", [])
                ms_inf = plus.ultimo.get("latencia_inferencia_ms")
            else:
                deteccoes = []

            n, janela = n + 1, janela + 1
            if time.time() - t_fps >= 1:
                fps, janela, t_fps = janela / (time.time() - t_fps), 0, time.time()
            agora = time.time()
            fim_a_fim = (agora - ts_captura) * 1000 if remoto and ts_captura else None
            rtt = monitor.rtt_ms if monitor else None

            criticas = [d for d in deteccoes if d["classe"] in CRITICAS]
            alerta = ", ".join(f'{d["classe"]} {d["conf"]:.2f}' for d in criticas)
            for d in criticas:
                log_alertas.writerow([sessao, f"{agora:.3f}", n, d["classe"], d["conf"], *d["bbox"]])
            if criticas:
                print(f"[{datetime.now():%H:%M:%S}] ALERTA: {alerta}")
            log_frames.writerow([
                sessao, f"{agora:.3f}", n, modo, round(fps, 2),
                round(fim_a_fim, 1) if fim_a_fim is not None else "",
                round(rtt, 1) if rtt is not None else "",
                round(ms_inf, 1) if ms_inf is not None else "",
                len(deteccoes), "|".join(sorted({d["classe"] for d in deteccoes})), int(bool(criticas)),
            ])

            if not args.sem_janela:
                painel(frame, [
                    f"modo {modo}  |  {fps:4.1f} FPS",
                    f"fim a fim: {fim_a_fim:.0f} ms" if fim_a_fim is not None else "fim a fim: n/d",
                    f"inferencia: {ms_inf:.0f} ms" if ms_inf is not None else "inferencia: n/d",
                    f"RTT rede: {rtt:.0f} ms" if rtt is not None else "",
                ], alerta)
                cv2.imshow("FarmTech | Ir Alem", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("s"):
                    caminho = Path("prints") / f"print_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(caminho), frame)
                    print(f"Quadro salvo em {caminho}")
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        fonte.fechar()
        if monitor:
            monitor.ativo = False
        f_frames.close()
        f_alertas.close()
        if not args.sem_janela:
            cv2.destroyAllWindows()
        print(f"{n} quadros processados. Logs gravados em {args.logs}/")


if __name__ == "__main__":
    main()
