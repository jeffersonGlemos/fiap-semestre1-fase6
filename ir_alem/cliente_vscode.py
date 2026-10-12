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

Onde a inferência roda no PC é parametrizável, porque desenvolvi no Mac (Apple Silicon) e o projeto
também precisa rodar em Windows e Linux:
    --dispositivo auto|cpu|mps|cuda          para o .pt (torch.hub); auto escolhe cuda > mps > cpu
    --provedor auto|cpu|coreml|cuda|directml para o .onnx (ONNX Runtime); ver backends/onnx_backend.py
O que foi efetivamente usado é impresso no início e gravado na coluna `execucao` do frames.csv.

Uso:
    python cliente_vscode.py --fonte http://192.168.0.42:8000/stream --modelo best.pt   # Pi, BASE
    python cliente_vscode.py --fonte http://192.168.0.50:81/stream --saude http://192.168.0.50 --modelo best.pt   # ESP32-CAM
    python cliente_vscode.py --fonte 0 --modelo best.pt                                 # webcam do PC
    python cliente_vscode.py --fonte http://192.168.0.42:8000/stream                    # Pi + Hailo, PLUS
    python cliente_vscode.py --fonte demo.mp4 --modelo best.onnx --sem-janela --max-frames 100
    python cliente_vscode.py --fonte http://<câmera>:<porta>/<rota MJPEG> --modelo best.pt --dispositivo mps
    python cliente_vscode.py --fonte <url> --modelo yolov5s.onnx --classes coco --provedor coreml
    python cliente_vscode.py --fonte https://<IP do Mac>:8000/stream --inseguro --modelo best.pt   # celular via navegador

Com o camera_server.py em --fonte navegador, o servidor usa HTTPS com certificado autoassinado. A
opção --inseguro desliga a verificação do certificado apenas nas conexões com esse servidor (stream e
/saude); o conteúdo continua cifrado, só a identidade do servidor deixa de ser conferida. Quando o
/saude informa essa fonte e o modelo roda aqui (BASE), o cliente devolve ao servidor as detecções de
cada quadro em coordenadas normalizadas (POST /deteccoes), e a página /celular desenha as caixas sobre
a prévia do próprio celular.

Na ESP32-CAM, o /stream fica em um servidor separado, na porta 81, e o /saude na porta 80 (ver
esp32cam/esp32cam_stream.ino). Como o cliente procura o /saude no mesmo endereço do stream, a opção
--saude informa a URL base do servidor quando ela é diferente da do stream.

Se a câmera parar de enviar, o servidor mantém o /stream vivo com quadros de keepalive; a janela
continua respondendo, mostra "aguardando quadros" e volta sozinha quando os quadros retornam.
"""

import argparse
import csv
import http.client
import json
import ssl
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from backends import CRITICAS, anotar, normalizar, resolver_classes

VERMELHO = (0, 0, 255)


class FonteMJPEG:
    """Lê um stream MJPEG por HTTP em uma thread separada e reconecta sozinha se a conexão cair.

    Optei por interpretar o multipart diretamente, em vez de usar cv2.VideoCapture, porque assim
    tenho acesso aos cabeçalhos de cada quadro: o X-Timestamp registra o instante da captura no
    dispositivo e é a base para medir a latência fim a fim; o X-Quadro identifica o quadro do celular
    a que as detecções devolvidas se referem; o X-Keepalive marca reenvios do servidor quando a câmera
    está parada, que mantêm a conexão viva e não contam como quadros novos.

    A leitura do stream e a inferência ficam em threads diferentes. A thread de leitura guarda apenas
    o JPEG mais recente, ainda sem decodificar; se a inferência estiver ocupada, os quadros
    intermediários são descartados (e contados), em vez de acumular atraso.
    """

    def __init__(self, url, timeout=10, contexto=None):
        self.url, self.timeout, self.contexto = url, timeout, contexto
        self.dados, self.ts_captura, self.quadro, self.novo = None, None, None, False
        self.cond, self.ativo = threading.Condition(), True
        self.descartados = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        falhas, conectou = 0, False
        while self.ativo:
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout, context=self.contexto) as resp:
                    print(f"Reconectado a {self.url}" if conectou else f"Conectado a {self.url}")
                    falhas, conectou = 0, True
                    self._consumir(resp)
            except Exception as erro:                        # queda do Wi-Fi, reinício do servidor etc.
                falhas += 1
                if falhas == 1:                              # uma mensagem por queda, e não uma por tentativa
                    print(f"Stream indisponível ({erro.__class__.__name__}); tentando reconectar em segundo plano")
                    if isinstance(getattr(erro, "reason", None), ssl.SSLCertVerificationError):
                        print("  O certificado do servidor é autoassinado: acrescente --inseguro ao comando.")
                time.sleep(min(0.5 * 2 ** (falhas - 1), 5))  # 0,5 s, 1 s, 2 s, 4 s e depois a cada 5 s

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
            if "x-keepalive" in cabecalho:                   # reenvio com a câmera parada: só mantém a conexão
                continue
            ts, quadro = cabecalho.get("x-timestamp"), cabecalho.get("x-quadro")
            with self.cond:
                if self.novo:                                # o anterior não chegou a ser processado
                    self.descartados += 1
                self.dados, self.novo = dados, True
                # Só serve como relógio absoluto se estiver em epoch (o ESP32 sem NTP envia o tempo desde o boot)
                self.ts_captura = float(ts) if ts and float(ts) > 1e9 else None
                self.quadro = int(quadro) if quadro and quadro.isdigit() else None
                self.cond.notify()

    @staticmethod
    def _ler_ate_eoi(resp):
        """Alternativa para servidores que não enviam Content-Length: lê até o marcador de fim do JPEG (FFD9)."""
        dados = bytearray()
        while not dados.endswith(b"\xff\xd9"):
            dados += resp.read(1)
        return bytes(dados)

    def ler(self, espera=0.25):
        """Devolve (frame, ts_captura, quadro) do quadro mais recente, ou None após `espera` segundos sem quadro.

        A espera é curta para que o laço principal continue atualizando a janela enquanto a câmera está parada.
        """
        with self.cond:
            if not self.cond.wait_for(lambda: self.novo, timeout=espera):
                return None, None, None
            self.novo = False
            dados, ts, quadro = self.dados, self.ts_captura, self.quadro
        frame = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)   # decodifica só o que será usado
        return (frame, ts, quadro) if frame is not None else (None, None, None)

    def fechar(self):
        self.ativo = False


class FonteLocal:
    """Webcam do PC (por índice) ou arquivo de vídeo, lidos pelo OpenCV."""

    def __init__(self, origem):
        self.cap = cv2.VideoCapture(int(origem) if origem.isdigit() else origem)
        if not self.cap.isOpened():
            raise SystemExit(f"Não foi possível abrir {origem!r}.")
        self.descartados = 0

    def ler(self, espera=None):
        ok, frame = self.cap.read()
        return (frame, time.time(), None) if ok else (None, None, None)

    def fechar(self):
        self.cap.release()


class EnviadorDeteccoes:
    """Modo BASE com o celular pelo navegador: devolve ao camera_server.py as detecções de cada quadro,
    em coordenadas normalizadas (POST /deteccoes), para que a página /celular desenhe as caixas sobre a
    prévia do próprio celular.

    Roda em uma thread própria, com conexão HTTP persistente (sem um handshake TLS por quadro), e envia
    apenas o resultado mais recente: se a rede atrasar, o resultado antigo é substituído, e a inferência
    nunca espera pelo envio.
    """

    def __init__(self, base, contexto=None):
        partes = urllib.parse.urlsplit(base)
        self.https, self.host, self.porta = partes.scheme == "https", partes.hostname, partes.port
        self.contexto, self.conexao = contexto, None
        self.cond, self.pendente, self.ativo = threading.Condition(), None, True
        self.enviados, self.falhas = 0, 0
        threading.Thread(target=self._loop, daemon=True).start()

    def enviar(self, carga):
        with self.cond:
            self.pendente = carga
            self.cond.notify()

    def _conectar(self):
        if self.https:
            return http.client.HTTPSConnection(self.host, self.porta, timeout=2, context=self.contexto)
        return http.client.HTTPConnection(self.host, self.porta, timeout=2)

    def _loop(self):
        while self.ativo:
            with self.cond:
                self.cond.wait_for(lambda: self.pendente is not None, timeout=0.5)
                carga, self.pendente = self.pendente, None
            if carga is None:
                continue
            try:
                self.conexao = self.conexao or self._conectar()
                self.conexao.request("POST", "/deteccoes", json.dumps(carga).encode(),
                                     {"Content-Type": "application/json"})
                self.conexao.getresponse().read()
                self.enviados += 1
            except (OSError, http.client.HTTPException):    # servidor reiniciado ou Wi-Fi instável
                if self.conexao:
                    self.conexao.close()
                self.conexao = None
                self.falhas += 1
                if self.falhas == 1:
                    print("Não foi possível devolver as detecções ao servidor; novas tentativas em segundo plano")

    def fechar(self):
        self.ativo = False


class MonitorRemoto:
    """Acompanha o servidor de câmera: mede o RTT da rede via /saude e, no modo PLUS, lê as detecções do Pi."""

    def __init__(self, base, intervalo=0.1, contexto=None):
        self.base, self.intervalo, self.contexto = base, intervalo, contexto
        self.ultimo = {"deteccoes": [], "latencia_inferencia_ms": None}
        self.rtt_ms, self.ativo = None, True
        self.info = self._tentar("/saude") or {}             # tanto o ESP32 quanto o camera_server respondem /saude
        self.infere = self.info.get("inferencia", "nenhuma") != "nenhuma"   # indica se o Pi está inferindo
        threading.Thread(target=self._loop, daemon=True).start()

    def _get(self, rota):
        with urllib.request.urlopen(self.base + rota, timeout=2, context=self.contexto) as resp:
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


def contexto_tls(url, inseguro):
    """Contexto TLS para urlopen. Com --inseguro e URL https, aceita o certificado autoassinado do
    camera_server.py; o contexto só é passado às conexões com esse servidor."""
    if not (url.startswith("https") and inseguro):
        return None                                          # padrão do urllib: verificação completa
    contexto = ssl.create_default_context()
    contexto.check_hostname = False
    contexto.verify_mode = ssl.CERT_NONE
    return contexto


def escolher_dispositivo(pedido):
    """Resolve --dispositivo para o PyTorch: auto prefere CUDA (NVIDIA), depois MPS (Apple Silicon), depois CPU."""
    import torch
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if pedido == "auto":
        return "cuda" if cuda else ("mps" if mps else "cpu")
    if (pedido == "cuda" and not cuda) or (pedido == "mps" and not mps):
        raise SystemExit(f"--dispositivo {pedido} não está disponível nesta máquina; use auto ou cpu.")
    return pedido


class ModeloLocal:
    """Modo BASE: carrega o best.pt (YOLOv5 via torch.hub) ou um .onnx (dispensa o PyTorch)."""

    def __init__(self, caminho, conf, classes="fase6", provedor="auto", dispositivo="auto"):
        self.onnx = caminho.suffix == ".onnx"
        if self.onnx:
            from backends.onnx_backend import OnnxBackend
            self.modelo = OnnxBackend(caminho, conf=conf, classes=resolver_classes(classes), provedor=provedor)
            self.execucao = f"onnx:{self.modelo.provedor}"   # ex.: onnx:CoreMLExecutionProvider
        else:
            import torch                                     # import tardio: o modo PLUS não usa PyTorch
            from backends.compat import carregamento_portavel
            disp = escolher_dispositivo(dispositivo)
            with carregamento_portavel():                    # best.pt do Colab (Python 3.13, Linux) em outro Python/SO
                self.modelo = torch.hub.load("ultralytics/yolov5", "custom", path=str(caminho), device=disp,
                                             trust_repo=True)   # evita a confirmação interativa das versões novas do torch
            self.modelo.conf = conf
            self.nomes = self.modelo.names                   # nomes das classes gravados no best.pt (ignora --classes)
            self.execucao = f"pt:{disp}"                     # ex.: pt:mps

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


def tela_espera(ultima, parado):
    """Tela exibida enquanto não chegam quadros: o último quadro escurecido e o tempo de espera."""
    tela = (ultima * 0.35).astype(np.uint8) if ultima is not None else np.zeros((480, 640, 3), np.uint8)
    texto = f"aguardando quadros da camera ({parado:.0f} s)"
    y = tela.shape[0] // 2
    cv2.putText(tela, texto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(tela, texto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return tela


def abrir_log(caminho, cabecalho):
    """Abre o CSV em modo de acréscimo. Se ele foi criado com outras colunas (versão anterior do cliente),
    o arquivo antigo é renomeado, para que o notebook nunca leia colunas desalinhadas."""
    if caminho.exists() and caminho.stat().st_size:
        with open(caminho, newline="", encoding="utf-8", errors="replace") as f:
            atual = next(csv.reader(f), [])
        if atual != cabecalho:
            antigo = caminho.with_name(f"{caminho.stem}_antigo_{datetime.now():%Y%m%d_%H%M%S}.csv")
            caminho.rename(antigo)
            print(f"{caminho.name} tinha outras colunas; o arquivo anterior foi preservado como {antigo.name}")
    arquivo = open(caminho, "a", newline="", encoding="utf-8", buffering=1)   # utf-8 explícito (no Windows seria cp1252); grava linha a linha, para não perder registros num encerramento forçado
    escritor = csv.writer(arquivo)
    if arquivo.tell() == 0:                                  # o cabeçalho só é escrito em arquivo novo
        escritor.writerow(cabecalho)
    return arquivo, escritor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", required=True, help="URL do stream MJPEG, índice da webcam ou arquivo de vídeo")
    ap.add_argument("--modelo", type=Path, help="best.pt (ou .onnx): inferência no PC, modo BASE")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="fase6",
                    help="nomes das classes do .onnx: fase6 (padrão), coco (yolov5s.onnx, para teste) ou arquivo .txt; "
                         "o .pt já traz os nomes gravados")
    ap.add_argument("--provedor", default="auto", choices=["auto", "cpu", "coreml", "cuda", "directml"],
                    help="execution provider do ONNX Runtime para o .onnx (auto: cuda > coreml > directml > cpu)")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="dispositivo do PyTorch para o .pt (auto: cuda > mps > cpu)")
    ap.add_argument("--logs", type=Path, default=Path("logs"), help="pasta onde ficam frames.csv e alertas.csv")
    ap.add_argument("--sem-janela", action="store_true", help="não abre a janela de vídeo (testes ou SSH)")
    ap.add_argument("--max-frames", type=int, default=0, help="encerra após N quadros (0 = sem limite)")
    ap.add_argument("--inseguro", action="store_true",
                    help="aceita o certificado autoassinado de um servidor https (camera_server.py --fonte navegador)")
    ap.add_argument("--saude", help="URL base do servidor para /saude e /deteccoes quando difere da do stream "
                                    "(ESP32-CAM: stream em http://<IP>:81/stream, /saude em http://<IP>)")
    args = ap.parse_args()

    remoto = args.fonte.startswith("http")
    contexto = contexto_tls(args.fonte, args.inseguro) if remoto else None
    fonte = FonteMJPEG(args.fonte, contexto=contexto) if remoto else FonteLocal(args.fonte)
    modelo = (ModeloLocal(args.modelo, args.conf, args.classes, args.provedor, args.dispositivo)
              if args.modelo else None)
    base = (args.saude or args.fonte.rsplit("/", 1)[0]).rstrip("/") if remoto else None   # /saude e /deteccoes
    monitor = MonitorRemoto(base, contexto=contexto) if remoto else None
    plus = monitor if monitor and monitor.infere and not modelo else None
    modo = "BASE" if modelo else ("PLUS" if plus else "SÓ VÍDEO")
    if modelo:
        execucao = modelo.execucao
    elif plus:                                               # no PLUS, quem informa onde roda é o servidor (/saude)
        execucao = ":".join(str(v) for v in ("remoto", plus.info.get("inferencia"), plus.info.get("provedor")) if v)
    else:
        execucao = ""
    print(f"Modo {modo} | fonte {args.fonte} | execução {execucao or 'sem inferência'}")
    # Com o celular pelo navegador e a inferência aqui (BASE), as detecções voltam ao servidor para a
    # página /celular desenhar as caixas. No PLUS, o próprio servidor já usa as detecções dele
    enviador = None
    if modelo and monitor and monitor.info.get("fonte") == "navegador" and not monitor.infere:
        enviador = EnviadorDeteccoes(base, contexto)
        print("As detecções voltam ao servidor (POST /deteccoes) para a página /celular desenhar as caixas")

    args.logs.mkdir(parents=True, exist_ok=True)
    if not args.sem_janela:
        Path("prints").mkdir(exist_ok=True)                  # pasta usada pela tecla s
    sessao = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_frames, log_frames = abrir_log(args.logs / "frames.csv", [
        "sessao", "timestamp", "frame", "modo", "execucao", "fps", "latencia_fim_a_fim_ms",
        "rtt_rede_ms", "latencia_inferencia_ms", "n_deteccoes", "classes", "alerta"])
    f_alertas, log_alertas = abrir_log(args.logs / "alertas.csv", [
        "sessao", "timestamp", "frame", "classe", "conf", "x1", "y1", "x2", "y2"])

    n, fps, t_fps, janela = 0, 0.0, time.time(), 0
    ultimo_quadro, ultima_tela, em_espera = time.time(), None, False
    try:
        while True:
            frame, ts_captura, quadro = fonte.ler()
            if frame is None:
                if not remoto:
                    break                                    # fim do arquivo ou da webcam
                # Stream sem quadros novos (celular parado ou reconexão em curso, a cargo da FonteMJPEG):
                # a janela continua respondendo e mostra a espera, em vez de congelar no último quadro
                parado = time.time() - ultimo_quadro
                if parado > 2 and not em_espera:
                    print(f"Sem quadros novos há {parado:.0f} s; aguardando a câmera")
                    em_espera = True
                if not args.sem_janela:
                    if parado > 0.5:
                        cv2.imshow("FarmTech | Ir Alem", tela_espera(ultima_tela, parado))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            if em_espera:
                print(f"Quadros retomados após {time.time() - ultimo_quadro:.1f} s")
                em_espera = False
            ultimo_quadro = time.time()

            ms_inf = None
            if modelo:                                       # BASE: a inferência e o desenho acontecem aqui
                t0 = time.perf_counter()
                deteccoes = modelo.detect(frame)
                ms_inf = (time.perf_counter() - t0) * 1000
                if enviador and quadro:                      # coordenadas 0-1: valem para qualquer tela
                    altura, largura = frame.shape[:2]
                    enviador.enviar({"quadro": quadro, "largura": largura, "altura": altura,
                                     "latencia_inferencia_ms": round(ms_inf, 1),
                                     "deteccoes": normalizar(deteccoes, largura, altura)})
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
                sessao, f"{agora:.3f}", n, modo, execucao, round(fps, 2),
                round(fim_a_fim, 1) if fim_a_fim is not None else "",
                round(rtt, 1) if rtt is not None else "",
                round(ms_inf, 1) if ms_inf is not None else "",
                len(deteccoes), "|".join(sorted({d["classe"] for d in deteccoes})), int(bool(criticas)),
            ])

            if not args.sem_janela:
                painel(frame, [
                    f"modo {modo}  |  {fps:4.1f} FPS  |  {execucao}",
                    f"fim a fim: {fim_a_fim:.0f} ms" if fim_a_fim is not None else "fim a fim: n/d",
                    f"inferencia: {ms_inf:.0f} ms" if ms_inf is not None else "inferencia: n/d",
                    f"RTT rede: {rtt:.0f} ms" if rtt is not None else "",
                ], alerta)
                cv2.imshow("FarmTech | Ir Alem", frame)
                ultima_tela = frame
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
        if enviador:
            enviador.fechar()
        f_frames.close()
        f_alertas.close()
        if not args.sem_janela:
            cv2.destroyAllWindows()
        print(f"{n} quadros processados. Logs gravados em {args.logs}/")
        if fonte.descartados:                                # quadros que chegaram enquanto a inferência estava ocupada
            print(f"{fonte.descartados} quadros descartados para processar sempre o mais recente")
        if enviador:
            print(f"{enviador.enviados} conjuntos de detecções devolvidos ao servidor para a página /celular")


if __name__ == "__main__":
    main()
