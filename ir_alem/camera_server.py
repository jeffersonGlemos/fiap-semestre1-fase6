"""Servidor de câmera: captura as imagens e as transmite por Wi-Fi como MJPEG sobre HTTP.

Roda no dispositivo que tem a câmera (Raspberry Pi 5 ou qualquer PC com webcam). Usei apenas a
biblioteca padrão para o HTTP, sem Flask, para reduzir as dependências no Pi. Rotas expostas:
    /stream       vídeo MJPEG contínuo (abre no navegador, no VLC ou no cliente_vscode.py)
    /frame.jpg    último quadro capturado, como imagem única
    /deteccoes    últimas detecções em JSON (somente com --inferencia hailo|onnx)
    /saude        status do servidor em JSON (o cliente usa esta rota para medir o RTT da rede)

Modos:
    BASE  (--inferencia nenhuma): apenas transmite. O reconhecimento acontece no PC, com o best.pt.
    PLUS  (--inferencia hailo):   o Pi 5 detecta na NPU Hailo-8L e envia o vídeo já anotado + JSON.
          (--inferencia onnx):    plano B do PLUS, o mesmo modelo em ONNX Runtime na CPU do Pi.

A opção --classes coco permite validar o pipeline com um .hef pré-treinado no COCO enquanto o
best.hef do projeto ainda não foi compilado.

Uso:
    python camera_server.py --fonte picamera2                                  # Pi, modo BASE
    python camera_server.py --fonte 0 --porta 8000                             # webcam do PC
    python camera_server.py --fonte demo.mp4 --loop                            # vídeo (testes)
    python camera_server.py --fonte picamera2 --inferencia hailo --modelo best.hef
    python camera_server.py --fonte picamera2 --inferencia onnx  --modelo best.onnx
    python camera_server.py --fonte 0 --inferencia onnx --modelo best.onnx --provedor coreml   # PLUS simulado no Mac
    python camera_server.py --fonte teste.mp4 --loop --inferencia hailo \
        --modelo /usr/share/hailo-models/yolov8s_h8l.hef --classes coco   # teste sem best.hef
"""

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from backends import anotar, carregar_backend, resolver_classes

FRONTEIRA = "quadro"                                         # delimitador entre as partes do MJPEG


class Camera:
    """Esconde as diferenças entre picamera2 (Pi) e OpenCV (webcam/arquivo) atrás de um único `ler()`."""

    def __init__(self, fonte, largura, altura, loop):
        self.fonte, self.loop, self.picam = fonte, loop, None
        if fonte == "picamera2":
            from picamera2 import Picamera2                  # import tardio: o pacote só existe no Pi
            self.picam = Picamera2()
            # No picamera2, "RGB888" já entrega os bytes em ordem BGR, a mesma usada pelo OpenCV
            self.picam.configure(self.picam.create_video_configuration(
                main={"size": (largura, altura), "format": "RGB888"}))
            self.picam.start()
            time.sleep(2)                                    # tempo para a exposição e o balanço de branco convergirem
        else:
            origem = int(fonte) if fonte.isdigit() else fonte  # "0" indica webcam; o resto é arquivo/URL
            self.cap = cv2.VideoCapture(origem)
            if not self.cap.isOpened():
                raise SystemExit(f"Não foi possível abrir a fonte {fonte!r}.")
            if isinstance(origem, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    def ler(self):
        if self.picam:
            return self.picam.capture_array()
        ok, frame = self.cap.read()
        if not ok and self.loop:                             # fim do arquivo: volta ao primeiro quadro
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None


class Estado:
    """Último quadro e últimas detecções, compartilhados entre a thread de captura e as conexões HTTP."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.numero, self.ts = None, 0, 0.0
        self.deteccoes = {"timestamp": 0.0, "frame": 0, "latencia_inferencia_ms": None, "deteccoes": []}
        self.fps = 0.0

    def publicar(self, jpeg, deteccoes):
        with self.cond:
            self.jpeg, self.numero, self.ts = jpeg, self.numero + 1, time.time()
            if deteccoes is not None:
                self.deteccoes = deteccoes
            self.cond.notify_all()                           # libera todas as conexões que aguardam em /stream


def capturar(camera, estado, backend, nome_backend, qualidade, fps_max, parar):
    """Laço de captura: lê o quadro, infere e anota (no modo PLUS), codifica em JPEG e publica."""
    intervalo = 1.0 / fps_max if fps_max else 0
    janela, t_janela = 0, time.time()
    while not parar.is_set():
        t0 = time.time()
        frame = camera.ler()
        if frame is None:
            print("A fonte parou de fornecer quadros; captura encerrada.")
            parar.set()
            break

        deteccoes = None
        if backend:
            ti = time.perf_counter()
            lista = backend.detect(frame)
            ms = (time.perf_counter() - ti) * 1000
            anotar(frame, lista)
            deteccoes = {"timestamp": time.time(), "frame": estado.numero + 1, "backend": nome_backend,
                         "latencia_inferencia_ms": round(ms, 2), "deteccoes": lista}

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
        if ok:
            estado.publicar(jpeg.tobytes(), deteccoes)

        janela += 1
        if time.time() - t_janela >= 1:                      # FPS de captura calculado em janelas de 1 s
            estado.fps, janela, t_janela = janela / (time.time() - t_janela), 0, time.time()
        if intervalo:
            time.sleep(max(0, intervalo - (time.time() - t0)))


def criar_handler(estado, info):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):                        # desliga o log padrão de cada requisição
            pass

        def _json(self, obj):
            corpo = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self):
            rota = self.path.split("?")[0]
            if rota == "/saude":
                self._json({**info, "timestamp": time.time(), "fps_captura": round(estado.fps, 1),
                            "quadros": estado.numero})
            elif rota == "/deteccoes":
                with estado.cond:
                    self._json(estado.deteccoes)
            elif rota == "/frame.jpg":
                with estado.cond:
                    estado.cond.wait_for(lambda: estado.jpeg is not None, timeout=5)
                    jpeg = estado.jpeg
                if jpeg is None:
                    self.send_error(503, "a câmera ainda não produziu quadros")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif rota == "/stream":
                self._stream()
            else:
                self.send_error(404, "rotas disponíveis: /stream /frame.jpg /deteccoes /saude")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={FRONTEIRA}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            visto = 0
            try:
                while True:
                    with estado.cond:                        # aguarda um quadro novo para não reenviar o mesmo
                        estado.cond.wait_for(lambda: estado.numero != visto, timeout=5)
                        jpeg, visto, ts = estado.jpeg, estado.numero, estado.ts
                    if jpeg is None:
                        continue
                    # O X-Timestamp (epoch) permite ao cliente medir a latência da captura até a tela
                    self.wfile.write(
                        f"--{FRONTEIRA}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\nX-Timestamp: {ts:.6f}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):  # o cliente fechou a conexão; caso esperado
                pass

    return Handler


def ip_local():
    """IP da interface Wi-Fi/LAN. O connect em UDP não envia pacotes, apenas faz o sistema escolher a rota."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", default="picamera2", help="picamera2, índice da webcam (0) ou arquivo de vídeo")
    ap.add_argument("--porta", type=int, default=8000)
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--altura", type=int, default=720)
    ap.add_argument("--qualidade", type=int, default=80, help="qualidade do JPEG (0-100)")
    ap.add_argument("--fps-max", type=float, default=30, help="limite de FPS da captura (0 = sem limite)")
    ap.add_argument("--loop", action="store_true", help="reinicia o arquivo de vídeo quando ele termina")
    ap.add_argument("--inferencia", choices=["nenhuma", "hailo", "onnx"], default="nenhuma")
    ap.add_argument("--modelo", type=Path, help="best.hef (hailo) ou best.onnx (onnx)")
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima para manter uma detecção")
    ap.add_argument("--provedor", default="auto", choices=["auto", "cpu", "coreml", "cuda", "directml"],
                    help="execution provider do ONNX Runtime (só com --inferencia onnx; no Pi, auto resulta em cpu)")
    ap.add_argument("--classes", default="fase6",
                    help="fase6 (padrão, best.hef/onnx), coco (modelos pré-treinados, para teste) ou arquivo .txt")
    args = ap.parse_args()

    backend = None
    if args.inferencia != "nenhuma":
        if not args.modelo or not args.modelo.exists():
            ap.error(f"--inferencia {args.inferencia} exige um --modelo que exista")
        backend = carregar_backend(args.inferencia, args.modelo, conf=args.conf,
                                   classes=resolver_classes(args.classes), provedor=args.provedor)

    camera = Camera(args.fonte, args.largura, args.altura, args.loop)
    estado, parar = Estado(), threading.Event()
    threading.Thread(target=capturar, daemon=True,
                     args=(camera, estado, backend, args.inferencia, args.qualidade, args.fps_max, parar)).start()

    info = {"servidor": "farmtech-camera", "fonte": args.fonte, "inferencia": args.inferencia,
            "modelo": args.modelo.name if args.modelo else None,
            "provedor": getattr(backend, "provedor", None)}  # execution provider do ONNX; o cliente grava no log
    servidor = ThreadingHTTPServer(("0.0.0.0", args.porta), criar_handler(estado, info))
    servidor.daemon_threads = True                           # conexões abertas não impedem o encerramento
    ip = ip_local()
    extra = f", {info['provedor']}" if info["provedor"] else ""
    print(f"Transmitindo em http://{ip}:{args.porta}/stream  (inferência: {args.inferencia}{extra})")
    print(f"No PC:  python cliente_vscode.py --fonte http://{ip}:{args.porta}/stream")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    finally:
        parar.set()
        servidor.server_close()


if __name__ == "__main__":
    main()
