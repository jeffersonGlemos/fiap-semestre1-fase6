"""Servidor de câmera: captura as imagens e as transmite por Wi-Fi como MJPEG sobre HTTP.

Roda no dispositivo que tem a câmera (Raspberry Pi 5 ou qualquer PC com webcam). Usei apenas a
biblioteca padrão para o HTTP, sem Flask, para reduzir as dependências no Pi. Rotas expostas:
    /stream       vídeo MJPEG contínuo (abre no navegador, no VLC ou no cliente_vscode.py)
    /frame.jpg    último quadro capturado, como imagem única
    /deteccoes    últimas detecções em JSON, em pixels (somente com --inferencia hailo|onnx);
                  com ?formato=normalizado, em coordenadas 0-1 (em qualquer modo)
    /saude        status do servidor em JSON (o cliente usa esta rota para medir o RTT da rede)
Com --fonte navegador, também:
    /celular      página que transforma o navegador do celular em câmera (getUserMedia)
    /quadro       POST com um JPEG capturado pela página; cabeçalho X-Timestamp em ms. A resposta já
                  traz as detecções mais recentes em coordenadas normalizadas, que a página desenha
                  sobre a própria prévia (sem requisição extra e sem reenviar vídeo ao celular)
    /deteccoes    POST do cliente_vscode.py no modo BASE com as detecções normalizadas de cada quadro
    /qr.svg       QR code com a URL de /celular (exige o pacote segno)

No /stream, quando não há quadro novo por KEEPALIVE segundos (o celular parou de enviar ou a câmera
ainda não começou), o servidor reenvia o último quadro, ou um aviso de "aguardando câmera", marcado
com o cabeçalho X-Keepalive. A conexão continua ativa, e o cliente não estoura o timeout de leitura.

Modos:
    BASE  (--inferencia nenhuma): apenas transmite. O reconhecimento acontece no PC, com o best.pt.
    PLUS  (--inferencia hailo):   o Pi 5 detecta na NPU Hailo-8L e envia o vídeo já anotado + JSON.
          (--inferencia onnx):    plano B do PLUS, o mesmo modelo em ONNX Runtime na CPU do Pi.

A opção --classes coco permite validar o pipeline com um .hef pré-treinado no COCO enquanto o
best.hef do projeto ainda não foi compilado.

Fonte "navegador" (celular sem app): o celular abre https://<IP>:<porta>/celular no Safari ou no
Chrome, e a página envia os quadros da câmera traseira por POST /quadro. Escolhi esse caminho porque
o iPhone usado na demonstração não permite instalar um app de câmera IP. Os navegadores só liberam a
câmera em contexto seguro (HTTPS), por isso o --https fica ligado por padrão nessa fonte: na primeira
execução, o servidor gera um certificado autoassinado em certs/ (com o IP local nas SANs), usando o
pacote cryptography ou, na falta dele, o openssl. Por ser autoassinado, o navegador exibe um aviso
na primeira visita. No Safari do iPhone: "Mostrar Detalhes" > "visitar este site" > "Visitar Site";
no Chrome do Android: "Avançado" > "Ir para <IP> (não seguro)". O aviso volta a aparecer quando o IP
muda (outra rede gera outro certificado). O terminal mostra um QR code com a URL, para abrir pela
câmera nativa do celular sem digitar o IP.

Uso:
    python camera_server.py --fonte picamera2                                  # Pi, modo BASE
    python camera_server.py --fonte 0 --porta 8000                             # webcam do PC
    python camera_server.py --fonte demo.mp4 --loop                            # vídeo (testes)
    python camera_server.py --fonte navegador                                  # celular pelo navegador (HTTPS)
    python camera_server.py --fonte picamera2 --inferencia hailo --modelo best.hef
    python camera_server.py --fonte picamera2 --inferencia onnx  --modelo best.onnx
    python camera_server.py --fonte 0 --inferencia onnx --modelo best.onnx --provedor coreml   # PLUS simulado no Mac
    python camera_server.py --fonte navegador --inferencia onnx --modelo best.onnx             # celular + PLUS
    python camera_server.py --fonte teste.mp4 --loop --inferencia hailo \
        --modelo /usr/share/hailo-models/yolov8s_h8l.hef --classes coco   # teste sem best.hef
"""

import argparse
import json
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from backends import CRITICAS, anotar, carregar_backend, cores_css, normalizar, resolver_classes

FRONTEIRA = "quadro"                                         # delimitador entre as partes do MJPEG
PASTA = Path(__file__).resolve().parent
PAGINA_CELULAR = PASTA / "celular.html"
PASTA_CERTS = PASTA / "certs"                                # fora do git (ir_alem/.gitignore)
MAX_QUADRO = 8 * 1024 * 1024                                 # limite do corpo do POST /quadro (bytes)
MAX_DETECCOES = 64 * 1024                                    # limite do corpo do POST /deteccoes (bytes)
KEEPALIVE = 0.5                                              # s sem quadro novo até reenviar o último no /stream (2 Hz)
PAUSA_CELULAR = 2.0                                          # s sem POST /quadro para considerar o celular desconectado


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


class FonteNavegador:
    """Câmera do celular: os quadros chegam por POST /quadro e ficam aqui até o laço de captura lê-los.

    Guardo apenas o quadro mais recente. Se a inferência for mais lenta do que o envio do celular,
    os quadros intermediários são descartados, em vez de formar uma fila que aumentaria a latência.
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.numero, self.visto = None, 0, 0
        self.ts_captura = None                               # epoch (s) da captura, já no relógio do servidor
        self.lido = (None, None)                             # (número, ts_captura) do último quadro entregue à captura
        # Número do quadro -> X-Timestamp original (relógio do celular, ms). A página usa esse instante
        # para saber a idade das caixas que recebe, sem depender da correção de relógio
        self.ts_celular = {}
        self.ultimo_post, self.ausente, self.origem = None, False, None
        self.relogio = {"offset_ms": None, "incerteza_ms": None, "latencia_envio_ms": None, "origem": None}
        threading.Thread(target=self._vigiar, daemon=True).start()

    def receber(self, jpeg, ts_celular_ms, offset_ms, incerteza_ms, origem):
        recebido = time.time()
        if ts_celular_ms is None:                            # sem X-Timestamp: o instante da chegada é a melhor estimativa
            ts = recebido
        else:
            # O offset (relógio do servidor - relógio do celular) é estimado pela página com GET /saude
            ts = (ts_celular_ms + (offset_ms or 0)) / 1000
        anterior = self.relogio["offset_ms"]
        if offset_ms is not None and (anterior is None or abs(offset_ms - anterior) > 50):
            print(f"Celular {origem}: relógio {offset_ms:+.0f} ms em relação ao servidor (±{incerteza_ms or 0:.0f} ms)")
        with self.cond:
            if self.ultimo_post is None:
                aviso = f"Celular {origem} conectado; recebendo quadros"
            elif self.ausente:
                aviso = f"Celular {origem} voltou a enviar quadros após {recebido - self.ultimo_post:.1f} s de pausa"
            else:
                aviso = None
            self.ultimo_post, self.ausente, self.origem = recebido, False, origem
            self.jpeg, self.numero, self.ts_captura = jpeg, self.numero + 1, ts
            self.ts_celular[self.numero] = ts_celular_ms
            self.ts_celular.pop(self.numero - 300, None)     # guarda só os últimos quadros (cerca de 10 s a 30 fps)
            self.relogio = {"offset_ms": offset_ms, "incerteza_ms": incerteza_ms,
                            "latencia_envio_ms": round((recebido - ts) * 1000, 1), "origem": origem}
            self.cond.notify_all()
            numero = self.numero
        if aviso:
            print(aviso)
        return numero

    def ts_do_quadro(self, numero):
        with self.cond:
            return self.ts_celular.get(numero)

    def conectado(self):
        return self.ultimo_post is not None and not self.ausente

    def _vigiar(self):
        """Registra uma única vez no terminal quando o celular para de enviar (aba fechada, tela bloqueada)."""
        while True:
            time.sleep(0.5)
            with self.cond:
                parou = (self.ultimo_post is not None and not self.ausente
                         and time.time() - self.ultimo_post > PAUSA_CELULAR)
                if parou:
                    self.ausente = True
            if parou:
                print(f"Celular {self.origem} parou de enviar quadros (nenhum há {PAUSA_CELULAR:.0f} s); "
                      "o /stream segue com keepalive até ele voltar")

    def ler_jpeg(self):
        """Bloqueia até chegar um quadro novo e devolve os bytes JPEG como vieram do celular."""
        with self.cond:
            self.cond.wait_for(lambda: self.numero != self.visto)
            self.visto = self.numero
            self.lido = (self.numero, self.ts_captura)
            return self.jpeg

    def ler(self):
        while True:                                          # um JPEG corrompido é ignorado, sem encerrar a captura
            frame = cv2.imdecode(np.frombuffer(self.ler_jpeg(), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame


class Estado:
    """Último quadro e últimas detecções, compartilhados entre a thread de captura e as conexões HTTP."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.numero, self.ts, self.publicado = None, 0, 0.0, 0.0
        self.quadro_fonte = None                             # número do quadro na fonte navegador (X-Quadro no /stream)
        self.deteccoes = {"timestamp": 0.0, "frame": 0, "latencia_inferencia_ms": None, "deteccoes": []}
        # Últimas detecções em coordenadas 0-1, devolvidas ao celular na resposta do POST /quadro.
        # quadro e ts_celular identificam o quadro analisado; origem diz quem inferiu (cliente, onnx ou hailo)
        self.normalizadas = {"quadro": 0, "ts_celular": None, "origem": None, "largura": None, "altura": None,
                             "latencia_inferencia_ms": None, "deteccoes": []}
        self.fps = 0.0

    def publicar(self, jpeg, deteccoes, ts=None, quadro=None):
        with self.cond:
            # ts: instante da captura; na fonte navegador vem do celular, nas demais é o momento da leitura
            self.jpeg, self.numero, self.ts = jpeg, self.numero + 1, ts or time.time()
            self.publicado, self.quadro_fonte = time.time(), quadro
            if deteccoes is not None:
                self.deteccoes = deteccoes
            self.cond.notify_all()                           # libera todas as conexões que aguardam em /stream

    def registrar_normalizadas(self, quadro, **campos):
        """Guarda as detecções normalizadas, a menos que sejam de um quadro mais antigo que as atuais."""
        with self.cond:
            if quadro >= self.normalizadas["quadro"]:
                self.normalizadas = {"quadro": quadro, **campos}


def capturar(camera, estado, backend, nome_backend, qualidade, fps_max, parar):
    """Laço de captura: lê o quadro, infere e anota (no modo PLUS), codifica em JPEG e publica."""
    intervalo = 1.0 / fps_max if fps_max else 0
    janela, t_janela = 0, time.time()
    navegador = isinstance(camera, FonteNavegador)
    # Celular sem inferência: o JPEG recebido segue direto para o /stream, sem decodificar e recodificar
    repassar = navegador and not backend
    while not parar.is_set():
        t0 = time.time()
        frame = camera.ler_jpeg() if repassar else camera.ler()
        if frame is None:
            print("A fonte parou de fornecer quadros; captura encerrada.")
            parar.set()
            break
        quadro, ts = camera.lido if navegador else (None, None)

        deteccoes = None
        if backend:
            ti = time.perf_counter()
            lista = backend.detect(frame)
            ms = (time.perf_counter() - ti) * 1000
            altura, largura = frame.shape[:2]
            estado.registrar_normalizadas(
                quadro or estado.numero + 1, ts_celular=camera.ts_do_quadro(quadro) if navegador else None,
                origem=nome_backend, largura=largura, altura=altura, latencia_inferencia_ms=round(ms, 1),
                deteccoes=normalizar(lista, largura, altura))
            anotar(frame, lista)
            deteccoes = {"timestamp": time.time(), "frame": estado.numero + 1, "backend": nome_backend,
                         "latencia_inferencia_ms": round(ms, 2), "deteccoes": lista}

        if repassar:
            ok, jpeg = True, frame
        else:
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
            jpeg = jpeg.tobytes() if ok else None
        if ok:
            estado.publicar(jpeg, deteccoes, ts, quadro)

        janela += 1
        if time.time() - t_janela >= 1:                      # FPS de captura calculado em janelas de 1 s
            estado.fps, janela, t_janela = janela / (time.time() - t_janela), 0, time.time()
        if intervalo:
            time.sleep(max(0, intervalo - (time.time() - t0)))


def quadro_espera(largura=640, altura=480):
    """JPEG de aviso enviado no keepalive do /stream enquanto nenhuma câmera produziu quadros."""
    img = np.full((altura, largura, 3), (16, 18, 13), np.uint8)
    cv2.putText(img, "aguardando camera", (150, altura // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (147, 163, 155), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def validar_deteccoes(dados):
    """Confere o JSON enviado pelo cliente ao POST /deteccoes e limita cada caixa ao intervalo 0-1."""
    lista = []
    for d in dados["deteccoes"][:300]:
        caixa = [min(max(float(v), 0.0), 1.0) for v in d["bbox"]]
        if len(caixa) != 4:
            raise ValueError("bbox deve ter 4 valores")
        lista.append({"classe": str(d["classe"])[:40], "conf": round(float(d["conf"]), 3),
                      "bbox": [round(v, 4) for v in caixa]})
    return lista


def criar_handler(estado, info, navegador=None, qr_svg=None):
    espera_jpeg = quadro_espera()
    # No modo PLUS, quem detecta é o próprio servidor; as caixas enviadas por um cliente são ignoradas
    aceita_deteccoes = info["inferencia"] == "nenhuma"

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 mantém a conexão aberta entre os POSTs do celular; sem isso, cada quadro
        # pagaria um novo handshake TCP + TLS
        protocol_version = "HTTP/1.1"
        timeout = 60                                         # fecha conexões ociosas

        def log_message(self, *args):                        # desliga o log padrão de cada requisição
            pass

        def _corpo(self, corpo, tipo, extra=()):
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            for chave, valor in extra:
                self.send_header(chave, valor)
            self.end_headers()
            self.wfile.write(corpo)

        def _json(self, obj):
            self._corpo(json.dumps(obj).encode(), "application/json", [("Access-Control-Allow-Origin", "*")])

        def do_GET(self):
            partes = urlsplit(self.path)
            rota = partes.path
            if rota == "/saude":
                extra = {"celular": {**navegador.relogio, "conectado": navegador.conectado()}} if navegador else {}
                self._json({**info, **extra, "timestamp": time.time(), "fps_captura": round(estado.fps, 1),
                            "quadros": estado.numero})
            elif rota == "/deteccoes":
                normalizado = parse_qs(partes.query).get("formato") == ["normalizado"]
                with estado.cond:
                    corpo = estado.normalizadas if normalizado else estado.deteccoes
                self._json(corpo)
            elif rota == "/frame.jpg":
                with estado.cond:
                    estado.cond.wait_for(lambda: estado.jpeg is not None, timeout=5)
                    jpeg = estado.jpeg
                if jpeg is None:
                    self.send_error(503, "a câmera ainda não produziu quadros")
                    return
                self._corpo(jpeg, "image/jpeg")
            elif rota == "/stream":
                self._stream()
            elif rota in ("/", "/celular") and navegador:
                self._corpo(PAGINA_CELULAR.read_bytes(), "text/html; charset=utf-8")
            elif rota == "/qr.svg" and qr_svg:
                self._corpo(qr_svg, "image/svg+xml")
            else:
                self.send_error(404, "rotas disponíveis: /stream /frame.jpg /deteccoes /saude"
                                + (" /celular /quadro /qr.svg" if navegador else ""))

        def do_POST(self):
            rota = self.path.split("?")[0]
            if rota == "/quadro" and navegador:
                self._receber_quadro()
            elif rota == "/deteccoes" and navegador:
                self._receber_deteccoes()
            else:
                self.send_error(404)

        def _ler_corpo(self, limite):
            tamanho = int(self.headers.get("Content-Length") or 0)
            if not 0 < tamanho <= limite:
                self.send_error(413 if tamanho else 411)
                return None
            return self.rfile.read(tamanho)

        def _receber_deteccoes(self):
            corpo = self._ler_corpo(MAX_DETECCOES)
            if corpo is None:
                return
            try:
                dados = json.loads(corpo)
                quadro, lista = int(dados["quadro"]), validar_deteccoes(dados)
            except (ValueError, KeyError, TypeError) as erro:
                self.send_error(400, f"JSON inválido: {erro}")
                return
            if aceita_deteccoes:
                estado.registrar_normalizadas(
                    quadro, ts_celular=navegador.ts_do_quadro(quadro), origem="cliente",
                    largura=dados.get("largura"), altura=dados.get("altura"),
                    latencia_inferencia_ms=dados.get("latencia_inferencia_ms"), deteccoes=lista)
            self._json({"ok": aceita_deteccoes})

        def _receber_quadro(self):
            jpeg = self._ler_corpo(MAX_QUADRO)
            if jpeg is None:
                return
            if not jpeg.startswith(b"\xff\xd8"):            # marcador SOI: o corpo precisa ser um JPEG
                self.send_error(415, "o corpo deve ser image/jpeg")
                return

            def numero(chave):
                try:
                    return float(self.headers[chave])
                except (TypeError, ValueError):
                    return None

            n = navegador.receber(jpeg, numero("X-Timestamp"), numero("X-Relogio-Offset"),
                                  numero("X-Relogio-Incerteza"), self.client_address[0])
            with estado.cond:
                deteccoes = estado.normalizadas
            # As detecções mais recentes seguem de carona na resposta, sem uma requisição a mais por quadro
            self._json({"ok": True, "quadro": n, "deteccoes": deteccoes})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={FRONTEIRA}")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")         # sem Content-Length: o fim do stream é o fim da conexão
            self.end_headers()
            self.close_connection = True
            visto = 0
            try:
                while True:
                    with estado.cond:                        # aguarda um quadro novo para não reenviar o mesmo
                        novo = estado.cond.wait_for(lambda: estado.numero != visto, timeout=KEEPALIVE)
                        jpeg, visto, ts, quadro = estado.jpeg, estado.numero, estado.ts, estado.quadro_fonte
                        # Ao conectar, o último quadro pode ser antigo (celular parado): vale como keepalive
                        novo = novo and jpeg is not None and time.time() - estado.publicado < 1.0
                    if novo:
                        # O X-Timestamp (epoch) permite ao cliente medir a latência da captura até a tela; o
                        # X-Quadro identifica o quadro do celular a que o cliente devolve as detecções
                        extra = f"X-Timestamp: {ts:.6f}\r\n" + (f"X-Quadro: {quadro}\r\n" if quadro else "")
                    else:
                        # Sem quadro novo: reenvio o último (ou o aviso) marcado como keepalive, que o cliente
                        # descarta; um navegador simplesmente continua exibindo a mesma imagem
                        jpeg, extra = jpeg or espera_jpeg, "X-Keepalive: 1\r\n"
                    self.wfile.write(
                        f"--{FRONTEIRA}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n{extra}\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError, TimeoutError):  # o cliente fechou a conexão
                pass

    return Handler


class Servidor(ThreadingHTTPServer):
    """ThreadingHTTPServer com TLS opcional. O handshake é feito na thread de cada conexão, e não no
    accept(), para que um navegador que ainda não aceitou o certificado não trave as demais conexões."""

    daemon_threads = True                                    # conexões abertas não impedem o encerramento

    def __init__(self, endereco, handler, contexto=None):
        super().__init__(endereco, handler)
        self.contexto = contexto

    def finish_request(self, request, client_address):
        if not self.contexto:
            return super().finish_request(request, client_address)
        request.settimeout(10)
        try:
            tls = self.contexto.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):                      # aviso de certificado recusado: o navegador encerra o handshake
            return
        tls.settimeout(None)
        try:
            super().finish_request(tls, client_address)
        finally:
            self.shutdown_request(tls)

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, ssl.SSLError, TimeoutError)):
            return                                           # queda de conexão do celular ou do cliente; caso esperado
        super().handle_error(request, client_address)


def gerar_certificado(ip):
    """Devolve (crt, key) de um certificado autoassinado para este IP, gerando-o se ainda não existir.

    Um arquivo por IP: ao trocar de rede, o certificado antigo não serviria (a SAN não confere).
    Validade de 397 dias, RSA 2048 e extendedKeyUsage serverAuth, dentro dos limites que o iOS aceita.
    """
    crt, chave = PASTA_CERTS / f"servidor_{ip}.crt", PASTA_CERTS / f"servidor_{ip}.key"
    if crt.exists() and chave.exists() and time.time() - crt.stat().st_mtime < 390 * 86400:
        return crt, chave
    PASTA_CERTS.mkdir(exist_ok=True)
    nomes_dns = sorted({"localhost", socket.gethostname()})
    ips = sorted({ip, "127.0.0.1"})
    try:
        _certificado_cryptography(crt, chave, ips, nomes_dns)
        via = "cryptography"
    except ImportError:
        if not shutil.which("openssl"):
            raise SystemExit("Para o --https, instale o pacote cryptography (pip install cryptography) "
                             "ou o openssl; ou use --no-https (o celular não libera a câmera sem HTTPS).")
        san = ",".join([f"IP:{i}" for i in ips] + [f"DNS:{d}" for d in nomes_dns])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "397",
                        "-keyout", str(chave), "-out", str(crt), "-subj", "/CN=farmtech-camera",
                        "-addext", f"subjectAltName={san}", "-addext", "extendedKeyUsage=serverAuth",
                        "-addext", "basicConstraints=critical,CA:FALSE"],
                       check=True, capture_output=True)
        via = "openssl"
    chave.chmod(0o600)
    print(f"Certificado autoassinado gerado com {via}: {crt.relative_to(PASTA)} (SAN: {', '.join(ips + nomes_dns)})")
    return crt, chave


def _certificado_cryptography(crt, chave, ips, nomes_dns):
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    privada = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nome = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "farmtech-camera")])
    agora = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.IPAddress(ipaddress.ip_address(i)) for i in ips] + [x509.DNSName(d) for d in nomes_dns]
    cert = (x509.CertificateBuilder()
            .subject_name(nome).issuer_name(nome)            # autoassinado: emissor = sujeito
            .public_key(privada.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(agora - datetime.timedelta(minutes=5))   # tolera pequena diferença de relógio do celular
            .not_valid_after(agora + datetime.timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(privada, hashes.SHA256()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    chave.write_bytes(privada.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption()))


def preparar_qr(url):
    """Mostra a URL como QR code no terminal e devolve o SVG para a rota /qr.svg (None se faltar o segno)."""
    try:
        import segno
    except ImportError:
        print("(Para ver um QR code com essa URL aqui no terminal, instale o pacote segno: pip install segno)")
        return None
    import io
    qr = segno.make(url, error="m")
    qr.terminal(compact=True)                                # meio-blocos Unicode: cabe em um terminal pequeno
    svg = io.BytesIO()
    qr.save(svg, kind="svg", scale=8, border=2)
    return svg.getvalue()


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
    ap.add_argument("--fonte", default="picamera2",
                    help="picamera2, índice da webcam (0), arquivo de vídeo ou navegador (câmera do celular via /celular)")
    ap.add_argument("--porta", type=int, default=8000)
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--altura", type=int, default=720)
    ap.add_argument("--qualidade", type=int, default=80, help="qualidade do JPEG (0-100)")
    ap.add_argument("--fps-max", type=float, default=30, help="limite de FPS da captura (0 = sem limite)")
    ap.add_argument("--loop", action="store_true", help="reinicia o arquivo de vídeo quando ele termina")
    ap.add_argument("--https", action=argparse.BooleanOptionalAction, default=None,
                    help="serve por HTTPS com certificado autoassinado em certs/ (padrão: ligado só com --fonte "
                         "navegador, porque o celular exige contexto seguro para liberar a câmera)")
    ap.add_argument("--inferencia", choices=["nenhuma", "hailo", "onnx"], default="nenhuma")
    ap.add_argument("--modelo", type=Path, help="best.hef (hailo) ou best.onnx (onnx)")
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima para manter uma detecção")
    ap.add_argument("--provedor", default="auto", choices=["auto", "cpu", "coreml", "cuda", "directml"],
                    help="execution provider do ONNX Runtime (só com --inferencia onnx; no Pi, auto resulta em cpu)")
    ap.add_argument("--classes", default="fase6",
                    help="fase6 (padrão, best.hef/onnx), coco (modelos pré-treinados, para teste) ou arquivo .txt")
    args = ap.parse_args()
    navegador = args.fonte == "navegador"
    https = navegador if args.https is None else args.https

    backend = None
    if args.inferencia != "nenhuma":
        if not args.modelo or not args.modelo.exists():
            ap.error(f"--inferencia {args.inferencia} exige um --modelo que exista")
        backend = carregar_backend(args.inferencia, args.modelo, conf=args.conf,
                                   classes=resolver_classes(args.classes), provedor=args.provedor)

    camera = FonteNavegador() if navegador else Camera(args.fonte, args.largura, args.altura, args.loop)
    estado, parar = Estado(), threading.Event()
    threading.Thread(target=capturar, daemon=True,
                     args=(camera, estado, backend, args.inferencia, args.qualidade, args.fps_max, parar)).start()

    ip = ip_local()
    contexto = None
    if https:
        crt, chave = gerar_certificado(ip)
        contexto = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        contexto.load_cert_chain(crt, chave)
    esquema = "https" if https else "http"

    qr_svg = None
    if navegador:
        url_celular = f"{esquema}://{ip}:{args.porta}/celular"
        print(f"\nNo celular (mesma rede Wi-Fi), abra {url_celular}  ou leia o QR code:")
        qr_svg = preparar_qr(url_celular)
        if qr_svg:
            print(f"(o mesmo QR code está em {esquema}://localhost:{args.porta}/qr.svg)")
        if https:
            print("Aviso de certificado na primeira visita: no Safari, Mostrar Detalhes > visitar este site; "
                  "no Chrome, Avançado > Ir para o site.\n")

    info = {"servidor": "farmtech-camera", "fonte": args.fonte, "inferencia": args.inferencia,
            "modelo": args.modelo.name if args.modelo else None,
            "provedor": getattr(backend, "provedor", None),  # execution provider do ONNX; o cliente grava no log
            "cores": cores_css(), "criticas": sorted(CRITICAS)}   # a página /celular desenha com as mesmas cores
    servidor = Servidor(("0.0.0.0", args.porta),
                        criar_handler(estado, info, camera if navegador else None, qr_svg), contexto)
    extra = f", {info['provedor']}" if info["provedor"] else ""
    inseguro = " --inseguro" if https else ""
    print(f"Transmitindo em {esquema}://{ip}:{args.porta}/stream  (inferência: {args.inferencia}{extra})")
    print(f"No PC:  python cliente_vscode.py --fonte {esquema}://{ip}:{args.porta}/stream{inseguro}")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    finally:
        parar.set()
        servidor.server_close()


if __name__ == "__main__":
    main()
