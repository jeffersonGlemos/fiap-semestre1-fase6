"""Simula a página /celular sem celular: lê um vídeo e envia os quadros ao camera_server.py por POST /quadro.

Usei este script durante o desenvolvimento da fonte "navegador" para validar, no próprio Mac, o mesmo
protocolo que a página celular.html executa no Safari:
    - envio sequencial: o próximo quadro só sai depois da resposta do anterior, no ritmo de --fps;
    - JPEG redimensionado para --largura x --altura com a qualidade de --qualidade (padrões da página);
    - relógio do "celular" deliberadamente deslocado de --desvio-ms, com o offset estimado pelo método
      de Cristian (várias leituras de GET /saude e a amostra de menor RTT), refeito a cada 30 s;
    - leitura das detecções normalizadas que voltam na resposta do POST /quadro, como a página faz para
      desenhar as caixas sobre a prévia.

No fim, o script imprime um resumo em JSON: FPS enviado, FPS de detecções novas recebidas, RTT do POST,
idade das caixas ao chegar (no relógio do "celular", já corrigido), quantidade de caixas fora do
intervalo 0-1 e as classes mais frequentes. Com --amostras, salva alguns quadros enviados com as caixas
normalizadas redesenhadas por cima, para conferir visualmente se elas coincidem com os objetos.
Com --pausa-em e --pausa-por, o "celular" para de enviar por um tempo, o que permite observar o
keepalive do /stream e a recuperação do cliente_vscode.py.

Os números deste script vêm do próprio Mac, sem o trecho de Wi-Fi entre o celular e o servidor; ele
valida o protocolo, mas não substitui a medição com o celular real (seção 7.1 do notebook).

Uso (dentro de ir_alem/, com o servidor já iniciado em --fonte navegador):
    python camera_server.py --fonte navegador --porta 8443
    python ferramentas/simula_celular.py --base https://localhost:8443 --video teste.mp4
    python ferramentas/simula_celular.py --base https://localhost:8443 --video teste.mp4 --pausa-em 15 --pausa-por 10
Depende de opencv-python e requests (requirements-pc.txt). O certificado autoassinado do servidor é
aceito sem verificação, como faz o --inseguro do cliente.
"""

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import cv2
import requests
import urllib3

RESSINCRONIZAR_S = 30                                        # mesmo intervalo de ressincronização da página


def argumentos():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="https://localhost:8000", help="URL do camera_server.py em --fonte navegador")
    ap.add_argument("--video", required=True, help="vídeo que faz o papel da câmera do celular")
    ap.add_argument("--fps", type=float, default=15, help="quadros por segundo enviados (padrão da página: 15)")
    ap.add_argument("--segundos", type=float, default=40, help="duração do envio")
    ap.add_argument("--desvio-ms", type=float, default=2500,
                    help="quanto o relógio do 'celular' está adiantado em relação ao servidor (ms)")
    ap.add_argument("--qualidade", type=int, default=70, help="qualidade do JPEG, 0-100 (a página usa 0,7)")
    ap.add_argument("--largura", type=int, default=640)
    ap.add_argument("--altura", type=int, default=480)
    ap.add_argument("--pausa-em", type=float, default=0, help="segundos após o início em que o envio para")
    ap.add_argument("--pausa-por", type=float, default=0, help="duração da pausa em segundos (0 = sem pausa)")
    ap.add_argument("--amostras", type=Path, help="pasta para salvar quadros com as caixas devolvidas desenhadas")
    ap.add_argument("--json", type=Path, help="arquivo em que o resumo também é gravado")
    return ap.parse_args()


def percentil(valores, p):
    return sorted(valores)[min(len(valores) - 1, int(p * len(valores)))] if valores else None


def main():
    a = argumentos()
    urllib3.disable_warnings()                               # certificado autoassinado: o aviso repetiria a cada POST
    sessao = requests.Session()
    sessao.verify = False

    def agora_celular():
        return time.time() * 1000 + a.desvio_ms              # relógio do "celular", em ms, fora de sincronia

    def sincronizar(n=5):
        """Método de Cristian: o horário do servidor corresponde ao ponto médio da requisição de menor RTT."""
        melhor = None
        for _ in range(n):
            t0 = agora_celular()
            saude = sessao.get(a.base + "/saude", timeout=3).json()
            t1 = agora_celular()
            if melhor is None or t1 - t0 < melhor[0]:
                melhor = (t1 - t0, saude["timestamp"] * 1000 - (t0 + t1) / 2)
        return melhor[1], melhor[0] / 2                      # offset e incerteza (±RTT/2), em ms

    offset, incerteza = sincronizar()
    print(f"offset estimado {offset:.1f} ms (esperado {-a.desvio_ms:.0f} ms) ±{incerteza:.1f} ms")
    if a.amostras:
        a.amostras.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir o vídeo {a.video!r}.")
    enviados = collections.OrderedDict()                     # número do quadro no servidor → imagem enviada
    rtts, idades, n, ultima_sinc = [], [], 0, time.time()
    ultimo_det, n_det, invalidas, caixas = 0, 0, 0, 0
    classes = collections.Counter()
    por_segundo = collections.defaultdict(lambda: [0, 0])    # segundo → [quadros enviados, detecções novas]
    salvas, pausou, resposta = 0, False, {}
    inicio = time.time()
    fim = inicio + a.segundos

    while time.time() < fim:
        t_ini = time.time()
        if a.pausa_por and not pausou and t_ini - inicio >= a.pausa_em:
            print(f"[{t_ini - inicio:5.1f} s] o 'celular' para de enviar por {a.pausa_por:.0f} s")
            time.sleep(a.pausa_por)
            pausou = True
            print(f"[{time.time() - inicio:5.1f} s] o 'celular' volta a enviar")
            continue
        ok, quadro = cap.read()
        if not ok:                                           # fim do vídeo: recomeça, como uma câmera contínua
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        quadro = cv2.resize(quadro, (a.largura, a.altura))
        ts_captura = agora_celular()                         # a página registra o instante do desenho no canvas
        jpeg = cv2.imencode(".jpg", quadro, [cv2.IMWRITE_JPEG_QUALITY, a.qualidade])[1].tobytes()

        t0 = time.perf_counter()
        r = sessao.post(a.base + "/quadro", data=jpeg, timeout=5, headers={
            "Content-Type": "image/jpeg", "X-Timestamp": f"{ts_captura:.0f}",
            "X-Relogio-Offset": f"{offset:.1f}", "X-Relogio-Incerteza": f"{incerteza:.1f}"})
        r.raise_for_status()
        rtts.append((time.perf_counter() - t0) * 1000)
        n += 1
        resposta = r.json()
        segundo = int(time.time() - inicio)
        por_segundo[segundo][0] += 1
        enviados[resposta["quadro"]] = quadro
        while len(enviados) > 60:                            # só os quadros recentes podem receber caixas
            enviados.popitem(last=False)

        # As detecções vêm de carona na resposta; conto apenas as de um quadro analisado ainda não visto
        det = resposta.get("deteccoes") or {}
        q = det.get("quadro") or 0
        if q and q != ultimo_det:
            ultimo_det, n_det = q, n_det + 1
            por_segundo[segundo][1] += 1
            if det.get("ts_celular"):                        # idade da caixa no relógio do "celular"
                idades.append(agora_celular() - det["ts_celular"])
            for d in det.get("deteccoes", []):
                caixas += 1
                x1, y1, x2, y2 = d["bbox"]
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    invalidas += 1
                classes[d["classe"]] += 1
            if a.amostras and det.get("deteccoes") and q in enviados and salvas < 12 and q % 15 == 0:
                img = enviados[q].copy()                     # o mesmo quadro que o modelo analisou
                h, w = img.shape[:2]
                for d in det["deteccoes"]:
                    x1, y1, x2, y2 = d["bbox"]
                    cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 2)
                    cv2.putText(img, f'{d["classe"]} {d["conf"]:.2f}', (int(x1 * w), max(12, int(y1 * h) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imwrite(str(a.amostras / f"q{q}_{det.get('origem')}.jpg"), img)
                salvas += 1

        if time.time() - ultima_sinc > RESSINCRONIZAR_S:
            offset, incerteza = sincronizar(3)
            ultima_sinc = time.time()
        time.sleep(max(0, 1 / a.fps - (time.time() - t_ini)))

    if not rtts:
        raise SystemExit("Nenhum quadro foi enviado.")
    duracao = a.segundos - (a.pausa_por if pausou else 0)
    completos = [por_segundo[k] for k in sorted(por_segundo)][1:-1]   # descarta o 1º e o último segundo, incompletos
    fps_env = [e for e, _ in completos if e]
    fps_det = [d for e, d in completos if e]
    resumo = {
        "enviados": n, "fps_enviado_medio": round(n / duracao, 2),
        "fps_enviado_mediana_por_s": statistics.median(fps_env) if fps_env else None,
        "deteccoes_novas": n_det, "fps_deteccoes_medio": round(n_det / duracao, 2),
        "fps_deteccoes_mediana_por_s": statistics.median(fps_det) if fps_det else None,
        "rtt_post_mediana_ms": round(statistics.median(rtts), 1), "rtt_post_p95_ms": round(percentil(rtts, .95), 1),
        "idade_caixas_mediana_ms": round(statistics.median(idades), 1) if idades else None,
        "idade_caixas_p95_ms": round(percentil(idades, .95), 1) if idades else None,
        "caixas": caixas, "caixas_invalidas": invalidas, "classes": dict(classes.most_common(8)),
        "ultima_resposta": resposta.get("deteccoes"),
    }
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    print("saude:", sessao.get(a.base + "/saude", timeout=3).json())
    if a.json:
        a.json.write_text(json.dumps(resumo, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
