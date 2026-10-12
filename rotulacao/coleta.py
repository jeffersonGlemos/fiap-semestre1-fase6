"""Coleta de fotos em campo para ampliar o lote A (cenas de pasto), executada no Raspberry Pi 5.

As fotos atuais do lote A vêm do Wikimedia Commons (baixar_commons.py); este script fica pronto
para acrescentar fotos próprias, tiradas com a câmera que o rover usaria.

O script tira uma foto a cada N segundos durante uma caminhada pelo pasto. Todas as fotos de uma
execução (uma caminhada) recebem o mesmo carimbo de início, para que o split.py as mantenha juntas:

    pasto_20261012_071530_000.jpg, pasto_20261012_071530_001.jpg, ...
          └─ data ─┘ └hora┘ └ nº ┘

`split.py --grupo "pasto_\\d{8}_\\d{6}"` agrupa por caminhada; `"pasto_\\d{8}_\\d{2}"` agrupa
por data e hora de início (caminhadas iniciadas na mesma hora caem no mesmo grupo).

Uso:
    uv run coleta.py brutas/pasto                       # Pi + picamera2, 1 foto a cada 3 s
    uv run coleta.py brutas/pasto --intervalo 5 --max 120
    uv run coleta.py teste/ --webcam --max 5            # testa no PC com a webcam (OpenCV)

No Pi, o picamera2 vem do apt (python3-picamera2), e não do PyPI. Por isso o script deve rodar
com o Python do sistema (`python3 coleta.py brutas/pasto`) ou num venv criado com
`uv venv --system-site-packages`.
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

LARGURA, ALTURA = 1920, 1080                                  # resolução do deploy (Camera Module)


def camera_pi():
    """Abre a câmera do Pi e devolve uma função que salva um quadro no caminho dado."""
    try:
        from picamera2 import Picamera2                       # só existe no Raspberry Pi OS
    except ImportError:
        sys.exit("picamera2 não encontrado: execute no Raspberry Pi (sudo apt install python3-picamera2) "
                 "ou use --webcam para testar no PC.")
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (LARGURA, ALTURA)}))
    cam.start()
    time.sleep(2)                                             # espera a exposição e o balanço de branco (AE/AWB) estabilizarem
    return lambda caminho: cam.capture_file(str(caminho)), cam.stop


def camera_webcam(indice):
    """Alternativa para testar o script no PC com a webcam, via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        sys.exit(f"Não foi possível abrir a webcam {indice}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
    time.sleep(2)                                             # mesma espera de exposição do Pi

    def salvar(caminho):
        ok, quadro = cap.read()
        if not ok:
            raise RuntimeError("falha ao ler quadro da webcam")
        cv2.imwrite(str(caminho), quadro, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return salvar, cap.release


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("saida", type=Path, help="pasta onde as fotos serão gravadas")
    ap.add_argument("--intervalo", type=float, default=3.0, help="segundos entre fotos (padrão 3)")
    ap.add_argument("--max", type=int, default=200, help="quantidade máxima de fotos (padrão 200)")
    ap.add_argument("--webcam", type=int, nargs="?", const=0, metavar="INDICE",
                    help="usa a webcam do PC (OpenCV) em vez da câmera do Pi")
    args = ap.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    salvar, fechar = camera_webcam(args.webcam) if args.webcam is not None else camera_pi()
    inicio = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # carimbo único da caminhada

    print(f"Caminhada {inicio}: até {args.max} fotos, 1 a cada {args.intervalo:g} s. Ctrl+C para parar.")
    tiradas = 0
    try:
        for i in range(args.max):
            caminho = args.saida / f"pasto_{inicio}_{i:03d}.jpg"
            salvar(caminho)
            tiradas += 1
            print(caminho.name)
            time.sleep(args.intervalo)
    except KeyboardInterrupt:                                 # Ctrl+C é a forma normal de encerrar a caminhada
        print()
    finally:
        fechar()
    print(f"{tiradas} fotos em {args.saida}. Antes de rotular, descarte as ruins e mantenha as mais variadas.")


if __name__ == "__main__":
    main()
