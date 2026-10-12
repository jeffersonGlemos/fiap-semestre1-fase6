"""Pré-rotulação automática com Grounding DINO (detecção zero-shot guiada por texto).

Gera um .txt no formato YOLO para cada imagem e uma prévia com as caixas desenhadas. Essas caixas
são apenas um ponto de partida: todas passam por revisão manual no Roboflow Annotate antes do treino.

Uso:
    uv run autolabel.py brutas/pasto  saida/  --classes vaca,arvore,cerca
    uv run autolabel.py brutas/cobras saida/  --classes cobra
    uv run autolabel.py brutas/caneca saida_caneca/ --classes caneca
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")     # operações sem suporte no MPS caem para a CPU

import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

CLASSES = ["vaca", "cobra", "escorpiao", "arvore", "cerca", "caneca"]  # mesma ordem do labels.txt/data.yaml

# Prompt em inglês (idioma do pré-treino do modelo) e limiar de confiança por classe.
# Para a cerca, o prompt pede o mourão: a caixa da cerca inteira ficaria quase toda preenchida por fundo.
PROMPTS = {
    "vaca":      ("a cow.",                     0.35),
    "cobra":     ("a snake.",                   0.30),
    "escorpiao": ("a scorpion.",                0.30),
    "arvore":    ("a tree.",                    0.35),
    "cerca":     ("a wooden fence post.",       0.30),
    "caneca":    ("a mug.",                     0.35),
}
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dispositivo():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def detectar(imagem, classes, processor, model, device):
    """Roda um prompt por classe e junta tudo num sv.Detections."""
    caixas, confs, ids = [], [], []
    for nome in classes:
        prompt, limiar = PROMPTS[nome]
        entrada = processor(images=imagem, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            saida = model(**entrada)
        r = processor.post_process_grounded_object_detection(
            saida, entrada.input_ids, threshold=limiar, text_threshold=limiar,
            target_sizes=[imagem.size[::-1]],                  # (altura, largura)
        )[0]
        caixas.append(r["boxes"].cpu().numpy())
        confs.append(r["scores"].cpu().numpy())
        ids.append(np.full(len(r["scores"]), CLASSES.index(nome)))

    det = sv.Detections(
        xyxy=np.concatenate(caixas).reshape(-1, 4),
        confidence=np.concatenate(confs),
        class_id=np.concatenate(ids).astype(int),
    )
    det = det.with_nms(threshold=0.5, class_agnostic=False)   # duplicatas da mesma classe
    det = det.with_nms(threshold=0.5, class_agnostic=True)    # mesmo objeto com 2 classes: mantém a mais confiante
    area = (det.xyxy[:, 2] - det.xyxy[:, 0]) * (det.xyxy[:, 3] - det.xyxy[:, 1])
    return det[area < 0.9 * imagem.width * imagem.height]     # caixa do tamanho da imagem é ruído


def salvar_yolo(det, largura, altura, destino):
    """Escreve 'classe xc yc w h' normalizado em [0, 1]."""
    linhas = []
    for (x1, y1, x2, y2), cls in zip(det.xyxy, det.class_id):
        xc, yc = (x1 + x2) / 2 / largura, (y1 + y2) / 2 / altura
        w, h = (x2 - x1) / largura, (y2 - y1) / altura
        linhas.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    destino.write_text("\n".join(linhas) + ("\n" if linhas else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", type=Path, help="pasta com as imagens brutas")
    ap.add_argument("saida", type=Path, help="gera saida/images, saida/labels e saida/previa")
    ap.add_argument("--classes", default=",".join(CLASSES),
                    help="classes procuradas neste lote (ex.: cobra); restringir a lista reduz falsos positivos")
    ap.add_argument("--modelo", default="IDEA-Research/grounding-dino-base")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    if desconhecidas := set(classes) - set(CLASSES):
        ap.error(f"classes desconhecidas: {desconhecidas}")

    for sub in ("images", "labels", "previa"):
        (args.saida / sub).mkdir(parents=True, exist_ok=True)

    device = dispositivo()
    print(f"Carregando {args.modelo} em {device}...")
    processor = AutoProcessor.from_pretrained(args.modelo)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.modelo).to(device).eval()

    caixa_ann = sv.BoxAnnotator(thickness=3)
    rotulo_ann = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)
    arquivos = sorted(p for p in args.entrada.iterdir() if p.suffix.lower() in EXT_IMG)
    total = np.zeros(len(CLASSES), dtype=int)

    for i, caminho in enumerate(arquivos, 1):
        # exif_transpose: fotos giradas pelo celular ou pela câmera ficam na orientação correta
        imagem = ImageOps.exif_transpose(Image.open(caminho)).convert("RGB")
        det = detectar(imagem, classes, processor, model, device)

        imagem.save(args.saida / "images" / f"{caminho.stem}.jpg", quality=95)  # sem EXIF de rotação
        salvar_yolo(det, imagem.width, imagem.height, args.saida / "labels" / f"{caminho.stem}.txt")

        textos = [f"{CLASSES[c]} {p:.2f}" for c, p in zip(det.class_id, det.confidence)]
        previa = rotulo_ann.annotate(caixa_ann.annotate(imagem.copy(), det), det, labels=textos)
        previa.save(args.saida / "previa" / f"{caminho.stem}.jpg", quality=85)

        np.add.at(total, det.class_id, 1)
        print(f"[{i}/{len(arquivos)}] {caminho.name}: {', '.join(textos) or 'nenhuma detecção'}")

    print("\nCaixas por classe:", {CLASSES[i]: int(n) for i, n in enumerate(total) if n})
    print(f"Revise as prévias em {args.saida / 'previa'} antes de treinar.")


if __name__ == "__main__":
    main()
