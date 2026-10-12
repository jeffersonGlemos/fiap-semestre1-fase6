"""Backends de inferência embarcada (modo PLUS), todos com a mesma interface.

Cada backend expõe `detect(frame_bgr) -> list[dict]`, e cada dict tem o formato:
    {"classe": str, "conf": float, "bbox": [x1, y1, x2, y2]}   # em pixels do frame original
"""

import cv2

# Mesma ordem do data.yaml do treino. A caneca entrou por último para não renumerar as cinco classes
# anteriores; ela é o objeto que levo para a demonstração ao vivo, já que não há vaca nem cobra à mão.
CLASSES = ["vaca", "cobra", "escorpiao", "arvore", "cerca", "caneca"]
CRITICAS = {"cobra", "escorpiao"}                            # classes que disparam alerta (a caneca não dispara)

# As 80 classes do COCO, na ordem usada pelos modelos pré-treinados (yolov5s.pt, .hef do Model Zoo).
# Usei esta lista para validar o pipeline antes de ter o best.pt/best.hef do projeto compilado.
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def resolver_classes(valor):
    """None ou "fase6" → as 6 classes do projeto; "coco" → as 80 do COCO; caminho .txt → um nome por linha."""
    if valor in (None, "fase6"):
        return CLASSES
    if valor == "coco":
        return COCO
    return [l.strip() for l in open(valor, encoding="utf-8") if l.strip()]

# Cor BGR de cada classe: as críticas em vermelho/laranja para se destacarem, as demais em tons frios
CORES = {
    "vaca": (80, 200, 80),
    "cobra": (0, 0, 255),
    "escorpiao": (0, 140, 255),
    "arvore": (60, 120, 30),
    "cerca": (200, 160, 40),
    "caneca": (200, 80, 200),
}


def carregar_backend(nome, modelo, conf=0.25, classes=CLASSES, provedor="auto"):
    """Importa o backend apenas quando ele é pedido, para que um Pi sem Hailo não precise do hailo_platform.

    `provedor` só vale para o ONNX (auto, cpu, coreml, cuda ou directml); o Hailo sempre usa a NPU.
    """
    if nome == "onnx":
        from .onnx_backend import OnnxBackend
        return OnnxBackend(modelo, conf=conf, classes=classes, provedor=provedor)
    if nome == "hailo":
        from .hailo_backend import HailoBackend
        return HailoBackend(modelo, conf=conf, classes=classes)
    raise ValueError(f"backend desconhecido: {nome}")


def anotar(frame, deteccoes):
    """Desenha as caixas e os rótulos diretamente no frame e devolve o próprio frame."""
    for d in deteccoes:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cor = CORES.get(d["classe"], (255, 255, 255))
        espessura = 3 if d["classe"] in CRITICAS else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, espessura)
        texto = f'{d["classe"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), cor, -1)  # fundo sólido atrás do rótulo
        cv2.putText(frame, texto, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame
