"""Backend ONNX Runtime na CPU do Pi: plano B do Hailo, com o mesmo best.pt exportado para ONNX.

Exportação feita no Colab:  python yolov5/export.py --weights best.pt --include onnx --imgsz 640 --opset 11
"""

import numpy as np

from . import CLASSES
from .yolo_utils import letterbox, pos_processar


class OnnxBackend:
    def __init__(self, modelo, conf=0.25, iou=0.45, classes=CLASSES):
        import onnxruntime as ort                            # import tardio: só é necessário quando o ONNX é usado

        self.sessao = ort.InferenceSession(str(modelo), providers=["CPUExecutionProvider"])
        self.entrada = self.sessao.get_inputs()[0].name
        self.tamanho = self.sessao.get_inputs()[0].shape[2] or 640   # entrada no formato (1, 3, 640, 640)
        self.conf, self.iou = conf, iou
        self.classes = classes                               # nomes na mesma ordem das saídas do modelo

    def detect(self, frame):
        img, escala, pad = letterbox(frame, self.tamanho)
        tensor = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB e HWC→CHW
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0
        saida = self.sessao.run(None, {self.entrada: tensor})[0]
        return pos_processar(saida, escala, pad, frame.shape, self.classes, self.conf, self.iou)
