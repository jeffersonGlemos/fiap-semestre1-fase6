"""Pré e pós-processamento do YOLOv5 em numpy puro, compartilhado pelos backends ONNX e Hailo.

Reimplementei estas etapas para que o Pi não precise do PyTorch apenas para decodificar as saídas.
"""

import cv2
import numpy as np


def letterbox(frame, tamanho=640, cor=(114, 114, 114)):
    """Redimensiona mantendo a proporção e completa com bordas cinza, como no treino do YOLOv5.

    Devolve (imagem quadrada, escala, (pad_x, pad_y)), que servem para desfazer a transformação depois.
    """
    h, w = frame.shape[:2]
    escala = min(tamanho / h, tamanho / w)
    nw, nh = int(round(w * escala)), int(round(h * escala))
    pad_x, pad_y = (tamanho - nw) / 2, (tamanho - nh) / 2
    img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=cor)
    return img, escala, (left, top)


def nms(caixas, scores, limiar_iou=0.45):
    """Non-Maximum Suppression clássico: em cada grupo de caixas sobrepostas, mantém a de maior confiança."""
    x1, y1, x2, y2 = caixas.T
    areas = (x2 - x1) * (y2 - y1)
    ordem = scores.argsort()[::-1]
    manter = []
    while ordem.size:
        i = ordem[0]
        manter.append(i)
        xx1 = np.maximum(x1[i], x1[ordem[1:]])
        yy1 = np.maximum(y1[i], y1[ordem[1:]])
        xx2 = np.minimum(x2[i], x2[ordem[1:]])
        yy2 = np.minimum(y2[i], y2[ordem[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[ordem[1:]] - inter + 1e-9)
        ordem = ordem[1:][iou <= limiar_iou]
    return np.array(manter, dtype=int)


def pos_processar(saida, escala, pad, forma_original, classes, conf=0.25, iou=0.45):
    """Converte a saída crua do YOLOv5 (1, N, 5 + nc) em uma lista de detecções no frame original.

    Cada linha contém cx, cy, w, h, objectness e prob_classe_0..nc-1, já com sigmoid aplicada (como no export padrão).
    """
    pred = np.asarray(saida).reshape(-1, 5 + len(classes))
    scores_cls = pred[:, 5:] * pred[:, 4:5]                 # confiança final = objectness × probabilidade da classe
    ids = scores_cls.argmax(1)
    scores = scores_cls[np.arange(len(pred)), ids]
    filtro = scores >= conf
    pred, ids, scores = pred[filtro], ids[filtro], scores[filtro]
    if not len(pred):
        return []

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    caixas = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    caixas[:, [0, 2]] -= pad[0]                             # desfaz o letterbox
    caixas[:, [1, 3]] -= pad[1]
    caixas /= escala
    alt, larg = forma_original[:2]
    caixas[:, [0, 2]] = caixas[:, [0, 2]].clip(0, larg)
    caixas[:, [1, 3]] = caixas[:, [1, 3]].clip(0, alt)

    # NMS por classe: desloco as caixas de cada classe para regiões distintas, assim classes diferentes não se suprimem
    deslocadas = caixas + ids[:, None] * 4096.0
    manter = nms(deslocadas, scores, iou)
    return [
        {"classe": classes[ids[i]], "conf": round(float(scores[i]), 4),
         "bbox": [round(float(v), 1) for v in caixas[i]]}
        for i in manter
    ]
