"""Backend Hailo-8L (modo PLUS): destinado a executar na NPU do Pi 5 o best.pt compilado em .hef.

Situação: implementado a partir da documentação do HailoRT e do Hailo Model Zoo, mas NÃO testado. Até a
entrega, este código não foi executado no Raspberry Pi 5 com o AI Kit e o best.hef não foi compilado;
o comportamento descrito abaixo é o esperado, e não um resultado medido.

Escolhi a NPU para que a inferência aconteça no próprio dispositivo, sem depender do PC nem de nuvem,
com a expectativa de latência baixa e consumo compatível com um equipamento embarcado.

Pré-requisitos no Pi (Raspberry Pi OS Bookworm 64-bit):
    sudo apt install hailo-all && sudo reboot
    hailortcli fw-control identify          # confirma que o Hailo foi reconhecido

O .hef é gerado pelo pipeline descrito em hailo/README_conversao.md. Com o config yolov5s do Model
Zoo, o NMS roda no próprio chip e a saída já vem separada por classe, no formato
[ymin, xmin, ymax, xmax, score] normalizado.
"""

import numpy as np

from . import CLASSES
from .yolo_utils import letterbox, pos_processar


class HailoBackend:
    def __init__(self, modelo, conf=0.25, iou=0.45, classes=CLASSES):
        try:                                                 # import tardio: o módulo só existe no Pi com hailo-all
            from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                        InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)
        except ImportError as erro:
            raise SystemExit(
                "hailo_platform não encontrado. No Pi 5, execute `sudo apt install hailo-all` e reinicie.\n"
                "Sem o Hailo, use --inferencia onnx (plano B na CPU) ou faça a inferência no PC."
            ) from erro

        self.conf, self.iou = conf, iou
        self.classes = classes                               # nomes na mesma ordem das saídas do modelo
        hef = HEF(str(modelo))
        self.vdevice = VDevice()                             # abre a NPU (ligada por PCIe no AI Kit)
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self.rede = self.vdevice.configure(hef, params)[0]
        entrada = InputVStreamParams.make(self.rede, format_type=FormatType.UINT8)
        saida = OutputVStreamParams.make(self.rede, format_type=FormatType.FLOAT32)

        info = hef.get_input_vstream_infos()[0]
        self.nome_entrada = info.name
        self.tamanho = info.shape[0]                         # (640, 640, 3), layout NHWC

        # A rede fica ativa e os vstreams abertos durante toda a execução, sem reabri-los a cada quadro
        self._pipeline = InferVStreams(self.rede, entrada, saida)
        self._pipeline.__enter__()
        self._ativacao = self.rede.activate(self.rede.create_params())
        self._ativacao.__enter__()

    def detect(self, frame):
        img, escala, pad = letterbox(frame, self.tamanho)
        rgb = np.ascontiguousarray(img[:, :, ::-1])[None]    # BGR→RGB, uint8, lote de tamanho 1
        saidas = self._pipeline.infer({self.nome_entrada: rgb})
        saida = next(iter(saidas.values()))[0]

        if isinstance(saida, (list, tuple)):                 # NMS feito no chip: uma lista de caixas por classe
            return self._decodificar_nms(saida, escala, pad, frame.shape)
        return pos_processar(saida, escala, pad, frame.shape, self.classes, self.conf, self.iou)

    def _decodificar_nms(self, por_classe, escala, pad, forma):
        alt, larg = forma[:2]
        deteccoes = []
        for cls, caixas in enumerate(por_classe):
            for ymin, xmin, ymax, xmax, score in np.asarray(caixas).reshape(-1, 5):
                if score < self.conf:
                    continue
                # converte as coordenadas normalizadas do quadro 640×640 para pixels do frame original
                x1 = (xmin * self.tamanho - pad[0]) / escala
                y1 = (ymin * self.tamanho - pad[1]) / escala
                x2 = (xmax * self.tamanho - pad[0]) / escala
                y2 = (ymax * self.tamanho - pad[1]) / escala
                deteccoes.append({
                    "classe": self.classes[cls], "conf": round(float(score), 4),
                    "bbox": [round(float(np.clip(x1, 0, larg)), 1), round(float(np.clip(y1, 0, alt)), 1),
                             round(float(np.clip(x2, 0, larg)), 1), round(float(np.clip(y2, 0, alt)), 1)],
                })
        return deteccoes

    def __del__(self):
        for ctx in ("_ativacao", "_pipeline"):              # libera a NPU quando o processo termina
            if hasattr(self, ctx):
                getattr(self, ctx).__exit__(None, None, None)
