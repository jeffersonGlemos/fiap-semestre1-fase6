"""Backend ONNX Runtime: plano B do Hailo no Pi e alternativa sem PyTorch no PC.

O .onnx sai do exportar_modelo.py (o mesmo arquivo serve de entrada para o Hailo DFC):
    python exportar_modelo.py --pesos best.pt                # gera best.onnx (opset 11, 640 px)

O parâmetro `provedor` escolhe o execution provider do ONNX Runtime, porque desenvolvo no Mac
(Apple Silicon) e o projeto também precisa rodar em Windows e Linux:
    auto      escolhe o melhor disponível nesta ordem: CUDA, CoreML, DirectML e, por fim, CPU
    cpu       CPUExecutionProvider (todos os sistemas, inclusive o Pi)
    coreml    CoreMLExecutionProvider (macOS; já vem no pacote onnxruntime)
    cuda      CUDAExecutionProvider (Linux/Windows com NVIDIA; pacote onnxruntime-gpu)
    directml  DmlExecutionProvider (Windows com qualquer GPU DirectX 12; pacote onnxruntime-directml)
A CPU fica sempre no fim da lista como reserva para os operadores que o acelerador não suporta.
"""

import numpy as np

from . import CLASSES
from .yolo_utils import letterbox, pos_processar

PROVEDORES = {
    "cpu": "CPUExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}
ORDEM_AUTO = ["cuda", "coreml", "directml", "cpu"]           # do mais rápido ao mais genérico


def escolher_provedores(provedor, disponiveis):
    """Traduz o nome curto em uma lista de providers do ONNX Runtime, com a CPU sempre como reserva."""
    if provedor == "auto":
        nome = next(p for p in ORDEM_AUTO if PROVEDORES[p] in disponiveis)
    elif provedor in PROVEDORES:
        nome = provedor
        if PROVEDORES[nome] not in disponiveis:
            raise SystemExit(
                f"O provedor {provedor!r} não está disponível neste onnxruntime ({', '.join(disponiveis)}). "
                "Veja em requirements-pc.txt qual pacote instalar em cada sistema, ou use --provedor auto.")
    else:
        raise SystemExit(f"provedor desconhecido: {provedor!r} (use {', '.join(['auto', *PROVEDORES])})")
    lista = [PROVEDORES[nome]]
    if nome != "cpu":
        lista.append(PROVEDORES["cpu"])
    return lista


class OnnxBackend:
    def __init__(self, modelo, conf=0.25, iou=0.45, classes=CLASSES, provedor="auto"):
        import onnxruntime as ort                            # import tardio: só é necessário quando o ONNX é usado

        lista = escolher_provedores(provedor, ort.get_available_providers())
        opcoes = ort.SessionOptions()
        if lista[0] == PROVEDORES["directml"]:               # o DirectML não aceita estas duas otimizações
            opcoes.enable_mem_pattern = False
            opcoes.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sessao = ort.InferenceSession(str(modelo), sess_options=opcoes, providers=lista)
        self.provedor = self.sessao.get_providers()[0]      # o que o ONNX Runtime de fato ativou
        self.entrada = self.sessao.get_inputs()[0].name
        self.tamanho = self.sessao.get_inputs()[0].shape[2] or 640   # entrada no formato (1, 3, 640, 640)
        self.conf, self.iou = conf, iou
        self.classes = classes                               # nomes na mesma ordem das saídas do modelo

        # Cada saída tem 5 + nc valores. Se a lista de classes não bater com o modelo (por exemplo, um
        # modelo COCO sem --classes coco), o reshape do pós-processamento embaralharia tudo sem erro.
        largura = self.sessao.get_outputs()[0].shape[-1]
        if isinstance(largura, int) and largura != 5 + len(classes):
            raise SystemExit(
                f"O modelo tem {largura - 5} classes, mas foram informados {len(classes)} nomes. "
                "Use --classes coco para um modelo COCO ou um .txt com os nomes na ordem do treino.")

    def detect(self, frame):
        img, escala, pad = letterbox(frame, self.tamanho)
        tensor = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB e HWC→CHW
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0
        saida = self.sessao.run(None, {self.entrada: tensor})[0]
        return pos_processar(saida, escala, pad, frame.shape, self.classes, self.conf, self.iou)
