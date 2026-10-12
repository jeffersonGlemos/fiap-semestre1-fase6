# Conversão best.pt → .hef (Hailo-8L)

Este documento registra o caminho que segui para levar o modelo treinado na Entrega 1 até a NPU do
Raspberry Pi 5. Optei pelo Hailo-8L porque ele executa a inferência no próprio dispositivo, sem
depender do PC nem de nuvem, com 13 TOPS em INT8 e consumo de poucos watts, o que mantém a latência
baixa e é compatível com a alimentação de um equipamento embarcado.

A compilação **não roda no Pi**: o Hailo Dataflow Compiler (DFC) exige Linux x86_64 (Ubuntu 22.04,
WSL2 ou Docker).

```
best.pt ──export──> best.onnx ──parse──> .har ──optimize (INT8 + calibração)──> .har quantizado ──compile──> best.hef
```

## 0. Pré-requisitos

- Conta gratuita no [Hailo Developer Zone](https://hailo.ai/developer-zone/). A liberação do download não é imediata.
- DFC e Hailo Model Zoo (`hailo_model_zoo`) na **mesma versão** do HailoRT instalado no Pi
  (`hailortcli --version` no Pi). Com versões diferentes, o `.hef` gerado é recusado pelo Pi.
- Cerca de 100 imagens de **calibração**. Usei os descartes do lote A (fotos do pasto que não
  entraram no dataset) e deixei o conjunto de teste de fora, para que ele continue servindo de
  medida independente depois da quantização.

## 1. Exportar para ONNX (no Colab, junto do treino)

```bash
python yolov5/export.py --weights best.pt --include onnx --imgsz 640 --opset 11
```

## 2. Compilar com o Model Zoo (x86)

Usei o config pronto do `yolov5s` como base, porque ele já define os *end nodes* e o NMS no chip.

```bash
hailomz compile yolov5s \
    --ckpt best.onnx \
    --hw-arch hailo8l \
    --calib-path calibracao/ \
    --classes 5 \
    --performance
```

O comando executa as três etapas do DFC:

| Etapa | O que faz | Saída |
|---|---|---|
| `parse` | lê o ONNX e corta o grafo nos end nodes | `.har` |
| `optimize` | quantiza para INT8 usando as imagens de calibração | `.har` quantizado |
| `compile` | aloca as camadas nos recursos do chip | `best.hef` |

## 3. Rodar no Pi

```bash
sudo apt install hailo-all && sudo reboot
hailortcli fw-control identify                      # o chip foi reconhecido?
hailortcli benchmark best.hef                       # FPS e latência só da NPU
python camera_server.py --fonte picamera2 --inferencia hailo --modelo best.hef
```

Enquanto o `best.hef` não existe, o pipeline pode ser validado com um `.hef` pré-treinado no COCO
(por exemplo, os de `/usr/share/hailo-models/`) e a opção `--classes coco` do `camera_server.py`.

O `hailortcli benchmark` mede apenas o chip. O FPS de ponta a ponta (câmera → NPU → Wi-Fi → PC) vem
do `logs/frames.csv` do cliente. A diferença entre os dois números corresponde ao custo de captura,
pré-processamento e rede, e é um dos pontos que analiso no notebook e no vídeo complementar sobre a
escolha do Hailo.

## Pontos de atenção

1. **End nodes antes do `Detect`.** O grafo precisa ser cortado nas três convoluções de saída,
   antes do decode e do NMS do YOLOv5. Se o ONNX for compilado inteiro, com o `Detect` embutido, a
   compilação falha ou gera uma saída sem sentido. O config `yolov5s` do Model Zoo já corta no lugar certo.
2. **`nc=5` em todos os pontos:** no treino (`data.yaml`), na compilação (`--classes 5`) e no
   pós-processamento (`backends/__init__.py → CLASSES`). Se um deles divergir, as detecções saem com
   o rótulo trocado e o erro é difícil de rastrear.
3. **Ordem das classes:** `vaca, cobra, escorpiao, arvore, cerca`, a mesma do `labels.txt`.
4. **O INT8 pode derrubar as classes pequenas** (`escorpiao`, `cobra`). Por isso comparo o mAP do
   FP32 (Colab) com o do INT8 (Hailo) no mesmo conjunto de teste. Uma eventual queda é resultado do
   estudo, não um defeito da conversão.
5. **Plano B:** se a compilação não for concluída, o servidor roda com `--inferencia onnx --modelo best.onnx`
   (CPU do Pi), mantendo a mesma ponte Wi-Fi e o mesmo cliente.
