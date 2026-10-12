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

## 1. Exportar para ONNX

Escrevi o `exportar_modelo.py` (na pasta `ir_alem/`) para que a exportação seja a mesma em qualquer
máquina: ele chama o `export.py` oficial do `ultralytics/yolov5` com o mesmo interpretador Python,
usa apenas `pathlib` e funciona igual em macOS, Linux e Windows. O ONNX que ele gera é a entrada do
DFC nesta conversão e também o modelo do plano B (ONNX Runtime).

```bash
python exportar_modelo.py --pesos best.pt                  # best.onnx: opset 11, 640 px, forma fixa (1, 3, 640, 640)
```

Os padrões foram escolhidos para o DFC: opset 11, entrada de 640 px e sem `--dynamic`, porque o
compilador precisa de dimensões fixas. A saída tem forma `(1, 25200, 11)`: 4 coordenadas, a
objectness e as 6 classes. O repositório do YOLOv5 é reaproveitado do cache do `torch.hub` (ou
clonado na primeira execução); o caminho de um clone local pode ser informado com `--yolov5`.

Validei o script com o `yolov5s.pt` do COCO no Mac: o ONNX saiu com opset 11 e saída `(1, 25200, 85)`,
e as detecções do ONNX Runtime coincidiram com as do PyTorch nos mesmos quadros (por exemplo, `cow`
com confiança 0,94 nos dois). A opção `--formato coreml` também gera um `.mlpackage` para comparação
no Mac, mas ele não participa da conversão para o Hailo.

## 2. Compilar com o Model Zoo (x86)

Usei o config pronto do `yolov5s` como base, porque ele já define os *end nodes* e o NMS no chip.

```bash
hailomz compile yolov5s \
    --ckpt best.onnx \
    --hw-arch hailo8l \
    --calib-path calibracao/ \
    --classes 6 \
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
2. **`nc=6` em todos os pontos:** no treino (`data.yaml`), na compilação (`--classes 6`) e no
   pós-processamento (`backends/__init__.py → CLASSES`). Se um deles divergir, as detecções saem com
   o rótulo trocado e o erro é difícil de rastrear.
3. **Ordem das classes:** `vaca, cobra, escorpiao, arvore, cerca, caneca`, a mesma do `labels.txt`.
   Acrescentei a `caneca` no fim da lista para não renumerar as cinco classes que já
   existiam; ela é o objeto da demonstração ao vivo e não dispara alerta.
4. **O INT8 pode derrubar as classes pequenas** (`escorpiao`, `cobra`). Por isso comparo o mAP do
   FP32 (Colab) com o do INT8 (Hailo) no mesmo conjunto de teste. Uma eventual queda é resultado do
   estudo, não um defeito da conversão.
5. **Plano B:** se a compilação não for concluída, o servidor roda com `--inferencia onnx --modelo best.onnx`
   (CPU do Pi), mantendo a mesma ponte Wi-Fi e o mesmo cliente. É exatamente o mesmo arquivo gerado no
   passo 1. No Pi, `--provedor auto` resulta em `CPUExecutionProvider`; no PC de desenvolvimento, o mesmo
   comando simula o PLUS com aceleração (CoreML no Mac, CUDA ou DirectML no Windows/Linux).
