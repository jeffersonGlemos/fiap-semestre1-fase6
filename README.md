# FarmTech Solutions: visão computacional para um rover de pasto

**FIAP · Inteligência Artificial · Fase 6 (Redes Neurais e Visão Computacional)**
Autor: Jefferson Gonçalves Lemos (RM572399)

A FarmTech está prototipando um rover autônomo de inspeção de pasto. Neste repositório está o módulo
de visão que desenvolvi para esse rover: uma YOLO customizada que, a partir da câmera, responde onde
está o rebanho, se há risco biológico no caminho e por onde o rover pode andar. A solução completa
(dataset, treino, validação, teste, comparação com outras abordagens e conclusões) está documentada
passo a passo no notebook. Este README apenas apresenta o projeto e indica onde está cada parte.

| ID | Classe | Pergunta do rover | Observação |
|---:|---|---|---|
| 0 | `vaca` | Onde está o rebanho? | |
| 1 | `cobra` | Há risco biológico no caminho? | |
| 2 | `escorpiao` | Há risco biológico no caminho? | objeto pequeno, a classe mais difícil |
| 3 | `arvore` | Onde posso andar? | obstáculo |
| 4 | `cerca` | Onde posso andar? | a caixa é desenhada no **mourão**, não na cerca inteira |
| 5 | `caneca` | Nenhuma: não faz parte da cena do pasto | objeto da demonstração ao vivo com câmera no Ir Além |

A `caneca` não responde a uma pergunta do rover. Incluí essa classe porque a demonstração ao vivo do
Ir Além usa o celular como câmera Wi-Fi, e no ambiente de gravação não há vacas, cobras nem
escorpiões. Com a caneca, a detecção em tempo real pode ser mostrada com um objeto que está de fato
diante da câmera. Ela ocupa o último índice para que os rótulos já feitos das cinco classes do pasto
não precisem ser renumerados.

## Entregáveis

| Peça | Link |
|---|---|
| Notebook principal (Entregas 1 e 2) | [`JeffersonGoncalvesLemos_rm572399_pbl_fase6.ipynb`](JeffersonGoncalvesLemos_rm572399_pbl_fase6.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/JeffersonGoncalvesLemos_rm572399_pbl_fase6.ipynb) <!-- TODO: trocar USUARIO/REPOSITORIO --> |
| Notebook Ir Além | [`ir_alem/JeffersonGoncalvesLemos_rm572399_pbl_fase6_ir_alem.ipynb`](ir_alem/JeffersonGoncalvesLemos_rm572399_pbl_fase6_ir_alem.ipynb) |
| Vídeo: Entregas 1 e 2 (até 5 min) | <!-- TODO: link YouTube não listado --> _em breve_ |
| Vídeo: Ir Além (até 5 min) | <!-- TODO: link YouTube não listado --> _em breve_ |
| Vídeo: Por que Hailo (plus) | <!-- TODO: link YouTube não listado --> _em breve_ |

O notebook principal cobre:

- **Entrega 1:** YOLOv5 customizada, com duas simulações de quantidade de épocas bem diferentes
  (30 e 60) e análise de validação e teste por classe.
- **Entrega 2:** comparação com a YOLO padrão (pesos COCO) e com uma CNN treinada do zero, nos
  critérios de facilidade de uso, precisão, tempo de treino e tempo de inferência.

No **Ir Além**, segui a opção 3.1 do enunciado: uma câmera transmite por Wi-Fi para um script Python
no VS Code que roda o `best.pt`. Como plus, a inferência também roda embarcada num Raspberry Pi 5
com acelerador Hailo-8L, que envia o vídeo anotado e as detecções ao mesmo cliente.

## Estrutura do repositório

```
.
├── JeffersonGoncalvesLemos_rm572399_pbl_fase6.ipynb   notebook principal (Entregas 1 e 2)
├── README.md
├── assets/                    prints das imagens de teste e gráficos gerados pelo notebook
├── rotulacao/                 dataset: coleta, download, pré-rotulação, revisão, split e contagem
│   ├── coleta.py              captura com câmera própria (Pi ou webcam) para ampliar o dataset;
│   │                          não foi a origem do lote A atual
│   ├── baixar_bancos.py       cobras e escorpiões do iNaturalist (licenças CC) + creditos.csv
│   ├── baixar_commons.py      fotos de pasto e de canecas do Wikimedia Commons + creditos.csv
│   ├── autolabel.py           pré-rotulação zero-shot com Grounding DINO
│   ├── roboflow_sync.py       envio para a revisão no Roboflow Annotate e download da versão revisada
│   ├── roboflow.env.exemplo   modelo da configuração do Roboflow (chave, workspace e projeto)
│   ├── split.py               treino/val/teste por grupos de imagens, sem arquivo repetido entre conjuntos
│   ├── contagem.py            conferência do mínimo de 32/4/4 imagens por classe
│   └── publicar_dataset.py    geração do zip público baixado pelo notebook
└── ir_alem/                   opção 3.1: câmera → Wi-Fi → VS Code, e o plus com Pi 5 + Hailo-8L
    ├── JeffersonGoncalvesLemos_rm572399_pbl_fase6_ir_alem.ipynb
    ├── camera_server.py       roda na câmera (Pi/PC): stream MJPEG e, no plus, inferência
    ├── cliente_vscode.py      roda no PC: recebe o stream, detecta, alerta e registra
    ├── backends/              inferência embarcada: Hailo (.hef) e ONNX (plano B)
    ├── esp32cam/              sketch do ESP32-CAM com as mesmas rotas HTTP
    ├── hailo/                 conversão best.pt → ONNX → .hef
    └── figuras/               figura autoral da arquitetura
```

## Rotulação

```
autolabel.py → roboflow_sync.py → revisão no Roboflow Annotate → split.py → contagem.py
```

O enunciado sugere o Make Sense IA. Optei por outro fluxo porque, com seis classes e no mínimo 40
imagens por classe, desenhar todas as caixas do zero seria lento e tenderia a gerar caixas
inconsistentes entre sessões de trabalho. O Grounding DINO faz uma pré-rotulação zero-shot a partir
de um prompt de texto por classe, e essas caixas servem apenas de ponto de partida: revisei
manualmente todas elas no Roboflow Annotate, ajustando limites, apagando falsos positivos e
desenhando os objetos que o modelo deixou passar. A exportação sai no mesmo formato YOLO que o
Make Sense geraria, então o restante do pipeline não muda.

O `split.py` separa treino, validação e teste por grupos de imagens: uma expressão regular aplicada
ao nome do arquivo (`--grupo`) define o grupo, e o grupo inteiro vai para o mesmo conjunto. Isso serve
para manter juntas as fotos em sequência capturadas pelo `coleta.py`, cujo nome traz data e hora. As
fotos atuais, do Wikimedia Commons e do iNaturalist, não têm data e hora no nome, então cada uma forma
um grupo próprio. O `contagem.py` confere o mínimo de 32/4/4 imagens por classe antes de qualquer
treino.
A discussão completa dessa etapa está no notebook.

## Como reproduzir

1. Abra o notebook principal no Colab pelo badge acima e ative a GPU
   (*Ambiente de execução → Alterar o tipo de ambiente → T4 GPU*).
2. Execute todas as células. O dataset é baixado automaticamente de um link público, sem depender
   de acesso ao Drive do autor. <!-- TODO: link público do dataset -->
3. O Google Drive é montado apenas para guardar os resultados (`runs/` e pesos) entre sessões; essa
   etapa é opcional.

O dataset pode ser refeito do zero localmente com os scripts de `rotulacao/`, na ordem do fluxo
descrito acima (ambiente gerenciado com `uv`). O envio ao Roboflow exige copiar
`rotulacao/roboflow.env.exemplo` para `rotulacao/roboflow.env` e preencher a chave da API, já que
o `roboflow.env` fica fora do controle de versão.

## Créditos das imagens

As fotos de pasto (`vaca`, `arvore`, `cerca`) formam o lote A do dataset e vêm do Wikimedia Commons,
das categorias *Cattle in Brazil*, *Pastures in Brazil*, *Wooden fences in Brazil*, *Fence posts* e
*Barbed wire fences* (com as subcategorias diretas). Aceitei apenas arquivos JPEG com licença CC0,
CC BY, CC BY-SA ou em domínio público e acima de uma resolução mínima, e dividi o total em cotas por
categoria para que o lote não ficasse composto só de fotos de gado. O download é feito por
`rotulacao/baixar_commons.py`.
As imagens de `cobra` e `escorpiao` vêm do iNaturalist: usei apenas observações *research grade*
cujas fotos têm licença CC0, CC BY ou CC BY-SA, baixadas por `rotulacao/baixar_bancos.py`.
As fotos de `caneca` formam o lote D e também vêm do Wikimedia Commons, baixadas pelo mesmo
`rotulacao/baixar_commons.py` e com os mesmos filtros de formato, licença (CC0, CC BY, CC BY-SA ou
domínio público) e resolução. Usei as categorias *Coffee mugs*, *Mugs with logos*, *People with
mugs*, *Mugs with soup* e *Mugs* (com as subcategorias diretas), com a mesma divisão em cotas.
Escolhi categorias de canecas em uso, com café, sopa ou na mão de alguém, porque na demonstração
ao vivo a caneca aparece sobre a mesa e não isolada como num catálogo; pelo mesmo motivo, o script
descarta peças de acervo de museu pelo título e fotos de produto pela borda quase toda branca.
Fotos próprias de canecas, feitas com o celular no ambiente da demonstração, poderão complementar
esse lote.
Autor, licença e links de todas as imagens dos quatro lotes estão em
[`rotulacao/creditos.csv`](rotulacao/creditos.csv).
Dos bancos públicos baixei somente as imagens; todas as anotações foram feitas neste projeto.
