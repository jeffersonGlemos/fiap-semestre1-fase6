"""Exporta o best.pt (YOLOv5) para ONNX e, opcionalmente, para CoreML, usando o export.py oficial.

O ONNX gerado aqui tem dois usos: é o modelo do plano B (ONNX Runtime no Pi ou no PC, sem PyTorch)
e é a entrada do Hailo Dataflow Compiler na conversão para .hef (hailo/README_conversao.md). Por isso
o opset padrão é 11, o mesmo usado nos exemplos do Hailo Model Zoo para o YOLOv5.

O CoreML é opcional e só interessa no Mac: ele permite comparar a execução nativa da Apple com o
ONNX Runtime usando o CoreMLExecutionProvider.

O script não chama comandos de shell; usa pathlib e o próprio interpretador (sys.executable), de modo
que funciona igual em macOS, Linux e Windows. O repositório ultralytics/yolov5 é procurado nesta ordem:
--yolov5, cache do torch.hub (criado pelo cliente_vscode.py) e cache próprio; se nenhum existir, é
baixado com git ou, sem git, como zip do GitHub.

Antes da exportação, o script gera uma cópia "portável" dos pesos (<nome>_portavel.pt, ao lado do
original) e é ela que segue para o export.py. Treinei no Colab com Python 3.13 em Linux, e o
checkpoint do YOLOv5 guarda um objeto pathlib nos argumentos do treino: em Python 3.12 ele aparece
como o módulo inexistente pathlib._local e, no Windows, como um PosixPath que não pode ser criado.
O export.py roda em outro processo e não recebe o ajuste de backends/compat.py, por isso converto
esses objetos para str uma única vez e salvo a cópia. Com --so-portavel, o script gera apenas essa
cópia, que é o arquivo que distribuo para quem for executar o projeto em outra máquina.

Uso:
    python exportar_modelo.py --pesos best.pt                          # best.onnx ao lado do best.pt
    python exportar_modelo.py --pesos best.pt --so-portavel            # só o best_portavel.pt
    python exportar_modelo.py --pesos best.pt --formato ambos --saida modelos/
    python exportar_modelo.py --pesos best.pt --yolov5 ../yolov5       # clone já existente
Dependências da exportação: pip install onnx onnxslim (e coremltools para --formato coreml|ambos).
"""

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from backends.compat import carregamento_portavel, sem_pathlib

REPO_GIT = "https://github.com/ultralytics/yolov5.git"
REPO_ZIP = "https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip"
CACHE_HUB = Path.home() / ".cache" / "torch" / "hub" / "ultralytics_yolov5_master"
CACHE_PROPRIO = Path.home() / ".cache" / "farmtech" / "yolov5"


def localizar_yolov5(caminho=None):
    """Devolve uma pasta com o export.py do YOLOv5, baixando o repositório se for preciso."""
    if caminho:
        if not (caminho / "export.py").exists():
            raise SystemExit(f"{caminho} não contém export.py do YOLOv5.")
        return caminho
    for pasta in (CACHE_HUB, CACHE_PROPRIO):
        if (pasta / "export.py").exists():
            return pasta

    CACHE_PROPRIO.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        print(f"Clonando {REPO_GIT} em {CACHE_PROPRIO}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_GIT, str(CACHE_PROPRIO)], check=True)
    else:                                                    # Windows sem git, por exemplo
        print(f"git não encontrado; baixando {REPO_ZIP}")
        with urllib.request.urlopen(REPO_ZIP, timeout=120) as resp:
            dados = resp.read()
        with zipfile.ZipFile(io.BytesIO(dados)) as z:
            z.extractall(CACHE_PROPRIO.parent)
        (CACHE_PROPRIO.parent / "yolov5-master").rename(CACHE_PROPRIO)
    return CACHE_PROPRIO


def gerar_portavel(pesos, yolov5):
    """Salva <nome>_portavel.pt ao lado dos pesos, sem objetos pathlib no checkpoint, e devolve o caminho.

    O checkpoint é carregado neste processo com o ajuste de compatibilidade e regravado pelo torch.save
    local, de modo que a cópia não referencia pathlib._local nem PosixPath e carrega em qualquer versão
    do Python e em qualquer sistema operacional.
    """
    if pesos.stem.endswith("_portavel"):                     # já é uma cópia portável
        return pesos
    import torch                                             # import tardio: só a exportação precisa do PyTorch
    destino = pesos.with_name(f"{pesos.stem}_portavel{pesos.suffix}")
    sys.path.insert(0, str(yolov5))                          # o pickle referencia as classes de models/ do YOLOv5
    try:
        with carregamento_portavel():
            ckpt = torch.load(pesos, map_location="cpu", weights_only=False)
    finally:
        sys.path.remove(str(yolov5))
    torch.save(sem_pathlib(ckpt), destino)
    with zipfile.ZipFile(destino) as z:                      # confere no próprio pickle gravado
        pickle = next(n for n in z.namelist() if n.endswith("data.pkl"))
        if b"pathlib" in z.read(pickle):
            raise SystemExit(f"{destino.name} ainda referencia pathlib; a conversão não foi completa.")
    print(f"Pesos portáveis: {destino}", flush=True)
    return destino


def exportar(yolov5, pesos, formato, imgsz, opset):
    """Chama o export.py com o mesmo Python deste script e devolve o arquivo gerado ao lado dos pesos."""
    comando = [sys.executable, str(yolov5 / "export.py"), "--weights", str(pesos),
               "--include", formato, "--imgsz", str(imgsz), "--device", "cpu"]
    if formato == "onnx":
        comando += ["--opset", str(opset)]                  # sem --dynamic: o DFC e o CoreML EP exigem forma fixa
    gerado = pesos.with_suffix({"onnx": ".onnx", "coreml": ".mlpackage"}[formato])
    if gerado.exists():                                      # apaga a versão anterior para não confundi-la com a nova
        shutil.rmtree(gerado) if gerado.is_dir() else gerado.unlink()
    print("Executando:", " ".join(comando), flush=True)      # antes da saída do subprocesso
    subprocess.run(comando, check=True, cwd=yolov5)
    if not gerado.exists():                                  # o export.py registra a falha no log, mas não aborta
        raise SystemExit(f"A exportação para {formato} falhou: {gerado.name} não foi criado (veja o log acima).")
    return gerado


def mover(origem, pasta, nome=None):
    """Move o arquivo (ou o pacote .mlpackage, que é uma pasta) para a pasta de saída, com `nome` se informado."""
    destino = pasta / (nome or origem.name)
    if destino.resolve() == origem.resolve():
        return destino
    if destino.exists():
        shutil.rmtree(destino) if destino.is_dir() else destino.unlink()
    shutil.move(str(origem), str(destino))
    return destino


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, required=True, help="arquivo .pt do YOLOv5 (best.pt ou yolov5s.pt)")
    ap.add_argument("--formato", choices=["onnx", "coreml", "ambos"], default="onnx")
    ap.add_argument("--imgsz", type=int, default=640, help="lado da entrada em pixels (o mesmo do treino)")
    ap.add_argument("--opset", type=int, default=11, help="opset do ONNX (11 é o usado com o Hailo DFC)")
    ap.add_argument("--saida", type=Path, help="pasta de destino (padrão: a mesma pasta dos pesos)")
    ap.add_argument("--yolov5", type=Path, help="clone local do ultralytics/yolov5 (padrão: cache)")
    ap.add_argument("--so-portavel", action="store_true",
                    help="só gera <nome>_portavel.pt ao lado dos pesos, sem exportar")
    args = ap.parse_args()

    pesos = args.pesos.resolve()
    if not pesos.exists():
        ap.error(f"{pesos} não existe")
    saida = (args.saida or pesos.parent).resolve()
    saida.mkdir(parents=True, exist_ok=True)
    yolov5 = localizar_yolov5(args.yolov5.resolve() if args.yolov5 else None)
    portavel = gerar_portavel(pesos, yolov5)
    if args.so_portavel:
        return
    nome_base = pesos.stem.removesuffix("_portavel")          # best.onnx, e não best_portavel.onnx

    formatos = ["onnx", "coreml"] if args.formato == "ambos" else [args.formato]
    gerados, falhas = [], []
    for formato in formatos:                                 # uma falha no CoreML não descarta o ONNX já gerado
        try:
            gerado = exportar(yolov5, portavel, formato, args.imgsz, args.opset)
            gerados.append(mover(gerado, saida, nome_base + gerado.suffix))
        except (SystemExit, subprocess.CalledProcessError) as erro:
            falhas.append(f"{formato}: {erro}")
    print("\nArquivos gerados:" if gerados else "\nNenhum arquivo gerado.")
    for g in gerados:
        print(f"  {g}")
    if falhas:
        raise SystemExit("Falhas:\n  " + "\n  ".join(falhas))


if __name__ == "__main__":
    main()
