"""Envia as pré-rotulações ao Roboflow para revisão e baixa a versão revisada.

Fluxo:
  1. enviar: as imagens e os .txt do autolabel viram anotações editáveis no Roboflow Annotate;
  2. no navegador, todas as caixas são revisadas e corrigidas à mão, e uma versão é gerada
     (Generate, sem augmentation);
  3. baixar: traz a versão no formato YOLOv5 e junta tudo em images/ e labels/ para o split.py.

Requer a chave da API do Roboflow (Settings → API Keys em app.roboflow.com). A configuração
fica em rotulacao/roboflow.env, que está fora do controle de versão: copie o modelo versionado
rotulacao/roboflow.env.exemplo para roboflow.env e preencha ROBOFLOW_API_KEY. O mesmo arquivo
aceita ROBOFLOW_WORKSPACE (vazio usa o workspace padrão da conta) e ROBOFLOW_PROJETO (padrão
farmtech-fase6). Variáveis já definidas no ambiente têm precedência sobre o arquivo.

Uso:
    uv run roboflow_sync.py enviar saida/ --projeto farmtech-fase6
    uv run roboflow_sync.py baixar revisado/ --projeto farmtech-fase6 --versao 1
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import roboflow
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name("roboflow.env"))       # chave, workspace e projeto; não sobrescreve o ambiente

LABELMAP = str(Path(__file__).with_name("labels.txt"))       # 0=vaca ... 4=cerca


def conectar(nome_projeto):
    chave = os.environ.get("ROBOFLOW_API_KEY")
    if not chave:
        sys.exit("Defina ROBOFLOW_API_KEY em rotulacao/roboflow.env (modelo em roboflow.env.exemplo) "
                 "ou no ambiente (app.roboflow.com → Settings → API Keys).")
    # ROBOFLOW_WORKSPACE vazio vira None, e o SDK usa o workspace padrão da conta
    ws = roboflow.Roboflow(api_key=chave).workspace(os.environ.get("ROBOFLOW_WORKSPACE") or None)
    try:
        return ws.project(nome_projeto)
    except Exception:                                         # projeto ainda não existe
        print(f"Criando projeto {nome_projeto}...")
        # CC BY-SA 4.0 porque parte das imagens (iNaturalist, Wikimedia Commons) é CC BY-SA,
        # licença que obriga o conjunto derivado a manter os mesmos termos
        return ws.create_project(nome_projeto, "object-detection", "CC BY-SA 4.0", "objetos")


def enviar(args):
    projeto = conectar(args.projeto)
    imagens = sorted((args.pasta / "images").glob("*.jpg"))
    for i, img in enumerate(imagens, 1):
        txt = args.pasta / "labels" / f"{img.stem}.txt"
        projeto.single_upload(
            image_path=str(img),
            annotation_path=str(txt) if txt.exists() else None,
            annotation_labelmap=LABELMAP,
            batch_name=args.lote,
            num_retry_uploads=2,
        )
        print(f"[{i}/{len(imagens)}] {img.name}")
    print(f"\nEnvio concluído. Revisão em https://app.roboflow.com → {args.projeto} → Annotate → {args.lote}")


def baixar(args):
    projeto = conectar(args.projeto)
    with tempfile.TemporaryDirectory() as tmp:
        projeto.version(args.versao).download("yolov5", location=tmp, overwrite=True)
        for sub in ("images", "labels"):
            (args.pasta / sub).mkdir(parents=True, exist_ok=True)
        # O Roboflow devolve train/valid/test; junto tudo porque a separação sem vazamento fica com o split.py
        for arq in Path(tmp).glob("*/*/*"):
            if arq.parent.name in ("images", "labels"):
                shutil.copy2(arq, args.pasta / arq.parent.name / arq.name)

    # O Roboflow anexa um hash ao nome (pasto_1_jpg.rf.abc123.jpg); aqui o nome original é restaurado
    for arq in list((args.pasta / "images").iterdir()) + list((args.pasta / "labels").iterdir()):
        if ".rf." in arq.name:
            original = arq.name.split(".rf.")[0].rsplit("_", 1)[0] + arq.suffix
            arq.rename(arq.with_name(original))
    print(f"Versão {args.versao} salva em {args.pasta}. Próximo passo: uv run split.py {args.pasta} dataset/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enviar", help="envia as imagens e as pré-rotulações")
    e.add_argument("pasta", type=Path, help="pasta com images/ e labels/ (saída do autolabel)")
    # ROBOFLOW_PROJETO (roboflow.env) define o projeto padrão dos dois subcomandos
    e.add_argument("--projeto", default=os.environ.get("ROBOFLOW_PROJETO") or "farmtech-fase6")
    e.add_argument("--lote", default="autolabel", help="nome do batch no Roboflow")
    b = sub.add_parser("baixar", help="baixa a versão revisada no formato YOLOv5")
    b.add_argument("pasta", type=Path)
    b.add_argument("--projeto", default=os.environ.get("ROBOFLOW_PROJETO") or "farmtech-fase6")
    b.add_argument("--versao", type=int, required=True)
    args = ap.parse_args()
    enviar(args) if args.cmd == "enviar" else baixar(args)


if __name__ == "__main__":
    main()
