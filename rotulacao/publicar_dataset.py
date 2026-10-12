"""Empacota o dataset final num .zip público para o notebook baixar no Colab.

Quem executar o notebook não terá acesso ao Drive do autor. Por isso o dataset é distribuído
num zip compartilhado por link, que o notebook baixa com gdown:

    farmtech_fase6_dataset.zip
    ├── data.yaml                 (path: /content/dataset)
    ├── train/{images,labels}
    ├── val/{images,labels}
    └── test/{images,labels}

Antes de zipar, o script roda o contagem.py: se o dataset não atender ao mínimo de 32/4/4
imagens por classe, a execução é abortada (a opção --forcar existe só para testes).

Uso:
    uv run publicar_dataset.py dataset/
    uv run publicar_dataset.py dataset/ --saida ~/Desktop/farmtech_fase6_dataset.zip
"""

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

import contagem
from contagem import CLASSES

DESTINO_COLAB = "/content/dataset"                            # onde o notebook descompacta
SPLITS = ("train", "val", "test")


def validar(dataset):
    """Roda o contagem.py no dataset; True se estiver pronto para treinar."""
    argv = sys.argv
    sys.argv = ["contagem.py", str(dataset)]
    try:
        return contagem.main() == 0
    finally:
        sys.argv = argv


def data_yaml():
    return yaml.safe_dump({
        "path": DESTINO_COLAB,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES,                                     # mesma ordem do labels.txt
    }, sort_keys=False, allow_unicode=True)


def zipar(dataset, saida):
    total = 0
    with zipfile.ZipFile(saida, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.yaml", data_yaml())
        for split in SPLITS:
            for sub in ("images", "labels"):
                for arq in sorted((dataset / split / sub).glob("*")):
                    if arq.is_file() and not arq.name.startswith("."):
                        z.write(arq, f"{split}/{sub}/{arq.name}")
                        total += 1
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="pasta gerada pelo split.py (train/val/test)")
    ap.add_argument("--saida", type=Path, default=Path("farmtech_fase6_dataset.zip"))
    ap.add_argument("--forcar", action="store_true", help="zipa mesmo se o contagem.py reprovar")
    args = ap.parse_args()

    if not validar(args.dataset):
        if not args.forcar:
            sys.exit("\nDataset reprovado no contagem.py: nada foi zipado. Corrija ou use --forcar.")
        print("\n--forcar: zipando mesmo assim (não publique este zip como versão final).")

    (args.dataset / "data.yaml").write_text(data_yaml(), encoding="utf-8")  # cópia local, igual à do zip
    n = zipar(args.dataset, args.saida)
    mb = args.saida.stat().st_size / 1e6

    print(f"""
{args.saida} gerado: {n} arquivos, {mb:.1f} MB.

Próximos passos:
  1. Suba o zip no Google Drive (pasta FarmTech_Fase6/).
  2. Botão direito → Compartilhar → Acesso geral: "Qualquer pessoa com o link" (Leitor).
  3. Copie o link: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
  4. No notebook, cole o <FILE_ID> em DATASET_ID. A célula baixa com:
       !gdown $DATASET_ID -O dataset.zip && unzip -q dataset.zip -d {DESTINO_COLAB}
  5. Teste numa aba anônima: o link precisa baixar sem pedir login.""")


if __name__ == "__main__":
    main()
