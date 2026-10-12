"""Separa o dataset rotulado em train/val/test sem vazamento.

Imagens do mesmo GRUPO (ex.: mesma caminhada no pasto) vão sempre para o mesmo split, para que
fotos quase idênticas não fiquem ao mesmo tempo no treino e na avaliação. Val e test são
preenchidos primeiro, até cada classe atingir o mínimo de 4 imagens; o restante vai para train.

Uso:
    uv run split.py rotulado/ dataset/
    uv run split.py rotulado/ dataset/ --grupo "pasto_\\d{8}_\\d{2}"   # fotos do coleta.py: agrupa por data+hora

Sem --grupo, cada imagem forma um grupo próprio. É o caso do dataset atual: todas as fotos vêm de
bancos públicos (Wikimedia Commons e iNaturalist), cada uma de um arquivo ou de uma observação
independente, e nenhuma tem data e hora no nome.
"""

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import contagem
from contagem import CLASSES, MINIMO


def classes_da_imagem(txt):
    return {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origem", type=Path, help="pasta com images/ e labels/ (saída do autolabel ou do Roboflow)")
    ap.add_argument("destino", type=Path, help="gera destino/{train,val,test}/{images,labels}")
    ap.add_argument("--grupo", help="regex aplicada ao nome; o trecho casado define o grupo")
    ap.add_argument("--seed", type=int, default=42, help="semente do sorteio dos grupos (padrão 42)")
    args = ap.parse_args()

    # Agrupa as imagens e registra quais classes cada grupo contém
    grupos = defaultdict(list)
    for img in sorted((args.origem / "images").iterdir()):
        txt = args.origem / "labels" / f"{img.stem}.txt"
        if not txt.exists():
            print(f"ignorando {img.name}: sem rótulo")
            continue
        m = re.search(args.grupo, img.stem) if args.grupo else None
        grupos[m.group(0) if m else img.stem].append((img, txt, classes_da_imagem(txt)))

    chaves = list(grupos)
    random.Random(args.seed).shuffle(chaves)                  # seed fixa torna o split reprodutível

    faltam = {s: {c: MINIMO[s] for c in range(len(CLASSES))} for s in ("test", "val")}
    destino_do_grupo = {}
    for chave in chaves:
        classes = set().union(*(c for _, _, c in grupos[chave]))
        split = "train"
        for s in ("test", "val"):                             # vai para o primeiro split que ainda precisa dessas classes
            if any(faltam[s][c] > 0 for c in classes):
                split = s
                break
        if split != "train":
            for _, _, c in grupos[chave]:
                for cls in c:
                    faltam[split][cls] -= 1
        destino_do_grupo[chave] = split

    for chave, split in destino_do_grupo.items():
        for sub in ("images", "labels"):
            (args.destino / split / sub).mkdir(parents=True, exist_ok=True)
        for img, txt, _ in grupos[chave]:
            shutil.copy2(img, args.destino / split / "images" / img.name)
            shutil.copy2(txt, args.destino / split / "labels" / txt.name)

    print(f"{len(chaves)} grupos distribuídos em {args.destino}. Conferindo com contagem.py:\n")
    sys.argv = ["contagem.py", str(args.destino)]
    raise SystemExit(contagem.main())


if __name__ == "__main__":
    main()
