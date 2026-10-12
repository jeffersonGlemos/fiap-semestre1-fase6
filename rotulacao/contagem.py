"""Contagem do dataset rotulado (pipeline autolabel → Roboflow, formato YOLO).

Verifica, para cada split (train/val/test):
  - quantas IMAGENS contêm cada classe (é essa contagem que o enunciado usa no mínimo de 32/4/4);
  - quantas INSTÂNCIAS (caixas) existem de cada classe;
  - problemas de rotulação: imagem sem .txt, .txt sem imagem, classe inválida,
    coordenadas fora de [0, 1] e linhas mal formatadas.

Uso no Colab:   !python contagem.py /content/drive/MyDrive/FarmTech_Fase6/dataset
Uso local:      python contagem.py caminho/do/dataset
"""

import sys
from collections import Counter
from pathlib import Path

CLASSES = ["vaca", "cobra", "escorpiao", "arvore", "cerca", "caneca"]  # mesma ordem do labels.txt
MINIMO = {"train": 32, "val": 4, "test": 4}                   # mínimo de imagens por classe
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}          # extensões aceitas pelo YOLOv5


def ler_rotulos(txt):
    """Lê um .txt YOLO e devolve (lista de ids de classe, lista de problemas)."""
    ids, problemas = [], []
    for n, linha in enumerate(txt.read_text().splitlines(), start=1):
        partes = linha.split()
        if not partes:                                        # linha vazia: ignora
            continue
        if len(partes) != 5:                                  # formato: classe xc yc w h
            problemas.append(f"{txt.name}:{n} esperava 5 valores, achou {len(partes)}")
            continue
        try:
            cls = int(partes[0])
            coords = [float(v) for v in partes[1:]]
        except ValueError:
            problemas.append(f"{txt.name}:{n} valor não numérico")
            continue
        if not 0 <= cls < len(CLASSES):                       # id fora da lista de classes
            problemas.append(f"{txt.name}:{n} classe {cls} inexistente")
            continue
        if any(not 0 <= c <= 1 for c in coords):              # YOLO usa coordenadas normalizadas
            problemas.append(f"{txt.name}:{n} coordenada fora de [0, 1]")
        ids.append(cls)
    return ids, problemas


def contar_split(raiz, split):
    """Conta imagens e instâncias por classe em <raiz>/<split>/{images,labels}."""
    pasta_img, pasta_lbl = raiz / split / "images", raiz / split / "labels"
    if not pasta_img.is_dir():
        return None

    imagens = {p.stem: p for p in pasta_img.iterdir() if p.suffix.lower() in EXT_IMG}
    rotulos = {p.stem: p for p in pasta_lbl.glob("*.txt")} if pasta_lbl.is_dir() else {}

    por_imagem, instancias, problemas = Counter(), Counter(), []
    problemas += [f"{imagens[s].name} sem rótulo" for s in sorted(imagens.keys() - rotulos.keys())]
    problemas += [f"{rotulos[s].name} sem imagem" for s in sorted(rotulos.keys() - imagens.keys())]

    for stem in sorted(imagens.keys() & rotulos.keys()):
        ids, probs = ler_rotulos(rotulos[stem])
        problemas += probs
        if not ids:
            problemas.append(f"{rotulos[stem].name} vazio (nenhuma caixa)")
        instancias.update(ids)                                # cada caixa conta
        por_imagem.update(set(ids))                           # cada imagem conta 1x por classe

    return {"imagens": len(imagens), "por_imagem": por_imagem,
            "instancias": instancias, "problemas": problemas}


def main():
    raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")
    resultados = {s: contar_split(raiz, s) for s in MINIMO}
    ok = True

    for split, r in resultados.items():
        if r is None:
            print(f"\n[{split}] pasta {raiz / split / 'images'} não encontrada")
            ok = False
            continue
        print(f"\n[{split}] {r['imagens']} imagens, mínimo de {MINIMO[split]} por classe")
        print(f"  {'classe':<10} {'imagens':>8} {'caixas':>7}  status")
        for i, nome in enumerate(CLASSES):
            n = r["por_imagem"][i]
            status = "ok" if n >= MINIMO[split] else f"FALTAM {MINIMO[split] - n}"
            ok &= n >= MINIMO[split]
            print(f"  {nome:<10} {n:>8} {r['instancias'][i]:>7}  {status}")
        if r["problemas"]:
            ok = False
            print(f"  {len(r['problemas'])} problema(s):")
            for p in r["problemas"]:
                print(f"    - {p}")

    # A mesma foto em dois splits caracteriza vazamento entre treino e avaliação
    nomes = {s: {p.stem for p in (raiz / s / "images").glob("*")} for s, r in resultados.items() if r}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        repetidas = nomes.get(a, set()) & nomes.get(b, set())
        if repetidas:
            ok = False
            print(f"\nVAZAMENTO: {len(repetidas)} imagem(ns) em {a} e {b}: {sorted(repetidas)[:5]}")

    print("\n" + ("Dataset pronto para treinar." if ok else "Dataset ainda NÃO atende ao mínimo."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
