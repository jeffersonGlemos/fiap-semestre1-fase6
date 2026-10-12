"""Baixa fotos de cobras e escorpiões do iNaturalist para os lotes B e C.

Uso apenas observações "research grade" (espécie confirmada pela comunidade) cuja foto tenha
licença que permite reuso: CC0, CC BY ou CC BY-SA. O script prioriza observações do Brasil e,
se faltarem, completa com o resto do mundo. Só a IMAGEM é baixada, nunca anotações, porque toda
a rotulação é feita neste projeto.

Cada foto baixada gera uma linha em creditos.csv (autor, licença e links), que é a fonte dos
créditos citados no README.

Uso:
    uv run baixar_bancos.py brutas/                          # 50 de cada → brutas/cobra, brutas/escorpiao
    uv run baixar_bancos.py brutas/ --classe cobra --quantidade 60
    uv run baixar_bancos.py brutas/ --sem-brasil             # busca global direto

Uma nova execução continua a numeração e pula as observações já baixadas.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests

API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "farmtech-fase6/0.1 (projeto academico FIAP; uso educacional)"
LICENCAS = {"cc0", "cc-by", "cc-by-sa"}                       # permitem reuso com atribuição
BRASIL = 6878                                                 # place_id do Brasil no iNaturalist
PAUSA = 1.0                                                   # segundos entre requisições (a API pede ~1 req/s)

# classe → (taxon_id, prefixo do arquivo)
TAXONS = {
    "cobra":     (85553, "cobra"),                            # subordem Serpentes
    "escorpiao": (48894, "escorp"),                           # ordem Scorpiones
}
CAMPOS = ["arquivo", "classe", "autor", "licenca", "url_observacao", "url_foto"]

sessao = requests.Session()
sessao.headers["User-Agent"] = USER_AGENT


def ler_creditos(caminho):
    if not caminho.exists():
        return []
    with caminho.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def salvar_creditos(caminho, linhas):
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(sorted(linhas, key=lambda l: (l["classe"], l["arquivo"])))


def observacoes(taxon_id, place_id):
    """Gera observações research grade, paginando por id decrescente (sem limite de página)."""
    abaixo = None
    while True:
        params = {"taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
                  "photo_license": ",".join(sorted(LICENCAS)), "order_by": "id", "order": "desc",
                  "per_page": 100}
        if place_id:
            params["place_id"] = place_id
        if abaixo:
            params["id_below"] = abaixo
        r = sessao.get(API, params=params, timeout=30)
        r.raise_for_status()
        resultados = r.json()["results"]
        time.sleep(PAUSA)
        if not resultados:
            return
        yield from resultados
        abaixo = resultados[-1]["id"]


def foto_reutilizavel(obs):
    """Primeira foto da observação com licença aceita (uma foto por observação)."""
    for foto in obs.get("photos", []):
        if (foto.get("license_code") or "").lower() in LICENCAS:
            return foto
    return None


def baixar_classe(classe, destino, quantidade, buscar_brasil, creditos):
    taxon_id, prefixo = TAXONS[classe]
    pasta = destino / classe
    pasta.mkdir(parents=True, exist_ok=True)

    vistas = {l["url_observacao"] for l in creditos}          # evita baixar a mesma observação duas vezes
    numeros = [int(m.group(1)) for p in pasta.glob(f"{prefixo}_*.jpg")
               if (m := re.fullmatch(rf"{prefixo}_(\d+)", p.stem))]
    proximo = max(numeros, default=0) + 1                     # continua a numeração
    baixadas = 0

    for place_id in ([BRASIL, None] if buscar_brasil else [None]):
        origem = "Brasil" if place_id else "mundo"
        for obs in observacoes(taxon_id, place_id):
            if baixadas >= quantidade:
                break
            foto = foto_reutilizavel(obs)
            if foto is None or obs["uri"] in vistas:
                continue
            url = foto["url"].replace("/square.", "/large.")  # a API devolve a URL da miniatura
            r = sessao.get(url, timeout=60)
            time.sleep(PAUSA)
            if r.status_code != 200:
                continue
            arquivo = f"{prefixo}_{proximo:03d}.jpg"
            (pasta / arquivo).write_bytes(r.content)

            usuario = obs.get("user") or {}
            creditos.append({
                "arquivo": arquivo, "classe": classe,
                "autor": usuario.get("name") or usuario.get("login", ""),
                "licenca": foto["license_code"].upper(),
                "url_observacao": obs["uri"], "url_foto": url,
            })
            vistas.add(obs["uri"])
            proximo += 1
            baixadas += 1
            print(f"[{classe} {baixadas}/{quantidade}] {arquivo} ({origem}, {foto['license_code']})")
        if baixadas >= quantidade:
            break

    if baixadas < quantidade:
        print(f"Aviso: só {baixadas} fotos de {classe} disponíveis com esses filtros.")
    return baixadas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("destino", type=Path, help="gera destino/cobra e destino/escorpiao")
    ap.add_argument("--classe", choices=list(TAXONS), help="baixa só esta classe (padrão: as duas)")
    ap.add_argument("--quantidade", type=int, default=50,
                    help="fotos por classe (padrão 50, com margem para descartar as ruins)")
    ap.add_argument("--sem-brasil", action="store_true", help="não prioriza observações do Brasil")
    ap.add_argument("--creditos", type=Path, default=Path(__file__).with_name("creditos.csv"),
                    help="CSV de atribuição (padrão: rotulacao/creditos.csv)")
    args = ap.parse_args()

    creditos = ler_creditos(args.creditos)
    try:
        for classe in [args.classe] if args.classe else list(TAXONS):
            baixar_classe(classe, args.destino, args.quantidade, not args.sem_brasil, creditos)
    except requests.RequestException as e:
        print(f"Erro de rede: {e}. O que já foi baixado ficou registrado nos créditos.", file=sys.stderr)
    finally:
        salvar_creditos(args.creditos, creditos)              # grava os créditos mesmo se a execução for interrompida
    print(f"\nCréditos em {args.creditos}. Revise as fotos e descarte as ruins antes de rodar o autolabel.py.")


if __name__ == "__main__":
    main()
