"""Baixa fotos de pasto do Wikimedia Commons para o lote A (vaca, arvore, cerca).

As fotos vêm das categorias "Cattle in Brazil", "Pastures in Brazil", "Wooden fences in Brazil",
"Fence posts" e "Barbed wire fences", incluindo as subcategorias diretas de cada uma. Aceito apenas
arquivos JPEG com licença livre (CC0, CC BY, CC BY-SA ou domínio público) e com resolução mínima.
O total pedido é dividido em cotas por categoria, para que o lote não fique composto só de fotos
de gado e também traga cercas, mourões e pasto com árvores. Só a imagem é baixada; a rotulação
é feita neste projeto.

Cada foto baixada gera uma linha em creditos.csv (autor, licença e links), o mesmo arquivo usado
pelos lotes B e C do iNaturalist (baixar_bancos.py).

Uso:
    uv run baixar_commons.py brutas/pasto
    uv run baixar_commons.py brutas/pasto --quantidade 200 --largura 1600

Uma nova execução continua a numeração e pula os arquivos já registrados nos créditos.
"""

import argparse
import html
import re
import time
from pathlib import Path

import requests

from baixar_bancos import USER_AGENT, ler_creditos, salvar_creditos

API = "https://commons.wikimedia.org/w/api.php"
CATEGORIAS = [
    "Cattle in Brazil",
    "Pastures in Brazil",
    "Wooden fences in Brazil",
    "Fence posts",
    "Barbed wire fences",
]
# LicenseShortName do Commons: aceita CC0, CC BY x.x, CC BY-SA x.x e domínio público.
# Variantes como CC BY-NC ou CC BY-ND não casam e ficam de fora.
LICENCAS = re.compile(r"^(CC0|CC BY(-SA)? \d|Public domain)", re.I)
PAUSA = 1.0                                                   # segundos entre requisições

sessao = requests.Session()
sessao.headers["User-Agent"] = USER_AGENT


def obter(url, tentativas=5, **kwargs):
    """GET com novas tentativas quando o Commons responde 429 (limite de requisições)."""
    for tentativa in range(tentativas):
        resposta = sessao.get(url, timeout=60, **kwargs)
        if resposta.status_code != 429:
            resposta.raise_for_status()
            return resposta
        # respeita o Retry-After do servidor; sem ele, espera 5, 10, 20, 40 e 80 s (backoff exponencial)
        espera = int(resposta.headers.get("Retry-After", 0)) or 5 * 2 ** tentativa
        print(f"limite de requisições atingido, aguardando {espera} s")
        time.sleep(espera)
    resposta.raise_for_status()                               # esgotou as tentativas: propaga o erro 429


def consultar(**params):
    """Chamada action=query da API do MediaWiki, com pausa depois de cada requisição."""
    # maxlag=5 pede ao servidor que recuse a consulta quando a replicação estiver atrasada mais de
    # 5 s, como a política de uso da API recomenda para scripts automáticos
    params.update(action="query", format="json", formatversion=2, maxlag=5)
    resposta = obter(API, params=params)
    time.sleep(PAUSA)
    return resposta.json()


def membros(categoria, tipo):
    """Títulos dos membros da categoria do tipo pedido ("file" ou "subcat"), com paginação."""
    titulos, continuar = [], {}
    while True:
        dados = consultar(list="categorymembers", cmtitle=f"Category:{categoria}",
                          cmtype=tipo, cmlimit=500, **continuar)
        titulos += [m["title"] for m in dados["query"]["categorymembers"]]
        if "continue" not in dados:
            return titulos
        continuar = dados["continue"]                         # token da próxima página


def arquivos_candidatos(categorias):
    """Arquivos de cada categoria e das subcategorias diretas, sem repetir entre categorias."""
    por_categoria, vistos = {}, set()
    for categoria in categorias:
        subcats = [s.removeprefix("Category:") for s in membros(categoria, "subcat")]
        por_categoria[categoria] = []
        for cat in [categoria, *subcats]:
            for titulo in membros(cat, "file"):
                if titulo not in vistos:                      # arquivo em várias categorias conta só na primeira
                    vistos.add(titulo)
                    por_categoria[categoria].append(titulo)
    return por_categoria


def detalhes(titulos, largura):
    """Gera (título, imageinfo) em lotes de 50 títulos, o máximo por consulta da API."""
    for i in range(0, len(titulos), 50):
        dados = consultar(prop="imageinfo", titles="|".join(titulos[i:i + 50]),
                          iiprop="url|size|mime|extmetadata", iiurlwidth=largura)  # iiurlwidth gera o thumburl
        for pagina in dados["query"]["pages"]:
            if pagina.get("imageinfo"):
                yield pagina["title"], pagina["imageinfo"][0]


def texto_meta(meta, chave):
    """Valor de um campo do extmetadata como texto puro (o Commons devolve HTML)."""
    valor = meta.get(chave, {}).get("value", "")
    return html.unescape(re.sub(r"<[^>]+>", "", valor)).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("destino", type=Path, help="pasta onde as fotos são gravadas (ex.: brutas/pasto)")
    ap.add_argument("--quantidade", type=int, default=200,
                    help="total de fotos, dividido em cotas entre as categorias (padrão 200)")
    ap.add_argument("--largura", type=int, default=1600,
                    help="largura em pixels da versão baixada (padrão 1600)")
    ap.add_argument("--minimo", type=int, default=1000,
                    help="lado maior mínimo do original em pixels; o lado menor precisa ter "
                         "pelo menos 60%% disso (padrão 1000)")
    ap.add_argument("--creditos", type=Path, default=Path(__file__).with_name("creditos.csv"),
                    help="CSV de atribuição compartilhado com os lotes B e C (padrão: rotulacao/creditos.csv)")
    args = ap.parse_args()

    args.destino.mkdir(parents=True, exist_ok=True)
    creditos = ler_creditos(args.creditos)
    ja_baixadas = {c["url_observacao"] for c in creditos}     # evita baixar de novo o mesmo arquivo
    numero = 1 + sum(1 for c in creditos if c["classe"] == "pasto")  # continua a numeração

    por_categoria = arquivos_candidatos(CATEGORIAS)
    print(f"{sum(map(len, por_categoria.values()))} arquivos nas categorias")
    baixadas = 0
    for indice, (categoria, titulos) in enumerate(por_categoria.items()):
        # divide o que falta entre as categorias restantes: se uma categoria não tiver fotos
        # suficientes que passem nos filtros, a sobra vai para as seguintes
        cota = (args.quantidade - baixadas) // (len(por_categoria) - indice)
        na_categoria = 0
        for titulo, info in detalhes(titulos, args.largura):
            if na_categoria >= cota:
                break
            meta = info.get("extmetadata", {})
            licenca = texto_meta(meta, "LicenseShortName")
            # descarta o que não for JPEG, for pequeno demais, tiver licença não livre ou já estiver nos créditos
            if (info.get("mime") != "image/jpeg" or min(info["width"], info["height"]) < args.minimo * 0.6
                    or max(info["width"], info["height"]) < args.minimo or not LICENCAS.match(licenca)
                    or info["descriptionurl"] in ja_baixadas):
                continue
            # thumburl é a versão reduzida para a largura pedida (mais leve que o original de alta resolução);
            # se a imagem já for mais estreita que isso, o Commons não amplia e o thumburl aponta para o original
            url = info.get("thumburl") or info["url"]
            nome = f"pasto_commons_{numero:03d}.jpg"
            try:
                conteudo = obter(url)
            except requests.RequestException as erro:
                print(f"falhou {titulo}: {erro}")
                continue
            (args.destino / nome).write_bytes(conteudo.content)
            creditos.append({"arquivo": nome, "classe": "pasto", "autor": texto_meta(meta, "Artist") or "desconhecido",
                             "licenca": licenca, "url_observacao": info["descriptionurl"], "url_foto": url})
            salvar_creditos(args.creditos, creditos)          # grava a cada foto; se o script parar, nada se perde
            ja_baixadas.add(info["descriptionurl"])
            numero += 1
            na_categoria += 1
            baixadas += 1
            print(f"[{baixadas}/{args.quantidade}] {nome}  {licenca}  {categoria}: {titulo}")
            time.sleep(PAUSA)

    print(f"\n{baixadas} fotos em {args.destino}. Créditos em {args.creditos}")


if __name__ == "__main__":
    main()
