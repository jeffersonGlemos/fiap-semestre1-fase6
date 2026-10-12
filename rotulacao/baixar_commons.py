"""Baixa fotos do Wikimedia Commons: o lote A (pasto) e o lote D (caneca).

Sem opções, o script baixa o lote A (vaca, arvore, cerca) das categorias "Cattle in Brazil",
"Pastures in Brazil", "Wooden fences in Brazil", "Fence posts" e "Barbed wire fences". Com
--categorias, --classe e --prefixo, o mesmo processo serve a outros lotes, como o das canecas.
As subcategorias diretas de cada categoria também entram. Aceito apenas arquivos JPEG com licença
livre (CC0, CC BY, CC BY-SA ou domínio público) e com resolução mínima. O total pedido é dividido
em cotas por categoria, para que o lote não fique composto só de fotos de gado e também traga
cercas, mourões e pasto com árvores. Só a imagem é baixada; a rotulação é feita neste projeto.

Para a caneca, procurei fotos do objeto em uso ou sobre uma mesa, parecidas com a cena da demonstração
ao vivo, e não fotos de catálogo. Duas opções ajudam nisso: --excluir descarta títulos que casem
com uma expressão regular (peças de museu, por exemplo) e --sem-fundo-branco descarta fotos cuja
borda é quase toda branca, o padrão das fotos de produto.

Cada foto baixada gera uma linha em creditos.csv (autor, licença e links), o mesmo arquivo usado
pelos lotes B e C do iNaturalist (baixar_bancos.py).

Uso:
    uv run baixar_commons.py brutas/pasto
    uv run baixar_commons.py brutas/pasto --quantidade 200 --largura 1600
    uv run baixar_commons.py brutas/caneca --classe caneca --quantidade 60 --sem-fundo-branco \\
        --categorias "Coffee mugs" "Mugs with logos" "People with mugs" "Mugs with soup" "Mugs" \\
        --excluir "museum|collection|\\(AM "

Uma nova execução continua a numeração e pula os arquivos já registrados nos créditos.
"""

import argparse
import html
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

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
BRANCO = 230                                                  # tom de cinza a partir do qual o pixel conta como branco

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
                          iiprop="url|size|mime|extmetadata|user", iiurlwidth=largura)  # iiurlwidth gera o thumburl
        for pagina in dados["query"]["pages"]:
            if pagina.get("imageinfo"):
                yield pagina["title"], pagina["imageinfo"][0]


def texto_meta(meta, chave):
    """Valor de um campo do extmetadata como texto puro (o Commons devolve HTML)."""
    valor = meta.get(chave, {}).get("value", "")
    return html.unescape(re.sub(r"<[^>]+>", "", valor)).strip()


def autor(meta, info):
    """Autor da foto: Artist, Attribution ou Credit do extmetadata, nessa ordem.

    Algumas páginas antigas não preenchem nenhum desses campos. Nesse caso registro o usuário que
    enviou o arquivo ao Commons, marcado como tal, porque em transferências do Flickr ou de outra
    Wikipédia quem enviou pode não ser o autor e a página do arquivo precisa ser conferida.
    """
    for chave in ("Artist", "Attribution", "Credit"):
        if valor := texto_meta(meta, chave):
            return valor
    return f"{info['user']} (enviou ao Commons)" if info.get("user") else "desconhecido"


def fundo_branco(conteudo, fracao=0.6):
    """True se mais que `fracao` da borda da foto for quase branca (típico de foto de produto)."""
    cinza = Image.open(io.BytesIO(conteudo)).convert("L")
    cinza.thumbnail((200, 200))                               # reduzir basta para medir a borda e é bem mais rápido
    largura, altura = cinza.size
    m = max(1, min(largura, altura) // 10)                    # borda com 10% do lado menor
    pixels = cinza.load()
    borda = [pixels[x, y] for x in range(largura) for y in range(altura)
             if x < m or y < m or x >= largura - m or y >= altura - m]
    return sum(p >= BRANCO for p in borda) > fracao * len(borda)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("destino", type=Path, help="pasta onde as fotos são gravadas (ex.: brutas/pasto)")
    ap.add_argument("--categorias", nargs="+", default=CATEGORIAS, metavar="CATEGORIA",
                    help="categorias do Commons, sem o prefixo Category: (padrão: as cinco do lote de pasto)")
    ap.add_argument("--classe", default="pasto",
                    help="valor da coluna classe em creditos.csv (padrão pasto)")
    ap.add_argument("--prefixo", help="início do nome dos arquivos, PREFIXO_commons_NNN.jpg (padrão: o valor de --classe)")
    ap.add_argument("--excluir", type=lambda r: re.compile(r, re.I), metavar="REGEX",
                    help="descarta arquivos cujo título casa com esta expressão regular (sem diferenciar maiúsculas)")
    ap.add_argument("--sem-fundo-branco", action="store_true",
                    help="descarta fotos com a borda quase toda branca, típicas de catálogo de produto")
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
    prefixo = args.prefixo or args.classe
    numero = 1 + sum(1 for c in creditos if c["classe"] == args.classe)  # continua a numeração

    por_categoria = arquivos_candidatos(args.categorias)
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
            # descarta o que não for JPEG, for pequeno demais, tiver licença não livre, já estiver nos créditos
            # ou tiver o título barrado por --excluir
            if (info.get("mime") != "image/jpeg" or min(info["width"], info["height"]) < args.minimo * 0.6
                    or max(info["width"], info["height"]) < args.minimo or not LICENCAS.match(licenca)
                    or info["descriptionurl"] in ja_baixadas
                    or (args.excluir and args.excluir.search(titulo))):
                continue
            # thumburl é a versão reduzida para a largura pedida (mais leve que o original de alta resolução);
            # se a imagem já for mais estreita que isso, o Commons não amplia e o thumburl aponta para o original
            url = info.get("thumburl") or info["url"]
            nome = f"{prefixo}_commons_{numero:03d}.jpg"
            try:
                conteudo = obter(url)
            except requests.RequestException as erro:
                print(f"falhou {titulo}: {erro}")
                continue
            # o fundo só pode ser avaliado depois do download; a foto descartada não volta a ser tentada nesta execução
            if args.sem_fundo_branco and fundo_branco(conteudo.content):
                print(f"descartada (fundo branco) {titulo}")
                ja_baixadas.add(info["descriptionurl"])
                time.sleep(PAUSA)
                continue
            (args.destino / nome).write_bytes(conteudo.content)
            creditos.append({"arquivo": nome, "classe": args.classe, "autor": autor(meta, info),
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
