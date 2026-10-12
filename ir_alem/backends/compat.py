"""Compatibilidade para carregar no PC os pesos do YOLOv5 treinados no Colab.

O checkpoint do YOLOv5 não guarda só os tensores: é um pickle que também leva os argumentos do treino
(`opt`), e um deles (`evolve_population`) é um objeto `pathlib.PosixPath`. Isso cria dois problemas
quando o modelo sai do Colab:

1. **Versão do Python.** Treinei com Python 3.13, em que as classes de `pathlib` passaram para o
   submódulo `pathlib._local`, e é esse nome que fica gravado no pickle. Em Python 3.12 ou anterior
   esse submódulo não existe e o `torch.hub.load` falha com "No module named 'pathlib._local'".
2. **Sistema operacional.** O Colab é Linux, então o caminho foi gravado como `PosixPath`, que o
   Windows se recusa a instanciar ("cannot instantiate 'PosixPath' on your system"), um problema
   conhecido do YOLOv5.

`carregamento_portavel()` resolve os dois durante o `torch.load`/`torch.hub.load`. A solução
definitiva para distribuir o modelo é a cópia `*_portavel.pt` gerada pelo `exportar_modelo.py`, em
que esses objetos já foram convertidos para `str`; mesmo com ela, manter o ajuste no carregamento não
custa nada, porque ele não faz nada quando não é necessário.
"""

import contextlib
import importlib.util
import pathlib
import sys


def ajustar_pathlib():
    """Registra `pathlib._local` como apelido do módulo `pathlib` quando o Python não o tem (3.12 ou anterior).

    O pickle procura as classes por `sys.modules[modulo]`, então basta o apelido para que
    `pathlib._local.PosixPath` seja resolvido como `pathlib.PosixPath`. No Python 3.13+ o submódulo
    existe de verdade e nada é alterado. Chamar a função mais de uma vez não tem efeito adicional.
    """
    if "pathlib._local" in sys.modules:
        return
    if getattr(pathlib, "__path__", None) and importlib.util.find_spec("pathlib._local"):
        return                                               # Python 3.13+: o submódulo real existe
    sys.modules["pathlib._local"] = pathlib


@contextlib.contextmanager
def carregamento_portavel():
    """Contexto para envolver o carregamento de um checkpoint do YOLOv5 treinado em outra máquina.

    Aplica `ajustar_pathlib()` e, só no Windows, faz `PosixPath` apontar para `WindowsPath` enquanto
    o checkpoint é lido, restaurando o valor original na saída. A troca é feita tanto em `pathlib`
    quanto em `pathlib._local`, porque o pickle do Python 3.13 referencia o submódulo.
    """
    ajustar_pathlib()
    if sys.platform != "win32":
        yield
        return
    modulos = {id(m): m for m in (pathlib, sys.modules.get("pathlib._local")) if m is not None}.values()
    originais = [(m, m.PosixPath) for m in modulos]
    try:
        for m, _ in originais:
            m.PosixPath = pathlib.WindowsPath
        yield
    finally:
        for m, original in originais:                        # devolve o PosixPath mesmo se o carregamento falhar
            m.PosixPath = original


def sem_pathlib(obj, _vistos=None):
    """Troca, recursivamente e no próprio objeto, todo `pathlib.PurePath` por `str`, e devolve o resultado.

    Percorre dicionários, listas, tuplas, conjuntos e o `__dict__` de objetos comuns (tensores são
    ignorados). Uso no checkpoint do YOLOv5 (`opt`, `train_args` etc.) para gerar uma cópia que
    carrega em qualquer versão do Python e em qualquer sistema operacional.
    """
    vistos = set() if _vistos is None else _vistos
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, (str, bytes, int, float, bool, type(None))) or id(obj) in vistos:
        return obj
    vistos.add(id(obj))
    if isinstance(obj, dict):
        for chave, valor in list(obj.items()):
            novo = sem_pathlib(valor, vistos)
            if novo is not valor:
                obj[chave] = novo
        return obj
    if isinstance(obj, list):
        for i, valor in enumerate(obj):
            novo = sem_pathlib(valor, vistos)
            if novo is not valor:
                obj[i] = novo
        return obj
    if isinstance(obj, (tuple, set, frozenset)):
        itens = [sem_pathlib(v, vistos) for v in obj]
        if all(a is b for a, b in zip(itens, obj)):
            return obj                                       # nada mudou: mantém o objeto original
        if hasattr(obj, "_fields"):                          # namedtuple
            return type(obj)(*itens)
        return type(obj)(itens)
    if type(obj).__module__.startswith("torch") and not hasattr(obj, "_modules"):
        return obj                                           # tensores e afins: nada a converter
    if hasattr(obj, "__dict__"):                             # objetos comuns, inclusive os nn.Module do modelo
        sem_pathlib(vars(obj), vistos)
    return obj
