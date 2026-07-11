#!/usr/bin/env python3
"""Camada de dados do pipeline de treinamento de classificacao de tiles de satelite.

Le os CSVs de rotulos gerados por rotular.py, filtra/amostra/divide os dados em
treino/val/test e monta os Dataset/DataLoader do PyTorch usados por treinar.py.

O volume de dados rotulados muda com o tempo (mais cenas, mais tiles). Por isso
nada aqui depende de contagens fixas, numero de classes presentes ou nomes de
cena "hardcoded" -- tudo e derivado dos CSVs em tempo de execucao. As unicas
constantes fixas sao as classes-alvo do modelo (CLASSES) e a ordem dos splits.

Uso (demonstracao rapida, sem treinar nada):
    python3 dados.py sat1 sat2
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Classes-alvo do modelo. 'incerto' nunca entra no treino -- e sempre descartado
# em filtrar(). Se uma classe nao existir nos dados carregados (ex.: uma cena
# sem 'objeto'), o restante do modulo trata isso normalmente (contagem 0).
CLASSES = ["mar", "terra", "nuvem", "objeto"]
IDX = {c: i for i, c in enumerate(CLASSES)}

SPLITS = ["treino", "val", "test"]

# Extrai (y, x) do nome do arquivo, ex.: sat1_y000768_x009472.png
_REGEX_POSICAO = re.compile(r"_y(\d+)_x(\d+)\.")

# Normalizacao padrao ImageNet (usada tanto em eval quanto em treino).
_MEAN_IMAGENET = [0.485, 0.456, 0.406]
_STD_IMAGENET = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Carregamento e limpeza
# ---------------------------------------------------------------------------
def carregar_rotulos(cenas, pasta_dados=PASTA_DADOS):
    """Le data/<cena>_tiles/<cena>_labels.csv (nome exato) para cada cena e
    concatena tudo num unico DataFrame.

    Colunas de saida: arquivo, rotulo, borda(int), cena, x(int), y(int), caminho(abs).

    ATENCAO: le APENAS o arquivo com o nome exato "<cena>_labels.csv". Nunca usa
    glob, entao arquivos de lixo como "<cena>_labels_bak.csv" ou
    "<cena>_labels - Copy.csv" sao ignorados automaticamente.
    """
    partes = []
    for cena in cenas:
        pasta_cena = os.path.join(pasta_dados, f"{cena}_tiles")
        caminho_csv = os.path.join(pasta_cena, f"{cena}_labels.csv")
        if not os.path.exists(caminho_csv):
            raise FileNotFoundError(
                f"CSV de rotulos nao encontrado para a cena '{cena}': {caminho_csv}. "
                f"Rotule a cena com 'python3 rotular.py {cena}' antes de treinar."
            )

        df_cena = pd.read_csv(caminho_csv, dtype={"arquivo": str, "rotulo": str})
        colunas_esperadas = {"arquivo", "rotulo", "borda", "timestamp"}
        faltando = colunas_esperadas - set(df_cena.columns)
        if faltando:
            raise ValueError(
                f"CSV {caminho_csv} nao tem as colunas esperadas (faltam: {sorted(faltando)}). "
                f"Colunas encontradas: {list(df_cena.columns)}."
            )

        df_cena["borda"] = df_cena["borda"].fillna(0).astype(int)
        df_cena["cena"] = cena

        xs, ys = [], []
        for arquivo in df_cena["arquivo"]:
            m = _REGEX_POSICAO.search(str(arquivo))
            if not m:
                raise ValueError(
                    f"Nao foi possivel extrair a posicao (y, x) do nome de arquivo "
                    f"'{arquivo}' na cena '{cena}'. Esperado o padrao "
                    f"'..._y<Y>_x<X>.<ext>' (ex.: sat1_y000768_x009472.png)."
                )
            ys.append(int(m.group(1)))
            xs.append(int(m.group(2)))
        df_cena["y"] = ys
        df_cena["x"] = xs
        df_cena["caminho"] = df_cena["arquivo"].apply(
            lambda a: os.path.abspath(os.path.join(pasta_cena, a))
        )

        partes.append(df_cena[["arquivo", "rotulo", "borda", "cena", "x", "y", "caminho"]])

    if not partes:
        raise ValueError("carregar_rotulos precisa de pelo menos uma cena.")

    df = pd.concat(partes, ignore_index=True)

    duplicados = df.loc[df["arquivo"].duplicated(keep=False), "arquivo"]
    if len(duplicados) > 0:
        exemplos = sorted(duplicados.unique())[:5]
        raise ValueError(
            f"Nomes de arquivo duplicados entre cenas: {exemplos}"
            f"{' (e outros)' if len(duplicados.unique()) > 5 else ''}. "
            f"Verifique se as cenas nao se sobrepoem ou se ha tiles com nome repetido."
        )

    return df


def filtrar(df, excluir_borda):
    """Remove 'incerto' e qualquer rotulo fora de CLASSES; opcionalmente remove
    tambem os tiles marcados com borda=1. Imprime as contagens antes/depois."""
    antes = df["rotulo"].value_counts().to_dict()
    print(f"filtrar: antes -> total={len(df)} contagens={antes}")

    df2 = df[df["rotulo"].isin(CLASSES)].copy()
    if excluir_borda:
        df2 = df2[df2["borda"] != 1].copy()

    depois = df2["rotulo"].value_counts().to_dict()
    print(f"filtrar: depois -> total={len(df2)} contagens={depois} "
          f"(excluir_borda={excluir_borda})")

    return df2.reset_index(drop=True)


def limitar(df, fracao, max_por_classe, seed):
    """Amostragem estratificada por classe, reprodutivel via seed.

    fracao: mantem ~fracao de cada classe presente. Piso de 3 exemplos por
        classe (ou n, se a classe tiver menos) -- o minimo para que dividir()
        consiga colocar 1 exemplo em cada um dos 3 splits.
    max_por_classe: teto aplicado depois da fracao (None = sem teto). Classes
        cujo resultado ja esta dentro do teto passam intactas.
    """
    rng = np.random.default_rng(seed)
    partes = []
    for classe, grupo in df.groupby("rotulo"):
        n = len(grupo)
        n_manter = min(n, max(3, int(round(fracao * n))))
        idx = grupo.index.to_numpy()
        escolhidos = rng.choice(idx, size=n_manter, replace=False)
        if max_por_classe is not None and len(escolhidos) > max_por_classe:
            escolhidos = rng.choice(escolhidos, size=max_por_classe, replace=False)
        print(f"limitar: {classe}: {n} -> {len(escolhidos)}")
        partes.append(df.loc[escolhidos])

    if not partes:
        return df.iloc[0:0].copy()

    return pd.concat(partes).sort_index().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Divisao treino/val/test
# ---------------------------------------------------------------------------
def _dividir_aleatorio(df, fracoes, rng):
    split = pd.Series(index=df.index, dtype=object)
    for _, grupo in df.groupby("rotulo"):
        idx = grupo.index.to_numpy()
        n = len(idx)
        idx_perm = idx[rng.permutation(n)]
        c1 = min(n, int(round(fracoes[0] * n)))
        c2 = min(n, max(c1, int(round((fracoes[0] + fracoes[1]) * n))))
        if n >= 3:
            # garante pelo menos 1 exemplo da classe em cada split (com n
            # pequeno o arredondamento zeraria val e/ou test)
            c1 = min(max(c1, 1), n - 2)
            c2 = min(max(c2, c1 + 1), n - 1)
        split.loc[idx_perm[:c1]] = "treino"
        split.loc[idx_perm[c1:c2]] = "val"
        split.loc[idx_perm[c2:]] = "test"
    return split


def _dividir_espacial(df, fracoes, rng, bloco_px):
    blocos_id = list(zip(df["cena"], df["y"] // bloco_px, df["x"] // bloco_px))
    df_tmp = df.copy()
    df_tmp["_bloco"] = blocos_id

    contagens_classe = df_tmp["rotulo"].value_counts()
    presentes = [c for c in CLASSES if c in contagens_classe.index]
    if not presentes:
        raise ValueError("dividir (espacial): nenhuma classe valida presente no DataFrame.")
    classe_rara = contagens_classe.loc[presentes].idxmin()

    info_blocos = {}
    for bid, sub in df_tmp.groupby("_bloco"):
        info_blocos[bid] = {
            "indices": sub.index.to_numpy(),
            "total": len(sub),
            "rara": int((sub["rotulo"] == classe_rara).sum()),
        }

    blocos_com_rara = [b for b, info in info_blocos.items() if info["rara"] > 0]
    blocos_outros = [b for b, info in info_blocos.items() if info["rara"] == 0]

    total_rara = sum(info_blocos[b]["rara"] for b in blocos_com_rara)
    total_tiles = len(df_tmp)

    alvo_rara = {s: fracoes[i] * total_rara for i, s in enumerate(SPLITS)}
    alvo_total = {s: fracoes[i] * total_tiles for i, s in enumerate(SPLITS)}
    atual_rara = {s: 0 for s in SPLITS}
    atual_total = {s: 0 for s in SPLITS}

    bloco_split = {}

    # Passo 1: blocos com a classe mais rara global, atribuidos primeiro para
    # que cada split fique com uma proporcao dela proxima de 'fracoes'.
    ordem = rng.permutation(len(blocos_com_rara))
    for i in ordem:
        b = blocos_com_rara[i]
        info = info_blocos[b]
        escolhido = max(SPLITS, key=lambda s: alvo_rara[s] - atual_rara[s])
        bloco_split[b] = escolhido
        atual_rara[escolhido] += info["rara"]
        atual_total[escolhido] += info["total"]

    # Passo 2: demais blocos, embaralhados e atribuidos pelo deficit de tiles.
    ordem2 = rng.permutation(len(blocos_outros))
    for i in ordem2:
        b = blocos_outros[i]
        info = info_blocos[b]
        escolhido = max(SPLITS, key=lambda s: alvo_total[s] - atual_total[s])
        bloco_split[b] = escolhido
        atual_total[escolhido] += info["total"]

    split = df_tmp["_bloco"].map(bloco_split)
    split.index = df_tmp.index
    return split


def dividir(df, metodo, fracoes, seed, bloco_px=2048):
    """Adiciona a coluna 'split' (treino/val/test) ao DataFrame.

    metodo 'aleatorio': estratificado por classe (nao evita vazamento espacial).
    metodo 'espacial': agrupa tiles em blocos (cena, y//bloco_px, x//bloco_px) e
        atribui blocos inteiros a um unico split, evitando que o mesmo bloco
        aparece em dois splits (anti-vazamento espacial).

    Ao final, valida que toda classe presente no df tem pelo menos 1 exemplo em
    cada split -- senao levanta ValueError.
    """
    if len(df) == 0:
        raise ValueError("dividir recebeu um DataFrame vazio; nada para dividir.")
    if len(fracoes) != 3:
        raise ValueError(f"fracoes deve ter 3 valores (treino, val, test); recebido: {fracoes}")

    rng = np.random.default_rng(seed)

    if metodo == "aleatorio":
        split = _dividir_aleatorio(df, fracoes, rng)
    elif metodo == "espacial":
        split = _dividir_espacial(df, fracoes, rng, bloco_px)
    else:
        raise ValueError(f"metodo de split desconhecido: {metodo!r}. Use 'aleatorio' ou 'espacial'.")

    df = df.copy()
    df["split"] = split

    presentes = [c for c in CLASSES if (df["rotulo"] == c).any()]
    faltando = [
        (c, s) for c in presentes for s in SPLITS
        if not (((df["rotulo"] == c) & (df["split"] == s)).any())
    ]
    if faltando:
        detalhes = ", ".join(f"{c}/{s}" for c, s in faltando)
        raise ValueError(
            f"O split resultou em 0 exemplos para: {detalhes}. "
            f"Aumente 'fracao' em limitar(), rotule mais tiles dessas classes, "
            f"ou troque o metodo/parametros de split."
        )

    return df


def resumo_splits(df):
    """Contagem por classe em cada split, para salvar em splits_resumo.json e auditar."""
    presentes = [c for c in CLASSES if (df["rotulo"] == c).any()]
    resumo = {}
    for s in SPLITS:
        sub = df[df["split"] == s]
        resumo[s] = {c: int((sub["rotulo"] == c).sum()) for c in presentes}
    resumo["total"] = {c: int((df["rotulo"] == c).sum()) for c in presentes}
    return resumo


# ---------------------------------------------------------------------------
# Dataset e transforms
# ---------------------------------------------------------------------------
class TilesDataset(torch.utils.data.Dataset):
    """Dataset que abre cada tile do disco sob demanda (nunca pre-carrega tudo
    em RAM). Guarda apenas listas de caminhos/rotulos, nao o DataFrame."""

    def __init__(self, df, transform=None):
        self.caminhos = df["caminho"].tolist()
        self.rotulos = df["rotulo"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        img = Image.open(self.caminhos[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, IDX[self.rotulos[i]]


class _RotacaoMultiplo90:
    """Rotaciona a imagem por um multiplo aleatorio de 90 graus (0/90/180/270).

    Tiles de satelite vistos de nadir sao invariantes a rotacao, entao isso e
    uma augmentation segura (diferente de uma rotacao arbitraria). Usa o RNG
    global do torch; para reprodutibilidade total entre execucoes, o chamador
    deve fixar a seed com torch.manual_seed(...) antes de treinar.
    """

    def __call__(self, img):
        angulo = int(torch.randint(0, 4, (1,)).item()) * 90
        if angulo == 0:
            return img
        return T.functional.rotate(img, angulo)


def criar_transforms(nivel, img_size, treino):
    """Monta o pipeline de transforms do torchvision.

    - eval (treino=False), ou treino=True com nivel='nenhuma':
        Resize + ToTensor + Normalize (ImageNet).
    - treino=True, nivel='leve':
        flips horizontal/vertical + rotacao aleatoria de 90 graus + resize + normalize.
    - treino=True, nivel='pesada':
        leve + RandomResizedCrop(scale=0.6-1.0) + ColorJitter + GaussianBlur
        ocasional (p~0.2) + RandomErasing (p=0.1, depois do ToTensor).
    """
    normalizar = T.Normalize(mean=_MEAN_IMAGENET, std=_STD_IMAGENET)

    if (not treino) or nivel == "nenhuma":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalizar,
        ])

    if nivel not in ("leve", "pesada"):
        raise ValueError(f"nivel de augmentation desconhecido: {nivel!r}. Use 'nenhuma', 'leve' ou 'pesada'.")

    passos = [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        _RotacaoMultiplo90(),
    ]

    if nivel == "leve":
        passos.append(T.Resize((img_size, img_size)))
    else:  # pesada
        passos.append(T.RandomResizedCrop(img_size, scale=(0.6, 1.0)))
        passos.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
        passos.append(T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2))

    passos.append(T.ToTensor())
    passos.append(normalizar)

    if nivel == "pesada":
        passos.append(T.RandomErasing(p=0.1))

    return T.Compose(passos)


# ---------------------------------------------------------------------------
# Balanceamento e DataLoaders
# ---------------------------------------------------------------------------
def pesos_das_classes(df_treino, max_peso):
    """peso_c = n_total / (n_classes_presentes * n_c), com clip em max_peso.

    Classes ausentes em df_treino recebem peso 0. Retorna um tensor float na
    ordem de CLASSES.
    """
    contagens = df_treino["rotulo"].value_counts()
    n_total = len(df_treino)
    presentes = [c for c in CLASSES if contagens.get(c, 0) > 0]

    pesos = torch.zeros(len(CLASSES), dtype=torch.float32)
    if not presentes or n_total == 0:
        return pesos

    n_classes_presentes = len(presentes)
    for c in presentes:
        n_c = int(contagens[c])
        peso = n_total / (n_classes_presentes * n_c)
        if max_peso is not None:
            peso = min(peso, max_peso)
        pesos[IDX[c]] = peso

    return pesos


def criar_loaders(df, img_size, augmentation, batch, num_workers, metodo_balanceamento, max_peso, seed):
    """Monta os tres DataLoaders (treino, val, test) a partir de um DataFrame
    que ja tem a coluna 'split'.

    metodo_balanceamento == 'sampler': usa WeightedRandomSampler no treino, com
        peso por amostra = peso da classe (clipado em max_peso). Caso contrario,
        o treino usa shuffle simples.
    """
    df_treino = df[df["split"] == "treino"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    transform_treino = criar_transforms(augmentation, img_size, treino=True)
    transform_eval = criar_transforms("nenhuma", img_size, treino=False)

    ds_treino = TilesDataset(df_treino, transform_treino)
    ds_val = TilesDataset(df_val, transform_eval)
    ds_test = TilesDataset(df_test, transform_eval)

    gerador = torch.Generator()
    gerador.manual_seed(seed)

    sampler = None
    shuffle_treino = True
    if metodo_balanceamento == "sampler":
        pesos_classe = pesos_das_classes(df_treino, max_peso)
        indices_classe = torch.tensor([IDX[r] for r in df_treino["rotulo"]], dtype=torch.long)
        pesos_amostra = pesos_classe[indices_classe]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=pesos_amostra,
            num_samples=len(df_treino),
            replacement=True,
            generator=gerador,
        )
        shuffle_treino = False

    dl_treino = torch.utils.data.DataLoader(
        ds_treino,
        batch_size=batch,
        shuffle=shuffle_treino if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # drop_last so quando ha mais de um batch; senao um smoke test com
        # poucos dados (len(treino) < batch) produziria 0 batches por epoca.
        drop_last=(len(ds_treino) >= batch),
        generator=gerador if sampler is None else None,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return dl_treino, dl_val, dl_test


# ---------------------------------------------------------------------------
# Funcao de conveniencia usada por treinar.py
# ---------------------------------------------------------------------------
def preparar_dados(cfg):
    """Encadeia carregar_rotulos -> filtrar -> limitar -> dividir -> resumo_splits.

    Espera cfg com o formato:
        cfg["seed"]: int
        cfg["dados"]: {
            "cenas": [...],
            "pasta_dados": opcional (padrao PASTA_DADOS),
            "excluir_borda": bool (padrao False),
            "fracao": float (padrao 1.0),
            "max_por_classe": int ou None (padrao None),
        }
        cfg["split"]: {
            "metodo": "aleatorio" ou "espacial" (padrao "aleatorio"),
            "fracoes": [treino, val, test] (padrao [0.7, 0.15, 0.15]),
            "bloco_px": int (padrao 2048, so usado no metodo 'espacial'),
        }

    Retorna (df_com_split, resumo).
    """
    cfg_dados = cfg["dados"]
    cfg_split = cfg["split"]
    seed = cfg["seed"]

    pasta_dados = cfg_dados.get("pasta_dados", PASTA_DADOS)
    df = carregar_rotulos(cfg_dados["cenas"], pasta_dados=pasta_dados)
    df = filtrar(df, excluir_borda=cfg_dados.get("excluir_borda", False))
    df = limitar(
        df,
        fracao=cfg_dados.get("fracao", 1.0),
        max_por_classe=cfg_dados.get("max_por_classe", None),
        seed=seed,
    )
    df = dividir(
        df,
        metodo=cfg_split.get("metodo", "aleatorio"),
        fracoes=cfg_split.get("fracoes", [0.7, 0.15, 0.15]),
        seed=seed,
        bloco_px=cfg_split.get("bloco_px", 2048),
    )
    resumo = resumo_splits(df)
    return df, resumo


# ---------------------------------------------------------------------------
# Execucao direta: resumo rapido dos dados de uma ou mais cenas
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cenas = sys.argv[1:]
    if not cenas:
        print("Uso: python3 dados.py <cena> [<cena> ...]")
        sys.exit(1)

    try:
        df = carregar_rotulos(cenas)
        df = filtrar(df, excluir_borda=False)
        df = dividir(df, metodo="aleatorio", fracoes=[0.7, 0.15, 0.15], seed=42)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    print(f"\nCarregado {len(df)} tiles rotulados de {len(cenas)} cena(s): {cenas}")
    import json
    print(json.dumps(resumo_splits(df), indent=2, ensure_ascii=False))
