#!/usr/bin/env python3
"""Script de avaliacao do classificador de tiles de satelite.

Modo 1 -- avalia um experimento ja treinado (le config.yaml e splits.csv do
proprio diretorio do experimento, reconstroi o modelo com timm, carrega um
checkpoint e roda inferencia no split pedido). Salva metricas, matriz de
confusao e predicoes dentro de experimentos/<nome>/.

Modo 2 -- compara todos os experimentos em experimentos/*/ que ja tenham
metricas_<split>.json (gerado por este script) ou metricas_val.json (gerado
por treinar.py), e imprime/salva uma tabela ordenada por f1_macro.

Nunca hardcoda contagens, cenas ou classes presentes -- tudo vem dos
artefatos do experimento (config.yaml, splits.csv, checkpoints).

Uso:
    python3 avaliar.py experimentos/smoke
    python3 avaliar.py experimentos/smoke --split val --checkpoint ultimo
    python3 avaliar.py experimentos/smoke --permitir-cpu
    python3 avaliar.py --comparar
    python3 avaliar.py --comparar --split val --csv comparacao.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from dados import CLASSES, IDX, PASTA_DADOS, TilesDataset, criar_transforms

PASTA_EXPERIMENTOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experimentos")

VERMELHO = "\033[91m"
RESET_COR = "\033[0m"


# ---------------------------------------------------------------------------
# Modo 1: avaliar um experimento
# ---------------------------------------------------------------------------
def _carregar_config_e_splits(dir_exp):
    caminho_config = os.path.join(dir_exp, "config.yaml")
    caminho_splits = os.path.join(dir_exp, "splits.csv")
    if not os.path.exists(caminho_config):
        raise FileNotFoundError(f"config.yaml nao encontrado em {dir_exp}")
    if not os.path.exists(caminho_splits):
        raise FileNotFoundError(f"splits.csv nao encontrado em {dir_exp}")

    import yaml
    with open(caminho_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(caminho_splits, dtype={"arquivo": str, "rotulo": str, "cena": str, "split": str})
    faltando = {"arquivo", "rotulo", "cena", "split"} - set(df.columns)
    if faltando:
        raise ValueError(f"splits.csv em {dir_exp} nao tem as colunas esperadas (faltam: {sorted(faltando)}).")

    df["caminho"] = df.apply(
        lambda r: os.path.join(PASTA_DADOS, f"{r['cena']}_tiles", r["arquivo"]), axis=1
    )
    return cfg, df


def _montar_modelo(cfg, device):
    try:
        modelo = timm.create_model(
            cfg["modelo"]["arquitetura"],
            pretrained=False,
            num_classes=len(CLASSES),
        )
    except Exception as e:
        raise SystemExit(
            f"ERRO: falha ao criar o modelo '{cfg['modelo']['arquitetura']}': {e}\n"
            f"Verifique se o nome da arquitetura e valido para o timm."
        )
    return modelo.to(device)


def _carregar_checkpoint(dir_exp, nome_checkpoint, device):
    caminho = os.path.join(dir_exp, f"{nome_checkpoint}.pt")
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"checkpoint '{nome_checkpoint}.pt' nao encontrado em {dir_exp}."
        )
    ckpt = torch.load(caminho, map_location=device, weights_only=False)
    return ckpt, caminho


@torch.no_grad()
def _inferir(modelo, loader, device):
    """Roda inferencia (softmax) no loader. Retorna arrays numpy y_true, y_pred, y_prob."""
    modelo.eval()
    y_true, y_pred, y_prob = [], [], []
    for imgs, rotulos in loader:
        imgs = imgs.to(device, non_blocking=True)
        saidas = modelo(imgs)
        probs = F.softmax(saidas.float(), dim=1)
        y_true.append(rotulos.numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=int)
    y_pred = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=int)
    y_prob = np.concatenate(y_prob) if y_prob else np.zeros((0, len(CLASSES)))
    return y_true, y_pred, y_prob


def _calcular_metricas(y_true, y_pred, y_prob, classes_presentes, epoca, checkpoint_usado, n_split):
    todas_labels = list(range(len(CLASSES)))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, labels=todas_labels, average="macro", zero_division=0))

    idx_presentes = [IDX[c] for c in classes_presentes]
    precisoes, recalls, f1s, suportes = precision_recall_fscore_support(
        y_true, y_pred, labels=idx_presentes, average=None, zero_division=0
    )
    por_classe = {}
    for c, p, r, f1c, s in zip(classes_presentes, precisoes, recalls, f1s, suportes):
        por_classe[c] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1c),
            "suporte": int(s),
        }

    ap_objeto = None
    if "objeto" in classes_presentes:
        y_true_bin = (y_true == IDX["objeto"]).astype(int)
        ap_objeto = float(average_precision_score(y_true_bin, y_prob[:, IDX["objeto"]]))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "por_classe": por_classe,
        "average_precision_objeto": ap_objeto,
        "epoca_checkpoint": epoca,
        "checkpoint_usado": checkpoint_usado,
        "n": n_split,
    }


def _salvar_matriz_confusao(y_true, y_pred, caminho_csv):
    todas_labels = list(range(len(CLASSES)))
    matriz = confusion_matrix(y_true, y_pred, labels=todas_labels)
    df_matriz = pd.DataFrame(matriz, index=CLASSES, columns=CLASSES)
    df_matriz.index.name = "verdadeiro\\predito"
    df_matriz.to_csv(caminho_csv)


def _salvar_predicoes(df_split, y_pred, y_prob, caminho_csv):
    saida = pd.DataFrame({
        "arquivo": df_split["arquivo"].tolist(),
        "cena": df_split["cena"].tolist(),
        "rotulo": df_split["rotulo"].tolist(),
        "predito": [CLASSES[i] for i in y_pred],
    })
    for c in CLASSES:
        saida[f"prob_{c}"] = y_prob[:, IDX[c]]
    saida.to_csv(caminho_csv, index=False)


def _imprimir_metricas(metricas, split, nome_exp):
    print(f"\n=== metricas ({nome_exp}, split={split}) ===")
    print(f"checkpoint: {metricas['checkpoint_usado']}  (epoca {metricas['epoca_checkpoint']})")
    print(f"n: {metricas['n']}")
    print(f"accuracy:          {metricas['accuracy']:.4f}")
    print(f"balanced_accuracy: {metricas['balanced_accuracy']:.4f}")
    print(f"f1_macro:          {metricas['f1_macro']:.4f}")
    if metricas["average_precision_objeto"] is not None:
        print(f"average_precision (objeto, one-vs-rest): {metricas['average_precision_objeto']:.4f}")
    else:
        print("average_precision (objeto): N/A (classe 'objeto' ausente no split)")
    print("\npor classe:")
    for c, m in metricas["por_classe"].items():
        print(f"  {c:8s} precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}  suporte={m['suporte']}")


def modo_avaliar(args):
    dir_exp = os.path.normpath(args.experimento)
    nome_exp = os.path.basename(dir_exp)

    if not os.path.isdir(dir_exp):
        print(f"ERRO: diretorio de experimento nao encontrado: {dir_exp}")
        sys.exit(1)

    try:
        cfg, df = _carregar_config_e_splits(dir_exp)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    # --- device ---
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print(f"{VERMELHO}AVISO: CUDA nao disponivel neste ambiente. Avaliar em CPU e "
              f"mais lento.{RESET_COR}")
        if not args.permitir_cpu:
            print("ERRO: rode de novo com --permitir-cpu para forcar a execucao em CPU.")
            sys.exit(1)
        device = "cpu"
    print(f"device: {device}")

    # --- split pedido ---
    df_split = df[df["split"] == args.split].reset_index(drop=True)
    if len(df_split) == 0:
        print(f"ERRO: o split '{args.split}' nao tem nenhum exemplo em {dir_exp}/splits.csv.")
        sys.exit(1)

    faltando_arquivo = [c for c in df_split["caminho"] if not os.path.exists(c)]
    if faltando_arquivo:
        exemplos = faltando_arquivo[:3]
        print(f"ERRO: {len(faltando_arquivo)} tile(s) do split '{args.split}' nao foram "
              f"encontrados em disco (ex.: {exemplos}). Verifique PASTA_DADOS em dados.py.")
        sys.exit(1)

    classes_presentes = [c for c in CLASSES if (df_split["rotulo"] == c).any()]

    # --- modelo + checkpoint ---
    modelo = _montar_modelo(cfg, device)
    try:
        ckpt, caminho_ckpt = _carregar_checkpoint(dir_exp, args.checkpoint, device)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        sys.exit(1)
    modelo.load_state_dict(ckpt["state_dict"])
    epoca_ckpt = ckpt["epoca"]

    # --- loader ---
    img_size = cfg["dados"]["img_size"]
    transform_eval = criar_transforms("nenhuma", img_size, treino=False)
    ds_split = TilesDataset(df_split, transform_eval)
    dl_split = torch.utils.data.DataLoader(
        ds_split,
        batch_size=cfg["treino"]["batch"],
        shuffle=False,
        num_workers=cfg["treino"]["num_workers"],
        pin_memory=True,
    )

    # --- inferencia ---
    y_true, y_pred, y_prob = _inferir(modelo, dl_split, device)

    # --- metricas ---
    metricas = _calcular_metricas(
        y_true, y_pred, y_prob, classes_presentes,
        epoca=epoca_ckpt, checkpoint_usado=f"{args.checkpoint}.pt", n_split=len(df_split),
    )

    # --- salvar artefatos ---
    caminho_metricas = os.path.join(dir_exp, f"metricas_{args.split}.json")
    with open(caminho_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    caminho_matriz = os.path.join(dir_exp, f"matriz_confusao_{args.split}.csv")
    _salvar_matriz_confusao(y_true, y_pred, caminho_matriz)

    caminho_predicoes = os.path.join(dir_exp, f"predicoes_{args.split}.csv")
    _salvar_predicoes(df_split, y_pred, y_prob, caminho_predicoes)

    print(f"\nsalvo: {caminho_metricas}")
    print(f"salvo: {caminho_matriz}")
    print(f"salvo: {caminho_predicoes}")

    _imprimir_metricas(metricas, args.split, nome_exp)


# ---------------------------------------------------------------------------
# Modo 2: comparar experimentos
# ---------------------------------------------------------------------------
def _epocas_do_experimento(dir_exp):
    """Numero de epocas rodadas (linhas de historico.csv), se existir."""
    caminho_hist = os.path.join(dir_exp, "historico.csv")
    if not os.path.exists(caminho_hist):
        return "-"
    try:
        df_hist = pd.read_csv(caminho_hist)
        return int(len(df_hist))
    except Exception:
        return "-"


def _linha_de_metricas_split(nome_exp, dir_exp, split, metricas):
    """Formato produzido por este script (modo_avaliar): metricas_<split>.json."""
    por_classe = metricas.get("por_classe", {})
    m_objeto = por_classe.get("objeto", {})
    return {
        "experimento": nome_exp,
        "split": split,
        "acc": metricas.get("accuracy", np.nan),
        "bal_acc": metricas.get("balanced_accuracy", np.nan),
        "f1_macro": metricas.get("f1_macro", np.nan),
        "rec_objeto": m_objeto.get("recall", np.nan),
        "prec_objeto": m_objeto.get("precision", np.nan),
        "ap_objeto": metricas.get("average_precision_objeto", np.nan),
        "n_objeto": m_objeto.get("suporte", 0),
        "epocas": _epocas_do_experimento(dir_exp),
    }


def _linha_de_metricas_val_treino(nome_exp, dir_exp, metricas):
    """Formato produzido por treinar.py: metricas_val.json (sem precision/ap,
    recall_por_classe aninhado). O suporte de 'objeto' vem de splits_resumo.json."""
    recall_por_classe = metricas.get("recall_por_classe", {})

    # suporte real de 'objeto' no val (0 fixo seria enganoso: pareceria
    # "nenhum objeto no split" quando na verdade e "desconhecido")
    n_objeto = np.nan
    caminho_resumo = os.path.join(dir_exp, "splits_resumo.json")
    if os.path.exists(caminho_resumo):
        try:
            with open(caminho_resumo, encoding="utf-8") as f:
                n_objeto = json.load(f).get("val", {}).get("objeto", 0)
        except Exception:
            pass

    return {
        "experimento": nome_exp,
        "split": "val",
        "acc": metricas.get("acc", np.nan),
        "bal_acc": metricas.get("balanced_accuracy", np.nan),
        "f1_macro": metricas.get("f1_macro", np.nan),
        "rec_objeto": recall_por_classe.get("objeto", np.nan),
        "prec_objeto": np.nan,
        "ap_objeto": np.nan,
        "n_objeto": n_objeto,
        "epocas": metricas.get("epoca_melhor", _epocas_do_experimento(dir_exp)),
    }


def modo_comparar(args):
    dirs_exp = sorted(d for d in glob.glob(os.path.join(PASTA_EXPERIMENTOS, "*")) if os.path.isdir(d))
    if not dirs_exp:
        print(f"ERRO: nenhum experimento encontrado em {PASTA_EXPERIMENTOS}.")
        sys.exit(1)

    linhas = []
    for dir_exp in dirs_exp:
        nome_exp = os.path.basename(dir_exp)

        caminho_split = os.path.join(dir_exp, f"metricas_{args.split}.json")
        if os.path.exists(caminho_split):
            with open(caminho_split, encoding="utf-8") as f:
                metricas = json.load(f)
            # metricas_val.json pode ter sido gerado por treinar.py (chaves 'acc'/
            # 'recall_por_classe') em vez deste script ('accuracy'/'por_classe') --
            # detecta o formato pelas chaves antes de interpretar.
            if "accuracy" in metricas:
                linhas.append(_linha_de_metricas_split(nome_exp, dir_exp, args.split, metricas))
            else:
                linhas.append(_linha_de_metricas_val_treino(nome_exp, dir_exp, metricas))
            continue

        caminho_val_treino = os.path.join(dir_exp, "metricas_val.json")
        if os.path.exists(caminho_val_treino):
            with open(caminho_val_treino, encoding="utf-8") as f:
                metricas = json.load(f)
            linhas.append(_linha_de_metricas_val_treino(nome_exp, dir_exp, metricas))
            continue

        print(f"AVISO: experimento '{nome_exp}' sem metricas_{args.split}.json nem "
              f"metricas_val.json; pulando.")

    if not linhas:
        print("ERRO: nenhum experimento com metricas disponiveis para comparar.")
        sys.exit(1)

    tabela = pd.DataFrame(linhas)
    tabela = tabela.sort_values("f1_macro", ascending=False, na_position="last").reset_index(drop=True)

    colunas = ["experimento", "split", "acc", "bal_acc", "f1_macro",
               "rec_objeto", "prec_objeto", "ap_objeto", "n_objeto", "epocas"]
    tabela = tabela[colunas]

    print(tabela.to_string(index=False))

    if args.csv:
        tabela.to_csv(args.csv, index=False)
        print(f"\nsalvo: {args.csv}")


# ---------------------------------------------------------------------------
# Execucao principal
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Avaliacao do classificador de tiles de satelite (mar/terra/nuvem/objeto)."
    )
    p.add_argument("experimento", nargs="?", default=None,
                   help="Caminho do diretorio do experimento (ex.: experimentos/smoke)")
    p.add_argument("--split", default="test", choices=["treino", "val", "test"],
                   help="Split a avaliar/comparar (padrao: test)")
    p.add_argument("--checkpoint", default="melhor", choices=["melhor", "ultimo"],
                   help="Checkpoint a carregar no modo de avaliacao (padrao: melhor)")
    p.add_argument("--permitir-cpu", action="store_true",
                   help="Permite avaliar em CPU quando CUDA nao esta disponivel")
    p.add_argument("--comparar", action="store_true",
                   help="Modo comparacao: varre experimentos/*/ em vez de avaliar um so")
    p.add_argument("--csv", default=None,
                   help="Caminho para salvar a tabela de comparacao (usado com --comparar)")
    args = p.parse_args()

    if args.comparar:
        modo_comparar(args)
        return

    if not args.experimento:
        print("ERRO: informe o diretorio do experimento, ou use --comparar.")
        print("Uso: python3 avaliar.py experimentos/<nome> [--split test|val] [--checkpoint melhor|ultimo]")
        print("     python3 avaliar.py --comparar [--split test] [--csv comparacao.csv]")
        sys.exit(1)

    modo_avaliar(args)


if __name__ == "__main__":
    main()
