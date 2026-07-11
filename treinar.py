#!/usr/bin/env python3
"""Script de treinamento do classificador de tiles de satelite.

Le um config YAML (deep-merge sobre defaults sensatos), prepara os dados via
dados.py, treina um modelo timm e registra tudo (historico.csv, TensorBoard,
checkpoints) em experimentos/<nome>/. Nunca hardcoda contagens ou nomes de
cena -- tudo vem do config e do que preparar_dados() encontrar em disco.

Uso:
    python3 treinar.py --config configs/smoke.yaml
    python3 treinar.py --config configs/exp1.yaml --permitir-cpu
    python3 treinar.py --config configs/exp1.yaml --sobrescrever
    python3 treinar.py --config configs/exp1.yaml --retomar
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from torch.utils.tensorboard import SummaryWriter

from dados import CLASSES, IDX, criar_loaders, pesos_das_classes, preparar_dados

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
PASTA_EXPERIMENTOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experimentos")

VERMELHO = "\033[91m"
RESET_COR = "\033[0m"

DEFAULTS = {
    "nome": None,
    "seed": 42,
    "dados": {
        "cenas": ["sat1", "sat2"],
        "excluir_borda": True,
        "fracao": 1.0,
        "max_por_classe": None,
        "img_size": 224,
    },
    "split": {
        "metodo": "aleatorio",
        "fracoes": [0.70, 0.15, 0.15],
        "bloco_px": 2048,
    },
    "modelo": {
        "arquitetura": "resnet18",
        "pretrained": True,
    },
    "balanceamento": {
        "metodo": "nenhum",
        "max_peso": 20.0,
    },
    "loss": {
        "tipo": "ce",
        "gamma": 2.0,
        "label_smoothing": 0.0,
    },
    "augmentation": "leve",
    "treino": {
        "epocas": 30,
        "batch": 128,
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "otimizador": "adamw",
        "scheduler": "cosseno",
        "amp": True,
        "num_workers": 2,
        "early_stopping_paciencia": 7,
        "metrica_checkpoint": "f1_macro",
    },
}

_ENUMS_VALIDOS = {
    ("split", "metodo"): {"aleatorio", "espacial"},
    ("balanceamento", "metodo"): {"nenhum", "pesos", "sampler"},
    ("loss", "tipo"): {"ce", "focal"},
    ("treino", "otimizador"): {"adamw", "sgd"},
    ("treino", "scheduler"): {"cosseno", "nenhum"},
    ("treino", "metrica_checkpoint"): {"f1_macro", "balanced_accuracy", "loss_val"},
}
_AUGMENTATIONS_VALIDAS = {"nenhuma", "leve", "pesada"}


# ---------------------------------------------------------------------------
# Config: deep-merge sobre defaults e validacao
# ---------------------------------------------------------------------------
def _deep_merge(base, override):
    """Mescla 'override' sobre 'base' recursivamente (dicts aninhados; demais
    tipos, incluindo listas, sao substituidos por completo)."""
    resultado = copy.deepcopy(base)
    for chave, valor in override.items():
        if isinstance(valor, dict) and isinstance(resultado.get(chave), dict):
            resultado[chave] = _deep_merge(resultado[chave], valor)
        else:
            resultado[chave] = valor
    return resultado


def _validar_config(cfg):
    if not cfg.get("nome"):
        raise ValueError(
            "config invalido: o campo 'nome' e obrigatorio (define a pasta "
            "experimentos/<nome>/)."
        )

    for (secao, campo), validos in _ENUMS_VALIDOS.items():
        valor = cfg[secao][campo]
        if valor not in validos:
            raise ValueError(
                f"config invalido: {secao}.{campo}={valor!r} nao e uma opcao valida. "
                f"Use um destes: {sorted(validos)}."
            )

    if cfg["augmentation"] not in _AUGMENTATIONS_VALIDAS:
        raise ValueError(
            f"config invalido: augmentation={cfg['augmentation']!r} nao e uma opcao "
            f"valida. Use um destes: {sorted(_AUGMENTATIONS_VALIDAS)}."
        )

    fracoes = cfg["split"]["fracoes"]
    if len(fracoes) != 3 or abs(sum(fracoes) - 1.0) > 1e-3:
        raise ValueError(
            f"config invalido: split.fracoes deve ter 3 valores somando ~1.0; "
            f"recebido {fracoes!r}."
        )

    if not cfg["modelo"]["arquitetura"]:
        raise ValueError("config invalido: modelo.arquitetura nao pode ser vazio.")

    # arquitetura em si e livre (qualquer nome aceito pelo timm); validada de
    # fato na hora de criar o modelo, onde o erro do timm ja e claro.


def carregar_config(caminho):
    """Le o YAML em 'caminho', aplica deep-merge sobre DEFAULTS e valida.

    Retorna o dict de config totalmente resolvido (com todos os defaults
    aplicados) -- esse e o dict salvo depois em experimentos/<nome>/config.yaml.
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"arquivo de config nao encontrado: {caminho}")

    with open(caminho, encoding="utf-8") as f:
        cfg_usuario = yaml.safe_load(f) or {}

    if not isinstance(cfg_usuario, dict):
        raise ValueError(
            f"config invalido: {caminho} nao contem um mapeamento YAML no topo."
        )

    cfg = _deep_merge(DEFAULTS, cfg_usuario)
    _validar_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss simples: CE por amostra -> pt=exp(-ce) -> ((1-pt)**gamma*ce).mean().

    'alpha' (pesos por classe) e passado como peso da cross-entropy interna,
    igual ao que CrossEntropyLoss(weight=...) faz.
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def montar_criterio(cfg, pesos):
    tipo = cfg["loss"]["tipo"]
    ls = cfg["loss"]["label_smoothing"]
    if tipo == "ce":
        return nn.CrossEntropyLoss(weight=pesos, label_smoothing=ls)
    if tipo == "focal":
        return FocalLoss(gamma=cfg["loss"]["gamma"], alpha=pesos, label_smoothing=ls)
    raise ValueError(f"tipo de loss desconhecido: {tipo!r}")  # ja validado antes


# ---------------------------------------------------------------------------
# Otimizador e scheduler
# ---------------------------------------------------------------------------
def montar_otimizador(cfg, parametros):
    t = cfg["treino"]
    if t["otimizador"] == "adamw":
        return torch.optim.AdamW(parametros, lr=t["lr"], weight_decay=t["weight_decay"])
    if t["otimizador"] == "sgd":
        return torch.optim.SGD(
            parametros, lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"]
        )
    raise ValueError(f"otimizador desconhecido: {t['otimizador']!r}")  # ja validado antes


def montar_scheduler(cfg, otimizador):
    t = cfg["treino"]
    if t["scheduler"] == "nenhum":
        return None
    if t["scheduler"] != "cosseno":
        raise ValueError(f"scheduler desconhecido: {t['scheduler']!r}")  # ja validado antes

    epocas = t["epocas"]
    warmup = min(2, epocas)

    def lr_lambda(epoca_idx):
        if warmup > 0 and epoca_idx < warmup:
            return (epoca_idx + 1) / warmup
        progresso = (epoca_idx - warmup) / max(1, epocas - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progresso, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(otimizador, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Treino e avaliacao de uma epoca
# ---------------------------------------------------------------------------
def treinar_uma_epoca(modelo, loader, otimizador, criterio, scaler, device, amp_enabled):
    modelo.train()
    soma_loss = 0.0
    total_n = 0
    for imgs, rotulos in loader:
        imgs = imgs.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        otimizador.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp_enabled):
            saidas = modelo(imgs)
            loss = criterio(saidas, rotulos)

        scaler.scale(loss).backward()
        scaler.step(otimizador)
        scaler.update()

        soma_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

    return soma_loss / total_n if total_n else float("nan")


@torch.no_grad()
def avaliar(modelo, loader, criterio, device, amp_enabled):
    """Avalia no loader dado. Retorna (loss_val, y_true, y_pred, y_prob), todos
    concatenados em arrays numpy (y_prob: probabilidades softmax por classe)."""
    modelo.eval()
    soma_loss = 0.0
    total_n = 0
    y_true, y_pred, y_prob = [], [], []

    for imgs, rotulos in loader:
        imgs = imgs.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        with torch.autocast(device_type=device, enabled=amp_enabled):
            saidas = modelo(imgs)
            loss = criterio(saidas, rotulos)

        soma_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

        probs = F.softmax(saidas.float(), dim=1)
        y_true.append(rotulos.cpu().numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=int)
    y_pred = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=int)
    y_prob = np.concatenate(y_prob) if y_prob else np.zeros((0, len(CLASSES)))
    loss_val = soma_loss / total_n if total_n else float("nan")
    return loss_val, y_true, y_pred, y_prob


def calcular_metricas(y_true, y_pred, classes_presentes):
    """Metricas de classificacao via sklearn. f1_macro/acc/bal_acc consideram
    todas as CLASSES; o recall por classe e reportado so para as presentes."""
    todas_labels = list(range(len(CLASSES)))
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=todas_labels, average="macro", zero_division=0)

    idx_presentes = [IDX[c] for c in classes_presentes]
    recalls = recall_score(
        y_true, y_pred, labels=idx_presentes, average=None, zero_division=0
    )
    recall_por_classe = {c: float(r) for c, r in zip(classes_presentes, recalls)}

    return {
        "acc": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "recall_por_classe": recall_por_classe,
    }


# ---------------------------------------------------------------------------
# Preparacao do diretorio do experimento
# ---------------------------------------------------------------------------
def preparar_dir_experimento(nome, sobrescrever, retomar):
    dir_exp = os.path.join(PASTA_EXPERIMENTOS, nome)
    historico = os.path.join(dir_exp, "historico.csv")
    ja_existe = os.path.isdir(dir_exp) and os.path.exists(historico)

    if retomar and not ja_existe:
        raise SystemExit(
            f"ERRO: --retomar foi passado, mas nao ha historico.csv em {dir_exp}. "
            f"Nao ha nada para retomar; remova --retomar para comecar um experimento novo."
        )

    if ja_existe and not retomar:
        if sobrescrever:
            shutil.rmtree(dir_exp)
        else:
            raise SystemExit(
                f"ERRO: o experimento '{nome}' ja existe em {dir_exp} (historico.csv "
                f"presente). Escolha outro nome, apague a pasta manualmente, ou rode "
                f"de novo com --sobrescrever para recomecar do zero."
            )

    os.makedirs(dir_exp, exist_ok=True)
    os.makedirs(os.path.join(dir_exp, "tb"), exist_ok=True)
    return dir_exp


# ---------------------------------------------------------------------------
# Execucao principal
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Treinamento do classificador de tiles de satelite (mar/terra/nuvem/objeto)."
    )
    p.add_argument("--config", required=True, help="Caminho do YAML de configuracao")
    p.add_argument(
        "--permitir-cpu", action="store_true",
        help="Permite treinar em CPU quando CUDA nao esta disponivel (bem mais lento)",
    )
    p.add_argument(
        "--sobrescrever", action="store_true",
        help="Apaga o experimento existente (se houver) e recomeca do zero",
    )
    p.add_argument(
        "--retomar", action="store_true",
        help="Continua o experimento existente a partir de ultimo.pt",
    )
    args = p.parse_args()

    try:
        cfg = carregar_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    nome = cfg["nome"]
    dir_exp = preparar_dir_experimento(nome, args.sobrescrever, args.retomar)
    print(f"experimento '{nome}' -> {dir_exp}")

    # --- seeds ---
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # --- device ---
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print(f"{VERMELHO}AVISO: CUDA nao disponivel neste ambiente. Treinar em CPU e "
              f"MUITO mais lento.{RESET_COR}")
        if not args.permitir_cpu:
            print("ERRO: rode de novo com --permitir-cpu para forcar a execucao em CPU.")
            sys.exit(1)
        device = "cpu"
    print(f"device: {device}")

    # --- dados ---
    try:
        df, resumo = preparar_dados(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO ao preparar os dados: {e}")
        sys.exit(1)

    print("resumo dos splits:")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    df[["arquivo", "rotulo", "cena", "split"]].to_csv(
        os.path.join(dir_exp, "splits.csv"), index=False
    )
    with open(os.path.join(dir_exp, "splits_resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    with open(os.path.join(dir_exp, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    classes_presentes_val = [c for c in CLASSES if resumo.get("val", {}).get(c, 0) > 0]

    # --- loaders ---
    if cfg["balanceamento"]["metodo"] == "sampler" and cfg["augmentation"] == "nenhuma":
        print("AVISO: balanceamento.metodo='sampler' com augmentation='nenhuma' faz o "
              "WeightedRandomSampler repetir exatamente os mesmos tiles da(s) classe(s) "
              "minoritaria(s) varias vezes por epoca (risco de decorar/overfit). "
              "Considere augmentation 'leve' ou 'pesada'.")

    dl_treino, dl_val, dl_test = criar_loaders(
        df,
        img_size=cfg["dados"]["img_size"],
        augmentation=cfg["augmentation"],
        batch=cfg["treino"]["batch"],
        num_workers=cfg["treino"]["num_workers"],
        metodo_balanceamento=cfg["balanceamento"]["metodo"],
        max_peso=cfg["balanceamento"]["max_peso"],
        seed=seed,
    )
    df_treino = df[df["split"] == "treino"]

    # --- modelo ---
    try:
        modelo = timm.create_model(
            cfg["modelo"]["arquitetura"],
            pretrained=cfg["modelo"]["pretrained"],
            num_classes=len(CLASSES),
        )
    except Exception as e:
        if cfg["modelo"]["pretrained"]:
            print(
                f"ERRO: falha ao criar o modelo '{cfg['modelo']['arquitetura']}' com "
                f"pesos pre-treinados (provavel falha de rede ao baixar os pesos): {e}\n"
                f"Verifique a conexao com a internet ou defina modelo.pretrained: false."
            )
        else:
            print(
                f"ERRO: falha ao criar o modelo '{cfg['modelo']['arquitetura']}': {e}\n"
                f"Verifique se o nome da arquitetura e valido para o timm."
            )
        sys.exit(1)
    modelo = modelo.to(device)

    # --- loss, otimizador, scheduler, amp ---
    pesos = None
    if cfg["balanceamento"]["metodo"] == "pesos":
        pesos = pesos_das_classes(df_treino, cfg["balanceamento"]["max_peso"]).to(device)
    criterio = montar_criterio(cfg, pesos)

    otimizador = montar_otimizador(cfg, modelo.parameters())
    scheduler = montar_scheduler(cfg, otimizador)

    amp_enabled = bool(cfg["treino"]["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # --- estado inicial / retomada ---
    epocas_totais = cfg["treino"]["epocas"]
    metrica_chave = cfg["treino"]["metrica_checkpoint"]
    maior_e_melhor = metrica_chave != "loss_val"

    caminho_melhor = os.path.join(dir_exp, "melhor.pt")
    caminho_ultimo = os.path.join(dir_exp, "ultimo.pt")
    caminho_historico = os.path.join(dir_exp, "historico.csv")

    melhor_valor = -float("inf") if maior_e_melhor else float("inf")
    melhor_epoca = 0
    contas_sem_melhora = 0
    epoca_inicial = 1

    if args.retomar and os.path.exists(caminho_ultimo):
        print(f"retomando de {caminho_ultimo}")
        ckpt = torch.load(caminho_ultimo, map_location=device, weights_only=False)
        modelo.load_state_dict(ckpt["state_dict"])
        otimizador.load_state_dict(ckpt["otimizador"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        epoca_inicial = ckpt["epoca"] + 1
        if os.path.exists(caminho_melhor):
            ckpt_melhor = torch.load(caminho_melhor, map_location="cpu", weights_only=False)
            melhor_valor = ckpt_melhor["metricas"][metrica_chave]
            melhor_epoca = ckpt_melhor["epoca"]
        print(f"retomando a partir da epoca {epoca_inicial}")

    escreve_header = not os.path.exists(caminho_historico)
    campos_historico = (
        ["epoca", "loss_treino", "loss_val", "acc", "balanced_accuracy", "f1_macro"]
        + [f"recall_{c}" for c in classes_presentes_val]
        + ["lr", "tempo_s"]
    )

    writer_tb = SummaryWriter(log_dir=os.path.join(dir_exp, "tb"))
    paciencia = cfg["treino"]["early_stopping_paciencia"]

    # --- loop de treino ---
    for epoca in range(epoca_inicial, epocas_totais + 1):
        t0 = time.time()

        loss_tr = treinar_uma_epoca(modelo, dl_treino, otimizador, criterio, scaler, device, amp_enabled)
        loss_val, y_true, y_pred, _y_prob = avaliar(modelo, dl_val, criterio, device, amp_enabled)
        metricas = calcular_metricas(y_true, y_pred, classes_presentes_val)

        lr_atual = otimizador.param_groups[0]["lr"]
        tempo = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        # --- print de 1 linha ---
        partes = [
            f"epoca {epoca}/{epocas_totais}",
            f"loss_tr {loss_tr:.3f}",
            f"loss_val {loss_val:.3f}",
            f"acc {metricas['acc']:.3f}",
            f"bal_acc {metricas['balanced_accuracy']:.3f}",
            f"f1 {metricas['f1_macro']:.3f}",
        ]
        for c in classes_presentes_val:
            partes.append(f"rec_{c} {metricas['recall_por_classe'][c]:.3f}")
        partes.append(f"({tempo:.1f}s)")
        print("  ".join(partes))

        # --- tensorboard ---
        writer_tb.add_scalar("loss/treino", loss_tr, epoca)
        writer_tb.add_scalar("loss/val", loss_val, epoca)
        writer_tb.add_scalar("val/acc", metricas["acc"], epoca)
        writer_tb.add_scalar("val/balanced_acc", metricas["balanced_accuracy"], epoca)
        writer_tb.add_scalar("val/f1_macro", metricas["f1_macro"], epoca)
        for c in classes_presentes_val:
            writer_tb.add_scalar(f"val/recall_{c}", metricas["recall_por_classe"][c], epoca)
        writer_tb.add_scalar("lr", lr_atual, epoca)
        writer_tb.flush()

        # --- historico.csv ---
        linha = {
            "epoca": epoca, "loss_treino": loss_tr, "loss_val": loss_val,
            "acc": metricas["acc"], "balanced_accuracy": metricas["balanced_accuracy"],
            "f1_macro": metricas["f1_macro"], "lr": lr_atual, "tempo_s": tempo,
        }
        for c in classes_presentes_val:
            linha[f"recall_{c}"] = metricas["recall_por_classe"][c]
        with open(caminho_historico, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos_historico)
            if escreve_header:
                w.writeheader()
                escreve_header = False
            w.writerow(linha)

        # --- checkpoint ---
        metricas_completas = dict(linha)
        valor_atual = metricas_completas[metrica_chave]
        melhorou = (valor_atual > melhor_valor) if maior_e_melhor else (valor_atual < melhor_valor)

        if melhorou:
            melhor_valor = valor_atual
            melhor_epoca = epoca
            contas_sem_melhora = 0
            torch.save(
                {
                    "state_dict": modelo.state_dict(),
                    "epoca": epoca,
                    "metricas": metricas_completas,
                    "config": cfg,
                },
                caminho_melhor,
            )
        else:
            contas_sem_melhora += 1

        torch.save(
            {
                "state_dict": modelo.state_dict(),
                "otimizador": otimizador.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoca": epoca,
                "metricas": metricas_completas,
                "config": cfg,
            },
            caminho_ultimo,
        )

        if paciencia > 0 and contas_sem_melhora >= paciencia:
            print(f"early stopping: sem melhora em {metrica_chave} por {paciencia} epocas "
                  f"(melhor epoca: {melhor_epoca}, valor: {melhor_valor:.4f})")
            break

    writer_tb.close()

    # --- avaliacao final com o melhor checkpoint ---
    if not os.path.exists(caminho_melhor):
        print("AVISO: nenhum checkpoint 'melhor.pt' foi salvo (nenhuma epoca rodou); "
              "pulando avaliacao final.")
        return

    ckpt_melhor = torch.load(caminho_melhor, map_location=device, weights_only=False)
    modelo.load_state_dict(ckpt_melhor["state_dict"])
    loss_val_final, y_true_final, y_pred_final, _ = avaliar(modelo, dl_val, criterio, device, amp_enabled)
    metricas_final = calcular_metricas(y_true_final, y_pred_final, classes_presentes_val)
    metricas_final["loss_val"] = loss_val_final
    metricas_final["epoca_melhor"] = ckpt_melhor["epoca"]

    with open(os.path.join(dir_exp, "metricas_val.json"), "w", encoding="utf-8") as f:
        json.dump(metricas_final, f, indent=2, ensure_ascii=False)

    print("avaliacao final (melhor checkpoint) no val:")
    print(json.dumps(metricas_final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
