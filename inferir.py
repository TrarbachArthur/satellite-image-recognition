#!/usr/bin/env python3
"""Pre-rotulagem de tiles de satelite por inferencia com modelos ja treinados.

Le o manifesto de uma pasta de tiles (gerado por gerar_tiles.py), roda
inferencia com um ou mais experimentos ja treinados (ensemble por media das
probabilidades softmax quando ha mais de um) e escreve um CSV de rotulos no
MESMO formato que rotular.py produz (arquivo,rotulo,borda,timestamp), pronto
para revisao manual com corrigir.py.

NUNCA sobrescreve um <prefixo>_labels.csv existente sem --sobrescrever: esse
arquivo pode conter rotulagem manual e a protecao evita perda de trabalho.

Nunca hardcoda contagens, cenas ou experimentos -- tudo vem do manifesto de
tiles e dos diretorios de experimento informados na linha de comando.

Uso:
    python3 inferir.py sat3 --experimentos experimentos/vit_small_pretrained
    python3 inferir.py --tiles data/sat3_tiles --experimentos experimentos/resnet18_base experimentos/vit_small_pretrained
    python3 inferir.py sat3 --experimentos experimentos/resnet18_base --limiar-incerto 0.6
    python3 inferir.py sat3 --experimentos experimentos/resnet18_base --sobrescrever
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch

from dados import CLASSES, IDX, PASTA_DADOS, criar_transforms

VERMELHO = "\033[91m"
RESET_COR = "\033[0m"

CABECALHO_LABELS = ["arquivo", "rotulo", "borda", "timestamp"]


# ---------------------------------------------------------------------------
# Resolucao de pasta de tiles / prefixo (mesmo padrao de rotular.py)
# ---------------------------------------------------------------------------
def resolver_tiles_dir(args):
    """Determina a pasta de tiles e o prefixo a partir dos argumentos."""
    if args.tiles:
        tiles_dir = os.path.abspath(args.tiles)
        prefixo = args.prefixo or os.path.basename(tiles_dir.rstrip("/")).replace("_tiles", "")
    else:
        if not args.prefixo:
            args.prefixo = input("Digite o prefixo da imagem (ex: sat2): ").strip()
        prefixo = args.prefixo
        tiles_dir = os.path.join(PASTA_DADOS, f"{prefixo}_tiles")
    return tiles_dir, prefixo


# ---------------------------------------------------------------------------
# Dataset local para tiles sem rotulo
# ---------------------------------------------------------------------------
class TilesSemRotulo(torch.utils.data.Dataset):
    """Dataset minimo que abre cada tile sob demanda e retorna so o tensor da
    imagem (sem rotulo -- usado para inferencia em tiles ainda nao rotulados).
    """

    def __init__(self, caminhos, transform):
        from PIL import Image
        self._Image = Image
        self.caminhos = caminhos
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        img = self._Image.open(self.caminhos[i]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Manifesto de tiles
# ---------------------------------------------------------------------------
def carregar_manifesto(tiles_dir, prefixo):
    """Le <tiles_dir>/<prefixo>_tiles.csv (nome exato) e mantem so os PNGs que
    ainda existem em disco. Retorna a lista de nomes de arquivo (na ordem do
    manifesto)."""
    caminho_manifesto = os.path.join(tiles_dir, f"{prefixo}_tiles.csv")
    if not os.path.exists(caminho_manifesto):
        raise FileNotFoundError(
            f"manifesto de tiles nao encontrado: {caminho_manifesto}. "
            f"Gere os tiles com 'python3 gerar_tiles.py {prefixo}' antes de inferir."
        )

    df_manifesto = pd.read_csv(caminho_manifesto, dtype={"arquivo": str})
    if "arquivo" not in df_manifesto.columns:
        raise ValueError(
            f"{caminho_manifesto} nao tem a coluna 'arquivo'. "
            f"Colunas encontradas: {list(df_manifesto.columns)}."
        )

    nomes = df_manifesto["arquivo"].tolist()
    existentes = [n for n in nomes if os.path.exists(os.path.join(tiles_dir, n))]
    faltando = len(nomes) - len(existentes)
    if faltando > 0:
        print(f"AVISO: {faltando} tile(s) listado(s) no manifesto nao foram "
              f"encontrados em disco e serao ignorados.")

    if not existentes:
        raise ValueError(f"Nenhum tile do manifesto {caminho_manifesto} existe em disco.")

    return existentes


# ---------------------------------------------------------------------------
# Protecao do labels.csv existente
# ---------------------------------------------------------------------------
def checar_protecao_labels(labels_csv, sobrescrever):
    if os.path.exists(labels_csv) and not sobrescrever:
        raise SystemExit(
            f"ERRO: {labels_csv} ja existe e pode conter rotulagem manual; "
            f"use --sobrescrever se tiver certeza de que quer substitui-lo."
        )


# ---------------------------------------------------------------------------
# Experimentos: validacao e carregamento
# ---------------------------------------------------------------------------
def validar_experimentos(dirs_exp, nome_checkpoint):
    """Confere que cada diretorio de experimento existe e tem <checkpoint>.pt."""
    for dir_exp in dirs_exp:
        if not os.path.isdir(dir_exp):
            raise SystemExit(f"ERRO: diretorio de experimento nao encontrado: {dir_exp}")
        caminho_ckpt = os.path.join(dir_exp, f"{nome_checkpoint}.pt")
        if not os.path.exists(caminho_ckpt):
            raise SystemExit(
                f"ERRO: checkpoint '{nome_checkpoint}.pt' nao encontrado em {dir_exp}."
            )


def _montar_modelo_do_checkpoint(ckpt, device):
    cfg = ckpt["config"]
    arquitetura = cfg["modelo"]["arquitetura"]
    try:
        modelo = timm.create_model(arquitetura, pretrained=False, num_classes=len(CLASSES))
    except Exception as e:
        raise SystemExit(
            f"ERRO: falha ao criar o modelo '{arquitetura}': {e}\n"
            f"Verifique se o nome da arquitetura e valido para o timm."
        )
    modelo.load_state_dict(ckpt["state_dict"])
    modelo = modelo.to(device)
    modelo.eval()
    return modelo, cfg


@torch.no_grad()
def _inferir_probs(modelo, loader, device, n_tiles, n_classes):
    """Roda inferencia softmax no loader inteiro. Retorna array numpy
    (n_tiles, n_classes) float32 na CPU, na ordem do loader (shuffle=False)."""
    probs_todas = np.zeros((n_tiles, n_classes), dtype=np.float32)
    offset = 0
    for i, imgs in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        saidas = modelo(imgs)
        probs = torch.nn.functional.softmax(saidas.float(), dim=1).cpu().numpy()
        n = probs.shape[0]
        probs_todas[offset:offset + n] = probs
        offset += n
        if (i + 1) % 50 == 0:
            print(f"    batch {i + 1}: {offset}/{n_tiles} tiles processados")
    return probs_todas


def rodar_experimento(dir_exp, nome_checkpoint, caminhos, device, batch, num_workers):
    """Carrega um experimento, roda inferencia em todos os tiles e retorna as
    probabilidades softmax (n_tiles, n_classes) em float32 na CPU. Libera o
    modelo da VRAM ao final."""
    caminho_ckpt = os.path.join(dir_exp, f"{nome_checkpoint}.pt")
    ckpt = torch.load(caminho_ckpt, map_location=device, weights_only=False)
    modelo, cfg = _montar_modelo_do_checkpoint(ckpt, device)

    img_size = cfg["dados"]["img_size"]
    transform = criar_transforms("nenhuma", img_size, treino=False)
    ds = TilesSemRotulo(caminhos, transform)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  experimento '{os.path.basename(dir_exp)}': arquitetura={cfg['modelo']['arquitetura']} "
          f"img_size={img_size} checkpoint={nome_checkpoint}.pt")
    t0 = time.time()
    probs = _inferir_probs(modelo, loader, device, len(caminhos), len(CLASSES))
    tempo = time.time() - t0
    print(f"  experimento '{os.path.basename(dir_exp)}' concluido em {tempo:.1f}s "
          f"({len(caminhos) / tempo:.1f} tiles/s)" if tempo > 0 else
          f"  experimento '{os.path.basename(dir_exp)}' concluido em {tempo:.1f}s")

    del modelo
    if device == "cuda":
        torch.cuda.empty_cache()

    return probs


# ---------------------------------------------------------------------------
# Escrita atomica dos CSVs de saida
# ---------------------------------------------------------------------------
def escrever_labels_csv(labels_csv, nomes, rotulos):
    """Escreve <prefixo>_labels.csv no mesmo formato de rotular.py: escrita
    atomica (temp + os.replace), cabecalho arquivo,rotulo,borda,timestamp."""
    tmp = labels_csv + ".tmp"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CABECALHO_LABELS)
        for nome, rotulo in zip(nomes, rotulos):
            w.writerow([nome, rotulo, 0, ts])
    os.replace(tmp, labels_csv)


def escrever_predicoes_csv(caminho_csv, nomes, preditos, confiancas, prob_final):
    tmp = caminho_csv + ".tmp"
    df_saida = pd.DataFrame({
        "arquivo": nomes,
        "predito": preditos,
        "confianca": confiancas,
    })
    for c in CLASSES:
        df_saida[f"prob_{c}"] = prob_final[:, IDX[c]]
    df_saida.to_csv(tmp, index=False)
    os.replace(tmp, caminho_csv)


# ---------------------------------------------------------------------------
# Execucao principal
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Pre-rotulagem de tiles de satelite por inferencia com modelos ja treinados."
    )
    p.add_argument("prefixo", nargs="?", default=None,
                   help="Prefixo da cena (ex: sat3); tiles em data/<prefixo>_tiles")
    p.add_argument("--tiles", default=None,
                   help="Pasta de tiles alternativa (prefixo derivado do nome da pasta)")
    p.add_argument("--experimentos", nargs="+", required=True,
                   help="1+ diretorios de experimentos treinados (ensemble se houver mais de um)")
    p.add_argument("--checkpoint", default="melhor", choices=["melhor", "ultimo"],
                   help="Qual checkpoint carregar de cada experimento (padrao: melhor)")
    p.add_argument("--limiar-incerto", type=float, default=0.0,
                   help="0.0 = desligado. Se >0, tiles com prob. maxima do ensemble abaixo "
                        "desse valor recebem rotulo 'incerto' (padrao: 0.0)")
    p.add_argument("--batch", type=int, default=128, help="Batch de inferencia (padrao: 128)")
    p.add_argument("--num-workers", type=int, default=2, help="Workers do DataLoader (padrao: 2)")
    p.add_argument("--permitir-cpu", action="store_true",
                   help="Permite inferir em CPU quando CUDA nao esta disponivel")
    p.add_argument("--sobrescrever", action="store_true",
                   help="Sobrescreve <prefixo>_labels.csv mesmo se ja existir")
    args = p.parse_args()

    tiles_dir, prefixo = resolver_tiles_dir(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERRO: pasta de tiles nao encontrada: {tiles_dir}")
        sys.exit(1)

    # --- manifesto ---
    try:
        nomes = carregar_manifesto(tiles_dir, prefixo)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)
    caminhos = [os.path.join(tiles_dir, n) for n in nomes]
    print(f"manifesto: {len(nomes)} tile(s) encontrados em {tiles_dir}")

    # --- protecao do labels.csv (antes de gastar tempo de inferencia) ---
    labels_csv = os.path.join(tiles_dir, f"{prefixo}_labels.csv")
    try:
        checar_protecao_labels(labels_csv, args.sobrescrever)
    except SystemExit as e:
        print(e)
        sys.exit(1)

    # --- device ---
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print(f"{VERMELHO}AVISO: CUDA nao disponivel neste ambiente. Inferir em CPU e "
              f"mais lento.{RESET_COR}")
        if not args.permitir_cpu:
            print("ERRO: rode de novo com --permitir-cpu para forcar a execucao em CPU.")
            sys.exit(1)
        device = "cpu"
    print(f"device: {device}")

    # --- validacao dos experimentos ---
    try:
        validar_experimentos(args.experimentos, args.checkpoint)
    except SystemExit as e:
        print(e)
        sys.exit(1)

    # --- inferencia: um passe sequencial por experimento (ensemble por soma
    # das probabilidades softmax, um modelo por vez para economizar VRAM) ---
    soma_probs = np.zeros((len(caminhos), len(CLASSES)), dtype=np.float32)
    print(f"\nrodando inferencia com {len(args.experimentos)} experimento(s):")
    for dir_exp in args.experimentos:
        probs = rodar_experimento(
            dir_exp, args.checkpoint, caminhos, device, args.batch, args.num_workers
        )
        soma_probs += probs

    prob_final = soma_probs / len(args.experimentos)

    # --- decisao final por tile ---
    idx_argmax = prob_final.argmax(axis=1)
    confiancas = prob_final[np.arange(len(caminhos)), idx_argmax]
    preditos = [CLASSES[i] for i in idx_argmax]

    if args.limiar_incerto > 0.0:
        rotulos = [
            p if c >= args.limiar_incerto else "incerto"
            for p, c in zip(preditos, confiancas)
        ]
    else:
        rotulos = list(preditos)

    # --- saidas ---
    escrever_labels_csv(labels_csv, nomes, rotulos)
    predicoes_csv = os.path.join(tiles_dir, f"{prefixo}_predicoes_auto.csv")
    escrever_predicoes_csv(predicoes_csv, nomes, preditos, confiancas, prob_final)

    print(f"\nsalvo: {labels_csv}")
    print(f"salvo: {predicoes_csv}")

    # --- resumo ---
    serie_rotulos = pd.Series(rotulos)
    distribuicao = serie_rotulos.value_counts().to_dict()
    n_objeto_candidato = int((prob_final[:, IDX["objeto"]] >= 0.5).sum()) if "objeto" in IDX else 0

    print(f"\n=== resumo ({prefixo}) ===")
    print(f"n tiles: {len(caminhos)}")
    print(f"distribuicao dos rotulos gravados: {distribuicao}")
    print(f"confianca media (prob. maxima do ensemble): {confiancas.mean():.4f}")
    print(f"candidatos a 'objeto' (prob_objeto >= 0.5): {n_objeto_candidato}")
    print(f"\nproximas etapas sugeridas:")
    print(f"  python3 visualizar.py {prefixo}   # visualizar a rotulagem gerada")
    print(f"  python3 corrigir.py {prefixo}     # revisar/corrigir tiles incertos ou mal classificados")


if __name__ == "__main__":
    main()
