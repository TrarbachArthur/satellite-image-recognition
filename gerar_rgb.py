#!/usr/bin/env python3
"""Converte imagens de satelite no formato "pan" (bandas r, g, b, p, nir separadas)
em uma imagem RGB normal, usando o algoritmo de pansharpening de Brovey.

Uso:
    python gerar_rgb.py                # pergunta o prefixo interativamente
    python gerar_rgb.py sat1           # processa data/sat1_{r,g,b,p,nir}.tif
    python gerar_rgb.py sat1 --escala 0.25   # gera saida em 1/4 da resolucao da Pan

Assume que os arquivos seguem o padrao <prefixo>_r.tif, <prefixo>_g.tif,
<prefixo>_b.tif, <prefixo>_p.tif e <prefixo>_nir.tif dentro da pasta 'data'.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import tifffile

PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUFIXOS = ["r", "g", "b", "p", "nir"]


def ler_banda(prefixo, sufixo):
    """Le uma banda a partir do prefixo e sufixo."""
    caminho = os.path.join(PASTA_DADOS, f"{prefixo}_{sufixo}.tif")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    print(f"  lendo {os.path.basename(caminho)} ...")
    banda = tifffile.imread(caminho)
    # Se a banda vier com um eixo de canal (ex: HxWx1), achata para 2D.
    if banda.ndim == 3 and banda.shape[2] == 1:
        banda = banda[:, :, 0]
    return banda


def brovey(prefixo, escala=1.0, saida=None):
    print(f"Processando prefixo '{prefixo}' ...")

    # --- Leitura das bandas ---
    R = ler_banda(prefixo, "r")
    G = ler_banda(prefixo, "g")
    B = ler_banda(prefixo, "b")
    Pan = ler_banda(prefixo, "p")

    classe_original = Pan.dtype
    linhas_pan, colunas_pan = Pan.shape[:2]

    # Resolucao de saida (permite reduzir para economizar memoria).
    largura = max(1, int(round(colunas_pan * escala)))
    altura = max(1, int(round(linhas_pan * escala)))
    print(f"Resolucao da Pan: {colunas_pan}x{linhas_pan} -> saida: {largura}x{altura}")

    # --- Redimensiona todas as bandas para a resolucao de saida ---
    # Redimensiona no tipo de dado nativo (uint8/uint16) e so depois converte
    # para float64. Isso evita criar uma copia float64 gigante em resolucao total
    # (para a Pan seriam dezenas de GB de RAM). cv2.resize usa (largura, altura).
    print("Redimensionando as bandas...")

    def redimensionar(banda):
        # INTER_AREA quando reduz (melhor qualidade); INTER_CUBIC (bicubica) quando amplia.
        h, w = banda.shape[:2]
        interp = cv2.INTER_AREA if (largura < w or altura < h) else cv2.INTER_CUBIC
        return cv2.resize(banda, (largura, altura), interpolation=interp).astype(np.float64)

    R = redimensionar(R)
    G = redimensionar(G)
    B = redimensionar(B)
    Pan = redimensionar(Pan)

    # --- Algoritmo de Brovey ---
    print("Aplicando pansharpening (Brovey)...")
    intensidade = (R + G + B) / 3.0
    intensidade[intensidade == 0] = np.finfo(np.float64).eps  # evita divisao por zero

    fator = Pan / intensidade
    R *= fator
    G *= fator
    B *= fator

    # --- Recorta (clip) conforme o tipo de dado original ---
    if classe_original == np.uint8:
        valor_max = 255
    elif classe_original == np.uint16:
        valor_max = 65535
    else:
        valor_max = float(Pan.max())

    rgb = np.stack([R, G, B], axis=-1)
    np.clip(rgb, 0, valor_max, out=rgb)

    # --- Converte para imagem "normal" de 8 bits para visualizacao ---
    # Escala do intervalo [0, valor_max] para [0, 255].
    rgb8 = (rgb / valor_max * 255.0).astype(np.uint8)

    # --- Salvamento ---
    if saida is None:
        saida = os.path.join(PASTA_DADOS, f"{prefixo}_rgb.png")
    # cv2 espera BGR na escrita.
    cv2.imwrite(saida, cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    print(f"Concluido! Imagem salva em: {saida}")
    return saida


def main():
    parser = argparse.ArgumentParser(description="Gera RGB (Brovey) a partir de bandas de satelite.")
    parser.add_argument("prefixo", nargs="?", help="Prefixo da imagem (ex: sat1)")
    parser.add_argument("--escala", type=float, default=1.0,
                        help="Fator de escala da resolucao de saida (ex: 0.25). Padrao: 1.0")
    parser.add_argument("--saida", help="Caminho do arquivo de saida (padrao: data/<prefixo>_rgb.png)")
    args = parser.parse_args()

    prefixo = args.prefixo or input("Digite o prefixo da imagem (ex: sat1): ").strip()
    if not prefixo:
        print("Prefixo vazio. Abortando.")
        sys.exit(1)

    try:
        brovey(prefixo, escala=args.escala, saida=args.saida)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
