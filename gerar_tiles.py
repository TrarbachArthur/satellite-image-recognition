#!/usr/bin/env python3
"""Converte imagens de satelite no formato "pan" (bandas r, g, b, p, nir separadas)
em uma GRADE DE TILES RGB (imagens normais), usando pansharpening de Brovey.

A imagem final (fusao RGB + Pancromatica em resolucao total) NUNCA e montada
inteira na memoria: o Pan e lido em faixas de linhas (streaming) e a saida e
recortada em tiles menores, ideais para deteccao de objetos.

Uso:
    python gerar_tiles.py                     # pergunta o prefixo
    python gerar_tiles.py sat2                # tiles 1024, escala 1.0, PNG
    python gerar_tiles.py sat2 --tile 640     # tiles de 640x640
    python gerar_tiles.py sat2 --overlap 64   # 64px de sobreposicao entre tiles
    python gerar_tiles.py sat2 --formato jpg  # salva JPG em vez de PNG
    python gerar_tiles.py sat2 --pular-vazias # descarta tiles quase uniformes (mar aberto)

Assume os sufixos <prefixo>_r/g/b/p/nir.tif dentro da pasta 'data'.
O RGB Brovey usa apenas r, g, b, p (o nir nao entra na imagem RGB visivel).
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import tifffile

PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def abrir_pagina(prefixo, sufixo):
    """Abre a pagina TIFF de uma banda (sem carregar os dados)."""
    caminho = os.path.join(PASTA_DADOS, f"{prefixo}_{sufixo}.tif")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    return tifffile.TiffFile(caminho).pages[0]


def ler_faixa(page, r0, r1):
    """Le as linhas [r0:r1) de uma pagina TIFF decodificando apenas os strips
    necessarios (nao carrega o arquivo inteiro)."""
    H, W = page.shape[0], page.shape[1]
    r0, r1 = max(0, r0), min(H, r1)
    rps = page.rowsperstrip or 1
    offs, cnts = page.dataoffsets, page.databytecounts
    fh = page.parent.filehandle
    s0, s1 = r0 // rps, (r1 - 1) // rps
    partes = []
    for s in range(s0, s1 + 1):
        fh.seek(offs[s])
        seg, _, shp = page.decode(fh.read(cnts[s]), s)
        partes.append(np.asarray(seg).reshape(shp[-3], shp[-2]))
    bloco = np.concatenate(partes, axis=0)
    inicio = r0 - s0 * rps
    return bloco[inicio:inicio + (r1 - r0)]


def ler_completo(page):
    """Le uma banda inteira (usar so em bandas pequenas: R, G, B)."""
    arr = page.asarray()
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


# ---------------------------------------------------------------------------
# Brovey
# ---------------------------------------------------------------------------
def brovey_bandas(R, G, B, Pan):
    """Aplica a fusao de Brovey a arrays float32 ja alinhados (mesmo shape)."""
    I = (R + G + B) / 3.0
    I[I == 0] = np.finfo(np.float32).eps  # evita divisao por zero
    f = Pan / I
    return R * f, G * f, B * f


def calcular_limites(pan_page, R, G, B, alvo=1600):
    """Estima limites de contraste (percentis 2 e 98) por canal a partir de uma
    versao reduzida da imagem, para que TODOS os tiles usem a mesma escala 8-bit."""
    print("Estimando contraste global (amostra reduzida)...")
    H = pan_page.shape[0]
    passo = max(1, H // alvo)
    linhas = [ler_faixa(pan_page, i, i + 1)[0] for i in range(0, H, passo)]
    pan_s = np.asarray(linhas, dtype=np.float32)
    h2 = pan_s.shape[0]
    w2 = min(pan_page.shape[1], alvo)
    pan_s = cv2.resize(pan_s, (w2, h2), interpolation=cv2.INTER_AREA)
    Rr = cv2.resize(R.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    Gr = cv2.resize(G.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    Br = cv2.resize(B.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    bR, bG, bB = brovey_bandas(Rr, Gr, Br, pan_s)
    lo, hi = [], []
    for banda in (bR, bG, bB):
        l, h = np.percentile(banda, (2, 98))
        lo.append(l)
        hi.append(max(h, l + 1e-6))
    print(f"  limites por canal (R,G,B): lo={np.round(lo,1)} hi={np.round(hi,1)}")
    return np.array(lo, np.float32), np.array(hi, np.float32)


# ---------------------------------------------------------------------------
# Processamento principal (streaming + tiles)
# ---------------------------------------------------------------------------
def processar(prefixo, tile=1024, escala=1.0, overlap=0,
              formato="png", saida_dir=None, pular_vazias=False, limiar_vazio=3.0):
    print(f"Processando '{prefixo}' (tile={tile}, escala={escala}, overlap={overlap})")

    pan_page = abrir_pagina(prefixo, "p")
    R = ler_completo(abrir_pagina(prefixo, "r")).astype(np.float32)
    G = ler_completo(abrir_pagina(prefixo, "g")).astype(np.float32)
    B = ler_completo(abrir_pagina(prefixo, "b")).astype(np.float32)

    pan_h, pan_w = pan_page.shape[0], pan_page.shape[1]
    rgb_h, rgb_w = R.shape[:2]
    out_h = max(1, int(round(pan_h * escala)))
    out_w = max(1, int(round(pan_w * escala)))
    print(f"  Pan {pan_w}x{pan_h}  ->  saida {out_w}x{out_h}")

    limites_lo, limites_hi = calcular_limites(pan_page, R, G, B)
    faixa = limites_hi - limites_lo

    if saida_dir is None:
        saida_dir = os.path.join(PASTA_DADOS, f"{prefixo}_tiles")
    os.makedirs(saida_dir, exist_ok=True)

    # Coordenadas de origem dos tiles (com sobreposicao opcional).
    passo = max(1, tile - overlap)
    xs_tiles = list(range(0, out_w, passo))
    ys_tiles = list(range(0, out_h, passo))
    print(f"  Grade: {len(xs_tiles)} x {len(ys_tiles)} = "
          f"{len(xs_tiles) * len(ys_tiles)} tiles (antes de filtros)")

    # Fatores de reamostragem (saida -> coordenadas nativas das bandas).
    sx_r, sy_r = rgb_w / out_w, rgb_h / out_h
    sx_p, sy_p = pan_w / out_w, pan_h / out_h

    manifesto = []
    salvos = descartados = 0

    # Processa uma faixa (linha de tiles) por vez: le o Pan da faixa uma unica
    # vez e recorta os tiles em colunas. O remap e feito POR TILE (dimensoes
    # pequenas) porque cv2.remap so aceita ate 32767px por eixo.
    for ty in ys_tiles:
        bh = min(tile, out_h - ty)

        # Pan da faixa: leitura direta em escala 1.0; reamostrado caso contrario.
        if escala == 1.0:
            pan_faixa = ler_faixa(pan_page, ty, ty + bh).astype(np.float32)
            pan_src_y0 = ty  # origem em coordenadas do Pan
        else:
            py0 = max(0, int(np.floor(ty * sy_p)) - 2)
            py1 = min(pan_h, int(np.ceil((ty + bh) * sy_p)) + 3)
            pan_faixa = ler_faixa(pan_page, py0, py1).astype(np.float32)
            pan_src_y0 = py0

        # Coordenadas-fonte (globais) das linhas de saida desta faixa.
        ys_rgb = (np.arange(ty, ty + bh, dtype=np.float32) + 0.5) * sy_r - 0.5

        for tx in xs_tiles:
            bw = min(tile, out_w - tx)

            # --- RGB: remap direto das bandas nativas (fonte < 32767px) ---
            xs_rgb = (np.arange(tx, tx + bw, dtype=np.float32) + 0.5) * sx_r - 0.5
            mx = np.ascontiguousarray(np.broadcast_to(xs_rgb, (bh, bw)))
            my = np.ascontiguousarray(np.broadcast_to(ys_rgb[:, None], (bh, bw)))
            Ru = cv2.remap(R, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            Gu = cv2.remap(G, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            Bu = cv2.remap(B, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # --- Pan do tile ---
            if escala == 1.0:
                Pan = pan_faixa[:, tx:tx + bw]
            else:
                cx0 = max(0, int(np.floor(tx * sx_p)) - 2)
                cx1 = min(pan_w, int(np.ceil((tx + bw) * sx_p)) + 3)
                sub = pan_faixa[:, cx0:cx1]
                pmx = np.ascontiguousarray(np.broadcast_to(
                    ((np.arange(tx, tx + bw, dtype=np.float32) + 0.5) * sx_p - 0.5 - cx0), (bh, bw)))
                pmy = np.ascontiguousarray(np.broadcast_to(
                    (ys_rgb * 0 + (np.arange(ty, ty + bh, dtype=np.float32) + 0.5) * sy_p - 0.5 - pan_src_y0)[:, None],
                    (bh, bw)))
                Pan = cv2.remap(sub, pmx, pmy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # --- Brovey + normalizacao 8-bit com limites globais ---
            bR, bG, bB = brovey_bandas(Ru, Gu, Bu, Pan)
            chip = np.empty((bh, bw, 3), dtype=np.uint8)
            for c, banda in enumerate((bR, bG, bB)):
                v = (banda - limites_lo[c]) / faixa[c] * 255.0
                chip[:, :, c] = np.clip(v, 0, 255).astype(np.uint8)

            if pular_vazias and float(chip.std()) < limiar_vazio:
                descartados += 1
                continue

            nome = f"{prefixo}_y{ty:06d}_x{tx:06d}.{formato}"
            cv2.imwrite(os.path.join(saida_dir, nome), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
            manifesto.append([nome, tx, ty, bw, bh])
            salvos += 1

        print(f"  faixa y={ty:6d} ({bh}px) -> {salvos} tiles salvos ate agora")

    # Manifesto para mapear cada tile de volta a imagem completa.
    with open(os.path.join(saida_dir, f"{prefixo}_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "x", "y", "largura", "altura"])
        w.writerows(manifesto)

    print(f"Concluido! {salvos} tiles salvos em {saida_dir}"
          + (f" ({descartados} vazios descartados)" if pular_vazias else ""))
    return saida_dir


def main():
    p = argparse.ArgumentParser(description="Gera tiles RGB (Brovey) a partir de bandas de satelite.")
    p.add_argument("prefixo", nargs="?", help="Prefixo da imagem (ex: sat2)")
    p.add_argument("--tile", type=int, default=1024, help="Tamanho do tile em pixels (padrao 1024)")
    p.add_argument("--escala", type=float, default=1.0, help="Escala da resolucao de saida (padrao 1.0)")
    p.add_argument("--overlap", type=int, default=0, help="Sobreposicao entre tiles em pixels (padrao 0)")
    p.add_argument("--formato", choices=["png", "jpg"], default="png", help="Formato de saida (padrao png)")
    p.add_argument("--saida-dir", help="Pasta de saida (padrao data/<prefixo>_tiles)")
    p.add_argument("--pular-vazias", action="store_true", help="Descarta tiles quase uniformes (mar aberto)")
    args = p.parse_args()

    prefixo = args.prefixo or input("Digite o prefixo da imagem (ex: sat2): ").strip()
    if not prefixo:
        print("Prefixo vazio. Abortando.")
        sys.exit(1)

    try:
        processar(prefixo, tile=args.tile, escala=args.escala, overlap=args.overlap,
                  formato=args.formato, saida_dir=args.saida_dir, pular_vazias=args.pular_vazias)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
