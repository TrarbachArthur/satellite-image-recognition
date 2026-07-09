#!/usr/bin/env python3
"""Visualizacao dos tiles rotulados.

Gera, a partir dos rotulos de uma cena:
  - Um COMPOSITE por classe: a imagem real de cada tile colada na sua posicao
    correta na cena, com o resto preto (uma imagem por rotulo).
  - Um composite dedicado aos tiles marcados com borda.
  - Um OVERVIEW: a cena real reduzida com cada tile contornado por um quadrado
    colorido por classe, e os tiles de borda destacados (contorno magenta).

Como a resolucao real e inviavel, tudo e gerado numa escala configuravel.

Uso:
    python3 visualizar.py sat1
    python3 visualizar.py sat1 --escala 0.15
    python3 visualizar.py --tiles data/sat1_tiles
"""

import argparse
import csv
import os
import sys

from PIL import Image, ImageDraw

PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Cores por classe (RGB). Classes ausentes no CSV sao simplesmente ignoradas.
CORES = {
    "mar": (40, 110, 230),
    "terra": (60, 160, 70),
    "nuvem": (235, 235, 235),
    "objeto": (235, 45, 45),
    "incerto": (240, 205, 40),
}
COR_BORDA = (235, 45, 235)  # magenta
COR_DESCONHECIDO = (150, 150, 150)


def carregar(tiles_dir, prefixo):
    """Junta manifesto (posicoes) + labels (rotulo/borda) pela coluna 'arquivo'."""
    man = os.path.join(tiles_dir, f"{prefixo}_tiles.csv")
    lab = os.path.join(tiles_dir, f"{prefixo}_labels.csv")
    if not os.path.exists(man):
        raise FileNotFoundError(f"Manifesto nao encontrado: {man}")
    if not os.path.exists(lab):
        raise FileNotFoundError(f"CSV de rotulos nao encontrado: {lab}")

    rotulos = {}
    with open(lab, newline="") as f:
        for r in csv.DictReader(f):
            rotulos[r["arquivo"]] = (
                r.get("rotulo", "") or "",
                1 if str(r.get("borda", "0")) in ("1", "True", "true") else 0,
            )

    tiles = []
    with open(man, newline="") as f:
        for r in csv.DictReader(f):
            nome = r["arquivo"]
            rotulo, borda = rotulos.get(nome, ("", 0))
            if not rotulo:
                continue  # ignora tiles ainda nao rotulados
            tiles.append({
                "arquivo": nome,
                "x": int(r["x"]), "y": int(r["y"]),
                "w": int(r["largura"]), "h": int(r["altura"]),
                "rotulo": rotulo, "borda": borda,
            })
    return tiles


def gerar(prefixo, tiles_dir, escala, saida_dir, espessura_extra=2):
    tiles = carregar(tiles_dir, prefixo)
    if not tiles:
        raise RuntimeError("Nenhum tile rotulado encontrado.")

    # Dimensao da cena (posicoes absolutas preservadas a partir de 0,0).
    cena_w = max(t["x"] + t["w"] for t in tiles)
    cena_h = max(t["y"] + t["h"] for t in tiles)
    out_w = max(1, round(cena_w * escala))
    out_h = max(1, round(cena_h * escala))

    classes = sorted({t["rotulo"] for t in tiles})
    print(f"{prefixo}: {len(tiles)} tiles rotulados | cena {cena_w}x{cena_h} "
          f"-> saida {out_w}x{out_h} | classes: {classes}")

    # Canvases: overview (fundo real) + um por classe + borda.
    overview = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    composites = {c: Image.new("RGB", (out_w, out_h), (0, 0, 0)) for c in classes}
    comp_borda = Image.new("RGB", (out_w, out_h), (0, 0, 0))

    # --- Passo unico de I/O: abre cada tile uma vez e reaproveita a miniatura ---
    print("Colando tiles (passo de I/O)...")
    for i, t in enumerate(tiles):
        px, py = round(t["x"] * escala), round(t["y"] * escala)
        tw, th = max(1, round(t["w"] * escala)), max(1, round(t["h"] * escala))
        try:
            img = Image.open(os.path.join(tiles_dir, t["arquivo"])).convert("RGB")
            if img.size != (tw, th):
                img = img.resize((tw, th))
        except Exception as e:
            print(f"  aviso: falha ao abrir {t['arquivo']}: {e}")
            continue
        overview.paste(img, (px, py))
        composites[t["rotulo"]].paste(img, (px, py))
        if t["borda"]:
            comp_borda.paste(img, (px, py))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(tiles)} tiles")

    # --- Passo de desenho (sem I/O): contornos no overview ---
    print("Desenhando contornos no overview...")
    draw = ImageDraw.Draw(overview)
    for t in tiles:
        px, py = round(t["x"] * escala), round(t["y"] * escala)
        tw, th = max(1, round(t["w"] * escala)), max(1, round(t["h"] * escala))
        cor = CORES.get(t["rotulo"], COR_DESCONHECIDO)
        # 'objeto' (raro) recebe contorno mais grosso para nao se perder na grade.
        largura = 1 + (espessura_extra if t["rotulo"] == "objeto" else 0)
        draw.rectangle([px, py, px + tw - 1, py + th - 1], outline=cor, width=largura)

    # Contorno magenta (por cima) nos tiles de borda.
    for t in tiles:
        if not t["borda"]:
            continue
        px, py = round(t["x"] * escala), round(t["y"] * escala)
        tw, th = max(1, round(t["w"] * escala)), max(1, round(t["h"] * escala))
        draw.rectangle([px, py, px + tw - 1, py + th - 1], outline=COR_BORDA, width=1)

    _desenhar_legenda(draw, classes)

    # --- Salvamento ---
    os.makedirs(saida_dir, exist_ok=True)
    salvos = []

    def salvar(img, nome):
        caminho = os.path.join(saida_dir, nome)
        img.save(caminho)
        salvos.append(caminho)

    salvar(overview, f"{prefixo}_overview.png")
    for c in classes:
        salvar(composites[c], f"{prefixo}_composite_{c}.png")
    salvar(comp_borda, f"{prefixo}_composite_borda.png")

    print(f"Concluido! {len(salvos)} imagens salvas em {saida_dir}:")
    for p in salvos:
        print("  " + os.path.basename(p))
    return saida_dir


def _desenhar_legenda(draw, classes):
    """Desenha uma legenda simples no canto superior esquerdo."""
    x0, y0, passo, sw = 8, 8, 22, 16
    itens = [(c, CORES.get(c, COR_DESCONHECIDO)) for c in classes] + [("borda", COR_BORDA)]
    # fundo semi-opaco para leitura
    draw.rectangle([x0 - 4, y0 - 4, x0 + 130, y0 + passo * len(itens) + 2], fill=(0, 0, 0))
    for i, (nome, cor) in enumerate(itens):
        yy = y0 + i * passo
        draw.rectangle([x0, yy, x0 + sw, yy + sw], fill=cor, outline=(255, 255, 255))
        draw.text((x0 + sw + 6, yy + 2), nome, fill=(255, 255, 255))


def resolver(args):
    if args.tiles:
        tiles_dir = os.path.abspath(args.tiles)
        prefixo = args.prefixo or os.path.basename(tiles_dir.rstrip("/")).replace("_tiles", "")
    else:
        if not args.prefixo:
            args.prefixo = input("Digite o prefixo da imagem (ex: sat1): ").strip()
        prefixo = args.prefixo
        tiles_dir = os.path.join(PASTA_DADOS, f"{prefixo}_tiles")
    return tiles_dir, prefixo


def main():
    p = argparse.ArgumentParser(description="Visualizacao dos tiles rotulados.")
    p.add_argument("prefixo", nargs="?", help="Prefixo da imagem (ex: sat1)")
    p.add_argument("--tiles", help="Caminho da pasta de tiles (alternativa ao prefixo)")
    p.add_argument("--escala", type=float, default=0.1, help="Escala da saida (padrao 0.1)")
    p.add_argument("--saida-dir", help="Pasta de saida (padrao data/<prefixo>_viz)")
    args = p.parse_args()

    tiles_dir, prefixo = resolver(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERRO: pasta de tiles nao encontrada: {tiles_dir}")
        sys.exit(1)
    saida_dir = args.saida_dir or os.path.join(PASTA_DADOS, f"{prefixo}_viz")

    try:
        gerar(prefixo, tiles_dir, args.escala, saida_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
