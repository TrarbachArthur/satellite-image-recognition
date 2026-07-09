#!/usr/bin/env python3
"""Correcao visual de rotulos.

Mostra a cena inteira reduzida (imagem real + overlay colorido por classe) num
canvas rolavel. Permite clicar/arrastar para selecionar tiles e reatribuir o
rotulo (ou a marca de borda) de todos os selecionados de uma vez, gravando de
volta no mesmo <prefixo>_labels.csv.

Uso:
    python3 corrigir.py sat1                 # escala 0.25 (padrao)
    python3 corrigir.py sat1 --escala 0.2
    python3 corrigir.py --tiles data/sat1_tiles
"""

import argparse
import csv
import os
import sys

from PIL import Image

# Reuso: persistencia de rotulos e cores por classe ja definidas no projeto.
from rotular import Rotulador, CLASSES
from visualizar import CORES, COR_BORDA, COR_DESCONHECIDO

Image.MAX_IMAGE_PIXELS = None  # cenas reduzidas ainda sao grandes (>178 MP)

PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ALFA_OVERLAY = 105  # 0-255: intensidade do overlay colorido


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def ler_manifesto(tiles_dir, prefixo):
    """Retorna (coords, passo, cena_w, cena_h).
    coords: {arquivo: (x, y, w, h)}; passo: (passo_x, passo_y) do grid."""
    man = os.path.join(tiles_dir, f"{prefixo}_tiles.csv")
    if not os.path.exists(man):
        raise FileNotFoundError(f"Manifesto nao encontrado: {man}")
    coords = {}
    with open(man, newline="") as f:
        for r in csv.DictReader(f):
            coords[r["arquivo"]] = (int(r["x"]), int(r["y"]),
                                    int(r["largura"]), int(r["altura"]))
    if not coords:
        raise RuntimeError("Manifesto vazio.")
    algum = next(iter(coords.values()))
    passo = (algum[2], algum[3])
    cena_w = max(x + w for x, y, w, h in coords.values())
    cena_h = max(y + h for x, y, w, h in coords.values())
    return coords, passo, cena_w, cena_h


# ---------------------------------------------------------------------------
# Imagem de fundo (cacheada) + overlay
# ---------------------------------------------------------------------------
def montar_base(coords, tiles_dir, escala, out_w, out_h):
    """Cola todos os tiles reais numa imagem reduzida (passo caro de I/O)."""
    base = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    total = len(coords)
    for i, (arquivo, (x, y, w, h)) in enumerate(coords.items()):
        px, py = round(x * escala), round(y * escala)
        tw, th = max(1, round(w * escala)), max(1, round(h * escala))
        try:
            img = Image.open(os.path.join(tiles_dir, arquivo)).convert("RGB")
            if img.size != (tw, th):
                img = img.resize((tw, th))
        except Exception as e:
            print(f"  aviso: falha ao abrir {arquivo}: {e}")
            continue
        base.paste(img, (px, py))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{total} tiles")
    return base


def obter_base(prefixo, tiles_dir, coords, escala, out_w, out_h):
    """Carrega a base do cache ou monta e salva (cache ao lado da pasta de tiles)."""
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(tiles_dir)), f"{prefixo}_viz")
    os.makedirs(viz_dir, exist_ok=True)
    cache = os.path.join(viz_dir, f"{prefixo}_base_e{escala:g}.png")
    if os.path.exists(cache):
        img = Image.open(cache).convert("RGB")
        if img.size == (out_w, out_h):
            print(f"Base carregada do cache: {cache}")
            return img
    print(f"Montando base (escala {escala}) — colando {len(coords)} tiles...")
    base = montar_base(coords, tiles_dir, escala, out_w, out_h)
    base.save(cache)
    print(f"Base salva em {cache}")
    return base


def montar_overlay(base, rot, coords, escala, passo):
    """Sobre a base real, mistura uma cor translucida por classe (grade pequena
    ampliada em NEAREST). Reflete o estado atual dos rotulos."""
    px, py = passo
    cols = max(x // px for x, y, w, h in coords.values()) + 1
    rows = max(y // py for x, y, w, h in coords.values()) + 1
    cor = Image.new("RGB", (cols, rows), (0, 0, 0))
    alfa = Image.new("L", (cols, rows), 0)
    for arquivo, (x, y, w, h) in coords.items():
        rotulo, _ = rot.estado(arquivo)
        if not rotulo:
            continue
        cor.putpixel((x // px, y // py), CORES.get(rotulo, COR_DESCONHECIDO))
        alfa.putpixel((x // px, y // py), ALFA_OVERLAY)
    cor_up = cor.resize(base.size, Image.NEAREST)
    alfa_up = alfa.resize(base.size, Image.NEAREST)
    over = base.copy()
    over.paste(cor_up, (0, 0), alfa_up)
    return over


# ---------------------------------------------------------------------------
# Interface grafica
# ---------------------------------------------------------------------------
def rodar_gui(prefixo, rot, coords, escala, passo, base_real, base_over):
    """Visualizador com zoom e pan sobre a base reduzida (escala fixa).

    Apenas a regiao visivel e recortada da base e ampliada -> leve e fluido em
    qualquer nivel de zoom. Coordenadas internas em "px da base" (= cena*escala)."""
    import tkinter as tk
    from PIL import ImageTk

    px_passo, py_passo = passo
    origem = {(x, y): a for a, (x, y, w, h) in coords.items()}
    base_w, base_h = base_real.size
    Z_MAX = 8.0  # ampliacao maxima da base (base ja e reduzida; alem disso borra)

    janela = tk.Tk()
    janela.title(f"Correcao de rotulos — {prefixo}")

    topo = tk.Label(janela, anchor="w", bg="#222", fg="#eee",
                    font=("TkDefaultFont", 11), padx=6, pady=4)
    topo.pack(fill="x")

    canvas = tk.Canvas(janela, bg="#000", highlightthickness=0, width=1100, height=760)
    canvas.pack(fill="both", expand=True)

    legenda = " | ".join([f"{t}={r}" for t, r in CLASSES] + [
        "b/n=borda", "O=overlay", "roda=zoom", "btn-dir/setas=pan", "0=ajustar",
        "Esc=limpar", "q=sair"])
    rodape = tk.Label(janela, text=legenda, anchor="w", bg="#222", fg="#aaa",
                      font=("TkDefaultFont", 9), padx=6, pady=3)
    rodape.pack(fill="x")

    # z: px de tela por px da base; (ox, oy): canto sup-esq visivel, em px da base.
    estado = {"overlay": True, "photo": None, "z": None, "ox": 0.0, "oy": 0.0,
              "sel": set(), "arrasto": None, "rubber": None, "pan": None}

    def z_fit(W, H):
        return min(W / base_w, H / base_h)

    def clamp_offsets(W, H):
        z = estado["z"]
        vw, vh = W / z, H / z
        estado["ox"] = (base_w - vw) / 2 if vw >= base_w else min(max(estado["ox"], 0), base_w - vw)
        estado["oy"] = (base_h - vh) / 2 if vh >= base_h else min(max(estado["oy"], 0), base_h - vh)

    # --- conversoes de coordenadas ---
    def disp_para_base(dx, dy):
        return estado["ox"] + dx / estado["z"], estado["oy"] + dy / estado["z"]

    def base_para_disp(bx, by):
        return (bx - estado["ox"]) * estado["z"], (by - estado["oy"]) * estado["z"]

    def bbox_disp(arquivo):
        x, y, w, h = coords[arquivo]
        x0, y0 = base_para_disp(x * escala, y * escala)
        x1, y1 = base_para_disp((x + w) * escala, (y + h) * escala)
        return x0, y0, x1, y1

    def tile_em(dx, dy):
        bx, by = disp_para_base(dx, dy)
        sx, sy = bx / escala, by / escala
        return origem.get((int(sx // px_passo) * px_passo, int(sy // py_passo) * py_passo))

    # --- render ---
    def render():
        W, H = canvas.winfo_width(), canvas.winfo_height()
        if W < 10 or H < 10:
            return
        if estado["z"] is None:
            estado["z"] = z_fit(W, H)
        estado["z"] = max(z_fit(W, H), min(estado["z"], Z_MAX))
        clamp_offsets(W, H)
        z, ox, oy = estado["z"], estado["ox"], estado["oy"]

        src = base_over if estado["overlay"] else base_real
        vw, vh = W / z, H / z
        regiao = src.crop((round(ox), round(oy), round(ox + vw), round(oy + vh)))
        regiao = regiao.resize((W, H))
        estado["photo"] = ImageTk.PhotoImage(regiao)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=estado["photo"])

        # marca os tiles corrigidos/selecionados visiveis (conjuntos pequenos)
        for a in estado["sel"]:
            x0, y0, x1, y1 = bbox_disp(a)
            if x1 >= 0 and y1 >= 0 and x0 <= W and y0 <= H:
                canvas.create_rectangle(x0, y0, x1, y1, outline="#fff", width=2)
        if estado["rubber"]:
            canvas.coords(estado["rubber"], *estado["rubber_xy"])
        atualizar_status()

    def atualizar_status(msg=""):
        n = len(estado["sel"])
        if n == 1:
            a = next(iter(estado["sel"]))
            r, b = rot.estado(a)
            det = f"  |  {a}: rotulo={r or '-'} borda={b}"
        else:
            det = f"  |  {n} selecionados" if n else "  |  nada selecionado"
        zpct = (estado["z"] / z_fit(canvas.winfo_width() or 1, canvas.winfo_height() or 1))
        topo.config(text=f"{prefixo} ({len(coords)} tiles)  zoom {zpct:.1f}x{det}   {msg}")

    # --- selecao (botao esquerdo) ---
    def on_press(event):
        estado["arrasto"] = (event.x, event.y, bool(event.state & 0x0001))
        estado["rubber_xy"] = (event.x, event.y, event.x, event.y)
        estado["rubber"] = canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                   outline="#ff0", dash=(4, 2))

    def on_drag(event):
        if not estado["arrasto"]:
            return
        x0, y0, _ = estado["arrasto"]
        estado["rubber_xy"] = (x0, y0, event.x, event.y)
        canvas.coords(estado["rubber"], x0, y0, event.x, event.y)

    def on_release(event):
        if not estado["arrasto"]:
            return
        x0, y0, shift = estado["arrasto"]
        estado["arrasto"] = None
        if estado["rubber"]:
            canvas.delete(estado["rubber"]); estado["rubber"] = None
        if not shift:
            estado["sel"] = set()
        if abs(event.x - x0) < 5 and abs(event.y - y0) < 5:
            a = tile_em(event.x, event.y)
            if a:
                estado["sel"].add(a)
        else:
            bx0, by0 = disp_para_base(min(x0, event.x), min(y0, event.y))
            bx1, by1 = disp_para_base(max(x0, event.x), max(y0, event.y))
            sx0, sy0, sx1, sy1 = bx0 / escala, by0 / escala, bx1 / escala, by1 / escala
            for a, (x, y, w, h) in coords.items():
                if sx0 <= x + w / 2 <= sx1 and sy0 <= y + h / 2 <= sy1:
                    estado["sel"].add(a)
        render()

    # --- zoom (roda) e pan (botao direito / setas) ---
    def on_wheel(event):
        amplia = getattr(event, "delta", 0) > 0 or event.num == 4
        fator = 1.25 if amplia else 0.8
        bx, by = disp_para_base(event.x, event.y)
        estado["z"] = max(z_fit(canvas.winfo_width(), canvas.winfo_height()),
                          min(estado["z"] * fator, Z_MAX))
        estado["ox"] = bx - event.x / estado["z"]
        estado["oy"] = by - event.y / estado["z"]
        render()

    def on_pan_press(event):
        estado["pan"] = (event.x, event.y)

    def on_pan_move(event):
        if not estado["pan"]:
            return
        lx, ly = estado["pan"]
        estado["ox"] -= (event.x - lx) / estado["z"]
        estado["oy"] -= (event.y - ly) / estado["z"]
        estado["pan"] = (event.x, event.y)
        render()

    def pan_teclas(ddx, ddy):
        estado["ox"] += ddx / estado["z"]
        estado["oy"] += ddy / estado["z"]
        render()

    def ajustar():
        estado["z"] = None
        estado["ox"] = estado["oy"] = 0.0
        render()

    # --- acoes de rotulo ---
    def aplicar_rotulo(rotulo):
        if estado["sel"]:
            rot.definir_varios(list(estado["sel"]), rotulo=rotulo)
            render(); atualizar_status(f"-> {len(estado['sel'])} tile(s) = {rotulo}")

    def aplicar_borda(valor):
        if estado["sel"]:
            rot.definir_varios(list(estado["sel"]), borda=valor)
            atualizar_status(f"-> borda={valor} em {len(estado['sel'])} tile(s)")

    def alternar_overlay():
        estado["overlay"] = not estado["overlay"]
        render()

    def limpar_selecao():
        estado["sel"] = set()
        render()

    # --- bindings ---
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<ButtonPress-3>", on_pan_press)
    canvas.bind("<B3-Motion>", on_pan_move)
    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        canvas.bind(seq, on_wheel)
    canvas.bind("<Configure>", lambda e: render())

    for tecla, rotulo in CLASSES:
        janela.bind(tecla, lambda e, r=rotulo: aplicar_rotulo(r))
    janela.bind("b", lambda e: aplicar_borda(1))
    janela.bind("n", lambda e: aplicar_borda(0))
    janela.bind("o", lambda e: alternar_overlay())
    janela.bind("O", lambda e: alternar_overlay())
    janela.bind("<Left>", lambda e: pan_teclas(80, 0))
    janela.bind("<Right>", lambda e: pan_teclas(-80, 0))
    janela.bind("<Up>", lambda e: pan_teclas(0, 80))
    janela.bind("<Down>", lambda e: pan_teclas(0, -80))
    janela.bind("0", lambda e: ajustar())
    janela.bind("<Escape>", lambda e: limpar_selecao())
    janela.bind("q", lambda e: janela.destroy())

    janela.after(50, render)
    janela.mainloop()


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
    p = argparse.ArgumentParser(description="Correcao visual de rotulos.")
    p.add_argument("prefixo", nargs="?", help="Prefixo da imagem (ex: sat1)")
    p.add_argument("--tiles", help="Caminho da pasta de tiles (alternativa ao prefixo)")
    p.add_argument("--escala", type=float, default=0.25, help="Escala da exibicao (padrao 0.25)")
    args = p.parse_args()

    tiles_dir, prefixo = resolver(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERRO: pasta de tiles nao encontrada: {tiles_dir}")
        sys.exit(1)

    try:
        coords, passo, cena_w, cena_h = ler_manifesto(tiles_dir, prefixo)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    rot = Rotulador(tiles_dir, prefixo)
    out_w, out_h = round(cena_w * args.escala), round(cena_h * args.escala)
    print(f"{prefixo}: {len(coords)} tiles | cena {cena_w}x{cena_h} "
          f"-> exibicao {out_w}x{out_h}")

    base_real = obter_base(prefixo, tiles_dir, coords, args.escala, out_w, out_h)
    print("Montando overlay de classes...")
    base_over = montar_overlay(base_real, rot, coords, args.escala, passo)

    rodar_gui(prefixo, rot, coords, args.escala, passo, base_real, base_over)


if __name__ == "__main__":
    main()
