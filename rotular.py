#!/usr/bin/env python3
"""Aplicacao de rotulagem rapida de tiles de satelite.

Carrega os tiles de uma pasta (na ordem esquerda->direita, cima->baixo) e permite
rotular cada um com uma tecla/botao (mar, terra, nuvem, objeto, incerto), alem de
uma marca independente de "borda". Rotula E avanca com um unico toque; permite
voltar para corrigir. O progresso e salvo continuamente em <prefixo>_labels.csv
dentro da pasta de tiles, entao e possivel parar e retomar de onde parou.

Uso:
    python3 rotular.py sat2                 # usa data/sat2_tiles/
    python3 rotular.py --tiles data/sat2_tiles
"""

import argparse
import csv
import os
import sys
import time

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# tecla -> rotulo. A ordem tambem define os botoes exibidos.
CLASSES = [
    ("1", "mar"),
    ("2", "terra"),
    ("3", "nuvem"),
    ("4", "objeto"),
    ("5", "incerto"),
]
ROTULOS_VALIDOS = {r for _, r in CLASSES}
CABECALHO = ["arquivo", "rotulo", "borda", "timestamp"]


# ---------------------------------------------------------------------------
# Nucleo (sem GUI) - testavel de forma isolada
# ---------------------------------------------------------------------------
class Rotulador:
    """Gerencia a lista ordenada de tiles e a persistencia dos rotulos."""

    def __init__(self, tiles_dir, prefixo):
        self.tiles_dir = tiles_dir
        self.prefixo = prefixo
        self.manifesto_csv = os.path.join(tiles_dir, f"{prefixo}_tiles.csv")
        self.labels_csv = os.path.join(tiles_dir, f"{prefixo}_labels.csv")
        self.tiles = self._carregar_ordem()          # lista de nomes de arquivo ordenados
        self.labels = self._carregar_labels()         # {arquivo: {"rotulo","borda"}}
        self.indice = self.primeiro_nao_rotulado()

    # --- carregamento ---
    def _carregar_ordem(self):
        """Le o manifesto e ordena por (y, x): cima->baixo, esquerda->direita.
        Se nao houver manifesto, cai para ordem alfabetica dos PNGs da pasta."""
        if os.path.exists(self.manifesto_csv):
            linhas = []
            with open(self.manifesto_csv, newline="") as f:
                for row in csv.DictReader(f):
                    linhas.append((int(row["y"]), int(row["x"]), row["arquivo"]))
            linhas.sort(key=lambda t: (t[0], t[1]))
            nomes = [a for _, _, a in linhas]
        else:
            nomes = sorted(x for x in os.listdir(self.tiles_dir)
                           if x.lower().endswith((".png", ".jpg", ".jpeg")))
        # mantem apenas tiles que existem em disco
        return [n for n in nomes if os.path.exists(os.path.join(self.tiles_dir, n))]

    def _carregar_labels(self):
        d = {}
        if os.path.exists(self.labels_csv):
            with open(self.labels_csv, newline="") as f:
                for row in csv.DictReader(f):
                    d[row["arquivo"]] = {
                        "rotulo": row.get("rotulo", "") or "",
                        "borda": 1 if str(row.get("borda", "0")) in ("1", "True", "true") else 0,
                    }
        return d

    # --- consultas ---
    def total(self):
        return len(self.tiles)

    def total_rotulados(self):
        return sum(1 for n in self.tiles
                   if self.labels.get(n, {}).get("rotulo"))

    def primeiro_nao_rotulado(self):
        for i, n in enumerate(self.tiles):
            if not self.labels.get(n, {}).get("rotulo"):
                return i
        return max(0, len(self.tiles) - 1)  # tudo rotulado: fica no ultimo

    def nome_atual(self, indice=None):
        i = self.indice if indice is None else indice
        return self.tiles[i] if 0 <= i < len(self.tiles) else None

    def caminho(self, nome):
        return os.path.join(self.tiles_dir, nome)

    def estado(self, nome):
        """Retorna (rotulo, borda) salvos para um tile (ou ('', 0))."""
        e = self.labels.get(nome)
        return (e["rotulo"], e["borda"]) if e else ("", 0)

    # --- mutacoes (persistem imediatamente) ---
    def definir_rotulo(self, nome, rotulo, borda):
        assert rotulo in ROTULOS_VALIDOS, f"rotulo invalido: {rotulo}"
        self.labels[nome] = {"rotulo": rotulo, "borda": int(bool(borda))}
        self._salvar()

    def definir_borda(self, nome, borda):
        """Atualiza a borda; se o tile ja tem rotulo, persiste na hora."""
        e = self.labels.get(nome)
        if e:
            e["borda"] = int(bool(borda))
            self._salvar()

    def definir_varios(self, nomes, rotulo=None, borda=None):
        """Aplica rotulo e/ou borda a varios tiles de uma vez, salvando o CSV
        UMA unica vez ao final (evita N reescritas ao corrigir um lote)."""
        if rotulo is not None:
            assert rotulo in ROTULOS_VALIDOS, f"rotulo invalido: {rotulo}"
        for nome in nomes:
            e = self.labels.get(nome, {"rotulo": "", "borda": 0})
            if rotulo is not None:
                e["rotulo"] = rotulo
            if borda is not None:
                e["borda"] = int(bool(borda))
            self.labels[nome] = e
        self._salvar()

    def _salvar(self):
        """Reescreve o CSV inteiro de forma atomica (temp + replace)."""
        tmp = self.labels_csv + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CABECALHO)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            for nome in self.tiles:
                e = self.labels.get(nome)
                if e and e.get("rotulo"):
                    w.writerow([nome, e["rotulo"], e["borda"], ts])
        os.replace(tmp, self.labels_csv)

    # --- navegacao ---
    def avancar(self):
        if self.indice < len(self.tiles) - 1:
            self.indice += 1

    def voltar(self):
        if self.indice > 0:
            self.indice -= 1


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
# Interface grafica (Tkinter)
# ---------------------------------------------------------------------------
def rodar_gui(rot):
    import tkinter as tk
    from PIL import Image, ImageTk

    TAM_EXIBICAO = 750  # lado maximo da imagem exibida

    janela = tk.Tk()
    janela.title(f"Rotulagem — {rot.prefixo}")
    janela.configure(bg="#222")

    # estado de trabalho do tile atual
    estado = {"borda": 0, "cache_nome": None, "cache_img": None, "zoom": None}

    # --- widgets ---
    lbl_topo = tk.Label(janela, font=("TkDefaultFont", 12, "bold"),
                        bg="#222", fg="#eee", pady=6)
    lbl_topo.pack(fill="x")

    lbl_sub = tk.Label(janela, font=("TkDefaultFont", 10),
                       bg="#222", fg="#aaa")
    lbl_sub.pack(fill="x")

    # moldura que vira vermelha quando borda=1
    moldura = tk.Frame(janela, bg="#222", bd=6)
    moldura.pack(padx=10, pady=8)
    canvas_img = tk.Label(moldura, bg="#000")
    canvas_img.pack()

    lbl_borda = tk.Label(janela, font=("TkDefaultFont", 11, "bold"),
                         bg="#222", pady=2)
    lbl_borda.pack(fill="x")

    barra = tk.Frame(janela, bg="#222")
    barra.pack(pady=8)

    def carregar_imagem(nome):
        """Carrega e redimensiona o tile para exibicao, com cache de 1."""
        if estado["cache_nome"] == nome and estado["cache_img"] is not None:
            return estado["cache_img"]
        img = Image.open(rot.caminho(nome)).convert("RGB")
        img.thumbnail((TAM_EXIBICAO, TAM_EXIBICAO))
        tkimg = ImageTk.PhotoImage(img)
        estado["cache_nome"] = nome
        estado["cache_img"] = tkimg
        return tkimg

    def prefetch(indice):
        """Pre-carrega a proxima imagem em cache para avanco instantaneo."""
        prox = rot.nome_atual(indice + 1)
        if prox and estado["cache_nome"] != prox:
            try:
                img = Image.open(rot.caminho(prox)).convert("RGB")
                img.thumbnail((TAM_EXIBICAO, TAM_EXIBICAO))
                estado["cache_nome"] = prox
                estado["cache_img"] = ImageTk.PhotoImage(img)
            except Exception:
                pass

    def atualizar():
        nome = rot.nome_atual()
        if nome is None:
            return
        rotulo_salvo, borda_salvo = rot.estado(nome)
        estado["borda"] = borda_salvo

        # topo: progresso
        feitos = rot.total_rotulados()
        total = rot.total()
        pct = (100.0 * feitos / total) if total else 0.0
        lbl_topo.config(text=f"{rot.prefixo} — {feitos}/{total} rotuladas ({pct:.1f}%)")

        # subtitulo: posicao + arquivo + rotulo atual (se revisitando)
        marca = f"  •  rotulo atual: {rotulo_salvo.upper()}" if rotulo_salvo else "  •  (sem rotulo)"
        lbl_sub.config(text=f"tile {rot.indice + 1}/{total}   {nome}{marca}")

        # imagem
        try:
            tkimg = carregar_imagem(nome)
            canvas_img.config(image=tkimg, text="")
            canvas_img.image = tkimg
        except Exception as e:
            canvas_img.config(image="", text=f"[erro ao abrir {nome}]\n{e}", fg="#f88")

        atualizar_borda()
        janela.after(1, lambda: prefetch(rot.indice))

    def atualizar_borda():
        if estado["borda"]:
            moldura.config(bg="#e33")
            lbl_borda.config(text="BORDA: SIM  (B para alternar)", fg="#f66")
        else:
            moldura.config(bg="#222")
            lbl_borda.config(text="borda: nao  (B para alternar)", fg="#888")

    # --- acoes ---
    def rotular(rotulo):
        nome = rot.nome_atual()
        if nome is None:
            return
        rot.definir_rotulo(nome, rotulo, estado["borda"])
        rot.avancar()
        atualizar()

    def alternar_borda():
        nome = rot.nome_atual()
        if nome is None:
            return
        estado["borda"] = 0 if estado["borda"] else 1
        rot.definir_borda(nome, estado["borda"])  # persiste se ja tiver rotulo
        atualizar_borda()

    def voltar():
        rot.voltar()
        atualizar()

    def proxima():
        rot.avancar()
        atualizar()

    def zoom():
        nome = rot.nome_atual()
        if nome is None:
            return
        if estado["zoom"] is not None and tk.Toplevel.winfo_exists(estado["zoom"]):
            estado["zoom"].destroy()
            estado["zoom"] = None
            return
        top = tk.Toplevel(janela)
        top.title(f"Zoom 1:1 — {nome}")
        img = Image.open(rot.caminho(nome)).convert("RGB")
        tkimg = ImageTk.PhotoImage(img)
        cv = tk.Canvas(top, width=min(img.width, 1000), height=min(img.height, 800),
                       bg="#000")
        cv.grid(row=0, column=0, sticky="nsew")
        sx = tk.Scrollbar(top, orient="horizontal", command=cv.xview)
        sy = tk.Scrollbar(top, orient="vertical", command=cv.yview)
        cv.config(xscrollcommand=sx.set, yscrollcommand=sy.set,
                  scrollregion=(0, 0, img.width, img.height))
        sx.grid(row=1, column=0, sticky="ew")
        sy.grid(row=0, column=1, sticky="ns")
        cv.create_image(0, 0, anchor="nw", image=tkimg)
        cv.image = tkimg
        top.bind("<Escape>", lambda e: top.destroy())
        estado["zoom"] = top

    def sair(*_):
        janela.destroy()

    # --- botoes ---
    for tecla, rotulo in CLASSES:
        tk.Button(barra, text=f"{tecla} {rotulo.capitalize()}", width=9,
                  command=lambda r=rotulo: rotular(r)).pack(side="left", padx=3)
    tk.Label(barra, text="  ", bg="#222").pack(side="left")
    tk.Button(barra, text="B Borda", width=8, command=alternar_borda).pack(side="left", padx=3)
    tk.Button(barra, text="← Voltar", width=8, command=voltar).pack(side="left", padx=3)
    tk.Button(barra, text="→ Próxima", width=9, command=proxima).pack(side="left", padx=3)
    tk.Button(barra, text="Z Zoom", width=8, command=zoom).pack(side="left", padx=3)

    # --- teclado ---
    for tecla, rotulo in CLASSES:
        janela.bind(tecla, lambda e, r=rotulo: rotular(r))
    janela.bind("b", lambda e: alternar_borda())
    janela.bind("B", lambda e: alternar_borda())
    janela.bind("<Left>", lambda e: voltar())
    janela.bind("<BackSpace>", lambda e: voltar())
    janela.bind("<Right>", lambda e: proxima())
    janela.bind("z", lambda e: zoom())
    janela.bind("Z", lambda e: zoom())
    janela.bind("<Escape>", sair)
    janela.bind("q", sair)

    atualizar()
    janela.mainloop()


def main():
    p = argparse.ArgumentParser(description="Rotulagem rapida de tiles de satelite.")
    p.add_argument("prefixo", nargs="?", help="Prefixo da imagem (ex: sat2)")
    p.add_argument("--tiles", help="Caminho da pasta de tiles (alternativa ao prefixo)")
    args = p.parse_args()

    tiles_dir, prefixo = resolver_tiles_dir(args)
    if not os.path.isdir(tiles_dir):
        print(f"ERRO: pasta de tiles nao encontrada: {tiles_dir}")
        sys.exit(1)

    rot = Rotulador(tiles_dir, prefixo)
    if rot.total() == 0:
        print(f"ERRO: nenhum tile encontrado em {tiles_dir}")
        sys.exit(1)

    print(f"Rotulando '{prefixo}': {rot.total_rotulados()}/{rot.total()} ja rotulados. "
          f"Iniciando no tile {rot.indice + 1}. Rotulos salvos em {rot.labels_csv}")
    rodar_gui(rot)


if __name__ == "__main__":
    main()
