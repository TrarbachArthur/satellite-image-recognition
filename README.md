# Satellite Image Recognition

Converte imagens de satélite em "formato pan" (bandas separadas) em tiles RGB de alta
resolução e permite rotulá-los, produzindo um conjunto de dados para treinar um modelo
de reconhecimento.

## O que o projeto faz

1. **Fusão + recorte** (`gerar_tiles.py`): funde as bandas de cor com a banda pancromática
   (pansharpening) em resolução total e recorta o resultado numa grade de tiles PNG.
2. **Rotulagem** (`rotular.py`): interface para classificar cada tile rapidamente e salvar
   os rótulos em CSV.
3. **Visualização** (`visualizar.py`): monta imagens que mostram o que foi rotulado —
   uma imagem por classe e um overview da cena com os tiles contornados por cor.

Há também o `gerar_rgb.py`, que gera uma única imagem RGB da cena inteira (visão geral),
em vez de tiles.

## Instalação

```bash
pip install -r requirements.txt
```

## Passo a passo

### 1. Colocar as imagens em `data/`

A pasta `data/` já existe no repositório, mas fica **vazia** — você coloca aqui as bandas de
cada cena. Cada cena tem um **prefixo** e cinco arquivos TIFF, um por banda:

```
data/
  <prefixo>_r.tif     # vermelho
  <prefixo>_g.tif     # verde
  <prefixo>_b.tif     # azul
  <prefixo>_p.tif     # pancromática (alta resolução, escala de cinza)
  <prefixo>_nir.tif   # infravermelho próximo
```

Ex.: para a cena `sat2`, os arquivos são `sat2_r.tif`, `sat2_g.tif`, ... `sat2_nir.tif`.

### 2. Gerar os tiles

```bash
python3 gerar_tiles.py <prefixo>
```

Cria a pasta `data/<prefixo>_tiles/` com os tiles PNG de 1024×1024 e um índice
`<prefixo>_tiles.csv` (`arquivo, x, y, largura, altura`).

Opções úteis:

```bash
python3 gerar_tiles.py <prefixo> --tile 640      # tamanho do tile
python3 gerar_tiles.py <prefixo> --overlap 128   # sobreposição entre tiles
python3 gerar_tiles.py <prefixo> --formato jpg   # salvar em JPG
python3 gerar_tiles.py <prefixo> --pular-vazias  # descartar tiles uniformes (mar/preto)
```

### 3. Rotular os tiles

```bash
python3 rotular.py <prefixo>
```

Abre os tiles em ordem (esquerda→direita, cima→baixo). Rotule com o teclado ou os botões;
a rotulagem pode ser interrompida e retomada a qualquer momento.

| Tecla | Ação |
|---|---|
| `1`–`5` | mar / terra / nuvem / objeto / incerto → rotula **e avança** |
| `B` | alterna a marca de **borda** (independente da classe) |
| `←` / `Backspace` | volta para corrigir |
| `→` | avança sem alterar |
| `Z` | zoom 1:1 |
| `Esc` / `q` | sair (tudo salvo) |

Os rótulos são salvos em `data/<prefixo>_tiles/<prefixo>_labels.csv`
(`arquivo, rotulo, borda, timestamp`).

### 4. Visualizar os rótulos

```bash
python3 visualizar.py <prefixo>               # escala 0.1 (padrão)
python3 visualizar.py <prefixo> --escala 0.2  # mais detalhe (arquivos maiores)
```

Gera em `data/<prefixo>_viz/`:

- `<prefixo>_overview.png` — a cena reduzida com cada tile contornado por um quadrado da cor da
  sua classe; tiles marcados como borda recebem contorno magenta; inclui legenda.
- `<prefixo>_composite_<classe>.png` — uma imagem por classe (mar/terra/nuvem/objeto/incerto)
  com a imagem real dos tiles daquela classe na posição correta e o resto preto.
- `<prefixo>_composite_borda.png` — o mesmo, só com os tiles marcados como borda.

### 5. Corrigir rótulos errados

Se a visualização revelar erros de rotulagem, use a app de correção em vez de editar o CSV à mão:

```bash
python3 corrigir.py <prefixo>              # escala 0.25 (padrão)
python3 corrigir.py <prefixo> --escala 0.2 # mais leve, se ficar pesado
```

Mostra a cena com **zoom e pan** e um overlay colorido por classe sobre a imagem real.
Dê zoom para conferir detalhes, clique num tile para selecioná-lo, ou **arraste** um retângulo
(ou **Shift+clique**) para selecionar vários; então aplique o rótulo certo. As mudanças gravam
no mesmo `<prefixo>_labels.csv`. (Na primeira execução, monta e cacheia a imagem de fundo — pode
levar alguns minutos; as próximas são rápidas.)

| Controle | Ação |
|---|---|
| roda do mouse | zoom in/out no ponto do cursor |
| botão direito (arrastar) / setas | mover a imagem (pan) |
| `0` | ajustar a cena inteira à janela |
| clique / arrastar / Shift+clique | selecionar um / vários tiles |
| `1`–`5` | aplica mar / terra / nuvem / objeto / incerto aos selecionados |
| `B` / `N` | marca / remove a **borda** dos selecionados |
| `O` | liga/desliga o overlay de classes |
| `Esc` / `q` | limpa a seleção / sair (tudo já salvo) |

## Saída

Cada cena é processada de forma independente:

```
data/<prefixo>_tiles/
  <prefixo>_yYYYYYY_xXXXXXX.png   # tiles (o nome guarda a posição na cena original)
  <prefixo>_tiles.csv             # índice dos tiles
  <prefixo>_labels.csv            # rótulos (criado pela rotulagem)
```

## Utilitário extra

```bash
python3 gerar_rgb.py <prefixo> --escala 0.1   # uma única imagem RGB reduzida da cena
```
