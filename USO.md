# Guia de uso completo

Referência de todos os processos do pipeline: como executar, o que esperar de cada etapa e
todas as configurações disponíveis (linha de comando e YAML).

Fluxo geral:

```
bandas .tif → [1] gerar_tiles.py → tiles PNG → [2] rotular.py → labels.csv
                                                      ↓
              [5] treinar.py ← configs/*.yaml   [3] visualizar.py (conferir)
                     ↓                          [4] corrigir.py  (corrigir erros)
              [6] avaliar.py → métricas / comparação
                     ↓
              [7] inferir.py → pré-rotula cenas novas com os modelos treinados
                               (revisão com corrigir.py → mais dados → novo treino)
```

Convenção: cada cena de satélite tem um **prefixo** (ex. `sat1`) e vive em `data/`. Todos os
scripts aceitam o prefixo como argumento posicional; sem argumento, perguntam interativamente.

---

## 1. `gerar_tiles.py` — bandas → tiles RGB

Funde as bandas de cor (R, G, B) com a pancromática (algoritmo de Brovey) em resolução total,
por streaming (nunca carrega a cena inteira na RAM), e recorta o resultado numa grade de tiles.

**Entrada esperada:** `data/<prefixo>_r.tif`, `_g.tif`, `_b.tif`, `_p.tif`, `_nir.tif`
(o `nir` não é usado no RGB, mas é assumido presente no padrão de nomes).

```bash
python3 gerar_tiles.py <prefixo> [opções]
```

| Opção | Default | Efeito |
|---|---|---|
| `--tile N` | 1024 | Lado do tile em pixels. Para classificação com GSD 2 m/px, 256 é um bom valor (ver discussão no histórico do projeto). |
| `--escala F` | 1.0 | Escala da resolução de saída (1.0 = resolução da pancromática). |
| `--overlap N` | 0 | Sobreposição entre tiles vizinhos, em pixels. Útil para detecção; para classificação, deixe 0 (evita tiles quase duplicados e vazamento treino/val). |
| `--formato png\|jpg` | png | Formato dos tiles. JPG gera arquivos menores com perda leve. |
| `--saida-dir DIR` | `data/<prefixo>_tiles` | Pasta de saída. |
| `--pular-vazias` | off | Descarta tiles quase uniformes (mar aberto homogêneo, moldura preta). Reduz muito o volume. |

**Resultado esperado:** `data/<prefixo>_tiles/` com os PNGs `<prefixo>_yYYYYYY_xXXXXXX.png`
(o nome codifica a posição do tile na cena) e o índice `<prefixo>_tiles.csv`
(`arquivo,x,y,largura,altura`). Para uma cena de ~56k×58k px com tiles de 1024: ~3200 tiles
em ~5 min, RAM constante ~3.5 GB. O contraste é normalizado globalmente (percentis 2–98
estimados numa amostra), então todos os tiles compartilham a mesma escala de cor.

### Utilitário relacionado: `gerar_rgb.py`

Mesma fusão de Brovey, mas gera **uma única imagem** da cena em vez de tiles — útil para uma
visão geral rápida antes de decidir os parâmetros de tiling.

```bash
python3 gerar_rgb.py <prefixo> --escala 0.1   # PNG único reduzido (10% da resolução)
```

| Opção | Default | Efeito |
|---|---|---|
| `--escala F` | 1.0 | Escala da saída. Atenção: em cenas grandes, 1.0 é inviável (imagem de gigapixels); use ≤0.25. |
| `--saida PATH` | `data/<prefixo>_rgb.png` | Arquivo de saída. |

**Resultado esperado:** um PNG RGB de 8 bits da cena inteira na escala pedida.

---

## 2. `rotular.py` — rotulagem rápida

Interface Tkinter que mostra os tiles em ordem (esquerda→direita, cima→baixo) e rotula com um
toque. Interrompível: pode fechar e reabrir que continua do primeiro tile sem rótulo.

```bash
python3 rotular.py <prefixo>
python3 rotular.py --tiles data/outra_pasta_tiles   # pasta arbitrária
python3 rotular.py <prefixo> --classes incerto      # navega SÓ pelos tiles com esses rótulos
```

`--classes` (1+ valores entre mar/terra/nuvem/objeto/incerto) restringe a navegação aos tiles
cujo rótulo atual está no filtro — ideal para revisar uma classe específica em sequência
(ex. os 'incerto' da pré-rotulagem). Começa do primeiro tile do filtro; o progresso e o topo
da janela mostram o filtro ativo. O CSV continua completo: rotular um tile para fora do filtro
não remove os rótulos das outras classes.

| Controle | Ação |
|---|---|
| `1` `2` `3` `4` `5` | mar / terra / nuvem / objeto / incerto → **salva e avança** |
| `B` | alterna a marca de **borda** do tile atual (flag independente da classe; não avança) |
| `←` / `Backspace` | volta ao tile anterior (para corrigir; a nova tecla sobrescreve) |
| `→` | avança sem alterar |
| `Z` | zoom 1:1 do tile (janela com scroll); `Esc` fecha |
| `Esc` / `q` | sair (tudo já está salvo) |

**Resultado esperado:** `data/<prefixo>_tiles/<prefixo>_labels.csv`
(`arquivo,rotulo,borda,timestamp`), uma linha por tile rotulado, reescrito atomicamente a cada
ação (seguro parar a qualquer momento). O rótulo `incerto` marca tiles indecisos — eles ficam
fora do treino. A marca `borda` identifica tiles com ruído de borda da cena, filtráveis depois.

---

## 3. `visualizar.py` — conferir a rotulagem

Gera imagens de auditoria a partir dos rótulos.

```bash
python3 visualizar.py <prefixo> [--escala 0.1] [--tiles DIR] [--saida-dir DIR]
```

| Opção | Default | Efeito |
|---|---|---|
| `--escala F` | 0.1 | Escala das imagens geradas (0.1 → cena de 56k vira ~5.6k px). |
| `--tiles DIR` | `data/<prefixo>_tiles` | Pasta de tiles/labels de entrada. |
| `--saida-dir DIR` | `data/<prefixo>_viz` | Pasta de saída. |

**Resultado esperado**, em `data/<prefixo>_viz/` (~2 min para 34k tiles, RAM <1 GB):
- `<prefixo>_overview.png` — cena real com cada tile contornado pela cor da classe
  (mar=azul, terra=verde, nuvem=branco, objeto=vermelho com traço grosso, incerto=amarelo)
  + contorno magenta nos tiles de borda + legenda.
- `<prefixo>_composite_<classe>.png` — uma por classe: os tiles daquela classe na posição
  real, resto preto. Útil para ver "onde o modelo vai achar cada coisa".
- `<prefixo>_composite_borda.png` — só os tiles marcados como borda.

---

## 4. `corrigir.py` — corrigir rótulos visualmente

Mostra a cena inteira com zoom/pan e um overlay translúcido por classe; erros saltam aos olhos
(ex. um quadrado verde no meio do mar). Selecione tiles e reatribua a classe sem tocar no CSV.

```bash
python3 corrigir.py <prefixo> [--escala 0.25] [--tiles DIR]
```

| Opção | Default | Efeito |
|---|---|---|
| `--escala F` | 0.25 | Resolução da imagem de fundo. Na 1ª execução ela é montada (abre todos os tiles, minutos) e fica **cacheada** em `data/<prefixo>_viz/<prefixo>_base_e<escala>.png`; das próximas vezes abre na hora. |
| `--tiles DIR` | `data/<prefixo>_tiles` | Pasta de tiles/labels. |
| `--classes C [C ...]` | (todas) | Filtro de classes: só tiles com esses rótulos ganham overlay **e** só eles são selecionáveis — arraste sobre uma região inteira e apenas os tiles do filtro recebem o novo rótulo. Ex.: `--classes incerto` para revisar só os incertos da pré-rotulagem. |

| Controle | Ação |
|---|---|
| roda do mouse | zoom no ponto do cursor (até 8× sobre a base) |
| botão direito arrastando / setas | pan |
| `0` | ajustar a cena inteira à janela |
| clique | seleciona 1 tile |
| arrastar retângulo | seleciona todos os tiles da região |
| Shift+clique / Shift+arrastar | soma à seleção |
| `1`–`5` | aplica mar/terra/nuvem/objeto/incerto a **todos os selecionados** (salva na hora) |
| `B` / `N` | marca / remove borda dos selecionados |
| `O` | liga/desliga o overlay colorido |
| `Esc` / `q` | limpa seleção / sair |

**Resultado esperado:** o mesmo `<prefixo>_labels.csv` atualizado (uma reescrita atômica por
lote de correção). A barra superior mostra o rótulo do tile clicado e o total selecionado.

---

## 5. `treinar.py` — treinar um experimento

Treina um classificador de 4 classes (mar/terra/nuvem/objeto) com PyTorch + timm, controlado
por um YAML. GPU fortemente recomendada.

```bash
python3 treinar.py --config configs/<exp>.yaml [flags]
```

| Flag | Efeito |
|---|---|
| `--config PATH` | (obrigatório) YAML do experimento. |
| `--permitir-cpu` | Sem CUDA o script aborta por padrão; esta flag força treino em CPU (lento). |
| `--sobrescrever` | Apaga `experimentos/<nome>/` existente e recomeça. Sem ela, rodar um experimento já existente dá erro (proteção contra perda acidental). |
| `--retomar` | Continua um treino interrompido a partir de `ultimo.pt` (modelo, otimizador, época). |

### Todas as chaves do YAML (com defaults)

Só `nome` é obrigatório; o restante herda os defaults abaixo. Cada experimento deve ter um
`nome` único — vira a pasta `experimentos/<nome>/`.

```yaml
nome: meu_experimento     # (obrigatorio) nome/pasta do experimento
seed: 42                  # semente global (dados, split, torch) — mesmo seed = mesmo split

dados:
  cenas: [sat1, sat2]     # prefixos das cenas; le data/<p>_tiles/<p>_labels.csv de cada uma
  excluir_borda: true     # true: descarta tiles com borda=1 | false: mantem (hipotese testavel)
  fracao: 1.0             # fracao estratificada dos dados (0.02 = ~2% de cada classe;
                          #   piso de 3 exemplos/classe). Use pequeno p/ smoke tests.
  max_por_classe: null    # teto de exemplos por classe (ex. 4000). Nunca reduz classes
                          #   que ja tem menos que o teto. null = sem teto.
  img_size: 224           # lado da imagem de entrada do modelo. 224 = regime dos pesos
                          #   pre-treinados (recomendado); 256 = resolucao nativa do tile
                          #   (hipotese: perder 12% de resolucao importa?)

split:
  metodo: aleatorio       # aleatorio: estratificado por classe (baseline otimista)
                          # espacial: blocos inteiros de tiles vao para um unico split —
                          #   elimina vazamento entre tiles vizinhos quase identicos
  fracoes: [0.70, 0.15, 0.15]   # treino / val / test (soma ~1.0)
  bloco_px: 2048          # so p/ metodo espacial: lado do bloco em px da cena
                          #   (2048 = blocos de 8x8 tiles de 256)

modelo:
  arquitetura: resnet18   # qualquer nome do timm. Testados nesta GPU (8GB):
                          #   resnet18, resnet50, vit_small_patch16_224,
                          #   deit_small_patch16_224, convnext_tiny,
                          #   efficientnet_b0, efficientnet_b2
  pretrained: true        # true: pesos ImageNet (baixados na 1a vez) | false: do zero

balanceamento:            # EIXO 1 do desbalanceamento: age nos dados/pesos
  metodo: nenhum          # nenhum: baseline (classes raras tendem a recall 0)
                          # pesos: CrossEntropy ponderada pela raridade da classe
                          # sampler: oversampling da minoria (WeightedRandomSampler);
                          #   use augmentation leve/pesada junto (senao decora a minoria)
  max_peso: 20.0          # teto do peso por classe (sem teto, 'objeto' pesaria ~270x)

loss:                     # EIXO 2, ortogonal ao balanceamento
  tipo: ce                # ce: cross-entropy | focal: foca nos exemplos dificeis
  gamma: 2.0              # so p/ focal: intensidade do foco (maior = mais foco nos dificeis)
  label_smoothing: 0.0    # suavizacao dos alvos da CE (ex. 0.1)

augmentation: leve        # nenhuma: so resize+normalize
                          # leve: flips H/V + rotacoes de 90 (seguro p/ imagem nadir)
                          # pesada: leve + crop aleatorio (0.6-1.0) + color jitter +
                          #   blur ocasional + random erasing

treino:
  epocas: 30              # maximo de epocas (early stopping pode parar antes)
  batch: 128              # reduza se der OutOfMemory (ex. 96 p/ ViT/ConvNeXt, 64 p/ B2)
  lr: 3.0e-4              # learning rate (ViT pre-treinado prefere ~1e-4)
  weight_decay: 0.05
  otimizador: adamw       # adamw | sgd (momentum 0.9)
  scheduler: cosseno      # cosseno com warmup de 2 epocas | nenhum
  amp: true               # mixed precision (mais rapido e economico em VRAM)
  num_workers: 2          # processos de carga de dados. NAO subir alem de 2-3 (RAM 11GB)
  early_stopping_paciencia: 7   # epocas sem melhora antes de parar (0 = desligado)
  metrica_checkpoint: f1_macro  # o que define o "melhor" checkpoint:
                                #   f1_macro | balanced_accuracy | loss_val
```

**Resultado esperado**, em `experimentos/<nome>/`:

| Arquivo | Conteúdo |
|---|---|
| `config.yaml` | Cópia congelada do config resolvido (com defaults) — reprodutibilidade. |
| `splits.csv` | O split exato usado (`arquivo,rotulo,cena,split`) — fonte de verdade p/ avaliação. |
| `splits_resumo.json` | Contagem por classe em cada split — audite antes de confiar nas métricas. |
| `historico.csv` | Uma linha por época: losses, acc, balanced_acc, f1_macro, recall por classe, lr, tempo. |
| `melhor.pt` / `ultimo.pt` | Checkpoints (melhor pela `metrica_checkpoint` / último para retomar). |
| `metricas_val.json` | Métricas do melhor checkpoint no **val** (o test fica para o avaliar.py). |
| `tb/` | Logs do TensorBoard. |

Console: 1 linha por época, ex.
`epoca 5/30  loss_tr 0.016  loss_val 0.014  acc 0.997  bal_acc 0.708  f1 0.726  rec_mar 0.999 ... (32.6s)`.
Referência de tempo na RTX 2070 (AMP, 224px): resnet18 ≈ 33 s/época com 25% dos dados
(~2 min/época com 100%); ViT-S/ConvNeXt ≈ 2–4×; um experimento completo fecha em 30–90 min.

Acompanhe as curvas de todos os experimentos lado a lado:
```bash
tensorboard --logdir experimentos
```

### Configs prontos (`configs/`)

| Config | O que testa |
|---|---|
| `smoke.yaml` | Pipeline de ponta a ponta em ~1 min (2% dos dados, 2 épocas). Rode primeiro. |
| `resnet18_base.yaml` | Baseline: CNN pré-treinada, sem tratamento de desbalanceamento. |
| `resnet18_pesos.yaml` | Hipótese: class weights na loss. |
| `resnet18_sampler.yaml` | Hipótese: oversampling da minoria (+ augmentation pesada). |
| `resnet18_focal.yaml` | Hipótese: focal loss. |
| `resnet18_com_borda.yaml` | Hipótese: manter tiles de borda muda algo? |
| `resnet18_split_espacial.yaml` | Hipótese: quanto o split aleatório infla as métricas (vazamento)? |
| `vit_small_pretrained.yaml` | Visual Transformer pré-treinado. |
| `convnext_tiny.yaml` | CNN moderna. |
| `efficientnet_b0.yaml` | CNN eficiente (inferência barata). |

Para criar uma hipótese nova: copie o YAML mais próximo, troque o `nome` e mude 1–2 campos.

---

## 6. `avaliar.py` — avaliar e comparar

### Avaliar um experimento no conjunto de teste

```bash
python3 avaliar.py experimentos/<nome> [--split test|val|treino] [--checkpoint melhor|ultimo] [--permitir-cpu]
```

Usa o `splits.csv` **congelado** do experimento (não re-deriva nada) e o checkpoint escolhido.
O test só é tocado aqui — durante a iteração de hipóteses, compare pelo val e guarde o test
para a decisão final.

**Resultado esperado**, no diretório do experimento + impresso no console:
- `metricas_<split>.json` — accuracy, balanced_accuracy, f1_macro, precision/recall/f1/suporte
  por classe, average_precision da classe 'objeto'.
- `matriz_confusao_<split>.csv` — linhas = verdadeiro, colunas = predito.
- `predicoes_<split>.csv` — `arquivo,cena,rotulo,predito,prob_<classe>...` por tile. Use as
  probabilidades para minerar candidatos a 'objeto' e priorizar a próxima rotulagem.

### Comparar todos os experimentos

```bash
python3 avaliar.py --comparar [--split test|val] [--csv comparacao.csv]
```

Varre `experimentos/*/`, agrega as métricas numa tabela ordenada por f1_macro e imprime
(opcionalmente salva em CSV). Experimentos ainda sem avaliação no split pedido caem para o
`metricas_val.json` do treino (coluna `split` indica isso); sem nenhum, são pulados com aviso.

```
experimento            split  acc    bal_acc  f1_macro  rec_objeto  prec_objeto  ap_objeto  n_objeto  epocas
resnet18_sampler       test   0.981  0.912    0.874     0.750       0.600        0.55       8         22
...
```

A coluna `n_objeto` (suporte no split) fica sempre visível de propósito: com poucos exemplos,
`rec_objeto 0.750` significa "6 de 8" — leia as métricas de 'objeto' como indicativas, não
conclusivas, enquanto a classe tiver poucas dezenas de rótulos.

---

## 7. `inferir.py` — pré-rotulagem por inferência

Usa modelos já treinados para gerar um `<prefixo>_labels.csv` inicial de uma cena **nova**
(já tileada, sem rótulos). Em vez de rotular do zero, você apenas **revisa** as predições —
bem mais rápido. Também serve como teste de generalização dos modelos em cenas não vistas.

```bash
python3 inferir.py sat3 --experimentos experimentos/vit_small_pretrained
python3 inferir.py sat3 --experimentos experimentos/vit_small_pretrained experimentos/resnet18_sampler --limiar-incerto 0.7
```

| Opção | Default | Efeito |
|---|---|---|
| `--experimentos DIR [DIR ...]` | (obrigatório) | 1+ experimentos treinados. Com 2+, faz **ensemble** (média das probabilidades) — pré-rótulos melhores, custo de ~N passes de inferência (minutos cada). |
| `--checkpoint melhor\|ultimo` | melhor | Checkpoint de cada experimento. |
| `--limiar-incerto F` | 0.0 | 0.0 = sempre grava a classe prevista, mesmo com confiança baixa. >0 = tiles com prob. máxima abaixo de F recebem `incerto` (aparecem em amarelo no corrigir.py e ficam fora do treino até revisão). |
| `--batch N` / `--num-workers N` | 128 / 2 | Parâmetros de inferência. |
| `--sobrescrever` | off | **Proteção**: se a cena já tem `labels.csv` (possível rotulagem manual!), o script aborta; esta flag força a substituição. |
| `--tiles DIR` / `--permitir-cpu` | — | Como nos demais scripts. |

**Resultado esperado**, na pasta de tiles (~1–3 min por modelo por cena na 2070):
- `<prefixo>_labels.csv` — mesmo formato da rotulagem manual (`arquivo,rotulo,borda,timestamp`,
  `borda=0`), compatível direto com `corrigir.py`/`rotular.py`/`visualizar.py`/`treinar.py`.
- `<prefixo>_predicoes_auto.csv` — `arquivo,predito,confianca,prob_<classe>...` — registro do
  que foi automático; use para priorizar a revisão (menor confiança primeiro) e para minerar
  candidatos a 'objeto' (`prob_objeto` alta).
- Resumo no console: distribuição prevista, nº de incertos, confiança média, candidatos a objeto.

**Fluxo de revisão recomendado:**

```bash
python3 inferir.py sat3 --experimentos experimentos/vit_small_pretrained experimentos/resnet18_sampler --limiar-incerto 0.7
python3 visualizar.py sat3    # auditoria visual rápida das predições
python3 corrigir.py sat3 --classes incerto   # 1ª passada: revisar SÓ os incertos
python3 corrigir.py sat3      # 2ª passada: revisão geral; marque as bordas por
                              #   seleção de região (B)
# depois da revisão, inclua a cena no treino: dados.cenas: [sat1, sat2, sat3] no YAML
```

O modelo **não prevê a marca de borda** — os tiles saem com `borda=0` e as bordas da cena
devem ser marcadas na revisão (no corrigir.py: arraste selecionando a faixa da borda e tecle `B`).

---

## Interpretação e limitações conhecidas

- **Accuracy global engana**: com mar+terra dominando, ~99% de accuracy convive com recall 0
  nas classes raras. Olhe `balanced_accuracy`, `f1_macro` e o recall por classe.
- **Classe 'objeto'**: enquanto houver poucas dezenas de exemplos, o realista é: (i) um ótimo
  classificador mar/terra/nuvem (filtro de cena), (ii) evidência qualitativa sobre 'objeto',
  (iii) usar `predicoes_*.csv` para achar candidatos e rotular mais. Não tire conclusões
  estatísticas de 7–8 exemplos de val/test.
- **Split espacial vs aleatório**: se as métricas caírem muito no espacial, o número do
  aleatório estava inflado por vazamento entre tiles vizinhos — confie no espacial.
- **Recursos**: 8 GB VRAM (reduza `batch` se faltar), 11 GB RAM (não suba `num_workers`).
- Os dados crescem: nenhum script assume contagens atuais; novas cenas entram por
  `dados.cenas` no YAML depois de rotuladas.
