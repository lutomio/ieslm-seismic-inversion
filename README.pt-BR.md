# TCC 2 — Implementação do iES-LM para inversão sísmica acústica

Código da parte prática do TCC de Lucas Tomio (UFSC — Ciências da Computação).
Implementa o **iES-LM** (Ma e Bi, 2019) e o compara com o **ES-MDA**
(Emerick e Reynolds, 2013) na inversão sísmica acústica 1D para estimar o
perfil de impedância acústica $Z$.

O eixo da comparação é **como cada algoritmo regulariza o passo de atualização
do conjunto**: o ES-MDA usa fatores de inflação $\alpha_i$ fixos, definidos a
priori; o iES-LM ajusta $\alpha^i$ a cada iteração por uma regra de região de
confiança.

## Como rodar

Requer Python 3.9+ e a biblioteca [SeReMpy](https://github.com/dariograna/SeReMpy),
que fornece as primitivas do modelo direto e o ES-MDA de referência. Coloque-a
ao lado deste projeto ou aponte a variável `SEREMPY_PATH` para a raiz dela.

Suíte de testes completa:

```bash
python -m pytest tests/ -v
```

Experimento comparativo (imprime as métricas e gera as figuras em `figuras/`):

```bash
python experimento.py
```

## Organização

| Arquivo | Conteúdo |
|---|---|
| `dados.py` | Localiza a SeReMpy e carrega `data5seis.dat` / `data5log.dat`. |
| `forward.py` | Modelo direto acústico: $d = W\,[\tfrac12 D \ln Z]$. |
| `prior.py` | Conjunto a priori de impedância (tendência suave + correlação vertical). |
| `ieslm.py` | **Núcleo do iES-LM** (Algoritmo 2 de Ma e Bi, 2019). |
| `experimento.py` | Driver comparativo iES-LM × ES-MDA e figuras. |
| `tests/` | Suíte de testes (ver abaixo). |
| `PLANO.md` | Plano de desenvolvimento aprovado antes da implementação, com as divergências registradas. |

A biblioteca `SeReMpy/` e o `ESPetroInversionDriver.py` **não são modificados**.
O modelo direto reaproveita `DifferentialMatrix` e `WaveletMatrix` da
biblioteca, e o ES-MDA da comparação usa a `EnsembleSmootherMDA` original.

## Testes

| Arquivo | O que cobre |
|---|---|
| `test_dados.py` | Carregamento dos dados e dimensões. |
| `test_forward.py` | Modelo direto: formatos, refletividade conferida à mão, reprodução do dado real (correlação > 0,99), não-linearidade. |
| `test_regras_lm.py` | Regras do algoritmo isoladas: Eqs. 30–31, 34, 39, 40, 42 e a perturbação $\xi\sim\mathcal N(0,\alpha C_D)$. |
| `test_mabi_exemplo1.py` | **Validação contra o artigo** — Exemplo 1 da Seção 5.1. |
| `test_integracao.py` | iES-LM rodando no dado sísmico real; critérios de parada. |
| `test_experimento.py` | Regressão do resultado que vai para o TCC. |

O teste-âncora é `test_mabi_exemplo1.py`: reproduz o benchmark publicado
(modelo linear de um parâmetro, MLE analítico $= 4{,}76543$) para $N_e =
10, 100, 500$. Se ele falhar, o núcleo está errado.

## Duas observações sobre a leitura do artigo

**1. A Eq. 38 impressa omite o fator 1/2.** A Eq. 34 define o objetivo com
$\tfrac12$, e o artigo afirma que $L^i_j(m^i_j) = O^i_j(m^i_j)$. Derivando $L$
a partir da Eq. 36, com $\bar G\,C_{MD} = C_{DD}$, chega-se a

$$L^i_j(m^{i+1}_j) = \frac{(\alpha^i)^2}{2}\, v^\top C_D\, v, \qquad v = (C_{DD} + \alpha^i C_D)^{-1} r .$$

O fator $\tfrac12$ é necessário para a consistência entre $L$ e $O$. A
verificação é o teste `test_rho_vale_um_no_caso_linear`: com modelo linear a
linearização é exata, então $\rho_j$ tem de valer exatamente 1. Medido:
$\rho = 1{,}000000$ com o fator, $\rho = 1{,}103$ sem ele.

**2. A parada da Eq. 43 é o que evita o sobreajuste.** O iES-LM resolve um
problema de máxima verossimilhança (Eq. 10, sem termo de prior). Rodando só
com os critérios da Seção 4.3, ele reduz o desajuste até ajustar o próprio
ruído: no experimento, o desajuste caiu a $5{,}2\times10^{-9}$, o conjunto
colapsou (envelope P10–P90 de 0,047) e o RMSE contra o poço **piorou** para
0,724 — pior que o ES-MDA. Com o critério da Eq. 43 ($R^i < 4p$, Seção 4.5),
o resultado se inverte. É o parâmetro `fator_ruido` de `ieslm.ieslm`.

## Resultado do experimento

Conjunto de 200 membros, 99 amostras de poço, 98 amostras sísmicas, semente fixa:

| | ES-MDA | iES-LM |
|---|---|---|
| RMSE vs. poço | 0,478 | **0,476** |
| desajuste final $\bar O$ | 1,06e-01 | **7,70e-02** |
| largura P10–P90 | 0,539 | 0,567 |
| avaliações do modelo direto | 5 | **3** |
| iterações | 4 | 2 |

(RMSE do conjunto a priori: 0,713.)

A regularização adaptativa alcançou qualidade equivalente à do ES-MDA
**com menos avaliações do modelo direto**: $\alpha$ cai de 18,4 para
8,6e-03 em duas iterações, enquanto o ES-MDA mantém $\alpha_i = 4$ fixo nas
quatro assimilações.

> Resultado de uma única configuração e semente. Para o texto do TCC, convém
> repetir com várias sementes e tamanhos de conjunto antes de afirmar
> superioridade.

## Referências

- Ma, X. e Bi, L. (2019). *A robust iterative ensemble smoother method for
  efficient history matching and uncertainty quantification*.
  Computational Geosciences 23:415–442. — **artigo central**
- Emerick, A. e Reynolds, A. (2013). *Ensemble smoother with multiple data
  assimilation*. Computers & Geosciences 55:3–15.
- Grana, D., Mukerji, T. e Doyen, P. (2021). *Seismic Reservoir Modeling*.
  Wiley. — biblioteca SeReMpy e modelo convolucional.
