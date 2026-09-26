# Guia do código — iES-LM para inversão sísmica

Este guia explica o código arquivo por arquivo e função por função, ligando
cada variável ao símbolo correspondente na teoria e indicando **de qual artigo
e de qual equação** cada parte foi tirada.

A leitura sugerida é: seções 1 a 3 para o panorama, seção 4 como consulta de
variáveis, e a seção 5.5 (`ieslm.py`) com o artigo de Ma e Bi aberto ao lado.

---

## 1. O problema que o código resolve

Temos um **dado sísmico observado** $\mathbf{d}_{obs}$ e queremos descobrir as
**propriedades da rocha** $\mathbf{m}$ que o produziram. O caminho inverso —
de $\mathbf{m}$ para o dado — é conhecido e se chama **modelo direto**,
$g(\mathbf{m})$. Inverter é achar $\mathbf{m}$ tal que $g(\mathbf{m}) \approx
\mathbf{d}_{obs}$.

O problema é mal-posto (vários $\mathbf{m}$ explicam o mesmo dado), então não
buscamos um único modelo e sim um **conjunto** de $N_e$ modelos que representa
a incerteza. Os dois métodos comparados atualizam esse conjunto de formas
diferentes:

| | Regularização do passo |
|---|---|
| **ES-MDA** | fixa: sequência de $\alpha_l$ decidida antes de rodar |
| **iES-LM** | adaptativa: $\alpha^i$ ajustado a cada iteração pela razão de ganho $\rho_j$ |

### Fluxo do experimento

```
dados.py ──► poço (Vp, Vs, ρ) — a referência
referencia.py ──► ou um modelo sintético no lugar dele
                 │
forward_elastico.py ──► g(m): Vp, Vs, ρ ──► traços Near, Mid, Far
                 │
                 ├──► adiciona_ruido ──► d_obs e C_D
                 │
prior.py ──► conjunto a priori (N_e membros)
                 │
        ┌────────┴────────┐
   ES-MDA (SeReMpy)    iES-LM (ieslm.py)
        └────────┬────────┘
                 │
metricas.py ──► RMSE, cobertura, envelope, KS, custo
                 │
experimento_elastico.py ──► tabela + figuras
```

O ponto de projeto mais importante: **`ieslm.py` não conhece sísmica**. Ele
recebe `g` como uma função qualquer. Por isso o mesmo código roda tanto o
exemplo sintético do artigo (para validar) quanto o problema sísmico.

---

## 2. Referências e o papel de cada uma

| Chave no `.bib` | Referência | O que veio dela |
|---|---|---|
| `mabi2019robust` | Ma e Bi (2019), *Computational Geosciences* 23:415–442 | **Todo o iES-LM**: Algoritmo 2 e Eqs. 30–43; o benchmark de validação (Seção 5.1) |
| `emerick2013ensemble` | Emerick e Reynolds (2013), *Computers & Geosciences* 55:3–15 | O ES-MDA e a condição de consistência $\sum 1/\alpha_l = 1$ |
| `grana2021seismic` | Grana, Mukerji e Doyen (2021), *Seismic Reservoir Modeling*, Wiley | Biblioteca SeReMpy: modelo direto (Cap. 5.1), `EnsembleSmootherMDA` (Cap. 5.6), simulação correlacionada (Cap. 3.6), dados |
| `liu2018stochastic` | Liu e Grana (2018), *Geophysics* 83(3) | Uso do ES-MDA em inversão sísmica (base do driver de referência) |
| *(ainda não está no `.bib`)* | Caetano, Santiago, Freitas e Roisenberg — *CMA-ES … Compared to ES-MDA* | Protocolo de comparação: ruído com SNR, sequência decrescente de $\alpha_l$, taxa de cobertura, teste KS, custo em chamadas ao operador direto |

A ideia de **região de confiança** (razão entre redução real e prevista) é
anterior ao Ma e Bi — vem do Levenberg-Marquardt clássico (Moré, 1978; Nocedal e
Wright, 2006; ver também a dissertação de Kaltenbach em `arquivos_pratica/`).
Ma e Bi adaptam essa ideia para conjuntos. Nenhuma linha de código foi tirada
dessas fontes clássicas; elas servem como fundamentação conceitual.

---

## 3. As dimensões do problema

Vale fixar os números antes de ler o código, porque quase todo erro de leitura
é confusão de formato de matriz.

| Símbolo no código | Valor (caso elástico) | Significado |
|---|---|---|
| `nm` | 99 | amostras de tempo do perfil de poço |
| `nd` | 98 por ângulo | amostras sísmicas (uma por interface entre camadas) |
| `nv` | 3 | propriedades: $V_p$, $V_s$, $\rho$ |
| `ne` / `NE` | 200 | tamanho do conjunto, $N_e$ |
| parâmetros | $3 \times 99 = 297$ | vetor $\mathbf{m}$ empilhado |
| dados | $3 \times 98 = 294$ | vetor $\mathbf{d}$ empilhado (Near, Mid, Far) |

Por que `nd = nm − 1`: a refletividade existe **entre** duas amostras, então
99 camadas geram 98 interfaces.

**Convenção de matrizes:** um conjunto é sempre uma matriz com **uma coluna por
membro**. `M` tem forma `(297, 200)`: a coluna `j` é o modelo $\mathbf{m}_j$.

---

## 4. Glossário de variáveis

As variáveis centrais do algoritmo, com o símbolo usado no seu Cap. 2 e no
artigo.

| Variável | Símbolo | Forma (elástico) | Significado | Origem |
|---|---|---|---|---|
| `M` | $\{\mathbf{m}_j^i\}$ | (297, 200) | conjunto de modelos na iteração $i$ | Ma e Bi, Seção 4.1 |
| `G` | $\{g(\mathbf{m}_j^i)\}$ | (294, 200) | previsão do modelo direto para cada membro | — |
| `d_obs` | $\mathbf{d}_{obs}$ | (294, 1) | dado observado (já com ruído) | — |
| `C_D` | $C_D$ | (294, 294) | covariância do erro de medição | Cap. 2, Eq. `eq:likelihood` |
| `C_D_inv` | $C_D^{-1}$ | (294, 294) | inversa, calculada uma vez | — |
| `raiz_C_D` | $C_D^{1/2}$ | (294, 294) | fator de Cholesky, para sortear ruído | — |
| `C_MD` | $C_{MD}^i$ | (297, 294) | covariância cruzada modelo-dado | Ma e Bi, Eq. 30; Cap. 2 `eq:cmd` |
| `C_DD` | $C_{DD}^i$ | (294, 294) | autocovariância dos dados previstos | Ma e Bi, Eq. 31; Cap. 2 `eq:cdd` |
| `d_pert` | $\mathbf{d}_{i,j}^{o} = \mathbf{d}_{obs} + \boldsymbol{\xi}_j^i$ | (294, 200) | observação perturbada, uma por membro | Ma e Bi, Eq. 26–27 |
| `alpha` | $\alpha^i$ | escalar | parâmetro de regularização LM | Ma e Bi, Eqs. 32, 40–41 |
| `gamma` | $\gamma^i$ | escalar | parâmetro da região de confiança | Ma e Bi, Eqs. 40–41 |
| `gamma_j` | $\gamma_j^i$ | (200,) | candidato a $\gamma$ proposto por cada membro | Ma e Bi, Eq. 40 |
| `R` | $\mathbf{d}_{i,j}^{o} - g(\mathbf{m}_j^i)$ | (294, 200) | resíduo em relação ao dado perturbado | — |
| `V` | $(C_{DD} + \alpha C_D)^{-1} R$ | (294, 200) | resíduo "filtrado" pelo sistema regularizado | Ma e Bi, Eq. 32 |
| `O_atual`, `O_novo` | $\mathcal{O}_j^i(\mathbf{m}_j^i)$, $\mathcal{O}_j^i(\mathbf{m}_j^{i+1})$ | (200,) | objetivo por membro antes e depois do passo | Ma e Bi, Eq. 34 |
| `L_novo` | $L_j^i(\mathbf{m}_j^{i+1})$ | (200,) | objetivo **previsto** pela linearização | Ma e Bi, Eq. 38 |
| `reducao_real` | numerador de $\rho_j$ | (200,) | quanto o objetivo de fato caiu | Ma e Bi, Eq. 37 |
| `reducao_prevista` | denominador de $\rho_j$ | (200,) | quanto a linearização previa que cairia | Ma e Bi, Eq. 37 |
| `rho` | $\rho_j$ | (200,) | razão de ganho de cada membro | Ma e Bi, Eq. 37; Cap. 2 `eq:gain_ratio` |
| `O_barra` | $\bar{\mathcal{O}}^i$ | escalar | desajuste médio normalizado do conjunto | Ma e Bi, Eq. 39 |
| `eta1`, `eta2` | $\eta_1$, $\eta_2$ | escalar | tolerâncias de parada | Ma e Bi, Seção 4.3 |
| `fator_ruido` | a constante 4 de $4p$ | escalar | critério de parada por nível de ruído | Ma e Bi, Eq. 43 |
| `limites` | — | 2 × (297, 1) | faixa física para truncar os modelos | Ma e Bi, Seção 4.4 |

---

## 5. Arquivo por arquivo

### 5.1 `dados.py` — entrada de dados

Não tem teoria. Existe para que **só um lugar do código saiba ler os arquivos**.

Os arquivos `.dat` não têm cabeçalho: a informação de que a coluna 4 é $V_p$ e
a coluna 6 é $\rho$ existe apenas como posição. Se cada módulo repetisse esses
índices, um único índice trocado — $V_s$ no lugar de $\rho$, por exemplo —
passaria despercebido e contaminaria o experimento. Por isso os índices ficam
declarados uma vez, e os demais módulos pedem os dados por nome.

| Nome | O que faz |
|---|---|
| `DATA_DIR` | Pasta dos arquivos `.dat`. Usa a cópia local em `data/` (redistribuída sob licença MIT) ou, na falta dela, a da SeReMpy. |
| `_localiza_serempy()` | Procura a biblioteca SeReMpy (variável `SEREMPY_PATH` ou pastas vizinhas) e a coloca no `sys.path`, para que `from SeReMpy.Inversion import …` funcione. |
| `COLUNAS_POCO`, `COLUNAS_SISMICA` | O único lugar em que a posição de cada coluna está escrita. |
| `carrega_dados()` | Lê os dois arquivos e devolve um dicionário com tudo o que os experimentos usam. |

O que `carrega_dados()` devolve:

| Chave | Símbolo | Forma | Conteúdo |
|---|---|---|---|
| `Time` | $t$ | (99, 1) | tempo do poço |
| `Vp`, `Vs`, `Rho` | $V_p$, $V_s$, $\rho$ | (99, 1) cada | propriedades do poço: o **modelo de referência** |
| `Z` | $Z = \rho V_p$ | (99, 1) | impedância de referência (Cap. 2, `eq:impedancia`) |
| `TimeSeis` | — | (98, 1) | tempo da sísmica: ponto médio entre amostras do poço |
| `Snear`, `Smid`, `Sfar` | — | (98, 1) cada | traços a 15°, 30° e 45° |
| `dt` | $\Delta t$ | escalar | passo de amostragem, tirado do tempo do poço |

Os arquivos:

- `data5log.dat` — perfil de poço. Colunas $\phi$, $V_{clay}$, $S_w$, tempo,
  $V_p$, $V_s$, $\rho$. As três primeiras (propriedades petrofísicas) não são
  usadas neste trabalho.
- `data5seis.dat` — tempo e os três traços sísmicos.

**Quem usa os traços — e quem não usa.** Isto costuma confundir:

| Experimento | Usa os traços de `data5seis.dat`? |
|---|---|
| `experimento.py` (acústico) | **Sim**: `Snear` é o dado observado $\mathbf{d}_{obs}$. Alterar o arquivo altera o resultado. |
| `experimento_elastico.py` | **Não**. O dado observado é **gerado** a partir do poço, com ruído de SNR controlada. Alterar o arquivo não muda nada. |

O motivo do elástico está na seção 5.3: os traços do pacote foram gerados pelo
mesmo operador usado na inversão, sem ruído, e invertê-los seria *inverse
crime*.

Os números do `data5seis.dat` **não são independentes** do `data5log.dat`:
foram calculados a partir dele. Alterar um sem o outro rompe a relação entre
dado e referência, e o erro medido contra o poço deixa de ter sentido. Para
variar o dado de forma controlada, os caminhos são mudar a SNR, trocar o perfil
de poço ou especificar a $C_D$ errada de propósito.

**Fonte:** Grana et al. (2021), biblioteca SeReMpy. São os mesmos dados do
`ESPetroInversionDriver.py` e do artigo de Caetano et al.

---

### 5.2 `forward.py` — modelo direto acústico (caso simples)

Estima só a impedância $Z$ a partir do traço de incidência normal. Continua no
repositório como caso de comparação, mas o experimento principal agora é o
elástico (seção 5.3).

**Teoria (Cap. 2, Seção 2.1):**

- Impedância: $Z = \rho\,V_p$ — `eq:impedancia`
- Refletividade sob contraste fraco: $R \approx \tfrac12 \Delta(\ln Z)$ — `eq:reflexao_log`
- Modelo convolucional: $s(t) = w(t) * r(t) + \epsilon(t)$ — `eq:convolucao`

Em forma matricial, o código faz exatamente:

$$g(Z) = W \left( \tfrac{1}{2}\, D \ln Z \right)$$

| Função | Símbolo | O que faz |
|---|---|---|
| `operador_acustico(nm, wavelet)` | $D$, $W$ | Monta a matriz de diferenças $D$ (`DifferentialMatrix`, faz $x_{k+1}-x_k$) e a matriz de convolução $W$ (`WaveletMatrix`). Verifica se a wavelet cabe no perfil. |
| `refletividade(Z, D)` | $R = \tfrac12 D\ln Z$ | Coeficientes de reflexão. Rejeita $Z \le 0$, porque usa logaritmo. |
| `modelo_direto(Z, D, W)` | $g(Z)$ | Traço sintético. Aceita um modelo `(nm,1)` ou o conjunto inteiro `(nm, ne)` de uma vez. |
| `tempo_sismico(Time)` | — | Tempos das amostras sísmicas: ponto médio entre amostras do poço. |
| `monta_forward(nm, dt)` | $g$ | Atalho que cria a wavelet de Ricker e devolve a função `g` pronta. |

**Fonte:** as primitivas `DifferentialMatrix`, `WaveletMatrix` e
`RickerWavelet` são da SeReMpy (Grana et al., 2021, Cap. 5.1), usadas sem
modificação.

**Por que a incógnita é $Z$ e não $\ln Z$:** em $\ln Z$ o operador seria
$g = (\tfrac12 W D)\,m$, exatamente linear. Num problema linear o ES-MDA e o
iES-LM convergiriam para a mesma resposta e a comparação perderia o sentido.

---

### 5.3 `forward_elastico.py` — modelo direto elástico (experimento principal)

Estima $V_p$, $V_s$ e $\rho$ simultaneamente a partir dos três ângulos. É a
configuração do artigo de Caetano et al.

**Teoria:** aproximação de Aki-Richards para o coeficiente de reflexão em
função do ângulo $\theta$, convoluída com a wavelet:

$$R(\theta) \approx c_p(\theta)\,\Delta\ln V_p + c_s(\theta)\,\Delta\ln V_s + c_\rho(\theta)\,\Delta\ln\rho$$

com

$$c_p = \tfrac12\left(1+\tan^2\theta\right), \qquad
c_s = -4\,\frac{\bar V_s^2}{\bar V_p^2}\sin^2\theta, \qquad
c_\rho = \tfrac12\left(1 - 4\,\frac{\bar V_s^2}{\bar V_p^2}\sin^2\theta\right)$$

onde $\bar V_p$, $\bar V_s$ são as médias nas interfaces. Isso está
implementado dentro da `SeismicModel` da SeReMpy (Grana et al., 2021, Cap. 5.1).

Duas observações que conectam com o resto do trabalho:

- **Com $\theta = 0$**, $c_p = \tfrac12$, $c_s = 0$, $c_\rho = \tfrac12$, e a
  fórmula vira $R = \tfrac12\Delta\ln(V_p\rho) = \tfrac12\Delta\ln Z$ — o
  modelo acústico da seção 5.2 é o caso particular de incidência normal.
- **A não-linearidade** vem dos coeficientes, que dependem da razão
  $V_s/V_p$ do próprio modelo. Medida: desvio de superposição de
  $7{,}2\times10^{-2}$, cerca de 12 vezes maior que no caso acústico.

| Nome | Símbolo | O que faz |
|---|---|---|
| `ANGULOS` | $\theta$ | 15°, 30° e 45° |
| `empilha(Vp, Vs, Rho)` | $\mathbf{m} = [V_p;\,V_s;\,\rho]$ | Junta as três propriedades num vetor de 297 posições. |
| `desempilha(M)` | — | Operação inversa: separa em três blocos de 99. |
| `monta_forward(Time, dt)` | $g(\mathbf{m})$ | Devolve `g`. Para cada membro chama `SeismicModel` e empilha os três traços: saída `(294, ne)`. |
| `separa_angulos(d)` | — | Divide um vetor de dados em Near, Mid e Far. |
| `adiciona_ruido(d_limpo, snr, rng)` | $\mathbf{d}_{obs} = g(\mathbf{m}_{ref}) + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal N(0, C_D)$ | Contamina o dado com ruído gaussiano. Desvio do ruído = desvio do sinal ÷ SNR. Devolve também a $C_D$ **coerente com o ruído de fato adicionado**. |

**Por que `adiciona_ruido` existe:** o traço distribuído com a SeReMpy foi
gerado pela própria `SeismicModel`, sem ruído (resíduo do modelo verdadeiro:
$7{,}7\times10^{-9}$). Inverter esse dado com o mesmo operador seria *inverse
crime* — o método não enfrentaria erro nenhum. **Fonte do procedimento:**
Caetano et al., Seção 3.2.

> **Atenção para o texto do TCC:** o Cap. 2 apresenta só a formulação
> acústica. A equação de Aki-Richards acima ainda não está no documento e vai
> precisar entrar na fundamentação ou na metodologia.

---

### 5.4 `prior.py` — o conjunto *a priori*

Gera os $N_e$ modelos iniciais: o que se acredita sobre a subsuperfície
**antes** de olhar a sísmica.

**Ideia:** a sísmica é limitada em banda (Cap. 2, Seção 2.1) e não recupera as
baixas frequências. Então o *a priori* fornece a **tendência de baixa
frequência** e a inversão recupera o **detalhe**.

| Função | O que faz | Fonte |
|---|---|---|
| `tendencia_suave(Z)` | Filtra o perfil com Butterworth passa-baixa (ordem 3, corte 0,04). É a média do *a priori*. | Mesmo filtro do `ESPetroInversionDriver.py` (SeReMpy) |
| `covariancia_espacial(nm, dt, L)` | Correlação gaussiana entre amostras, $C_{kl} = \exp\!\left(-(\lvert t_k-t_l\rvert/L)^2\right)$. Por omissão $L = 5\,dt$. Amostras vizinhas ficam parecidas: rocha é contínua. | Mesma construção dos drivers da SeReMpy |
| `conjunto_prior(tendencia, ne, dt)` | Caso **acústico**: sorteia em $\ln Z$ e exponencia, garantindo $Z > 0$. | Adaptação própria |
| `limites_fisicos(Z)` | Faixa de truncamento para o caso acústico. | Ma e Bi, Seção 4.4 (truncar aos limites) |
| `conjunto_prior_multivariado(tendencias, ne, dt, sigma0)` | Caso **elástico**: sorteia $V_p$, $V_s$, $\rho$ juntas. | Grana et al. (2021), Cap. 3.6 (`CorrelatedSimulation`) |

**Como o multivariado funciona:** a covariância total é o produto de Kronecker

$$C_{prior} = \Sigma_0 \otimes C_{tempo}$$

- `sigma0` ($\Sigma_0$, 3×3) — covariância **entre** as propriedades, calculada
  do poço. Preserva, por exemplo, que $V_p$ e $V_s$ tendem a subir juntos.
- `C_tempo` (99×99) — covariância **ao longo do tempo**.

O código aplica o fator de Cholesky de cada uma em um eixo de uma matriz de
ruído `z ~ N(0, I)`, o que dá o mesmo resultado sem montar a matriz 297×297.

**Por que não usar a `CorrelatedSimulation` da SeReMpy diretamente:** ela usa o
gerador aleatório global do numpy, o que impede reprodutibilidade por semente.
A função aqui faz a mesma conta com `numpy.random.Generator`.

---

### 5.5 `ieslm.py` — o algoritmo

**Fonte: Ma e Bi (2019), Algoritmo 2 (Seção 4.3), Eqs. 30–43.** Todo este
arquivo vem desse artigo.

#### Funções auxiliares

---

**`covariancias(M, G)`** → `C_MD`, `C_DD`

$$C_{MD}^i = \frac{1}{N_e-1}\sum_{j=1}^{N_e}(\mathbf{m}_j - \bar{\mathbf{m}})(\mathbf{g}_j - \bar{\mathbf{g}})^{\top}
\qquad
C_{DD}^i = \frac{1}{N_e-1}\sum_{j=1}^{N_e}(\mathbf{g}_j - \bar{\mathbf{g}})(\mathbf{g}_j - \bar{\mathbf{g}})^{\top}$$

Ma e Bi, **Eqs. 30 e 31**. Cap. 2: `eq:cmd` e `eq:cdd`.

É o "truque" dos métodos de conjunto: $C_{MD} \approx C_{MM}\bar G^{\top}$ e
$C_{DD} \approx \bar G C_{MM}\bar G^{\top}$. A **sensibilidade do dado ao
modelo** sai da dispersão do próprio conjunto, sem calcular a Jacobiana $G$.

No código, `dM` e `dG` são os desvios em relação à média (as parcelas
$\mathbf{m}_j - \bar{\mathbf{m}}$ e $\mathbf{g}_j - \bar{\mathbf{g}}$), e o
somatório vira um produto de matrizes.

---

**`objetivo_por_membro(d_pert, G, C_D_inv)`** → vetor `(ne,)`

$$\mathcal{O}_j^i(\mathbf{m}) = \tfrac12\left[\mathbf{d}_{i,j}^{o} - g(\mathbf{m})\right]^{\top} C_D^{-1}\left[\mathbf{d}_{i,j}^{o} - g(\mathbf{m})\right]$$

Ma e Bi, **Eq. 34**. Mesma forma do `eq:misfit` do Cap. 2, mas com o dado
**perturbado** de cada membro.

A ponderação por $C_D^{-1}$ mede o erro **em unidades do ruído esperado**: um
resíduo de 0,01 é enorme se o ruído é 0,001 e desprezível se é 1.

---

**`desajuste_medio(d_obs, G, C_D_inv)`** → `O_barra`

$$\bar{\mathcal{O}}^i = \frac{1}{N_e}\sum_{j=1}^{N_e}\frac{1}{2N_d}\left[\mathbf{d}_{obs} - g(\mathbf{m}_j^i)\right]^{\top}C_D^{-1}\left[\mathbf{d}_{obs} - g(\mathbf{m}_j^i)\right]$$

Ma e Bi, **Eq. 39**.

Diferenças em relação ao anterior, ambas de propósito:

- usa o dado **não perturbado**, porque a perturbação muda a cada iteração e só
  o dado original permite comparar uma iteração com outra;
- divide por $2N_d$, o que dá uma escala interpretável: **$\bar{\mathcal{O}}
  \approx 0{,}5$ significa ajustar exatamente no nível do ruído**. Abaixo
  disso, o método está ajustando ruído.

É o que aparece no eixo vertical da figura de convergência.

---

**`desajuste_absoluto(d_obs, G, C_D_inv)`** → `R`

$$R^i = \frac{1}{N_e}\sum_{j}\left[\mathbf{d}_{obs} - g(\mathbf{m}_j^i)\right]^{\top}C_D^{-1}\left[\mathbf{d}_{obs} - g(\mathbf{m}_j^i)\right] = 2N_d\,\bar{\mathcal{O}}^i$$

Ma e Bi, **Eq. 42**. Serve só para o critério de parada da Eq. 43.

---

**`fator_lm(rho)`**

$$\max\left(\tfrac13,\ 1 - (2\rho_j - 1)^3\right)$$

Ma e Bi, **parte da Eq. 40**. Cap. 2: `eq:lm_update_rule`.

É o coração da região de confiança. Alguns valores conferíveis à mão:

| $\rho_j$ | Situação | Fator | Efeito em $\gamma$ |
|---|---|---|---|
| 1 | linearização perfeita | 1/3 | cai para um terço → passos maiores |
| 0,5 | linearização razoável | 1 | mantém |
| 0 | passo não reduziu o objetivo | 2 | dobra → passos menores |
| < 0 | passo **piorou** o objetivo | > 2 | amortece ainda mais |

O $\max(1/3, \cdot)$ impede que o amortecimento caia bruscamente numa única
iteração.

---

**`perturba_observacao(d_obs, alpha, raiz_C_D, ne, rng)`** → `d_pert`

$$\mathbf{d}_{i,j}^{o} = \mathbf{d}_{obs} + \boldsymbol{\xi}_j^i, \qquad \boldsymbol{\xi}_j^i \sim \mathcal N(0,\ \alpha^i C_D)$$

Ma e Bi, **Eqs. 26–27** e **Algoritmo 2**.

Detalhe que diferencia o iES-LM de outros métodos iterativos: a perturbação é
**regerada a cada iteração** e sua covariância é **escalada pelo $\alpha$
corrente** (Ma e Bi, Seção 4.4).

O ruído correlacionado é gerado como $C_D^{1/2}\mathbf{z}$ com $\mathbf{z}\sim\mathcal
N(0, I)$, usando o fator de Cholesky (`_raiz_covariancia`). Isso vale para
qualquer $C_D$, inclusive não diagonal.

---

**Auxiliares técnicas** (sem correspondente no artigo):

| Função | O que faz |
|---|---|
| `_raiz_covariancia(C_D)` | Fator $L$ com $LL^{\top} = C_D$. Usa Cholesky; se $C_D$ não for definida positiva, cai para a decomposição espectral. |
| `_resolve(A, B)` | Resolve $AX = B$ sem inverter $A$ explicitamente. Usa pseudo-inversa se $A$ for singular. |
| `_largura_envelope(M)` | Largura P10–P90 de cada parâmetro, guardada a cada iteração. |

---

**`ResultadoIESLM`** — o que a função devolve

| Campo | Conteúdo |
|---|---|
| `conjunto` | Conjunto final: o de **menor** $\bar{\mathcal{O}}$ entre todas as iterações (Ma e Bi, fim da Seção 4.3) |
| `iteracao_final` | De qual iteração ele veio |
| `desajuste` | Histórico de $\bar{\mathcal{O}}^i$ |
| `alpha`, `gamma` | Históricos de $\alpha^i$ e $\gamma^i$ |
| `rho_mediano` | Mediana de $\rho_j$ por iteração |
| `rho_por_membro` | Todos os $\rho_j$ de cada iteração |
| `espalhamento` | Largura P10–P90 de cada parâmetro por iteração |
| `motivo_parada` | Qual critério encerrou |
| `n_avaliacoes` | Quantas vezes `g` foi chamada: o **custo computacional** |

#### A função principal: `ieslm(...)`

Parâmetros de entrada:

| Parâmetro | Símbolo | Padrão | Significado |
|---|---|---|---|
| `prior` | $\{\mathbf{m}_j^0\}$ | — | conjunto inicial |
| `d_obs` | $\mathbf{d}_{obs}$ | — | dado observado |
| `g` | $g(\cdot)$ | — | modelo direto, como função |
| `C_D` | $C_D$ | — | covariância do erro |
| `gamma0` | $\gamma^0$ | 1,0 | valor inicial (Algoritmo 2) |
| `max_iter` | — | 20 | limite de iterações |
| `eta1` | $\eta_1$ | $10^{-4}$ | parada por variação do desajuste |
| `eta2` | $\eta_2$ | $10^{-2}$ | parada por variação dos parâmetros |
| `limites` | — | nenhum | truncamento físico |
| `fator_ruido` | 4 em $4p$ | nenhum | parada por nível de ruído (Eq. 43) |
| `rng` | — | — | gerador aleatório, para reprodutibilidade |

#### O algoritmo passo a passo

Abaixo, cada bloco do código na ordem em que roda, com a equação correspondente.

**Inicialização** — Algoritmo 2:

```python
M = trunca(M)
G = g(M)
O_barra = desajuste_medio(d_obs, G, C_D_inv)
gamma = float(gamma0)
alpha = gamma * O_barra
```

$$\gamma^0 = 1, \qquad \alpha^0 = \gamma^0\,\bar{\mathcal{O}}^0$$

O $\alpha$ inicial é proporcional ao desajuste inicial: se o conjunto começa
muito longe do dado, o primeiro passo é bem amortecido.

**A cada iteração $i$:**

**Passo 1 — perturbar a observação.**

```python
d_pert = perturba_observacao(d_obs, alpha, raiz_C_D, ne, rng)
```

$\mathbf{d}_{i,j}^{o} = \mathbf{d}_{obs} + \boldsymbol{\xi}_j^i$, com
$\boldsymbol{\xi}_j^i \sim \mathcal N(0, \alpha^i C_D)$.

**Passo 2 — estimar as covariâncias pelo conjunto.** Eqs. 30–31.

```python
C_MD, C_DD = covariancias(M, G)
```

**Passo 3 — atualizar cada membro.** Eq. 32 (Cap. 2: `eq:ies_lm_update`).

```python
R = d_pert - G
V = _resolve(C_DD + alpha * C_D, R)
M_novo = trunca(M + C_MD @ V)
```

$$\mathbf{m}_j^{i+1} = \mathbf{m}_j^{i} + C_{MD}^i\left(C_{DD}^i + \alpha^i C_D\right)^{-1}\left(\mathbf{d}_{i,j}^{o} - g(\mathbf{m}_j^i)\right)$$

Leitura da equação:

- `R` é o quanto cada membro erra;
- `V` é esse erro passado pelo sistema $(C_{DD} + \alpha C_D)^{-1}$;
- `C_MD @ V` traduz a correção do espaço dos dados para o espaço dos modelos.

**O papel do $\alpha$ está todo aqui.** Com $\alpha$ grande, o termo $\alpha
C_D$ domina, `V` fica pequeno e o passo é curto. Com $\alpha$ pequeno, o passo
se aproxima de Gauss-Newton, longo.

`trunca` aplica os limites físicos (Ma e Bi, Seção 4.4).

**Passo 4 — avaliar o novo conjunto.**

```python
G_novo = g(M_novo)
n_aval += 1
```

É a única chamada ao modelo direto por iteração — o custo que o experimento mede.

**Passo 5 — calcular a razão de ganho.** Eqs. 34, 37 e 38.

```python
O_atual = objetivo_por_membro(d_pert, G, C_D_inv)
O_novo  = objetivo_por_membro(d_pert, G_novo, C_D_inv)
L_novo  = 0.5 * alpha ** 2 * np.sum(V * (C_D @ V), axis=0)
reducao_real     = O_atual - O_novo
reducao_prevista = O_atual - L_novo
rho = reducao_real / reducao_prevista
```

$$\rho_j = \frac{\mathcal{O}_j^i(\mathbf{m}_j^i) - \mathcal{O}_j^i(\mathbf{m}_j^{i+1})}{L_j^i(\mathbf{m}_j^i) - L_j^i(\mathbf{m}_j^{i+1})}$$

- **Numerador** (`reducao_real`): o que de fato aconteceu, com o modelo
  direto verdadeiro.
- **Denominador** (`reducao_prevista`): o que a linearização previa. Usa
  $L_j(\mathbf{m}^i) = \mathcal{O}_j(\mathbf{m}^i)$ e a Eq. 38 para
  $L_j(\mathbf{m}^{i+1})$.
- Os dois objetivos usam **o mesmo** `d_pert` — Ma e Bi insistem nisso
  (Seção 4.3), senão a comparação misturaria perturbações diferentes.

Salvaguarda: se a redução prevista não for positiva, `rho` recebe 0, o que
dobra o amortecimento na regra seguinte.

**Passo 6 — desajuste do conjunto atualizado.** Eq. 39.

```python
M, G = M_novo, G_novo
O_barra = desajuste_medio(d_obs, G, C_D_inv)
```

**Passo 7 — atualizar $\gamma$ e $\alpha$.** Eqs. 40–41 (Cap. 2:
`eq:lm_update_rule`, `eq:median_rule`).

```python
gamma_j = gamma * fator_lm(rho)
gamma   = float(np.median(gamma_j))
alpha   = float(min(alpha, np.median(gamma_j * O_barra)))
```

$$\gamma_j^i = \gamma^i\max\left(\tfrac13,\ 1-(2\rho_j-1)^3\right), \qquad \alpha_j^i = \gamma_j^i\,\bar{\mathcal{O}}^i$$

$$\gamma^{i+1} = \operatorname{mediana}(\gamma_j^i), \qquad \alpha^{i+1} = \min\left(\alpha^i,\ \operatorname{mediana}(\alpha_j^i)\right)$$

Dois detalhes:

- **Mediana, não média**: membros com linearização ruim não devem contaminar o
  $\alpha$ aplicado a todos (Ma e Bi, texto após a Eq. 41).
- **O `min`**: $\alpha$ nunca cresce entre iterações.

**Passo 8 — guardar o melhor.**

```python
if O_barra < melhor_desajuste:
    melhor_conjunto = M.copy()
```

O iES-LM **sempre atualiza** o conjunto, mesmo com passo ruim, mas devolve no
fim a iteração de menor desajuste (Ma e Bi, Seção 4.3). É a diferença de
comportamento mais visível em relação ao LM clássico, que rejeita o passo.

**Passo 9 — critérios de parada.** Ma e Bi, Seções 4.3 e 4.5.

| Critério | Condição no código | Fonte |
|---|---|---|
| Nível de ruído | `desajuste_absoluto(...) < fator_ruido * nd`, isto é, $R^i < 4p$ | Eq. 43, Seção 4.5 |
| Estagnação do desajuste | $\lvert\bar{\mathcal{O}}^{i+1}-\bar{\mathcal{O}}^i\rvert/\bar{\mathcal{O}}^i < \eta_1$ | Seção 4.3 |
| Estagnação dos parâmetros | $\lVert\mathbf{M}^{i+1}-\mathbf{M}^i\rVert/\lVert\mathbf{M}^i\rVert < \eta_2$ | Seção 4.3 |
| Limite de iterações | laço termina | Seção 4.3 |

O critério da Eq. 43 é o que impede o sobreajuste: o iES-LM resolve um
problema de **máxima verossimilhança** (Ma e Bi, Eq. 10 — sem termo de *prior*),
então sem ele continua ajustando até modelar o ruído.

---

### 5.6 `metricas.py` — como os métodos são comparados

**Fonte do protocolo:** Caetano et al., Seção 3.4, que avalia três eixos:
adesão aos dados e ao modelo, qualidade da incerteza e custo computacional.

| Função | Fórmula / definição | O que responde |
|---|---|---|
| `rmse(conjunto, verdadeiro)` | $\sqrt{\frac{1}{n}\sum_k(\bar m_k - m_k^{ref})^2}$ | A média do conjunto está perto do poço? |
| `envelope(conjunto)` | percentis P10 e P90 amostra a amostra | Onde estão 80% dos membros? |
| `largura_envelope(conjunto)` | média de P90 − P10 | Quanta incerteza o conjunto ainda representa? |
| `taxa_cobertura(conjunto, verdadeiro)` | fração das amostras em que a referência cai dentro de P10–P90 | A incerteza está **calibrada**? O ideal é 0,8. |
| `teste_ks(conjunto, verdadeiro)` | Kolmogorov-Smirnov de duas amostras | O método reproduz a **distribuição** de valores da propriedade? |
| `sequencia_alpha_esmda(na)` | $\alpha_l \propto 2^{N_a-l}$, normalizada para $\sum 1/\alpha_l = 1$ | Os fatores do ES-MDA |

**Por que cobertura e largura juntas:** um conjunto estreito pode ter
convergido bem ou pode ter colapsado. Só a cobertura distingue — no colapso, o
envelope estreito fica longe da referência e a cobertura despenca.

**A sequência de $\alpha_l$:** com $N_a = 4$ dá $[15;\ 7{,}5;\ 3{,}75;\
1{,}875]$. A condição $\sum 1/\alpha_l = 1$ é de Emerick e Reynolds (2013); a
escolha de uma sequência **decrescente** (amortece mais no começo, quando a
linearização é pior) segue Caetano et al.

---

### 5.7 `experimento_elastico.py` — o experimento

Amarra tudo e produz a tabela e as figuras.

**Constantes de configuração:**

| Constante | Valor | Significado |
|---|---|---|
| `NE` | 200 | tamanho do conjunto, $N_e$ |
| `NITER_MDA` | 4 | número de assimilações do ES-MDA, $N_a$ |
| `MAX_ITER_LM` | 10 | limite de iterações do iES-LM |
| `SNR` | 10 | razão sinal-ruído do dado |
| `FATOR_RUIDO` | 4 | constante da Eq. 43 |
| `SEMENTE` | 42 | semente única para prior, ruído e perturbações |
| `JANELA_ZOOM` | 1,810–1,822 s | janela de maior divergência entre os métodos |

**Funções:**

| Função | O que faz |
|---|---|
| `carrega_cenario(modelo=None)` | Obtém o poço por `dados.carrega_dados()` — ou usa o `modelo` sintético recebido —, monta `g`, **gera o dado observado** passando o poço pelo modelo direto e somando ruído, cria o conjunto *a priori* multivariado e define os limites físicos ($0{,}7\times$ mínimo a $1{,}3\times$ máximo de cada propriedade). Não usa os traços de `data5seis.dat`. |
| `roda_esmda(c)` | Aplica a `EnsembleSmootherMDA` da SeReMpy **sem modificação**, uma vez para cada $\alpha_l$ da sequência decrescente. |
| `executa(modelo=None)` | Roda os dois métodos sobre **o mesmo** cenário, imprime a tabela e gera as figuras. |
| `_tabela(...)` | RMSE, cobertura, largura, KS, desajuste final e custo para *a priori*, ES-MDA e iES-LM. |
| `_painel`, `_painel_detalhe`, `_figuras` | Geração das figuras. |

**O ES-MDA para comparação.** A `EnsembleSmootherMDA` (Grana et al., 2021,
Cap. 5.6) faz, a cada assimilação $l$:

$$\mathbf{m}_j \leftarrow \mathbf{m}_j + C_{MD}\left(C_{DD} + \alpha_l C_D\right)^{-1}\left(\mathbf{d}_{obs} + \sqrt{\alpha_l}\,\boldsymbol{\epsilon}_j - g(\mathbf{m}_j)\right)$$

Repare que é **a mesma estrutura** da Eq. 32 do iES-LM. A única diferença de
fundo é quem escolhe o $\alpha$: aqui ele vem de uma lista fixa; no iES-LM,
da razão de ganho. É isso que o trabalho compara.

**Figuras geradas** (em `figuras/`):

| Arquivo | Mostra |
|---|---|
| `elastico_perfis.png` | $V_p$, $V_s$, $\rho$: *a priori* e *a posteriori* dos dois métodos sobrepostos |
| `elastico_detalhe.png` | ampliação da janela de maior divergência |
| `elastico_erro.png` | média do conjunto menos referência, por propriedade |
| `elastico_ganho.png` | distribuição de $\rho_j$ e o fator aplicado a $\gamma$ |
| `elastico_residuos.png` | resíduo sísmico por ângulo |
| `elastico_convergencia.png` | $\bar{\mathcal{O}}$ por iteração e trajetórias de $\alpha$ |

---

### 5.8 `experimento.py` — o experimento acústico

Versão anterior, só com impedância e traço Near, sem ruído adicionado e com
$\alpha_i = 4$ constante no ES-MDA. Mantida porque está testada e pode servir
como caso de comparação mais simples. As funções têm o mesmo papel das do
experimento elástico.

---

### 5.9 `referencia.py` — modelos de referência sintéticos

Gera o **modelo verdadeiro** por código, como alternativa ao perfil de poço do
pacote. Existe porque a natureza do alvo afeta muito o resultado, e editar os
arquivos `.dat` quebraria a coerência entre `data5log.dat` e `data5seis.dat`
(ver seção 5.1). Trocar a referência por aqui é seguro: o experimento elástico
gera a observação a partir dela, então dado e alvo continuam consistentes.

| Função | O que faz |
|---|---|
| `modelo_em_camadas(...)` | Perfil constante por trechos, com contrastes abruptos nas interfaces. Parâmetros: número de camadas, contraste, espessura mínima. |
| `suaviza(modelo, corte)` | Versão suavizada de um modelo, com contrastes mais fracos. |

Só $V_p$ é sorteado; $V_s$ e a densidade vêm dele por relações **calibradas ao
poço de referência**, para que o modelo sintético fique na mesma escala física:

$$V_s = \frac{V_p}{1{,}570}, \qquad \rho = 1{,}610\; V_p^{1/4}$$

A segunda é a relação de Gardner. O coeficiente 1,610 foi ajustado ao próprio
poço e reproduz a densidade com erro médio de 2,5%. A razão 1,570 é a média
medida no mesmo poço.

O parâmetro `contraste` é o **desvio padrão relativo de $V_p$ entre camadas**;
o padrão 0,06 é o contraste medido no poço real.

Uso:

```python
import referencia as ref, experimento_elastico as X

modelo = ref.modelo_em_camadas(n_camadas=8, contraste=0.10,
                               rng=np.random.default_rng(0))
X.executa(modelo)          # sem argumento, usa o poço do pacote
```

**Por que isso importa para o experimento.** A aspereza do alvo muda
substancialmente a qualidade da estimativa e, sobretudo, a calibração da
incerteza. Medições com o mesmo protocolo do experimento:

| Alvo | Aspereza de $V_p$ | RMSE $V_p$ (ES-MDA / iES-LM) | Cobertura |
|---|---|---|---|
| poço do pacote | 0,1517 | 0,145 / 0,148 | 0,61 / 0,62 |
| 6 camadas | 0,0753 | 0,054 / 0,053 | 0,95 / 0,95 |
| 12 camadas | 0,1626 | 0,107 / 0,111 | 0,77 / 0,80 |
| 6 camadas, suavizado | 0,0272 | 0,017 / 0,012 | 1,00 / 1,00 |
| 6 camadas, contraste 2× | 0,1507 | 0,107 / 0,105 | 0,94 / 0,96 |

Compare a primeira linha com a última: **mesma aspereza, cobertura muito
diferente**. Não é a magnitude da variação que degrada a calibração, e sim
onde ela está no espectro — o poço real varia a cada amostra, em escala que a
sísmica não resolve; o modelo em camadas concentra a variação em poucas
interfaces, que são resolvíveis.

Isso torna a aspereza do alvo um terceiro eixo do estudo de sensibilidade, ao
lado do tamanho do conjunto e do nível de ruído.

---

## 6. As duas decisões de interpretação

Dois pontos em que o artigo precisou ser interpretado. Ambos estão documentados
na *docstring* de `ieslm()`.

### 6.1 O fator ½ na Eq. 38

A Eq. 38, como impressa, é

$$L_j^i(\mathbf{m}_j^{i+1}) = (\alpha^i)^2\left\lVert C_D^{1/2}(C_{DD}+\alpha^i C_D)^{-1}\mathbf{r}\right\rVert^2$$

mas a Eq. 34 tem fator ½ e o artigo afirma $L_j(\mathbf{m}^i) =
\mathcal{O}_j(\mathbf{m}^i)$. Derivando da Eq. 36 com $\bar G C_{MD} = C_{DD}$:

$$\mathbf{r} - \bar G(\mathbf{m}^{i+1}-\mathbf{m}^i) = \alpha C_D (C_{DD}+\alpha C_D)^{-1}\mathbf{r}
\quad\Longrightarrow\quad
L_j(\mathbf{m}^{i+1}) = \frac{(\alpha^i)^2}{2}\,\mathbf{v}^{\top}C_D\,\mathbf{v}$$

com $\mathbf{v} = (C_{DD}+\alpha C_D)^{-1}\mathbf{r}$ — que é a variável `V`.

**Verificação independente:** com modelo linear a linearização é exata, então
$\rho$ tem de valer 1. Medido: **1,00000000** com o fator; **1,103** sem ele.
Teste: `test_rho_vale_um_no_caso_linear`.

### 6.2 Qual $\bar{\mathcal{O}}$ usar na Eq. 40

A Eq. 40 escreve $\alpha_j = \gamma_j\,\bar{\mathcal{O}}^i$. A implementação
usa o $\bar{\mathcal{O}}$ do conjunto **recém-atualizado** — o conjunto ao qual
esse $\alpha$ será aplicado —, seguindo o mesmo padrão da inicialização do
Algoritmo 2, $\alpha^0 = \gamma^0\,\bar{\mathcal{O}}^0$.

---

## 7. Onde cada coisa é verificada

| Arquivo de teste | Verifica |
|---|---|
| `test_dados.py` | Carregamento, dimensões, $Z = \rho V_p$, consistência entre o tempo do poço e o da sísmica, e plausibilidade física (que denuncia índices de coluna trocados) |
| `test_forward.py` | Modelo acústico: refletividade à mão, reprodução do dado real, não-linearidade |
| `test_regras_lm.py` | Eqs. 30, 31, 34, 39, 40, 42 e a perturbação, isoladas |
| `test_mabi_exemplo1.py` | **Benchmark do artigo** (Seção 5.1, MLE = 4,76543) e $\rho = 1$ no caso linear |
| `test_integracao.py` | Algoritmo no dado sísmico; critérios de parada |
| `test_experimento.py` | Regressão do experimento acústico |
| `test_elastico.py` | Modelo elástico, *prior* multivariado, cobertura, KS, sequência de $\alpha_l$ |
| `test_referencia.py` | Modelos sintéticos: formato, relações físicas calibradas, número e espessura das camadas, reprodutibilidade e integração com o experimento |

Para rodar:

```bash
cd /Users/lucas/Documents/TCC/desenvolvimento/tcc2_ieslm && ../SeReMpy-main/Examples/venv/bin/python -m pytest tests/ -v
```

---

## 8. O que ainda falta levar para o texto do TCC

- A equação de **Aki-Richards** (seção 5.3): o Cap. 2 só tem a formulação acústica.
- A referência de **Caetano et al.** no `references.bib`.
- As **métricas** de cobertura e KS e a sequência decrescente de $\alpha_l$ no protocolo experimental.
- O procedimento de **adição de ruído** e o motivo (*inverse crime*).
- As duas **decisões de interpretação** da seção 6.
