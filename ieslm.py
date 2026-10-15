#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iES-LM - Iterative Ensemble Smoother com Levenberg-Marquardt.

Implementacao do Algoritmo 2 de Ma e Bi (2019), "A robust adaptive iterative
ensemble smoother scheme for practical history matching applications",
Computational Geosciences 23:415-442.

O modulo e agnostico ao dominio: o modelo direto entra como uma funcao
g(M) que recebe um conjunto (nm, ne) e devolve as previsoes (nd, ne). Isso
permite testar o nucleo tanto no exemplo sintetico do artigo quanto no
problema de inversao sismica.

Diferenca essencial para o ES-MDA (Emerick e Reynolds, 2013), que e a linha
de base do TCC: o ES-MDA usa fatores de inflacao alpha_i FIXOS, definidos a
priori; aqui o alpha^i e ADAPTATIVO, ajustado a cada iteracao por uma regra
de regiao de confianca baseada na razao de ganho rho_j.

Onde cada parte esta no artigo:

    Texto das Eqs. 26-27  observacao perturbada: xi ~ N(0, alpha^i C_D), logo
                          apos a Eq. 26; d^o = d_obs + xi, logo apos a Eq. 27
                          (as Eqs. 26-27 em si sao atualizacoes, precursoras
                          da Eq. 32; o passo explicito esta no Algoritmo 2)
    Eqs. 30-31            covariancias C_MD e C_DD estimadas pelo conjunto
    Eq. 32                atualizacao de cada membro
    Eq. 34                objetivo de cada membro (dado perturbado)
    Eqs. 36-38            objetivo linearizado L e razao de ganho rho_j
    Eq. 39                desajuste medio normalizado O_barra (dado limpo)
    Eqs. 40-41            regra de gamma e alpha pela mediana
    Eqs. 42-43            parada pelo nivel do ruido (Secao 4.5)
    Algoritmo 2           inicializacao, laco e criterios eta1, eta2
    Secao 4.3             devolve o conjunto de menor desajuste medio
    Secao 4.4             truncamento nos limites; perturbacao a cada iteracao
    Secao 5.2             gamma0 = 0.25 em problema muito nao-linear;
                          eta1 = 1e-4 e eta2 = 1e-2

Variaveis do codigo e simbolos do artigo:

    M, M_novo      {m_j^i}, {m_j^(i+1)}   conjunto de modelos (uma coluna por membro)
    G, G_novo      {g(m_j^i)}             previsoes do modelo direto
    d_obs          d_obs                  dado observado
    d_pert         d^o_(i,j)              dado perturbado do membro j (texto apos a Eq. 27)
    C_D            C_D                    covariancia do erro de medicao
    C_MD, C_DD     C_MD^i, C_DD^i         covariancias do conjunto (Eqs. 30-31)
    alpha          alpha^i                parametro de regularizacao (Eqs. 32, 41)
    gamma, gamma_j gamma^i, gamma_j^i     parametro da regiao de confianca (Eqs. 40-41)
    O_atual, O_novo O_j^i(m^i), O_j^i(m^(i+1))  objetivo de cada membro (Eq. 34)
    L_novo         L_j^i(m_j^(i+1))       objetivo linearizado apos o passo (Eq. 38)
    rho            rho_j                  razao de ganho (Eq. 37)
    O_barra        O_barra^i              desajuste medio normalizado (Eq. 39)
    desajuste_absoluto  R^i               desajuste medio sem normalizar (Eq. 42)
    nd             N_d (Eq. 39), p (Eq. 43)  numero de dados
    ne             N_e                    tamanho do conjunto
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResultadoIESLM:
    """
    Resultado de uma execucao do iES-LM.

    Attributes
    ----------
    conjunto : array_like
        Conjunto final (nm, ne) - o de MENOR desajuste medio entre todas as
        iteracoes, conforme o final da Secao 4.3 do artigo.
    iteracao_final : int
        Indice da iteracao de onde veio `conjunto`.
    desajuste : list of float
        Historico de O_barra (Eq. 39), um valor por conjunto avaliado.
    alpha : list of float
        Historico do parametro de regularizacao alpha^i.
    gamma : list of float
        Historico de gamma^i.
    rho_mediano : list of float
        Mediana das razoes de ganho rho_j em cada iteracao.
    rho_por_membro : list of array_like
        Razao de ganho de CADA membro, em cada iteracao. Guardado porque a
        dispersao entre membros e o que justifica a regra da mediana da
        Eq. 41: alguns membros apresentam linearizacoes ruins sem que isso
        deva contaminar o alpha aplicado globalmente.
    espalhamento : list of array_like
        Largura do envelope P10-P90 de cada parametro, a cada iteracao.
    motivo_parada : str
        Criterio que encerrou o laco.
    n_avaliacoes : int
        Numero de avaliacoes do modelo direto (custo computacional).
    """

    conjunto: np.ndarray
    iteracao_final: int
    desajuste: list = field(default_factory=list)
    alpha: list = field(default_factory=list)
    gamma: list = field(default_factory=list)
    rho_mediano: list = field(default_factory=list)
    rho_por_membro: list = field(default_factory=list)
    espalhamento: list = field(default_factory=list)
    motivo_parada: str = ''
    n_avaliacoes: int = 0


def covariancias(M, G):
    """
    COVARIANCIAS
    Covariancia cruzada modelo-dado e autocovariancia dos dados previstos,
    estimadas a partir do conjunto (Eqs. 30 e 31).

    Como no artigo, os desvios dos dados sao tomados em relacao a media das
    previsoes, g_barra (Eq. 28), e nao a previsao da media: isso evita uma
    simulacao extra por iteracao.

    Parameters
    ----------
    M : array_like
        Conjunto de modelos (nm, ne).
    G : array_like
        Previsoes correspondentes (nd, ne).

    Returns
    -------
    C_MD : array_like
        Covariancia cruzada (nm, nd).
    C_DD : array_like
        Autocovariancia dos dados previstos (nd, nd).
    """
    ne = M.shape[1]  # tamanho do conjunto, N_e
    # desvios de cada membro em relacao a media do conjunto
    dM = M - M.mean(axis=1, keepdims=True)  # m_j^i - m_barra^i
    dG = G - G.mean(axis=1, keepdims=True)  # g(m_j^i) - g_barra^i (Eq. 28)

    C_MD = dM @ dG.T / (ne - 1)  # Eq. 30
    C_DD = dG @ dG.T / (ne - 1)  # Eq. 31

    return C_MD, C_DD


def desajuste_medio(d_obs, G, C_D_inv):
    """
    DESAJUSTE MEDIO
    Desajuste de dados medio normalizado, O_barra (Eq. 39):
    O_barra^i = (1/N_e) * soma_j [1/(2 N_d)] (d_obs - g(m_j))^T C_D^-1 (d_obs - g(m_j)).

    Usa a observacao NAO perturbada, de proposito: como a perturbacao muda a
    cada iteracao, so o dado original permite comparar iteracoes entre si.

    Parameters
    ----------
    d_obs : array_like
        Observacao (nd, 1).
    G : array_like
        Previsoes do conjunto (nd, ne).
    C_D_inv : array_like
        Inversa da covariancia do erro (nd, nd).

    Returns
    -------
    float
        O_barra.
    """
    nd = d_obs.shape[0]  # N_d, numero de dados: normaliza a escala
    R = d_obs - G  # d_obs - g(m_j^i), sem perturbacao

    return float(np.mean(np.sum(R * (C_D_inv @ R), axis=0)) / (2.0 * nd))


def desajuste_absoluto(d_obs, G, C_D_inv):
    """
    DESAJUSTE ABSOLUTO
    Desajuste medio SEM normalizar por 2*nd (Eq. 42), usado no
    criterio de parada por nivel de ruido (Eq. 43):
    R^i = (1/N_e) * soma_j (d_obs - g(m_j))^T C_D^-1 (d_obs - g(m_j)).

    Relacao com a Eq. 39: R = 2 * nd * O_barra.

    Parameters
    ----------
    d_obs : array_like
        Observacao nao perturbada (nd, 1).
    G : array_like
        Previsoes do conjunto (nd, ne).
    C_D_inv : array_like
        Inversa da covariancia do erro (nd, nd).

    Returns
    -------
    float
        R.
    """
    R = d_obs - G  # mesmo residuo da Eq. 39, aqui sem normalizar

    return float(np.mean(np.sum(R * (C_D_inv @ R), axis=0)))


def objetivo_por_membro(d_pert, G, C_D_inv):
    """
    OBJETIVO POR MEMBRO
    Funcao objetivo de cada membro do conjunto (Eq. 34):
    O_j^i(m) = 1/2 (d^o_(i,j) - g(m))^T C_D^-1 (d^o_(i,j) - g(m)).

    Diferente da Eq. 39, aqui entra o dado PERTURBADO: cada membro tem a sua
    propria realizacao da observacao.

    Parameters
    ----------
    d_pert : array_like
        Observacoes perturbadas, uma por membro (nd, ne).
    G : array_like
        Previsoes do conjunto (nd, ne).
    C_D_inv : array_like
        Inversa da covariancia do erro (nd, nd).

    Returns
    -------
    array_like
        Vetor (ne,) com o objetivo de cada membro.
    """
    R = d_pert - G  # d^o_(i,j) - g(m_j): contra o dado perturbado do membro

    return 0.5 * np.sum(R * (C_D_inv @ R), axis=0)  # o 1/2 da Eq. 34


def fator_lm(rho):
    """
    FATOR LM
    Fator de ajuste da regiao de confianca, max(1/3, 1 - (2*rho - 1)^3),
    parte da Eq. 40: gamma_j^i = gamma^i * fator_lm(rho_j).

    Interpretacao: rho e a razao entre a reducao real e a reducao prevista
    do objetivo. Passo perfeito (rho = 1) devolve 1/3, reduzindo gamma e
    portanto o amortecimento (passos maiores na proxima iteracao); passo ruim
    (rho = 0) devolve 2, dobrando o amortecimento.

    Parameters
    ----------
    rho : array_like or float
        Razao de ganho.

    Returns
    -------
    array_like or float
        Fator multiplicativo de gamma.
    """
    return np.maximum(1.0 / 3.0, 1.0 - (2.0 * np.asarray(rho, dtype=float) - 1.0) ** 3)


def _raiz_covariancia(C_D):
    """
    Fator L tal que L @ L.T = C_D, para gerar ruido correlacionado.
    Detalhe de implementacao: o artigo so diz xi ~ N(0, alpha^i C_D).
    """
    try:
        return np.linalg.cholesky(C_D)
    except np.linalg.LinAlgError:
        # C_D semidefinida: cai para a raiz simetrica via decomposicao espectral
        valores, vetores = np.linalg.eigh(C_D)  # autovalores e autovetores
        valores = np.clip(valores, 0.0, None)
        return vetores @ np.diag(np.sqrt(valores))


def perturba_observacao(d_obs, alpha, raiz_C_D, ne, rng):
    """
    PERTURBA OBSERVACAO
    Gera uma realizacao da observacao para cada membro,
    d^o_(i,j) = d_obs + xi_j^i, com xi_j^i ~ N(0, alpha^i C_D).

    Definida no texto que acompanha as Eqs. 26 e 27: a
    distribuicao de xi logo apos a Eq. 26, e d^o = d_obs + xi logo apos a
    Eq. 27. O passo aparece explicito no Algoritmo 2.

    O iES-LM regera essas perturbacoes A CADA ITERACAO, com a covariancia
    escalada pelo alpha corrente - e uma das diferencas em relacao aos
    metodos iES que perturbam uma unica vez no inicio (Secao 4.4).

    Parameters
    ----------
    d_obs : array_like
        Observacao (nd, 1).
    alpha : float
        Parametro de regularizacao corrente.
    raiz_C_D : array_like
        Fator da covariancia do erro (de _raiz_covariancia).
    ne : int
        Tamanho do conjunto.
    rng : numpy.random.Generator
        Gerador de numeros aleatorios.

    Returns
    -------
    array_like
        Observacoes perturbadas (nd, ne).
    """
    nd = d_obs.shape[0]  # numero de dados
    ruido = raiz_C_D @ rng.standard_normal((nd, ne))  # ~ N(0, C_D)

    return d_obs + np.sqrt(alpha) * ruido  # sqrt(alpha) escala para N(0, alpha C_D)


def _largura_envelope(M):
    """Largura do envelope P10-P90 de cada parametro (diagnostico; nao esta no artigo)."""
    p10, p90 = np.percentile(M, [10, 90], axis=1)  # um par por parametro

    return p90 - p10


def _resolve(A, B):
    """
    Resolve A X = B, com pseudo-inversa como reserva se A for singular.
    Calcula (C_DD + alpha C_D)^-1 (d^o - g) da Eq. 32 sem inverter a matriz.
    """
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ B


def ieslm(prior, d_obs, g, C_D, gamma0=1.0, max_iter=20, eta1=1e-4, eta2=1e-2,
          limites=None, fator_ruido=None, rng=None):
    """
    iES-LM
    Iterative Ensemble Smoother com Levenberg-Marquardt adaptativo
    (Ma e Bi, 2019, Algoritmo 2).

    Parameters
    ----------
    prior : array_like
        Conjunto a priori (nm, ne).
    d_obs : array_like
        Observacao (nd, 1).
    g : callable
        Modelo direto: recebe (nm, ne) e devolve (nd, ne).
    C_D : array_like
        Covariancia do erro de medicao (nd, nd).
    gamma0 : float, optional
        Valor inicial de gamma (o artigo usa 1.0 no Algoritmo 2; no
        Exemplo 2, muito nao-linear, usa 0.25 para convergir mais rapido no
        inicio - Secao 5.2).
    max_iter : int, optional
        Numero maximo de iteracoes.
    eta1 : float, optional
        Tolerancia na variacao relativa do desajuste medio (Eq. 39). Criterio
        do Algoritmo 2; o padrao 1e-4 e o do Exemplo 2.
    eta2 : float, optional
        Tolerancia na variacao relativa dos parametros. Criterio do Algoritmo
        2; o padrao 1e-2 e o do Exemplo 2. O artigo nao fixa
        a norma: aqui, norma de Frobenius do conjunto inteiro.
    limites : tuple of (float or array_like), optional
        (minimo, maximo) para truncar os modelos. O artigo trunca os valores
        aos limites quando saem da faixa fisica (Secao 4.4).
    fator_ruido : float, optional
        Ativa o criterio de parada por nivel de ruido (Eqs. 42-43): encerra
        quando R^i < fator_ruido * nd. O artigo usa 4 (Eq. 43). Sem ele, o
        iES-LM continua reduzindo o desajuste ate ajustar o proprio ruido
        das observacoes, colapsando o conjunto - ver Secao 4.5. Deixe em None
        para reproduzir o Algoritmo 2 puro (util nos testes com dado exato).
    rng : numpy.random.Generator, optional
        Gerador aleatorio; passe um com semente fixa para reprodutibilidade.

    Returns
    -------
    ResultadoIESLM

    Notes
    -----
    Duas decisoes de implementacao onde o artigo e ambiguo, ambas
    documentadas para a defesa:

    1. A Eq. 38, como impressa, nao traz o fator 1/2 que a Eq. 34
       tem. Derivando L a partir da Eq. 36 com G^i @ C_MD = C_DD (G^i e a
       Jacobiana aproximada da Eq. 35; ver Eqs. 30-31), chega-se a
       L(m^{i+1}) = (alpha^2 / 2) * v^T C_D v. O 1/2 e necessario para que L
       seja consistente com O (o artigo afirma L(m^i) = O(m^i)); sem ele,
       rho nao daria 1 no caso linear. Ver test_mabi_exemplo1.py.

    2. A Eq. 40 escreve alpha_j = gamma_j * O_barra^i. Aqui usa-se o
       O_barra do conjunto recem-atualizado, que e o conjunto ao qual esse
       alpha sera aplicado - mesma relacao da inicializacao do Algoritmo 2,
       onde alpha^0 = gamma^0 * O_barra^0.
    """
    rng = np.random.default_rng() if rng is None else rng

    M = np.array(prior, dtype=float, copy=True)
    d_obs = np.asarray(d_obs, dtype=float).reshape(-1, 1)
    C_D = np.asarray(C_D, dtype=float)
    ne = M.shape[1]  # N_e, tamanho do conjunto

    C_D_inv = np.linalg.inv(C_D)  # C_D^-1 das Eqs. 34, 39 e 42
    raiz_C_D = _raiz_covariancia(C_D)  # C_D^(1/2), para sortear xi ~ N(0, alpha C_D)

    def trunca(X):
        # Secao 4.4: valores fora da faixa fisica voltam ao limite
        if limites is None:
            return X
        return np.clip(X, limites[0], limites[1])

    M = trunca(M)  # {m_j^0}, conjunto inicial
    G = g(M)  # {g(m_j^0)}
    n_aval = 1  # chamadas ao modelo direto: o custo computacional medido

    # Inicializacao do Algoritmo 2: gamma^0 = 1, alpha^0 = gamma^0 * O_barra^0
    O_barra = desajuste_medio(d_obs, G, C_D_inv)  # O_barra^0 (Eq. 39)
    gamma = float(gamma0)
    alpha = gamma * O_barra

    res = ResultadoIESLM(conjunto=M.copy(), iteracao_final=0)
    res.desajuste.append(O_barra)
    res.alpha.append(alpha)
    res.gamma.append(gamma)
    res.espalhamento.append(_largura_envelope(M))

    melhor_desajuste = O_barra
    melhor_conjunto = M.copy()
    melhor_iter = 0
    motivo = 'numero maximo de iteracoes'

    for i in range(max_iter):  # laco "repeat ... until" do Algoritmo 2
        # Salvaguarda numerica (nao esta no artigo): alpha nulo trava a Eq. 32
        if alpha <= 0 or not np.isfinite(alpha):
            motivo = 'alpha degenerado (desajuste nulo)'
            break

        # Algoritmo 2: d^o_(i,j) = d_obs + xi_j^i, xi ~ N(0, alpha^i C_D)
        # (definido no texto apos as Eqs. 26 e 27). O Algoritmo 2
        # gera este dado no fim da iteracao anterior, ja com o alpha
        # atualizado; gera-lo aqui, no comeco da seguinte, e equivalente.
        d_pert = perturba_observacao(d_obs, alpha, raiz_C_D, ne, rng)

        # Eqs. 30-31 : C_MD^i e C_DD^i estimadas pelo conjunto
        C_MD, C_DD = covariancias(M, G)

        # Eq. 32 : m_j^(i+1) = m_j^i + C_MD (C_DD + alpha C_D)^-1 (d^o_(i,j) - g(m_j^i))
        R = d_pert - G  # d^o_(i,j) - g(m_j^i): o quanto cada membro erra
        # v = (C_DD + alpha C_D)^-1 r, reaproveitado na Eq. 38
        V = _resolve(C_DD + alpha * C_D, R) #decide o tamanho do passo
        M_novo = trunca(M + C_MD @ V)

        G_novo = g(M_novo)  # {g(m_j^(i+1))}: a unica simulacao da iteracao
        n_aval += 1

        # Eq. 34 , com o MESMO dado perturbado nos dois pontos, como o
        # artigo exige ao comentar o Algoritmo 2
        O_atual = objetivo_por_membro(d_pert, G, C_D_inv)  # O_j^i(m_j^i)
        O_novo = objetivo_por_membro(d_pert, G_novo, C_D_inv)  # O_j^i(m_j^(i+1))

        # Eq. 38: L_j^i(m^(i+1)) = (alpha^2/2) v^T C_D v
        L_novo = 0.5 * alpha ** 2 * np.sum(V * (C_D @ V), axis=0)
        # Eq. 37: rho_j = [O(m^i) - O(m^(i+1))] / [L(m^i) - L(m^(i+1))], com L(m^i) = O(m^i)
        reducao_real = O_atual - O_novo  # numerador
        reducao_prevista = O_atual - L_novo  # denominador

        # Salvaguarda (nao esta no artigo): reducao prevista nao positiva vira
        # rho = 0, que pela Eq. 40 dobra o amortecimento
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.where(reducao_prevista > 0, reducao_real / reducao_prevista, 0.0)
        rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

        # Variacao relativa dos parametros, para o criterio eta2 (Algoritmo 2)
        norma_M = np.linalg.norm(M)  # escala do conjunto, para medir a variacao
        var_parametros = (np.linalg.norm(M_novo - M) / norma_M) if norma_M > 0 else 0.0

        M, G = M_novo, G_novo
        O_barra_anterior = O_barra
        O_barra = desajuste_medio(d_obs, G, C_D_inv)  # O_barra do conjunto novo (Eq. 39; ver Notes, 2)

        # Eq. 40: gamma_j^i = gamma^i * max(1/3, 1 - (2 rho_j - 1)^3);
        #                  alpha_j^i = gamma_j^i * O_barra
        gamma_j = gamma * fator_lm(rho)
        # Eq. 41: gamma^(i+1) = mediana(gamma_j); alpha^(i+1) = min(alpha^i, mediana(alpha_j))
        gamma = float(np.median(gamma_j))
        alpha = float(min(alpha, np.median(gamma_j * O_barra)))

        res.desajuste.append(O_barra)
        res.alpha.append(alpha)
        res.gamma.append(gamma)
        res.rho_mediano.append(float(np.median(rho)))
        res.rho_por_membro.append(rho.copy())
        res.espalhamento.append(_largura_envelope(M))

        # Secao 4.3: guarda o conjunto de menor desajuste medio
        if O_barra < melhor_desajuste:
            melhor_desajuste = O_barra
            melhor_conjunto = M.copy()
            melhor_iter = i + 1

        # Criterios de parada: Algoritmo 2 e Secao 4.5
        if fator_ruido is not None:
            # Eqs. 42-43: R^i < 4p; desajuste no nivel do ruido, continuar seria sobreajuste
            if desajuste_absoluto(d_obs, G, C_D_inv) < fator_ruido * d_obs.shape[0]:
                motivo = 'desajuste no nivel do ruido (Eq. 43)'
                break
        if O_barra_anterior > 0:  # eta1: variacao relativa do desajuste medio (Algoritmo 2)
            variacao = abs(O_barra - O_barra_anterior) / O_barra_anterior
            if variacao < eta1:
                motivo = 'variacao do desajuste abaixo de eta1'
                break
        if var_parametros < eta2:  # eta2: variacao relativa dos parametros (Algoritmo 2)
            motivo = 'variacao dos parametros abaixo de eta2'
            break

    # Secao 4.3: o artigo devolve o conjunto de menor desajuste medio entre as iteracoes
    res.conjunto = melhor_conjunto
    res.iteracao_final = melhor_iter
    res.motivo_parada = motivo
    res.n_avaliacoes = n_aval

    return res
