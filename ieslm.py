#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iES-LM - Iterative Ensemble Smoother com Levenberg-Marquardt.

Implementacao do Algoritmo 2 de Ma e Bi (2019), "A robust iterative ensemble
smoother method for efficient history matching and uncertainty quantification",
Computational Geosciences 23:415-442.

O modulo e agnostico ao dominio: o modelo direto entra como uma funcao
g(M) que recebe um conjunto (nm, ne) e devolve as previsoes (nd, ne). Isso
permite testar o nucleo tanto no exemplo sintetico do artigo quanto no
problema de inversao sismica.

Diferenca essencial para o ES-MDA (Emerick e Reynolds, 2013), que e a linha
de base do TCC: o ES-MDA usa fatores de inflacao alpha_i FIXOS, definidos a
priori; aqui o alpha^i e ADAPTATIVO, ajustado a cada iteracao por uma regra
de regiao de confianca baseada na razao de ganho rho_j.

Equacoes implementadas (numeracao do artigo):
    (30) (31) covariancias cruzada e de dados previstos, via conjunto
    (32)      atualizacao de cada membro
    (34)      funcao objetivo por membro (dado perturbado)
    (37) (38) razao de ganho rho_j
    (39)      desajuste medio normalizado O_barra (dado NAO perturbado)
    (40) (41) atualizacao de gamma e alpha pela mediana entre membros
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
    motivo_parada: str = ''
    n_avaliacoes: int = 0


def covariancias(M, G):
    """
    COVARIANCIAS
    Covariancia cruzada modelo-dado e autocovariancia dos dados previstos,
    estimadas a partir do conjunto (Eqs. 30 e 31).

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
    ne = M.shape[1]
    dM = M - M.mean(axis=1, keepdims=True)
    dG = G - G.mean(axis=1, keepdims=True)

    C_MD = dM @ dG.T / (ne - 1)
    C_DD = dG @ dG.T / (ne - 1)

    return C_MD, C_DD


def desajuste_medio(d_obs, G, C_D_inv):
    """
    DESAJUSTE MEDIO
    Desajuste de dados medio normalizado, O_barra (Eq. 39).

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
    nd = d_obs.shape[0]
    R = d_obs - G

    return float(np.mean(np.sum(R * (C_D_inv @ R), axis=0)) / (2.0 * nd))


def desajuste_absoluto(d_obs, G, C_D_inv):
    """
    DESAJUSTE ABSOLUTO
    Desajuste medio SEM normalizar por 2*nd (Eq. 42), usado no criterio de
    parada por nivel de ruido (Eq. 43).

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
    R = d_obs - G

    return float(np.mean(np.sum(R * (C_D_inv @ R), axis=0)))


def objetivo_por_membro(d_pert, G, C_D_inv):
    """
    OBJETIVO POR MEMBRO
    Funcao objetivo de cada membro do conjunto (Eq. 34).

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
    R = d_pert - G

    return 0.5 * np.sum(R * (C_D_inv @ R), axis=0)


def fator_lm(rho):
    """
    FATOR LM
    Fator de ajuste da regiao de confianca, max(1/3, 1 - (2*rho - 1)^3),
    parte da Eq. 40.

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
    """Fator L tal que L @ L.T = C_D, para gerar ruido correlacionado."""
    try:
        return np.linalg.cholesky(C_D)
    except np.linalg.LinAlgError:
        # C_D semidefinida: cai para a raiz simetrica via decomposicao espectral
        valores, vetores = np.linalg.eigh(C_D)
        valores = np.clip(valores, 0.0, None)
        return vetores @ np.diag(np.sqrt(valores))


def perturba_observacao(d_obs, alpha, raiz_C_D, ne, rng):
    """
    PERTURBA OBSERVACAO
    Gera uma realizacao da observacao para cada membro,
    d^o_j = d_obs + xi_j, com xi_j ~ N(0, alpha * C_D).

    O iES-LM regera essas perturbacoes A CADA ITERACAO, com a covariancia
    escalada pelo alpha corrente - e uma das diferencas em relacao aos
    metodos iES que perturbam uma unica vez no inicio (Secao 4.4 do artigo).

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
    nd = d_obs.shape[0]
    ruido = raiz_C_D @ rng.standard_normal((nd, ne))

    return d_obs + np.sqrt(alpha) * ruido


def _resolve(A, B):
    """Resolve A X = B, com pseudo-inversa como reserva se A for singular."""
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
        Valor inicial de gamma (o artigo usa 1.0 no Algoritmo 2; valores
        menores, como 0.25, aceleram a convergencia em casos muito
        nao-lineares - ver Secao 5.2).
    max_iter : int, optional
        Numero maximo de iteracoes.
    eta1 : float, optional
        Tolerancia na variacao relativa do desajuste medio (Eq. 39).
    eta2 : float, optional
        Tolerancia na variacao relativa dos parametros.
    limites : tuple of (float or array_like), optional
        (minimo, maximo) para truncar os modelos. O artigo trunca os valores
        aos limites quando saem da faixa fisica (Secao 4.3).
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

    1. A Eq. 38, como impressa, nao traz o fator 1/2 que a Eq. 34 tem.
       Derivando L a partir da Eq. 36 com G_barra @ C_MD = C_DD, chega-se a
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
    ne = M.shape[1]

    C_D_inv = np.linalg.inv(C_D)
    raiz_C_D = _raiz_covariancia(C_D)

    def trunca(X):
        if limites is None:
            return X
        return np.clip(X, limites[0], limites[1])

    M = trunca(M)
    G = g(M)
    n_aval = 1

    # Inicializacao do Algoritmo 2: gamma^0 = 1, alpha^0 = gamma^0 * O_barra^0
    O_barra = desajuste_medio(d_obs, G, C_D_inv)
    gamma = float(gamma0)
    alpha = gamma * O_barra

    res = ResultadoIESLM(conjunto=M.copy(), iteracao_final=0)
    res.desajuste.append(O_barra)
    res.alpha.append(alpha)
    res.gamma.append(gamma)

    melhor_desajuste = O_barra
    melhor_conjunto = M.copy()
    melhor_iter = 0
    motivo = 'numero maximo de iteracoes'

    for i in range(max_iter):
        if alpha <= 0 or not np.isfinite(alpha):
            motivo = 'alpha degenerado (desajuste nulo)'
            break

        # Observacao perturbada desta iteracao: xi ~ N(0, alpha * C_D)
        d_pert = perturba_observacao(d_obs, alpha, raiz_C_D, ne, rng)

        # Eqs. 30-31: covariancias estimadas pelo conjunto
        C_MD, C_DD = covariancias(M, G)

        # Eq. 32: atualizacao de cada membro
        R = d_pert - G
        V = _resolve(C_DD + alpha * C_D, R)
        M_novo = trunca(M + C_MD @ V)

        G_novo = g(M_novo)
        n_aval += 1

        # Eq. 34, com o MESMO dado perturbado nos dois pontos
        O_atual = objetivo_por_membro(d_pert, G, C_D_inv)
        O_novo = objetivo_por_membro(d_pert, G_novo, C_D_inv)

        # Eq. 38 (com o fator 1/2 - ver Notes) e Eq. 37
        L_novo = 0.5 * alpha ** 2 * np.sum(V * (C_D @ V), axis=0)
        reducao_real = O_atual - O_novo
        reducao_prevista = O_atual - L_novo

        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.where(reducao_prevista > 0, reducao_real / reducao_prevista, 0.0)
        rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

        # Variacao relativa dos parametros, para o criterio eta2
        norma_M = np.linalg.norm(M)
        var_parametros = (np.linalg.norm(M_novo - M) / norma_M) if norma_M > 0 else 0.0

        M, G = M_novo, G_novo
        O_barra_anterior = O_barra
        O_barra = desajuste_medio(d_obs, G, C_D_inv)

        # Eqs. 40-41: regra por mediana entre os membros; alpha nunca cresce
        gamma_j = gamma * fator_lm(rho)
        gamma = float(np.median(gamma_j))
        alpha = float(min(alpha, np.median(gamma_j * O_barra)))

        res.desajuste.append(O_barra)
        res.alpha.append(alpha)
        res.gamma.append(gamma)
        res.rho_mediano.append(float(np.median(rho)))

        if O_barra < melhor_desajuste:
            melhor_desajuste = O_barra
            melhor_conjunto = M.copy()
            melhor_iter = i + 1

        # Criterios de parada (Secoes 4.3 e 4.5)
        if fator_ruido is not None:
            # Eq. 43: desajuste ja no nivel do ruido; continuar seria sobreajuste
            if desajuste_absoluto(d_obs, G, C_D_inv) < fator_ruido * d_obs.shape[0]:
                motivo = 'desajuste no nivel do ruido (Eq. 43)'
                break
        if O_barra_anterior > 0:
            variacao = abs(O_barra - O_barra_anterior) / O_barra_anterior
            if variacao < eta1:
                motivo = 'variacao do desajuste abaixo de eta1'
                break
        if var_parametros < eta2:
            motivo = 'variacao dos parametros abaixo de eta2'
            break

    # O artigo devolve o conjunto de menor desajuste medio entre as iteracoes
    res.conjunto = melhor_conjunto
    res.iteracao_final = melhor_iter
    res.motivo_parada = motivo
    res.n_avaliacoes = n_aval

    return res
