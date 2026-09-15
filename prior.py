#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conjunto a priori de impedancia acustica.

Gera as realizacoes iniciais que alimentam tanto o iES-LM quanto o ES-MDA no
experimento comparativo. A construcao segue a mesma ideia dos drivers da
SeReMpy (tendencia suave do poco + covariancia espacial gaussiana), com duas
diferencas deliberadas:

1. As realizacoes sao geradas em log(Z) e exponenciadas. Isso garante Z > 0
   em todos os membros, o que o modelo direto exige, e e coerente com o
   carater multiplicativo da impedancia.
2. Usa-se numpy.random.Generator em vez do gerador global legado que a
   CorrelatedSimulation da SeReMpy emprega, para que o experimento inteiro
   seja reprodutivel a partir de uma semente.
"""

import numpy as np
from scipy import signal


def tendencia_suave(Z, ordem=3, corte=0.04):
    """
    TENDENCIA SUAVE
    Tendencia de baixa frequencia do perfil, usada como media do a priori.

    Filtra o perfil verdadeiro com um Butterworth passa-baixa, como fazem os
    drivers da SeReMpy: o conjunto a priori conhece a tendencia regional mas
    nao os detalhes finos, que sao justamente o que a inversao deve recuperar.

    Parameters
    ----------
    Z : array_like
        Perfil de impedancia (nm, 1).
    ordem : int, optional
        Ordem do filtro.
    corte : float, optional
        Frequencia de corte normalizada.

    Returns
    -------
    array_like
        Tendencia (nm, 1).
    """
    b, a = signal.butter(ordem, corte)  # coeficientes do filtro Butterworth

    return signal.filtfilt(b, a, np.squeeze(Z)).reshape(-1, 1)


def covariancia_espacial(nm, dt, comprimento_correlacao):
    """
    COVARIANCIA ESPACIAL
    Matriz de correlacao gaussiana entre amostras, exp(-(h/L)^2).

    Parameters
    ----------
    nm : int
        Numero de amostras.
    dt : float
        Passo de amostragem em tempo (s).
    comprimento_correlacao : float
        Comprimento de correlacao (s).

    Returns
    -------
    array_like
        Matriz de correlacao (nm, nm).
    """
    t = np.arange(nm) * dt  # instante de cada amostra
    distancia = np.abs(t.reshape(-1, 1) - t.reshape(1, -1))  # entre cada par

    return np.exp(-((distancia / comprimento_correlacao) ** 2))


def conjunto_prior(tendencia, ne, dt, desvio_log=0.05, comprimento_correlacao=None,
                   rng=None):
    """
    CONJUNTO PRIOR
    Realizacoes correlacionadas de impedancia acustica em torno da tendencia.

    Parameters
    ----------
    tendencia : array_like
        Media do a priori (nm, 1), tipicamente de tendencia_suave.
    ne : int
        Numero de realizacoes.
    dt : float
        Passo de amostragem em tempo (s).
    desvio_log : float, optional
        Desvio padrao em log(Z); 0.05 corresponde a cerca de 5% de variacao
        relativa na impedancia.
    comprimento_correlacao : float, optional
        Comprimento de correlacao vertical (s). Por omissao, 5*dt, mesmo
        valor usado nos drivers da SeReMpy.
    rng : numpy.random.Generator, optional
        Gerador aleatorio.

    Returns
    -------
    array_like
        Conjunto a priori (nm, ne), estritamente positivo.
    """
    rng = np.random.default_rng() if rng is None else rng
    nm = tendencia.shape[0]

    if comprimento_correlacao is None:
        comprimento_correlacao = 5 * dt

    C = covariancia_espacial(nm, dt, comprimento_correlacao)  # (nm, nm)
    # Jitter: a covariancia gaussiana e mal condicionada e a Cholesky falha
    # sem uma pequena regularizacao na diagonal.
    L = np.linalg.cholesky(C + 1e-8 * np.eye(nm))  # L @ z: ruido correlacionado

    perturbacao = desvio_log * (L @ rng.standard_normal((nm, ne)))

    return np.exp(np.log(tendencia) + perturbacao)


def limites_fisicos(Z, folga_inferior=0.5, folga_superior=1.5):
    """
    LIMITES FISICOS
    Faixa de truncamento para os modelos, derivada do perfil de referencia.

    O artigo trunca os parametros aos limites quando saem da faixa admissivel
    (Secao 4.3). Aqui a faixa e uma folga em torno do perfil verdadeiro.

    Parameters
    ----------
    Z : array_like
        Perfil de referencia.
    folga_inferior, folga_superior : float, optional
        Multiplicadores do minimo e do maximo observados.

    Returns
    -------
    tuple of float
        (minimo, maximo).
    """
    return float(np.min(Z) * folga_inferior), float(np.max(Z) * folga_superior)


def conjunto_prior_multivariado(tendencias, ne, dt, sigma0, comprimento_correlacao=None,
                                rng=None):
    """
    CONJUNTO PRIOR MULTIVARIADO
    Realizacoes correlacionadas de varias propriedades simultaneamente.

    Generaliza conjunto_prior para o caso elastico, em que o vetor de
    parametros reune Vp, Vs e rho. A estrutura de covariancia e o produto de
    Kronecker entre a covariancia estacionaria ENTRE propriedades (sigma0, que
    preserva a correlacao fisica entre elas) e a covariancia espacial ao longo
    do tempo - mesma construcao usada pela CorrelatedSimulation da SeReMpy,
    porem com numpy.random.Generator, para que o experimento seja reprodutivel
    a partir de uma semente.

    Parameters
    ----------
    tendencias : array_like
        Media a priori de cada propriedade (nm, nv).
    ne : int
        Numero de realizacoes.
    dt : float
        Passo de amostragem em tempo (s).
    sigma0 : array_like
        Covariancia estacionaria entre as propriedades (nv, nv).
    comprimento_correlacao : float, optional
        Comprimento de correlacao vertical (s). Por omissao, 5*dt.
    rng : numpy.random.Generator, optional
        Gerador aleatorio.

    Returns
    -------
    array_like
        Conjunto a priori empilhado (nm*nv, ne), na ordem das colunas de
        `tendencias`.
    """
    rng = np.random.default_rng() if rng is None else rng
    nm, nv = tendencias.shape  # amostras no tempo, numero de propriedades

    if comprimento_correlacao is None:
        comprimento_correlacao = 5 * dt

    C_tempo = covariancia_espacial(nm, dt, comprimento_correlacao)
    L_tempo = np.linalg.cholesky(C_tempo + 1e-8 * np.eye(nm))  # fator no tempo
    # e o fator entre propriedades, que preserva a correlacao fisica
    L_prop = np.linalg.cholesky(np.atleast_2d(sigma0) + 1e-12 * np.eye(nv))

    # z ~ N(0, I) de forma (nv, nm, ne); aplica L_tempo no eixo do tempo e
    # L_prop no eixo das propriedades, o que equivale a kron(sigma0, C_tempo).
    z = rng.standard_normal((nv, nm, ne))
    z = np.einsum('ij,jkl->ikl', L_prop, z)
    z = np.einsum('km,imn->ikn', L_tempo, z)

    blocos = [tendencias[:, [i]] + z[i] for i in range(nv)]  # uma propriedade por bloco

    return np.vstack(blocos)
