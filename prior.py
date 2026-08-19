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
    b, a = signal.butter(ordem, corte)

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
    t = np.arange(nm) * dt
    distancia = np.abs(t.reshape(-1, 1) - t.reshape(1, -1))

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

    C = covariancia_espacial(nm, dt, comprimento_correlacao)
    # Jitter: a covariancia gaussiana e mal condicionada e a Cholesky falha
    # sem uma pequena regularizacao na diagonal.
    L = np.linalg.cholesky(C + 1e-8 * np.eye(nm))

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
