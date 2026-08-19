#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo direto (forward) sismico acustico 1D.

Relaciona um perfil de impedancia acustica Z(t) ao traco sismico de
incidencia normal, em duas etapas:

    1. refletividade (aproximacao de contraste fraco):
           R_k = 1/2 * [ln(Z_{k+1}) - ln(Z_k)]
    2. convolucao com a wavelet:
           d = W R

O operador e montado com as mesmas primitivas que a SeReMpy usa em
SeismicModel para o caso elastico de 3 variaveis (DifferentialMatrix e
WaveletMatrix), apenas com nv = 1 e ntheta = 1. A biblioteca nao e
modificada.

A incognita do problema inverso e Z (nao log Z). Como o log esta dentro
do operador, g e nao-linear em Z - e essa nao-linearidade que da sentido
a comparacao entre regularizacao fixa (ES-MDA) e adaptativa (iES-LM).

Referencias: Grana, Mukerji e Doyen (2021), Cap. 5.1 (modelo convolucional);
Maurya, Singh e Singh (2020) (inversao acustica).
"""

import numpy as np

import dados  # noqa: F401  (efeito colateral: poe a SeReMpy no sys.path)
from SeReMpy.Inversion import DifferentialMatrix, RickerWavelet, WaveletMatrix


def operador_acustico(nm, wavelet):
    """
    OPERADOR ACUSTICO
    Monta as matrizes do modelo direto acustico de incidencia normal.

    Parameters
    ----------
    nm : int
        Numero de amostras do perfil de impedancia.
    wavelet : array_like
        Wavelet (por exemplo, de RickerWavelet).

    Returns
    -------
    D : array_like
        Matriz de diferenciacao (nm-1, nm). D @ x = x[k+1] - x[k].
    W : array_like
        Matriz de convolucao com a wavelet (nm-1, nm-1).

    Raises
    ------
    ValueError
        Se a wavelet nao couber no perfil. WaveletMatrix da SeReMpy monta a
        Toeplitz assumindo wavelet mais curta que o numero de interfaces;
        fora disso ela falha com um erro de broadcast dificil de interpretar.
    """
    if len(wavelet) >= nm - 1:
        raise ValueError(
            'wavelet com %d amostras nao cabe em um perfil de %d amostras '
            '(precisa de len(wavelet) < nm-1 = %d)' % (len(wavelet), nm, nm - 1)
        )

    D = DifferentialMatrix(nm, 1)
    W = WaveletMatrix(wavelet, nm, 1)

    return D, W


def refletividade(Z, D):
    """
    REFLETIVIDADE
    Coeficientes de reflexao de incidencia normal (contraste fraco).

    Parameters
    ----------
    Z : array_like
        Impedancia acustica (nm, 1) ou conjunto (nm, ne). Deve ser positiva.
    D : array_like
        Matriz de diferenciacao de operador_acustico.

    Returns
    -------
    R : array_like
        Refletividade (nm-1, 1) ou (nm-1, ne).
    """
    if np.any(Z <= 0):
        raise ValueError('impedancia deve ser positiva (o modelo usa log(Z))')

    return 0.5 * np.dot(D, np.log(Z))


def modelo_direto(Z, D, W):
    """
    MODELO DIRETO
    Traco sismico sintetico a partir da impedancia acustica.

    Aceita um unico modelo (nm, 1) ou um conjunto inteiro (nm, ne); no
    segundo caso a conta e feita de uma vez so para todos os membros.

    Parameters
    ----------
    Z : array_like
        Impedancia acustica (nm, 1) ou (nm, ne).
    D, W : array_like
        Matrizes de operador_acustico.

    Returns
    -------
    d : array_like
        Sismica sintetica (nm-1, 1) ou (nm-1, ne).
    """
    return np.dot(W, refletividade(Z, D))


def tempo_sismico(Time):
    """
    TEMPO SISMICO
    Tempos das amostras sismicas: ponto medio entre amostras do poco.
    Mesma convencao de SeismicModel da SeReMpy.
    """
    return 0.5 * (Time[0:-1] + Time[1:])


def monta_forward(nm, dt, freq=45, ntw=64):
    """
    MONTA FORWARD
    Atalho que cria a wavelet, monta o operador e devolve g pronto para uso.

    Parameters
    ----------
    nm : int
        Numero de amostras do perfil de impedancia.
    dt : float
        Passo de amostragem em tempo (s).
    freq : int, optional
        Frequencia dominante da wavelet de Ricker (Hz).
    ntw : int, optional
        Numero de amostras da wavelet.

    Returns
    -------
    g : callable
        g(Z) -> sismica sintetica, aceitando (nm, 1) ou (nm, ne).
    wavelet : array_like
        A wavelet usada.
    """
    wavelet, _ = RickerWavelet(freq, dt, ntw)
    D, W = operador_acustico(nm, wavelet)

    def g(Z):
        return modelo_direto(Z, D, W)

    return g, wavelet
