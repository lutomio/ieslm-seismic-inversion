#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo direto sismico elastico (AVO, multiplos angulos).

Relaciona os perfis de propriedades elasticas (Vp, Vs, rho) aos tracos
sismicos de incidencia proxima, media e distante, por meio da aproximacao
linearizada de Zoeppritz (Aki-Richards) convoluida com a wavelet.

Diferente do modelo acustico de forward.py, que estima apenas a impedancia a
partir do traco de incidencia normal, aqui as tres propriedades sao estimadas
simultaneamente a partir dos tres angulos - mesma configuracao adotada por
Caetano et al. na comparacao entre UH-CMA-ES e ES-MDA.

A modelagem em si e a da SeReMpy (SeismicModel), usada sem modificacao; este
modulo apenas empilha o vetor de parametros e propaga o conjunto inteiro.

Convencao do vetor de parametros: m = [Vp; Vs; rho], de tamanho 3*nm.
Convencao do vetor de dados:      d = [Near; Mid; Far], de tamanho 3*(nm-1).
"""

import numpy as np

import dados  # noqa: F401  (efeito colateral: poe a SeReMpy no sys.path)
from SeReMpy.Inversion import RickerWavelet, SeismicModel

ANGULOS = np.linspace(15, 45, 3)
NOMES_ANGULOS = ('Near', 'Mid', 'Far')
NOMES_PROPRIEDADES = ('Vp', 'Vs', 'rho')


def empilha(Vp, Vs, Rho):
    """
    EMPILHA
    Monta o vetor (ou conjunto) de parametros a partir das tres propriedades.

    Parameters
    ----------
    Vp, Vs, Rho : array_like
        Perfis (nm, 1) ou conjuntos (nm, ne).

    Returns
    -------
    array_like
        Parametros empilhados (3*nm, 1) ou (3*nm, ne).
    """
    return np.vstack([Vp, Vs, Rho])


def desempilha(M):
    """
    DESEMPILHA
    Separa o vetor de parametros nas tres propriedades.

    Parameters
    ----------
    M : array_like
        Parametros empilhados (3*nm, ne).

    Returns
    -------
    tuple of array_like
        (Vp, Vs, Rho), cada um (nm, ne).
    """
    nm = M.shape[0] // 3

    return M[:nm], M[nm:2 * nm], M[2 * nm:]


def monta_forward(Time, dt, freq=45, ntw=64, theta=None):
    """
    MONTA FORWARD
    Constroi a funcao g do modelo direto elastico.

    Parameters
    ----------
    Time : array_like
        Tempo do poco (nm, 1).
    dt : float
        Passo de amostragem (s).
    freq : int, optional
        Frequencia dominante da wavelet de Ricker (Hz).
    ntw : int, optional
        Numero de amostras da wavelet.
    theta : array_like, optional
        Angulos de incidencia (graus). Por omissao, 15, 30 e 45.

    Returns
    -------
    g : callable
        g(M) -> previsoes (3*(nm-1), ne), aceitando M de forma (3*nm, ne).
    wavelet : array_like
        A wavelet usada.
    """
    theta = ANGULOS if theta is None else np.asarray(theta)
    wavelet, _ = RickerWavelet(freq, dt, ntw)

    def g(M):
        M = np.atleast_2d(M)
        if M.shape[1] == 1 and M.ndim == 1:
            M = M.reshape(-1, 1)
        Vp, Vs, Rho = desempilha(M)
        ne = M.shape[1]
        saidas = []
        for j in range(ne):
            seis, _ = SeismicModel(Vp[:, j], Vs[:, j], Rho[:, j], Time, theta, wavelet)
            saidas.append(seis[:, 0])

        return np.column_stack(saidas)

    return g, wavelet


def separa_angulos(d):
    """
    SEPARA ANGULOS
    Divide um vetor de dados empilhado nos tres angulos.

    Parameters
    ----------
    d : array_like
        Dados (3*nd, ...) com os angulos empilhados na ordem Near, Mid, Far.

    Returns
    -------
    list of array_like
        Tres blocos, um por angulo.
    """
    nd = d.shape[0] // 3

    return [d[i * nd:(i + 1) * nd] for i in range(3)]


def adiciona_ruido(d_limpo, snr, rng):
    """
    ADICIONA RUIDO
    Contamina o dado com ruido gaussiano nao correlacionado, a partir de uma
    razao sinal-ruido escolhida.

    O conjunto de dados da SeReMpy e sintetico e livre de ruido: o traco
    observado foi gerado pelo proprio SeismicModel a partir do perfil de poco,
    de modo que o residuo do modelo verdadeiro e da ordem da precisao de
    maquina. Inverter esse dado com o mesmo operador que o gerou configuraria
    um "inverse crime", em que o metodo nao enfrenta erro algum e o desempenho
    fica superestimado. Adicionar ruido restabelece o carater mal-posto do
    problema; e o procedimento adotado por Caetano et al. sobre este mesmo
    conjunto de dados.

    Parameters
    ----------
    d_limpo : array_like
        Dado sem ruido (nd, 1).
    snr : float
        Razao entre o desvio padrao do sinal e o do ruido.
    rng : numpy.random.Generator
        Gerador aleatorio.

    Returns
    -------
    d_ruidoso : array_like
        Dado contaminado (nd, 1).
    C_D : array_like
        Covariancia do erro correspondente (nd, nd), coerente com o ruido
        efetivamente adicionado.
    """
    sigma = float(np.std(d_limpo)) / float(snr)
    ruido = sigma * rng.standard_normal(d_limpo.shape)

    return d_limpo + ruido, (sigma ** 2) * np.eye(d_limpo.shape[0])
