#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metricas de comparacao entre metodos de inversao.

Os tres eixos seguem a avaliacao adotada por Caetano et al. na comparacao
entre UH-CMA-ES e ES-MDA sobre este mesmo conjunto de dados: adesao aos dados
e ao modelo, qualidade da incerteza a posteriori, e custo computacional
medido em chamadas ao operador direto.
"""

import numpy as np
from scipy import stats


def rmse(conjunto, verdadeiro):
    """
    RMSE
    Erro quadratico medio entre a media do conjunto e o modelo de referencia.

    Parameters
    ----------
    conjunto : array_like
        Conjunto de modelos (nm, ne).
    verdadeiro : array_like
        Modelo de referencia (nm, 1).

    Returns
    -------
    float
    """
    return float(np.sqrt(np.mean((conjunto.mean(axis=1, keepdims=True) - verdadeiro) ** 2)))


def envelope(conjunto, inferior=10, superior=90):
    """
    ENVELOPE
    Percentis inferior e superior do conjunto, amostra a amostra.

    Returns
    -------
    tuple of array_like
        (P_inferior, P_superior), cada um (nm,).
    """
    return (np.percentile(conjunto, inferior, axis=1),
            np.percentile(conjunto, superior, axis=1))


def largura_envelope(conjunto, inferior=10, superior=90):
    """
    LARGURA ENVELOPE
    Largura media do envelope de incerteza (espalhamento do conjunto).

    Returns
    -------
    float
    """
    p_inf, p_sup = envelope(conjunto, inferior, superior)  # percentis por amostra

    return float(np.mean(p_sup - p_inf))


def taxa_cobertura(conjunto, verdadeiro, inferior=10, superior=90):
    """
    TAXA DE COBERTURA
    Fracao das amostras em que o modelo de referencia cai dentro do envelope.

    Complementa a largura do envelope: um conjunto pode ser estreito por ter
    convergido bem ou por ter colapsado, e so a cobertura distingue os dois
    casos. Para um envelope P10-P90 bem calibrado, o valor esperado e 0,8;
    valores muito abaixo indicam excesso de confianca.

    Parameters
    ----------
    conjunto : array_like
        Conjunto de modelos (nm, ne).
    verdadeiro : array_like
        Modelo de referencia (nm, 1).
    inferior, superior : float, optional
        Percentis do envelope.

    Returns
    -------
    float
        Fracao entre 0 e 1.
    """
    p_inf, p_sup = envelope(conjunto, inferior, superior)  # limites do envelope
    v = np.asarray(verdadeiro).ravel()  # referencia, achatada para comparar

    return float(np.mean((v >= p_inf) & (v <= p_sup)))


def teste_ks(conjunto, verdadeiro):
    """
    TESTE KS
    Teste de Kolmogorov-Smirnov de duas amostras entre os valores do conjunto
    a posteriori e os do modelo de referencia.

    Mede a maxima distancia absoluta entre as funcoes de distribuicao
    acumulada. Diferente do RMSE, que compara ponto a ponto, avalia se o
    metodo reproduz a DISTRIBUICAO de valores da propriedade. Um p-valor alto
    indica que nao ha evidencia para distinguir as duas distribuicoes.

    Parameters
    ----------
    conjunto : array_like
        Conjunto de modelos (nm, ne).
    verdadeiro : array_like
        Modelo de referencia (nm, 1).

    Returns
    -------
    tuple of float
        (estatistica, p_valor).
    """
    r = stats.ks_2samp(  # devolve estatistica e p-valor
        np.asarray(conjunto).ravel(), np.asarray(verdadeiro).ravel())

    return float(r.statistic), float(r.pvalue)


def sequencia_alpha_esmda(n_assimilacoes, razao=2.0):
    """
    SEQUENCIA ALPHA ESMDA
    Sequencia decrescente de fatores de inflacao para o ES-MDA, satisfazendo
    a condicao de consistencia sum(1/alpha_l) = 1.

    A escolha de fatores maiores nas primeiras assimilacoes amortece a
    nao-linearidade nos passos iniciais, quando a linearizacao e menos
    confiavel - alternativa ao valor constante alpha_l = N_a e adotada por
    Emerick e Reynolds (2013) e por Caetano et al.

    Parameters
    ----------
    n_assimilacoes : int
        Numero de assimilacoes.
    razao : float, optional
        Razao geometrica entre fatores consecutivos.

    Returns
    -------
    array_like
        Vetor com os fatores, em ordem decrescente.
    """
    # pesos decrescentes; o fator seguinte os normaliza para sum(1/alpha) = 1
    pesos = razao ** np.arange(n_assimilacoes - 1, -1, -1, dtype=float)

    return pesos * float(np.sum(1.0 / pesos))
