#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acesso aos dados e a biblioteca SeReMpy.

Concentra num so lugar (a) a insercao da SeReMpy no sys.path e (b) a leitura
dos arquivos de poco e de sismica, para que os demais modulos do TCC nao
precisem lidar com caminhos relativos.

Dados: Grana, Mukerji e Doyen (2021), SeReMpy - Data/data5log.dat e
Data/data5seis.dat (mesmos dados usados pelo ESPetroInversionDriver.py).
"""

import os
import sys

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))

# Dados: preferencia para a copia local (redistribuida sob MIT, ver data/README.md);
# se ausente, procura a instalacao da SeReMpy ao lado do projeto.
_DATA_LOCAL = os.path.join(_AQUI, 'data')
DATA_DIR = _DATA_LOCAL if os.path.isdir(_DATA_LOCAL) else None


def _localiza_serempy():
    """
    LOCALIZA SEREMPY
    Encontra a raiz da biblioteca SeReMpy, necessaria para as primitivas do
    modelo direto (DifferentialMatrix, WaveletMatrix, RickerWavelet) e para o
    ES-MDA de referencia (EnsembleSmootherMDA).

    Ordem de busca: variavel de ambiente SEREMPY_PATH, depois os locais usuais
    ao lado deste projeto.

    Returns
    -------
    str
        Caminho da raiz da SeReMpy.

    Raises
    ------
    RuntimeError
        Se a biblioteca nao for encontrada, com instrucoes de instalacao.
    """
    candidatos = []
    if os.environ.get('SEREMPY_PATH'):
        candidatos.append(os.environ['SEREMPY_PATH'])
    candidatos += [
        os.path.join(_AQUI, '..', 'SeReMpy-main'),
        os.path.join(_AQUI, 'SeReMpy-main'),
        os.path.join(_AQUI, '..', 'SeReMpy'),
    ]

    for c in candidatos:
        if os.path.isdir(os.path.join(os.path.abspath(c), 'SeReMpy')):
            return os.path.abspath(c)

    raise RuntimeError(
        'biblioteca SeReMpy nao encontrada.\n'
        'Baixe-a de https://github.com/dariograna/SeReMpy e coloque a pasta\n'
        'SeReMpy-main ao lado deste projeto, ou aponte a variavel de ambiente\n'
        'SEREMPY_PATH para a raiz da biblioteca.'
    )


SEREMPY_ROOT = _localiza_serempy()
if DATA_DIR is None:
    DATA_DIR = os.path.join(SEREMPY_ROOT, 'Data')

# Mesma ideia do Examples/context.py da SeReMpy: torna 'SeReMpy' importavel
if SEREMPY_ROOT not in sys.path:
    sys.path.insert(0, SEREMPY_ROOT)


def carrega_dados():
    """
    CARREGA DADOS
    Le o poco e a sismica usados no experimento do TCC 2.

    Returns
    -------
    dict com as chaves:
        Time : array_like
            Tempo do poco (nm, 1).
        TimeSeis : array_like
            Tempo da sismica (nd, 1).
        Snear : array_like
            Traco sismico de incidencia proxima (nd, 1) - o dado observado.
        Vp : array_like
            Velocidade da onda P (nm, 1).
        Rho : array_like
            Densidade (nm, 1).
        Z : array_like
            Impedancia acustica de referencia, Z = Vp * Rho (nm, 1).
        dt : float
            Passo de amostragem em tempo (s).
    """
    ds = np.loadtxt(os.path.join(DATA_DIR, 'data5seis.dat'))
    dl = np.loadtxt(os.path.join(DATA_DIR, 'data5log.dat'))

    TimeSeis = ds[:, 0].reshape(-1, 1)
    Snear = ds[:, 1].reshape(-1, 1)

    Time = dl[:, 3].reshape(-1, 1)
    Vp = dl[:, 4].reshape(-1, 1)
    Rho = dl[:, 6].reshape(-1, 1)

    return {
        'Time': Time,
        'TimeSeis': TimeSeis,
        'Snear': Snear,
        'Vp': Vp,
        'Rho': Rho,
        'Z': Vp * Rho,
        'dt': float(TimeSeis[1, 0] - TimeSeis[0, 0]),
    }
