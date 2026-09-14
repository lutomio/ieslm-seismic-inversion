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


# Os arquivos .dat nao tem cabecalho: a posicao das colunas e a unica fonte de
# verdade sobre o que cada numero significa. Ficam declaradas aqui, e so aqui.
COLUNAS_POCO = {'Time': 3, 'Vp': 4, 'Vs': 5, 'Rho': 6}
COLUNAS_SISMICA = {'TimeSeis': 0, 'Snear': 1, 'Smid': 2, 'Sfar': 3}


def carrega_dados():
    """
    CARREGA DADOS
    Ponto unico de leitura dos arquivos de poco e de sismica.

    Todos os modulos obtem os dados por aqui, para que os indices de coluna
    nao se repitam pelo codigo: um indice trocado em um so lugar (por exemplo,
    Vs no lugar de rho) passaria despercebido e contaminaria o experimento.

    Os tracos sismicos sao devolvidos, mas o experimento elastico NAO os usa
    como observacao: ele gera o dado a partir do perfil de poco com ruido de
    SNR controlada (ver forward_elastico.adiciona_ruido). O experimento
    acustico, ao contrario, usa Snear diretamente.

    Returns
    -------
    dict com as chaves:
        Time : array_like
            Tempo do poco (nm, 1).
        Vp, Vs, Rho : array_like
            Velocidade da onda P (km/s), da onda S (km/s) e densidade
            (g/cm3), cada uma (nm, 1). Formam o modelo de referencia.
        Z : array_like
            Impedancia acustica de referencia, Z = Vp * Rho (nm, 1).
        TimeSeis : array_like
            Tempo da sismica (nd, 1): ponto medio entre amostras do poco.
        Snear, Smid, Sfar : array_like
            Tracos de incidencia proxima, media e distante (15, 30 e 45
            graus), cada um (nd, 1).
        dt : float
            Passo de amostragem em tempo (s), tomado do tempo do poco, que e
            a malha em que o modelo direto opera.
    """
    dl = np.loadtxt(os.path.join(DATA_DIR, 'data5log.dat'))
    ds = np.loadtxt(os.path.join(DATA_DIR, 'data5seis.dat'))

    d = {nome: dl[:, col].reshape(-1, 1) for nome, col in COLUNAS_POCO.items()}
    d.update({nome: ds[:, col].reshape(-1, 1) for nome, col in COLUNAS_SISMICA.items()})
    d['Z'] = d['Vp'] * d['Rho']
    d['dt'] = float(d['Time'][1, 0] - d['Time'][0, 0])

    return d
