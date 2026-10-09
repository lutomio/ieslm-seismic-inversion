#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelos de referencia sinteticos.

Gera o "modelo verdadeiro" do experimento por codigo, como alternativa ao
perfil de poco distribuido com a SeReMpy. Serve para variar a natureza do
alvo sem editar os arquivos de dados: o poco original permanece intacto como
caso de base, e os testes que verificam a coerencia entre data5log.dat e
data5seis.dat continuam validos.

Isso e possivel porque o experimento elastico NAO usa os tracos do pacote:
ele gera a observacao a partir do modelo de referencia, qualquer que seja a
origem dele (ver forward_elastico.adiciona_ruido). Trocar a referencia mantem
dado e alvo consistentes entre si.

A motivacao e experimental: a aspereza do perfil verdadeiro afeta diretamente
a qualidade da estimativa e a calibracao da incerteza, porque a sismica e
limitada em banda e nao resolve camadas finas. Poder controlar essa aspereza
por parametro transforma-a em mais um eixo do estudo de sensibilidade, ao
lado do tamanho do conjunto e do nivel de ruido.

As constantes abaixo foram ajustadas ao proprio conjunto de dados da SeReMpy,
para que os modelos sinteticos fiquem na mesma escala fisica do poco real.
"""

import numpy as np
from scipy import signal

# Razao Vp/Vs media do poco de referencia (varia entre 1,53 e 1,64).
RAZAO_VP_VS = 1.570

# Relacao de Gardner rho = a * Vp^(1/4), com Vp em km/s. O coeficiente foi
# ajustado ao poco de referencia e reproduz a densidade com erro medio de 2,5%.
GARDNER_A = 1.610

# Velocidade compressional media (km/s) e contraste relativo do poco, medidos
# no mesmo conjunto: desvio padrao de Vp dividido pela media.
VP_MEDIO = 4.05
CONTRASTE = 0.06


def _monta(Time, Vp, dt, razao_vp_vs, gardner_a):
    """
    MONTA
    Completa Vs e a densidade a partir de Vp e devolve o modelo no mesmo
    formato de dados.carrega_dados, para que os dois possam ser usados de
    forma intercambiavel pelo experimento.

    Vs vem da razao Vp/Vs e a densidade da relacao de Gardner, ambas
    calibradas ao poco de referencia.
    """
    Vp = np.asarray(Vp, dtype=float).reshape(-1, 1)
    Vs = Vp / razao_vp_vs
    Rho = gardner_a * Vp ** 0.25

    return {
        'Time': Time, 'Vp': Vp, 'Vs': Vs, 'Rho': Rho,
        'Z': Vp * Rho, 'dt': float(dt),
    }


def modelo_em_camadas(nm=99, dt=0.001, t0=1.8, n_camadas=6, contraste=CONTRASTE,
                      vp_medio=VP_MEDIO, razao_vp_vs=RAZAO_VP_VS,
                      gardner_a=GARDNER_A, espessura_minima=4, rng=None):
    """
    MODELO EM CAMADAS
    Perfil constante por trechos, com contrastes abruptos nas interfaces.

    E o oposto do poco real, que varia amostra a amostra: aqui a variacao se
    concentra em poucas interfaces, o que aproxima o alvo do que a sismica
    consegue de fato resolver.

    Parameters
    ----------
    nm : int, optional
        Numero de amostras do perfil.
    dt : float, optional
        Passo de amostragem em tempo (s).
    t0 : float, optional
        Tempo da primeira amostra (s).
    n_camadas : int, optional
        Numero de camadas constantes.
    contraste : float, optional
        Desvio padrao relativo de Vp entre camadas. O padrao, 0,06,
        corresponde ao contraste medido no poco de referencia.
    vp_medio : float, optional
        Velocidade compressional media (km/s).
    razao_vp_vs, gardner_a : float, optional
        Constantes que derivam Vs e a densidade a partir de Vp.
    espessura_minima : int, optional
        Numero minimo de amostras por camada, para que nenhuma fique fina
        demais para a wavelet.
    rng : numpy.random.Generator, optional
        Gerador aleatorio.

    Returns
    -------
    dict
        Mesmas chaves de dados.carrega_dados, menos os tracos sismicos:
        Time, Vp, Vs, Rho, Z e dt.

    Raises
    ------
    ValueError
        Se as camadas pedidas nao couberem no perfil com a espessura minima.
    """
    rng = np.random.default_rng() if rng is None else rng

    if n_camadas * espessura_minima > nm:
        raise ValueError(
            '%d camadas de no minimo %d amostras nao cabem em %d amostras'
            % (n_camadas, espessura_minima, nm)
        )

    # Sorteia as interfaces respeitando a espessura minima: distribui o
    # excedente de amostras entre as camadas.
    excedente = nm - n_camadas * espessura_minima
    cortes = np.sort(rng.integers(0, excedente + 1, size=n_camadas - 1))
    bordas = np.concatenate([[0], cortes + espessura_minima * np.arange(1, n_camadas), [nm]])

    # uniform(-1, 1) tem desvio 1/sqrt(3); o fator sqrt(3) faz o desvio de Vp
    # entre camadas ser exatamente `contraste` vezes a media.
    Vp = np.empty(nm)
    for k in range(n_camadas):
        desvio = contraste * np.sqrt(3.0) * rng.uniform(-1.0, 1.0)
        Vp[bordas[k]:bordas[k + 1]] = vp_medio * (1.0 + desvio)

    Time = (t0 + dt * np.arange(nm)).reshape(-1, 1)

    return _monta(Time, Vp, dt, razao_vp_vs, gardner_a)


def suaviza(modelo, corte=0.15, ordem=3):
    """
    SUAVIZA
    Versao suavizada de um modelo de referencia, com contrastes mais fracos.

    Aplica um passa-baixa a Vp e recalcula Vs e a densidade, preservando a
    coerencia fisica entre as tres propriedades. Util para investigar como a
    aspereza do alvo afeta a estimativa: perfis mais suaves estao dentro da
    banda que a sismica resolve e sao recuperados com erro menor.

    Parameters
    ----------
    modelo : dict
        Modelo de referencia, como o devolvido por modelo_em_camadas ou por
        dados.carrega_dados.
    corte : float, optional
        Frequencia de corte normalizada. Valores menores suavizam mais.
    ordem : int, optional
        Ordem do filtro Butterworth.

    Returns
    -------
    dict
        Modelo suavizado, com as mesmas chaves.
    """
    b, a = signal.butter(ordem, corte)  # coeficientes do filtro
    Vp = signal.filtfilt(b, a, np.squeeze(modelo['Vp']))

    razao = float(np.mean(modelo['Vp'] / modelo['Vs']))
    gardner = float(np.mean(modelo['Rho'] / modelo['Vp'] ** 0.25))

    return _monta(modelo['Time'], Vp, modelo['dt'], razao, gardner)


# Nomes aceitos pelo campo `alvo` da Configuracao. 'poço' e o perfil do
# pacote; os demais sao sinteticos e mudam com a semente.
ALVOS = ('poço', '6 camadas', '12 camadas', 'suavizado')


def alvo_por_nome(nome, semente):
    """
    ALVO POR NOME
    Modelo de referencia correspondente a um nome de ALVOS.

    Parameters
    ----------
    nome : str
        Um dos nomes de ALVOS.
    semente : int
        Semente da execucao. Os alvos sinteticos usam 1000 + semente, para
        que o sorteio do alvo seja independente do sorteio do a priori.

    Returns
    -------
    dict or None
        O modelo, no formato de dados.carrega_dados; None para 'poço', que
        sinaliza ao experimento que use o perfil do pacote.
    """
    if nome == 'poço':
        return None

    rng = np.random.default_rng(1000 + semente)
    if nome == '6 camadas':
        return modelo_em_camadas(n_camadas=6, rng=rng)
    if nome == '12 camadas':
        return modelo_em_camadas(n_camadas=12, rng=rng)
    if nome == 'suavizado':
        return suaviza(modelo_em_camadas(n_camadas=6, rng=rng))

    raise ValueError('alvo desconhecido: %s' % nome)
