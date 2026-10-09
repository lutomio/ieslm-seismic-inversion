#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuracao do experimento elastico: todas as entradas em um so lugar.

Antes desta classe, os numeros de entrada estavam espalhados em seis
arquivos, parte como constantes no topo e parte escondida dentro de funcoes
(o comprimento de correlacao do a priori, os limites fisicos, a frequencia
da wavelet, o gamma inicial). Mudar uma entrada exigia saber onde ela estava,
e nada registrava qual configuracao tinha produzido um resultado.

Agora cada execucao recebe uma Configuracao, e a configuracao usada e
gravada em JSON junto do resultado.

Uso:

    from dataclasses import replace
    import config

    cfg = replace(config.PADRAO, snr=20.0, ne=100)   # muda so o que precisa
    experimento_elastico.executa(cfg)

Os modulos de calculo (prior, forward_elastico, metricas, ieslm) continuam
recebendo numeros, e nao a Configuracao: e o experimento que le os campos e
os repassa. Assim aqueles modulos seguem utilizaveis isoladamente.
"""

import json
from dataclasses import asdict, dataclass, fields, replace  # noqa: F401
from typing import Optional

import referencia as ref


@dataclass(frozen=True)
class Configuracao:
    """
    CONFIGURACAO
    Entradas de uma execucao do experimento elastico.

    E imutavel (frozen): para variar um campo use dataclasses.replace, que
    devolve uma copia. Isso impede que uma execucao altere, sem querer, a
    configuracao de outra.

    Attributes
    ----------
    ne : int
        Tamanho do conjunto, N_e.
    semente : int
        Semente do conjunto a priori, do ruido e das perturbacoes.
    snr : float
        Razao entre o desvio padrao do sinal e o do ruido adicionado ao dado.
    freq_wavelet : float
        Frequencia dominante da wavelet de Ricker (Hz).
    amostras_wavelet : int
        Numero de amostras da wavelet.
    angulos : tuple of float
        Angulos de incidencia (graus).
    comprimento_correlacao : float
        Comprimento de correlacao vertical do a priori, em multiplos de dt.
    ordem_tendencia, corte_tendencia : int, float
        Filtro Butterworth que define a tendencia (media) do a priori.
    vies_prior : float
        Deslocamento relativo da tendencia do a priori: -0.05 a coloca 5%
        abaixo da tendencia verdadeira. Simula um conhecimento previo errado
        da baixa frequencia, que a sismica limitada em banda nao corrige.
        Zero reproduz o a priori sem vies.
    folga_inferior, folga_superior : float
        Limites fisicos: multiplicadores do minimo e do maximo de cada
        propriedade no modelo de referencia.
    n_assimilacoes : int
        Numero de assimilacoes do ES-MDA, N_a.
    razao_alpha : float
        Razao geometrica da sequencia decrescente de alpha do ES-MDA.
    gamma0 : float
        Valor inicial de gamma no iES-LM (Algoritmo 2 de Ma e Bi).
    max_iter : int
        Numero maximo de iteracoes do iES-LM.
    eta1, eta2 : float
        Tolerancias de parada do iES-LM (Secao 4.3 de Ma e Bi).
    fator_ruido : float or None
        Constante da parada por nivel de ruido (Eq. 43); None desliga.
    alvo : str
        Modelo de referencia, um dos nomes de referencia.ALVOS.
    """

    # conjunto e repeticao
    ne: int = 200
    semente: int = 42

    # dado
    snr: float = 10.0

    # modelo direto
    freq_wavelet: float = 45.0
    amostras_wavelet: int = 64
    angulos: tuple = (15.0, 30.0, 45.0)

    # a priori
    comprimento_correlacao: float = 5.0
    ordem_tendencia: int = 3
    corte_tendencia: float = 0.04
    vies_prior: float = 0.0

    # limites fisicos
    folga_inferior: float = 0.7
    folga_superior: float = 1.3

    # ES-MDA
    n_assimilacoes: int = 4
    razao_alpha: float = 2.0

    # iES-LM
    gamma0: float = 1.0
    max_iter: int = 10
    eta1: float = 1e-4
    eta2: float = 1e-2
    fator_ruido: Optional[float] = 4.0

    # modelo de referencia
    alvo: str = 'poço'

    def __post_init__(self):
        # Normaliza os angulos para tupla de floats: aceita lista, e garante
        # que a configuracao seja imutavel e comparavel por igualdade.
        object.__setattr__(self, 'angulos', tuple(float(a) for a in self.angulos))
        _valida(self)


def _valida(cfg):
    """
    VALIDA
    Rejeita valores sem sentido fisico ou numerico, com mensagem clara.

    Pega erros de digitacao ao mudar entradas antes que eles virem um
    resultado estranho, ou uma excecao obscura no meio da inversao.
    """
    erros = []

    def exige(condicao, mensagem):
        if not condicao:
            erros.append(mensagem)

    exige(cfg.ne >= 2, 'ne precisa ser ao menos 2 (a covariancia divide por ne-1)')
    exige(cfg.semente >= 0, 'semente precisa ser nao negativa')
    exige(cfg.snr > 0, 'snr precisa ser positiva')
    exige(cfg.freq_wavelet > 0, 'freq_wavelet precisa ser positiva')
    exige(cfg.amostras_wavelet >= 2, 'amostras_wavelet precisa ser ao menos 2')
    exige(len(cfg.angulos) >= 1, 'angulos precisa ter ao menos um angulo')
    exige(all(0 <= a < 90 for a in cfg.angulos), 'angulos precisam estar em [0, 90) graus')
    exige(cfg.comprimento_correlacao > 0, 'comprimento_correlacao precisa ser positivo')
    exige(cfg.ordem_tendencia >= 1, 'ordem_tendencia precisa ser ao menos 1')
    exige(0 < cfg.corte_tendencia < 1, 'corte_tendencia precisa estar entre 0 e 1')
    exige(-0.5 < cfg.vies_prior < 0.5, 'vies_prior precisa estar entre -0,5 e 0,5')
    exige(0 < cfg.folga_inferior <= 1, 'folga_inferior precisa estar em (0, 1]')
    exige(cfg.folga_superior >= 1, 'folga_superior precisa ser ao menos 1')
    exige(cfg.n_assimilacoes >= 1, 'n_assimilacoes precisa ser ao menos 1')
    exige(cfg.razao_alpha > 0, 'razao_alpha precisa ser positiva')
    exige(cfg.gamma0 > 0, 'gamma0 precisa ser positivo')
    exige(cfg.max_iter >= 1, 'max_iter precisa ser ao menos 1')
    exige(cfg.eta1 >= 0 and cfg.eta2 >= 0, 'eta1 e eta2 nao podem ser negativos')
    exige(cfg.fator_ruido is None or cfg.fator_ruido > 0,
          'fator_ruido precisa ser positivo, ou None para desligar')
    exige(cfg.alvo in ref.ALVOS,
          'alvo desconhecido: %r (opcoes: %s)' % (cfg.alvo, ', '.join(ref.ALVOS)))

    if erros:
        raise ValueError('configuracao invalida:\n  - ' + '\n  - '.join(erros))


PADRAO = Configuracao()


def para_dict(cfg):
    """Configuracao como dicionario simples, pronto para JSON."""
    d = asdict(cfg)
    d['angulos'] = list(cfg.angulos)
    return d


def de_dict(d):
    """
    DE DICT
    Reconstroi uma Configuracao a partir de um dicionario.

    Campos ausentes assumem o valor padrao, para que JSONs gravados antes de
    um campo novo existir continuem legiveis. Campos desconhecidos sao
    rejeitados, para nao ignorar silenciosamente um erro de digitacao.
    """
    conhecidos = {f.name for f in fields(Configuracao)}
    desconhecidos = set(d) - conhecidos
    if desconhecidos:
        raise ValueError('campos desconhecidos: %s' % ', '.join(sorted(desconhecidos)))

    return Configuracao(**d)


def salva(cfg, caminho, **extra):
    """
    SALVA
    Grava a configuracao em JSON.

    Parameters
    ----------
    cfg : Configuracao
    caminho : str
    **extra
        Informacao adicional sobre a execucao (por exemplo, as grades do
        estudo de sensibilidade), gravada em uma chave separada.
    """
    conteudo = {'configuracao': para_dict(cfg)}
    if extra:
        conteudo['extra'] = extra
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(conteudo, f, ensure_ascii=False, indent=2)


def carrega(caminho):
    """Le uma configuracao gravada por salva."""
    with open(caminho, encoding='utf-8') as f:
        return de_dict(json.load(f)['configuracao'])
