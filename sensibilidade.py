#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo de sensibilidade: iES-LM x ES-MDA sob condicoes variadas.

Atende ao objetivo especifico de analisar a sensibilidade do metodo ao
tamanho do conjunto e ao nivel de ruido, e acrescenta um terceiro eixo, a
aspereza do modelo de referencia, que se mostrou determinante para a
calibracao da incerteza.

O ponto central e estatistico. Uma unica execucao nao permite afirmar que um
metodo e melhor: a diferenca observada pode vir do sorteio do conjunto a
priori e do ruido. Aqui cada configuracao e repetida com varias sementes, e
os dois metodos recebem em cada repeticao EXATAMENTE o mesmo cenario. Isso
torna a comparacao pareada: a diferenca e medida semente a semente, o que
elimina a variabilidade comum e permite um teste de significancia.

Protocolo identico ao de experimento_elastico.py, cujas funcoes sao
reaproveitadas para que nao existam duas implementacoes do mesmo experimento.

Uso:
    python sensibilidade.py            # grade completa
    python sensibilidade.py --rapido   # grade reduzida, para conferir
"""

import os
import sys
import functools
import time
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import config
import experimento_elastico as X
import forward_elastico as fe
import metricas as mt
import referencia as ref

PASTA_FIGURAS = X.PASTA_FIGURAS
PASTA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

# Eixos do estudo. Cada um varia um fator com os demais fixos nos valores de
# referencia (NE = 200, SNR = 10, alvo = poco do pacote).
# Ponto de referencia: os eixos variam a partir dele. Todas as demais
# entradas (gamma inicial, numero de assimilacoes, ...) vem dele tambem.
REF = config.PADRAO

TAMANHOS = (25, 50, 100, 200, 400)
RUIDOS = (2.0, 5.0, 10.0, 20.0, 50.0)
ALVOS = ref.ALVOS

# Numero de assimilacoes do ES-MDA. O custo do ES-MDA e N_a + 1 avaliacoes do
# modelo direto, e N_a = 4 e uma escolha arbitraria: variar N_a permite
# comparar os dois metodos com o MESMO custo, em vez de atribuir ao iES-LM
# uma vantagem que viria so da configuracao escolhida para o ES-MDA.
ASSIMILACOES = (1, 2, 3, 4, 6)
NA_REF = REF.n_assimilacoes

N_SEMENTES = 20
COR_MDA, COR_LM = X.COR_MDA, X.COR_LM


def _metricas(M, verd, sufixo):
    """RMSE, cobertura e largura do envelope de cada propriedade."""
    prop = fe.desempilha(M)
    linha = {}
    for i, nome in enumerate(fe.NOMES_PROPRIEDADES):
        linha['rmse_%s_%s' % (nome, sufixo)] = mt.rmse(prop[i], verd[i])
        linha['cob_%s_%s' % (nome, sufixo)] = mt.taxa_cobertura(prop[i], verd[i])
        linha['env_%s_%s' % (nome, sufixo)] = mt.largura_envelope(prop[i])
    return linha


@functools.lru_cache(maxsize=None)
def _resultado_ieslm(cfg):
    """
    RESULTADO IESLM
    Metricas do iES-LM para uma configuracao, calculadas uma vez so.

    O iES-LM nao depende do numero de assimilacoes do ES-MDA: para a mesma
    semente, o resultado e identico em todo o eixo de N_a. A chamada recebe a
    configuracao com N_a normalizado, e o cache evita repetir a inversao.
    """
    c = X.carrega_cenario(cfg)
    res = X.roda_ieslm(c)
    linha = {'aval_lm': res.n_avaliacoes,
             'desajuste_lm': min(res.desajuste),
             'iter_lm': len(res.desajuste) - 1}
    linha.update(_metricas(res.conjunto, fe.desempilha(c['verdadeiro']), 'lm'))
    return linha


def avalia(ne, snr, alvo, semente, na=NA_REF):
    """
    AVALIA
    Roda os dois metodos sobre o mesmo cenario e devolve as metricas.

    Parameters
    ----------
    ne : int
        Tamanho do conjunto.
    snr : float
        Razao sinal-ruido.
    alvo : str
        Nome do modelo de referencia (ver ALVOS).
    semente : int
        Semente do cenario e dos dois metodos.
    na : int, optional
        Numero de assimilacoes do ES-MDA.

    Returns
    -------
    dict
        Uma linha de resultado, com as metricas de ambos os metodos.
    """
    cfg = replace(REF, ne=ne, snr=snr, alvo=alvo, semente=semente, n_assimilacoes=na)
    c = X.carrega_cenario(cfg)

    M_mda, hist_mda, aval_mda, _ = X.roda_esmda(c)
    verd = fe.desempilha(c['verdadeiro'])

    linha = {'ne': ne, 'snr': snr, 'alvo': alvo, 'na': na, 'semente': semente,
             'aspereza': float(np.std(np.diff(verd[0], axis=0))),
             'aval_mda': aval_mda, 'desajuste_mda': hist_mda[-1]}
    linha.update(_metricas(M_mda, verd, 'mda'))
    linha.update(_resultado_ieslm(replace(cfg, n_assimilacoes=NA_REF)))

    return linha


def grade(rapido=False):
    """
    GRADE
    Configuracoes a executar: um eixo por vez, os demais no valor de
    referencia. A configuracao central e compartilhada pelos quatro eixos e
    roda uma vez so.
    """
    tamanhos = (50, 200) if rapido else TAMANHOS
    ruidos = (5.0, 10.0) if rapido else RUIDOS
    alvos = ('poço', '6 camadas') if rapido else ALVOS
    assimilacoes = (2, NA_REF) if rapido else ASSIMILACOES
    n_sem = 4 if rapido else N_SEMENTES

    # Cada configuracao e (ne, snr, alvo, na)
    configs = set()
    for ne in tamanhos:
        configs.add((ne, REF.snr, 'poço', NA_REF))
    for snr in ruidos:
        configs.add((REF.ne, snr, 'poço', NA_REF))
    for alvo in alvos:
        configs.add((REF.ne, REF.snr, alvo, NA_REF))
    for na in assimilacoes:
        configs.add((REF.ne, REF.snr, 'poço', na))

    return sorted(configs), n_sem


def executa(rapido=False):
    configs, n_sem = grade(rapido)
    total = len(configs) * n_sem
    print('Estudo de sensibilidade: %d configuracoes x %d sementes = %d execucoes'
          % (len(configs), n_sem, total))

    linhas = []
    t0 = time.time()
    for k, (ne, snr, alvo, na) in enumerate(configs, 1):
        for semente in range(n_sem):
            linhas.append(avalia(ne, snr, alvo, semente, na))
        print('  [%2d/%2d] ne=%3d snr=%4.1f alvo=%-11s na=%d  (%.0f s)'
              % (k, len(configs), ne, snr, alvo, na, time.time() - t0))

    salva(linhas)
    config.salva(REF, os.path.join(PASTA_RESULTADOS, 'sensibilidade_config.json'),
                 tamanhos=sorted({l['ne'] for l in linhas}),
                 ruidos=sorted({l['snr'] for l in linhas}),
                 alvos=sorted({l['alvo'] for l in linhas}),
                 assimilacoes=sorted({l['na'] for l in linhas}),
                 sementes=n_sem)
    _resumo(linhas)
    _figuras(linhas)

    return linhas


def salva(linhas):
    """Grava as execucoes em CSV, para que as tabelas do texto sejam rastreaveis."""
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    caminho = os.path.join(PASTA_RESULTADOS, 'sensibilidade.csv')
    colunas = list(linhas[0])
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(','.join(colunas) + '\n')
        for linha in linhas:
            f.write(','.join(str(linha[c]) for c in colunas) + '\n')
    print('\n%d execucoes gravadas em %s' % (len(linhas), caminho))


def _filtra(linhas, **criterios):
    return [l for l in linhas if all(l[k] == v for k, v in criterios.items())]


def _pareado(sub, metrica):
    """
    PAREADO
    Compara os dois metodos semente a semente.

    Como ambos veem o mesmo cenario em cada repeticao, a diferenca pode ser
    analisada de forma pareada. Devolve as medias, a diferenca media e o
    p-valor do teste de postos sinalizados de Wilcoxon, que nao pressupoe
    normalidade e e apropriado para poucas amostras.
    """
    a = np.array([l['%s_mda' % metrica] for l in sub])
    b = np.array([l['%s_lm' % metrica] for l in sub])
    dif = b - a
    if np.allclose(dif, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(a, b).pvalue)

    return a.mean(), a.std(), b.mean(), b.std(), dif.mean(), p


def mesmo_custo(linhas):
    """
    MESMO CUSTO
    Pares iES-LM x ES-MDA com o mesmo numero de avaliacoes do modelo direto.

    O custo do iES-LM varia de semente para semente (depende de quando a
    parada dispara). Para cada semente, escolhe-se a execucao do ES-MDA cujo
    N_a da exatamente o mesmo custo, N_a = avaliacoes do iES-LM - 1. Como cada
    linha ja traz as metricas dos dois metodos, a linha escolhida e o par.

    Parameters
    ----------
    linhas : list of dict
        Execucoes do estudo.

    Returns
    -------
    list of dict
        Uma linha por semente para a qual existe o N_a correspondente.
    """
    eixo = _filtra(linhas, ne=REF.ne, snr=REF.snr, alvo='poço')
    por_semente = {}
    for l in eixo:
        por_semente.setdefault(l['semente'], {})[l['na']] = l

    pares = []
    for semente in sorted(por_semente):
        linhas_s = por_semente[semente]
        custo_lm = next(iter(linhas_s.values()))['aval_lm']
        par = linhas_s.get(custo_lm - 1)
        if par is not None:
            pares.append(par)
    return pares


def _linha_resumo(rotulo, sub, metrica='rmse_Vp'):
    m_a, s_a, m_b, s_b, dif, p = _pareado(sub, metrica)
    marca = '*' if p < 0.05 else ' '
    return ('%-14s %7.4f+-%.4f %7.4f+-%.4f %+8.4f %8.3f%s'
            % (rotulo, m_a, s_a, m_b, s_b, dif, p, marca))


def _resumo(linhas):
    cab = ('%-14s %15s %15s %8s %8s' % ('', 'ES-MDA', 'iES-LM', 'dif.', 'p'))
    for titulo, chave, valores, fixos in (
            ('Tamanho do conjunto', 'ne', sorted({l['ne'] for l in linhas}),
             {'snr': REF.snr, 'alvo': 'poço', 'na': NA_REF}),
            ('Nivel de ruido (SNR)', 'snr', sorted({l['snr'] for l in linhas}),
             {'ne': REF.ne, 'alvo': 'poço', 'na': NA_REF}),
            ('Alvo', 'alvo', [a for a in ALVOS if any(l['alvo'] == a for l in linhas)],
             {'ne': REF.ne, 'snr': REF.snr, 'na': NA_REF})):
        print('\n=== %s — RMSE de Vp (media +- desvio entre sementes) ===' % titulo)
        print(cab)
        for v in valores:
            sub = _filtra(linhas, **dict(fixos, **{chave: v}))
            if sub:
                print(_linha_resumo(str(v), sub))
        print('  * diferenca significativa ao nivel de 5% (Wilcoxon pareado)')

    nas = sorted({l['na'] for l in linhas})
    if len(nas) > 1:
        for metrica, nome in (('rmse_Vp', 'RMSE de Vp'), ('cob_Vp', 'Cobertura de Vp')):
            print('\n=== Assimilacoes do ES-MDA — %s (iES-LM fixo) ===' % nome)
            print('%-14s %15s %15s %8s %8s' % ('N_a (custo)', 'ES-MDA', 'iES-LM', 'dif.', 'p'))
            for na in nas:
                sub = _filtra(linhas, ne=REF.ne, snr=REF.snr, alvo='poço', na=na)
                if sub:
                    print(_linha_resumo('%d (%d aval.)' % (na, na + 1), sub, metrica))

        pares = mesmo_custo(linhas)
        custos = sorted({l['aval_mda'] for l in pares})
        print('\n=== Comparacao no MESMO custo: %d de %d sementes pareadas, '
              'custo %s avaliacoes ==='
              % (len(pares), len({l['semente'] for l in linhas}),
                 '/'.join(str(c) for c in custos)))
        print('%-14s %15s %15s %8s %8s' % ('', 'ES-MDA', 'iES-LM', 'dif.', 'p'))
        for metrica, nome in (('rmse_Vp', 'RMSE Vp'), ('cob_Vp', 'cobertura Vp'),
                              ('env_Vp', 'envelope Vp')):
            if pares:
                print(_linha_resumo(nome, pares, metrica))
        print('  * diferenca significativa ao nivel de 5% (Wilcoxon pareado)')

    print('\n=== Custo: avaliacoes do modelo direto ===')
    for v in sorted({l['ne'] for l in linhas}):
        sub = _filtra(linhas, ne=v, snr=REF.snr, alvo='poço', na=NA_REF)
        if sub:
            print('  ne=%3d   ES-MDA %.1f   iES-LM %.1f'
                  % (v, np.mean([l['aval_mda'] for l in sub]),
                     np.mean([l['aval_lm'] for l in sub])))


def _painel_eixo(ax, linhas, chave, valores, fixos, metrica, rotulo_y, log_x=True):
    """Media e desvio entre sementes, para os dois metodos, ao longo de um eixo."""
    for sufixo, cor, nome in (('mda', COR_MDA, 'ES-MDA'), ('lm', COR_LM, 'iES-LM')):
        m, s = [], []
        for v in valores:
            sub = _filtra(linhas, **dict(fixos, **{chave: v}))
            vals = [l['%s_%s' % (metrica, sufixo)] for l in sub]
            m.append(np.mean(vals))
            s.append(np.std(vals))
        ax.errorbar(valores, m, yerr=s, marker='o', capsize=3, color=cor, label=nome)
    if log_x:
        ax.set_xscale('log')
        ax.set_xticks(valores)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel(rotulo_y)
    ax.grid(alpha=0.3)


def _figuras(linhas):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    nes = sorted({l['ne'] for l in linhas})
    snrs = sorted({l['snr'] for l in linhas})

    # Figura 1: os dois eixos numericos, em qualidade e em calibracao
    fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
    _painel_eixo(eixos[0, 0], linhas, 'ne', nes, {'snr': REF.snr, 'alvo': 'poço', 'na': NA_REF},
                 'rmse_Vp', r'RMSE de $V_p$ (km/s)')
    eixos[0, 0].set_title('Qualidade × tamanho do conjunto')
    eixos[0, 0].set_xlabel(r'$N_e$')
    eixos[0, 0].legend(fontsize=8)

    _painel_eixo(eixos[0, 1], linhas, 'snr', snrs, {'ne': REF.ne, 'alvo': 'poço', 'na': NA_REF},
                 'rmse_Vp', r'RMSE de $V_p$ (km/s)')
    eixos[0, 1].set_title('Qualidade × nível de ruído')
    eixos[0, 1].set_xlabel('SNR')

    _painel_eixo(eixos[1, 0], linhas, 'ne', nes, {'snr': REF.snr, 'alvo': 'poço', 'na': NA_REF},
                 'cob_Vp', r'Cobertura de $V_p$')
    eixos[1, 0].axhline(0.8, color='k', ls=':', lw=1.2, label='calibração ideal (0,8)')
    eixos[1, 0].set_title('Calibração × tamanho do conjunto')
    eixos[1, 0].set_xlabel(r'$N_e$')
    eixos[1, 0].legend(fontsize=8)

    _painel_eixo(eixos[1, 1], linhas, 'snr', snrs, {'ne': REF.ne, 'alvo': 'poço', 'na': NA_REF},
                 'cob_Vp', r'Cobertura de $V_p$')
    eixos[1, 1].axhline(0.8, color='k', ls=':', lw=1.2)
    eixos[1, 1].set_title('Calibração × nível de ruído')
    eixos[1, 1].set_xlabel('SNR')

    fig.suptitle('Sensibilidade: média e desvio entre %d sementes'
                 % len({l['semente'] for l in linhas}), y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_eixos.png'), dpi=150)
    plt.close(fig)

    # Figura 2: diferenca pareada, que e o que sustenta (ou nao) a comparacao
    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (chave, valores, fixos, rotulo) in zip(eixos, (
            ('ne', nes, {'snr': REF.snr, 'alvo': 'poço', 'na': NA_REF}, r'$N_e$'),
            ('snr', snrs, {'ne': REF.ne, 'alvo': 'poço', 'na': NA_REF}, 'SNR'),
            ('alvo', [a for a in ALVOS if any(l['alvo'] == a for l in linhas)],
             {'ne': REF.ne, 'snr': REF.snr, 'na': NA_REF}, 'alvo'))):
        dados_cx, rotulos = [], []
        for v in valores:
            sub = _filtra(linhas, **dict(fixos, **{chave: v}))
            if sub:
                dados_cx.append([l['rmse_Vp_lm'] - l['rmse_Vp_mda'] for l in sub])
                rotulos.append(str(v))
        cx = ax.boxplot(dados_cx, tick_labels=rotulos, patch_artist=True, widths=0.55)
        for caixa in cx['boxes']:
            caixa.set_facecolor(COR_LM)
            caixa.set_alpha(0.35)
        for mediana in cx['medians']:
            mediana.set_color(COR_LM)
        ax.axhline(0, color='k', ls='--', lw=1.0)
        ax.set_xlabel(rotulo)
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel('RMSE iES-LM − RMSE ES-MDA')
    fig.suptitle('Diferença pareada: abaixo de zero, o iES-LM é melhor naquela semente',
                 y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_pareada.png'), dpi=150)
    plt.close(fig)

    # Figura 3: custo contra qualidade
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for sufixo, cor, nome in (('mda', COR_MDA, 'ES-MDA'), ('lm', COR_LM, 'iES-LM')):
        sub = _filtra(linhas, snr=REF.snr, alvo='poço', na=NA_REF)
        ax.scatter([l['aval_%s' % sufixo] for l in sub],
                   [l['rmse_Vp_%s' % sufixo] for l in sub],
                   s=[0.12 * l['ne'] for l in sub], color=cor, alpha=0.45,
                   edgecolors='none', label='%s (área ∝ $N_e$)' % nome)
    ax.set_xlabel('Avaliações do modelo direto')
    ax.set_ylabel(r'RMSE de $V_p$ (km/s)')
    ax.set_title('Custo × qualidade')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_custo.png'), dpi=150)
    plt.close(fig)

    _figura_assimilacoes(linhas)

    print('figuras salvas em %s' % PASTA_FIGURAS)


def _figura_assimilacoes(linhas):
    """
    FIGURA ASSIMILACOES
    Qualidade e calibracao em funcao do custo: a curva do ES-MDA ao variar
    N_a, e o iES-LM como um ponto. E a comparacao justa, no mesmo eixo de
    custo, que a figura antiga de custo nao fazia.
    """
    nas = sorted({l['na'] for l in linhas})
    if len(nas) < 2:
        return

    eixo = lambda na: _filtra(linhas, ne=REF.ne, snr=REF.snr, alvo='poço', na=na)
    base = eixo(NA_REF)
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metrica, rotulo in ((eixos[0], 'rmse_Vp', r'RMSE de $V_p$ (km/s)'),
                                (eixos[1], 'cob_Vp', r'Cobertura de $V_p$')):
        x = [na + 1 for na in nas]
        m = [np.mean([l['%s_mda' % metrica] for l in eixo(na)]) for na in nas]
        d = [np.std([l['%s_mda' % metrica] for l in eixo(na)]) for na in nas]
        ax.errorbar(x, m, yerr=d, marker='o', capsize=3, color=COR_MDA,
                    label=r'ES-MDA, variando $N_a$')
        for xi, mi, na in zip(x, m, nas):
            ax.annotate(r'$N_a$=%d' % na, (xi, mi), textcoords='offset points',
                        xytext=(6, 6), fontsize=8, color=COR_MDA)

        custo = [l['aval_lm'] for l in base]
        valor = [l['%s_lm' % metrica] for l in base]
        ax.errorbar([np.mean(custo)], [np.mean(valor)], xerr=[np.std(custo)],
                    yerr=[np.std(valor)], marker='s', ms=9, capsize=3, color=COR_LM,
                    label='iES-LM')
        ax.set_xlabel('Avaliações do modelo direto (custo)')
        ax.set_ylabel(rotulo)
        ax.grid(alpha=0.3)
    eixos[1].axhline(0.8, color='k', ls=':', lw=1.2, label='calibração ideal (0,8)')
    eixos[0].set_title('Qualidade × custo')
    eixos[1].set_title('Calibração × custo')
    eixos[1].legend(fontsize=8)
    fig.suptitle('Comparação no mesmo custo: média e desvio entre %d sementes'
                 % len({l['semente'] for l in base}), y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_assimilacoes.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    executa(rapido='--rapido' in sys.argv)
