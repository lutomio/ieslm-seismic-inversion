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
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import experimento_elastico as X
import forward_elastico as fe
import metricas as mt
import referencia as ref

PASTA_FIGURAS = X.PASTA_FIGURAS
PASTA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

# Eixos do estudo. Cada um varia um fator com os demais fixos nos valores de
# referencia (NE = 200, SNR = 10, alvo = poco do pacote).
TAMANHOS = (25, 50, 100, 200, 400)
RUIDOS = (2.0, 5.0, 10.0, 20.0, 50.0)
ALVOS = ('poço', '6 camadas', '12 camadas', 'suavizado')

N_SEMENTES = 20
COR_MDA, COR_LM = X.COR_MDA, X.COR_LM


def constroi_alvo(nome, semente):
    """
    CONSTROI ALVO
    Modelo de referencia correspondente ao nome pedido.

    O poco do pacote e fixo; os sinteticos sao sorteados, e mudam com a
    semente, de modo que a repeticao cobre tambem a variabilidade do alvo.
    """
    if nome == 'poço':
        return None  # carrega_cenario usa o perfil do pacote
    rng = np.random.default_rng(1000 + semente)
    if nome == '6 camadas':
        return ref.modelo_em_camadas(n_camadas=6, rng=rng)
    if nome == '12 camadas':
        return ref.modelo_em_camadas(n_camadas=12, rng=rng)
    if nome == 'suavizado':
        return ref.suaviza(ref.modelo_em_camadas(n_camadas=6, rng=rng))
    raise ValueError('alvo desconhecido: %s' % nome)


def avalia(ne, snr, alvo, semente):
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

    Returns
    -------
    dict
        Uma linha de resultado, com as metricas de ambos os metodos.
    """
    c = X.carrega_cenario(modelo=constroi_alvo(alvo, semente),
                          ne=ne, snr=snr, semente=semente)

    M_mda, hist_mda, aval_mda, _ = X.roda_esmda(c, semente=semente)
    res_lm = X.roda_ieslm(c, semente=semente)

    verd = fe.desempilha(c['verdadeiro'])
    linha = {'ne': ne, 'snr': snr, 'alvo': alvo, 'semente': semente,
             'aspereza': float(np.std(np.diff(verd[0], axis=0))),
             'aval_mda': aval_mda, 'aval_lm': res_lm.n_avaliacoes,
             'desajuste_mda': hist_mda[-1], 'desajuste_lm': min(res_lm.desajuste),
             'iter_lm': len(res_lm.desajuste) - 1}

    for rotulo, M in (('mda', M_mda), ('lm', res_lm.conjunto)):
        prop = fe.desempilha(M)
        for i, nome in enumerate(fe.NOMES_PROPRIEDADES):
            linha['rmse_%s_%s' % (nome, rotulo)] = mt.rmse(prop[i], verd[i])
            linha['cob_%s_%s' % (nome, rotulo)] = mt.taxa_cobertura(prop[i], verd[i])
            linha['env_%s_%s' % (nome, rotulo)] = mt.largura_envelope(prop[i])

    return linha


def grade(rapido=False):
    """
    GRADE
    Configuracoes a executar: um eixo por vez, os demais no valor de
    referencia. A configuracao central e compartilhada pelos tres eixos e
    roda uma vez so.
    """
    tamanhos = (50, 200) if rapido else TAMANHOS
    ruidos = (5.0, 10.0) if rapido else RUIDOS
    alvos = ('poço', '6 camadas') if rapido else ALVOS
    n_sem = 4 if rapido else N_SEMENTES

    configs = set()
    for ne in tamanhos:
        configs.add((ne, X.SNR, 'poço'))
    for snr in ruidos:
        configs.add((X.NE, snr, 'poço'))
    for alvo in alvos:
        configs.add((X.NE, X.SNR, alvo))

    return sorted(configs), n_sem


def executa(rapido=False):
    configs, n_sem = grade(rapido)
    total = len(configs) * n_sem
    print('Estudo de sensibilidade: %d configuracoes x %d sementes = %d execucoes'
          % (len(configs), n_sem, total))

    linhas = []
    t0 = time.time()
    for k, (ne, snr, alvo) in enumerate(configs, 1):
        for semente in range(n_sem):
            linhas.append(avalia(ne, snr, alvo, semente))
        print('  [%2d/%2d] ne=%3d snr=%4.1f alvo=%-11s  (%.0f s)'
              % (k, len(configs), ne, snr, alvo, time.time() - t0))

    salva(linhas)
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


def _linha_resumo(rotulo, sub, metrica='rmse_Vp'):
    m_a, s_a, m_b, s_b, dif, p = _pareado(sub, metrica)
    marca = '*' if p < 0.05 else ' '
    return ('%-14s %7.4f+-%.4f %7.4f+-%.4f %+8.4f %8.3f%s'
            % (rotulo, m_a, s_a, m_b, s_b, dif, p, marca))


def _resumo(linhas):
    cab = ('%-14s %15s %15s %8s %8s' % ('', 'ES-MDA', 'iES-LM', 'dif.', 'p'))
    for titulo, chave, valores, fixos in (
            ('Tamanho do conjunto', 'ne', sorted({l['ne'] for l in linhas}),
             {'snr': X.SNR, 'alvo': 'poço'}),
            ('Nivel de ruido (SNR)', 'snr', sorted({l['snr'] for l in linhas}),
             {'ne': X.NE, 'alvo': 'poço'}),
            ('Alvo', 'alvo', [a for a in ALVOS if any(l['alvo'] == a for l in linhas)],
             {'ne': X.NE, 'snr': X.SNR})):
        print('\n=== %s — RMSE de Vp (media +- desvio entre sementes) ===' % titulo)
        print(cab)
        for v in valores:
            sub = _filtra(linhas, **dict(fixos, **{chave: v}))
            if sub:
                print(_linha_resumo(str(v), sub))
        print('  * diferenca significativa ao nivel de 5% (Wilcoxon pareado)')

    print('\n=== Custo: avaliacoes do modelo direto ===')
    for v in sorted({l['ne'] for l in linhas}):
        sub = _filtra(linhas, ne=v, snr=X.SNR, alvo='poço')
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
    _painel_eixo(eixos[0, 0], linhas, 'ne', nes, {'snr': X.SNR, 'alvo': 'poço'},
                 'rmse_Vp', r'RMSE de $V_p$ (km/s)')
    eixos[0, 0].set_title('Qualidade × tamanho do conjunto')
    eixos[0, 0].set_xlabel(r'$N_e$')
    eixos[0, 0].legend(fontsize=8)

    _painel_eixo(eixos[0, 1], linhas, 'snr', snrs, {'ne': X.NE, 'alvo': 'poço'},
                 'rmse_Vp', r'RMSE de $V_p$ (km/s)')
    eixos[0, 1].set_title('Qualidade × nível de ruído')
    eixos[0, 1].set_xlabel('SNR')

    _painel_eixo(eixos[1, 0], linhas, 'ne', nes, {'snr': X.SNR, 'alvo': 'poço'},
                 'cob_Vp', r'Cobertura de $V_p$')
    eixos[1, 0].axhline(0.8, color='k', ls=':', lw=1.2, label='calibração ideal (0,8)')
    eixos[1, 0].set_title('Calibração × tamanho do conjunto')
    eixos[1, 0].set_xlabel(r'$N_e$')
    eixos[1, 0].legend(fontsize=8)

    _painel_eixo(eixos[1, 1], linhas, 'snr', snrs, {'ne': X.NE, 'alvo': 'poço'},
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
            ('ne', nes, {'snr': X.SNR, 'alvo': 'poço'}, r'$N_e$'),
            ('snr', snrs, {'ne': X.NE, 'alvo': 'poço'}, 'SNR'),
            ('alvo', [a for a in ALVOS if any(l['alvo'] == a for l in linhas)],
             {'ne': X.NE, 'snr': X.SNR}, 'alvo'))):
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
        sub = _filtra(linhas, snr=X.SNR, alvo='poço')
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

    print('figuras salvas em %s' % PASTA_FIGURAS)


if __name__ == '__main__':
    executa(rapido='--rapido' in sys.argv)
