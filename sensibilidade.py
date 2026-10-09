#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo de sensibilidade: iES-LM x ES-MDA sob condicoes variadas.

Atende ao objetivo especifico de analisar a sensibilidade do metodo ao
tamanho do conjunto e ao nivel de ruido, e o estende a outros fatores: a
aspereza do alvo, o numero de assimilacoes do ES-MDA (para comparar os
metodos com o mesmo custo), a constante da parada da Eq. 43, o gamma inicial,
um vies no a priori e o angulo maximo de incidencia.

O ponto central e estatistico. Uma unica execucao nao permite afirmar que um
metodo e melhor: a diferenca observada pode vir do sorteio do conjunto a
priori e do ruido. Aqui cada configuracao e repetida com varias sementes, e
os dois metodos recebem em cada repeticao EXATAMENTE o mesmo cenario. Isso
torna a comparacao pareada: a diferenca e medida semente a semente, o que
elimina a variabilidade comum e permite um teste de significancia.

Cada eixo varia um fator com todos os demais no valor de referencia
(config.PADRAO). Alem dos eixos, uma grade cruzada varia o tamanho do
conjunto e a SNR ao mesmo tempo, para revelar interacoes que a variacao de um
fator por vez esconde.

Protocolo identico ao de experimento_elastico.py, cujas funcoes sao
reaproveitadas para que nao existam duas implementacoes do mesmo experimento.

Uso:
    python sensibilidade.py             # grade completa
    python sensibilidade.py --rapido    # grade reduzida, para conferir
    python sensibilidade.py --relatorio # refaz resumo e figuras a partir do
                                        # CSV, sem rodar as inversoes de novo
"""

import functools
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable

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

# Ponto de referencia: os eixos variam a partir dele, e todas as entradas que
# nao pertencem a nenhum eixo vem dele tambem.
REF = config.PADRAO

N_SEMENTES = 20
COR_MDA, COR_LM = X.COR_MDA, X.COR_LM


def _angulos(maximo):
    """Tres angulos igualmente espacados de 15 graus ate o maximo pedido."""
    return tuple(float(a) for a in np.linspace(15.0, maximo, 3))


@dataclass(frozen=True)
class Eixo:
    """
    EIXO
    Um fator do estudo de sensibilidade.

    Attributes
    ----------
    campo : str
        Nome da coluna no CSV e da chave nas linhas de resultado.
    titulo : str
        Nome legivel, para tabelas e figuras.
    valores : tuple
        Valores da grade completa. Devem incluir o valor de referencia.
    rapido : tuple
        Valores da grade reduzida (--rapido).
    ajuste : callable
        Recebe um valor e devolve os campos da Configuracao a alterar.
    log_x : bool
        Se o eixo deve ser desenhado em escala logaritmica.
    rotulo : callable
        Formata um valor para tabelas e figuras.
    """

    campo: str
    titulo: str
    valores: tuple
    rapido: tuple
    ajuste: Callable
    log_x: bool = False
    rotulo: Callable = str


EIXOS = (
    Eixo('ne', 'Tamanho do conjunto', (25, 50, 100, 200, 400), (50, 200),
         lambda v: {'ne': v}, log_x=True),
    Eixo('snr', 'Nível de ruído (SNR)', (2.0, 5.0, 10.0, 20.0, 50.0), (5.0, 10.0),
         lambda v: {'snr': v}, log_x=True, rotulo=lambda v: '%g' % v),
    Eixo('alvo', 'Alvo', ref.ALVOS, ('poço', '6 camadas'),
         lambda v: {'alvo': v}),
    # O custo do ES-MDA e N_a + 1 avaliacoes do modelo direto. N_a = 4 e uma
    # escolha arbitraria; variar N_a permite comparar os metodos no MESMO custo.
    Eixo('na', 'Assimilações do ES-MDA', (1, 2, 3, 4, 6), (2, 4),
         lambda v: {'n_assimilacoes': v}),
    # A Eq. 43 encerra quando R < fator * nd. O artigo usa 4, o que corresponde
    # a parar com o desajuste medio em 2, quatro vezes acima do nivel do ruido.
    Eixo('fator_ruido', 'Constante da Eq. 43', (0.5, 1.0, 2.0, 4.0, 8.0), (2.0, 4.0),
         lambda v: {'fator_ruido': v}, log_x=True, rotulo=lambda v: '%g' % v),
    Eixo('gamma0', 'Gamma inicial', (0.1, 0.25, 0.5, 1.0, 2.0), (0.25, 1.0),
         lambda v: {'gamma0': v}, log_x=True, rotulo=lambda v: '%g' % v),
    Eixo('vies_prior', 'Viés do a priori', (-0.08, -0.05, -0.02, 0.0), (-0.05, 0.0),
         lambda v: {'vies_prior': v}, rotulo=lambda v: '%+.0f%%' % (100 * v)),
    Eixo('angulo_max', 'Ângulo máximo', (30.0, 45.0, 60.0), (45.0, 60.0),
         lambda v: {'angulos': _angulos(v)}, rotulo=lambda v: '%.0f°' % v),
)
EIXO = {e.campo: e for e in EIXOS}
CAMPOS = tuple(e.campo for e in EIXOS)



@dataclass(frozen=True)
class Cruzada:
    """
    CRUZADA
    Dois eixos variados ao mesmo tempo, para revelar interacoes que a
    variacao de um fator por vez esconde.

    Attributes
    ----------
    nome : str
        Identificador, usado no nome da figura.
    linha, coluna : str
        Campos dos eixos nas linhas e nas colunas da matriz.
    valores_linha, valores_coluna : tuple
        Valores da grade completa.
    rapido_linha, rapido_coluna : tuple
        Valores da grade reduzida.
    """

    nome: str
    linha: str
    coluna: str
    valores_linha: tuple
    valores_coluna: tuple
    rapido_linha: tuple
    rapido_coluna: tuple


CRUZADAS = (
    # A desvantagem do iES-LM com pouco ruido depende do tamanho do conjunto?
    Cruzada('ne_snr', 'ne', 'snr', (50, 100, 200, 400), (2.0, 5.0, 10.0, 20.0, 50.0),
            (50, 200), (5.0, 10.0)),
    # Um limiar diferente na Eq. 43 corrige o comportamento com pouco ruido? O
    # eixo da constante sozinho roda em SNR = 10, onde nao ha desvantagem.
    Cruzada('fator_snr', 'fator_ruido', 'snr', (1.0, 2.0, 4.0, 8.0), (5.0, 10.0, 20.0, 50.0),
            (2.0, 4.0), (10.0, 20.0)),
)


def campos_de(cfg):
    """
    CAMPOS DE
    Valores de uma configuracao em cada eixo do estudo.

    E a ponte entre a Configuracao e as linhas de resultado: o eixo
    'angulo_max', por exemplo, e lido como o maior dos angulos.
    """
    return {'ne': cfg.ne, 'snr': cfg.snr, 'alvo': cfg.alvo, 'na': cfg.n_assimilacoes,
            'fator_ruido': cfg.fator_ruido, 'gamma0': cfg.gamma0,
            'vies_prior': cfg.vies_prior, 'angulo_max': max(cfg.angulos)}


REF_CAMPOS = campos_de(REF)

for _e in EIXOS:
    assert REF_CAMPOS[_e.campo] in _e.valores, \
        'eixo %s nao contem o valor de referencia' % _e.campo


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


def avalia(cfg):
    """
    AVALIA
    Roda os dois metodos sobre o mesmo cenario e devolve as metricas.

    Parameters
    ----------
    cfg : config.Configuracao
        Configuracao completa da execucao, incluindo a semente.

    Returns
    -------
    dict
        Uma linha de resultado: o valor de cada eixo, a semente e as
        metricas de ambos os metodos.
    """
    c = X.carrega_cenario(cfg)
    M_mda, hist_mda, aval_mda, _ = X.roda_esmda(c)
    verd = fe.desempilha(c['verdadeiro'])

    linha = dict(campos_de(cfg))
    linha.update({'semente': cfg.semente,
                  'aspereza': float(np.std(np.diff(verd[0], axis=0))),
                  'aval_mda': aval_mda, 'desajuste_mda': hist_mda[-1]})
    linha.update(_metricas(M_mda, verd, 'mda'))
    linha.update(_resultado_ieslm(replace(cfg, n_assimilacoes=REF.n_assimilacoes)))

    return linha


def _ordem(cfg):
    """Chave de ordenacao das configuracoes, para uma execucao deterministica."""
    v = campos_de(cfg)
    return tuple(v[c] for c in CAMPOS)


def grade(rapido=False):
    """
    GRADE
    Configuracoes a executar.

    Cada eixo varia o seu fator com todos os demais na referencia; cada
    grade cruzada varia dois fatores juntos. Configuracoes repetidas (a de referencia,
    por exemplo, pertence a todos os eixos) rodam uma vez so: como a
    Configuracao e imutavel e comparavel, basta reuni-las num conjunto.

    Returns
    -------
    configs : list of config.Configuracao
        Configuracoes, com semente 0; executa substitui a semente.
    n_sementes : int
    """
    base = replace(REF, semente=0)
    configs = set()
    for eixo in EIXOS:
        for v in (eixo.rapido if rapido else eixo.valores):
            configs.add(replace(base, **eixo.ajuste(v)))

    for cz in CRUZADAS:
        linhas_cz = cz.rapido_linha if rapido else cz.valores_linha
        colunas_cz = cz.rapido_coluna if rapido else cz.valores_coluna
        for a in linhas_cz:
            for b in colunas_cz:
                configs.add(replace(base, **EIXO[cz.linha].ajuste(a),
                                    **EIXO[cz.coluna].ajuste(b)))

    return sorted(configs, key=_ordem), (4 if rapido else N_SEMENTES)


def _descricao(cfg):
    """O que distingue uma configuracao da referencia, para o progresso."""
    v = campos_de(cfg)
    dif = ['%s=%s' % (c, EIXO[c].rotulo(v[c])) for c in CAMPOS if v[c] != REF_CAMPOS[c]]
    return ', '.join(dif) if dif else 'referencia'


def executa(rapido=False):
    configs, n_sem = grade(rapido)
    print('Estudo de sensibilidade: %d configuracoes x %d sementes = %d execucoes'
          % (len(configs), n_sem, len(configs) * n_sem))

    linhas = []
    t0 = time.time()
    for k, cfg in enumerate(configs, 1):
        for semente in range(n_sem):
            linhas.append(avalia(replace(cfg, semente=semente)))
        print('  [%2d/%2d] %-38s (%.0f s)'
              % (k, len(configs), _descricao(cfg), time.time() - t0))

    salva(linhas)
    config.salva(REF, os.path.join(PASTA_RESULTADOS, 'sensibilidade_config.json'),
                 eixos={e.campo: list(e.rapido if rapido else e.valores) for e in EIXOS},
                 grades_cruzadas={
                     cz.nome: {cz.linha: list(cz.rapido_linha if rapido else cz.valores_linha),
                               cz.coluna: list(cz.rapido_coluna if rapido else cz.valores_coluna)}
                     for cz in CRUZADAS},
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


def _converte(texto):
    """Valor do CSV de volta ao tipo original: None, int, float ou texto."""
    if texto == 'None':
        return None
    for tipo in (int, float):
        try:
            return tipo(texto)
        except ValueError:
            pass
    return texto


def carrega_csv(caminho=None):
    """
    CARREGA CSV
    Le as execucoes gravadas por salva, com os tipos originais.

    Permite refazer resumo e figuras sem repetir as inversoes. Floats sao
    gravados com a representacao exata do Python, entao a leitura reproduz
    os mesmos valores.
    """
    caminho = caminho or os.path.join(PASTA_RESULTADOS, 'sensibilidade.csv')
    with open(caminho, encoding='utf-8') as f:
        colunas = f.readline().rstrip('\n').split(',')
        return [dict(zip(colunas, (_converte(v) for v in l.rstrip('\n').split(','))))
                for l in f if l.strip()]


def relatorio(caminho=None):
    """Resumo e figuras a partir do CSV, sem rodar as inversoes."""
    linhas = carrega_csv(caminho)
    print('%d execucoes lidas do CSV' % len(linhas))
    _resumo(linhas)
    _figuras(linhas)
    return linhas


# --------------------------------------------------------------- selecao ---

def no_eixo(linhas, *campos):
    """
    NO EIXO
    Linhas em que todos os fatores, EXCETO os indicados, estao na referencia.

    E o unico filtro do estudo. Ele existe porque a forma anterior, que
    listava a mao os fatores fixos de cada resumo, deixou de ser segura
    quando novos eixos foram acrescentados: um fator esquecido na lista faria
    linhas de outros eixos entrarem silenciosamente nas medias.

    Parameters
    ----------
    linhas : list of dict
    *campos : str
        Fatores livres. Um so para um eixo; 'ne' e 'snr' para a grade cruzada.
    """
    fixos = [c for c in CAMPOS if c not in campos]
    return [l for l in linhas if all(l[c] == REF_CAMPOS[c] for c in fixos)]


def _valores(linhas, eixo):
    """Valores do eixo presentes nas linhas, na ordem declarada."""
    presentes = {l[eixo.campo] for l in no_eixo(linhas, eixo.campo)}
    return [v for v in eixo.valores if v in presentes]


def _com(linhas, **valores):
    return [l for l in linhas if all(l[k] == v for k, v in valores.items())]


# ------------------------------------------------------------ estatistica ---

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
    por_semente = {}
    for l in no_eixo(linhas, 'na'):
        por_semente.setdefault(l['semente'], {})[l['na']] = l

    pares = []
    for semente in sorted(por_semente):
        linhas_s = por_semente[semente]
        custo_lm = next(iter(linhas_s.values()))['aval_lm']
        par = linhas_s.get(custo_lm - 1)
        if par is not None:
            pares.append(par)
    return pares


# ----------------------------------------------------------------- resumo ---

def _marca(p):
    return '*' if p < 0.05 else ' '


def _linha_resumo(rotulo, sub, metrica='rmse_Vp'):
    m_a, s_a, m_b, s_b, dif, p = _pareado(sub, metrica)
    return ('%-14s %7.4f+-%.4f %7.4f+-%.4f %+8.4f %8.3f%s'
            % (rotulo, m_a, s_a, m_b, s_b, dif, p, _marca(p)))


def _linha_eixo(rotulo, sub):
    """RMSE e cobertura de Vp numa linha so, com as diferencas pareadas."""
    r = _pareado(sub, 'rmse_Vp')
    c = _pareado(sub, 'cob_Vp')
    it = np.mean([l['iter_lm'] for l in sub])
    return ('%-11s | %.4f %.4f %+.4f %6.3f%s | %.3f %.3f %+.3f %6.3f%s | %4.1f'
            % (rotulo, r[0], r[2], r[4], r[5], _marca(r[5]),
               c[0], c[2], c[4], c[5], _marca(c[5]), it))


def _resumo(linhas):
    cab = ('%-11s | %-6s %-6s %-7s %7s | %-5s %-5s %-6s %7s | %s'
           % ('valor', 'MDA', 'LM', 'dif.', 'p', 'MDA', 'LM', 'dif.', 'p', 'iter LM'))
    for eixo in EIXOS:
        valores = _valores(linhas, eixo)
        if len(valores) < 2:
            continue
        print('\n=== %s ===' % eixo.titulo)
        print('%-11s | %-30s | %-27s |' % ('', 'RMSE de Vp', 'Cobertura de Vp'))
        print(cab)
        for v in valores:
            sub = _com(no_eixo(linhas, eixo.campo), **{eixo.campo: v})
            print(_linha_eixo(eixo.rotulo(v), sub))
    print('\n  * diferenca significativa ao nivel de 5% (Wilcoxon pareado)')

    pares = mesmo_custo(linhas)
    if pares:
        custos = sorted({l['aval_mda'] for l in pares})
        print('\n=== Comparacao no MESMO custo: %d de %d sementes pareadas, '
              'custo %s avaliacoes ==='
              % (len(pares), len({l['semente'] for l in linhas}),
                 '/'.join(str(c) for c in custos)))
        print('%-14s %15s %15s %8s %8s' % ('', 'ES-MDA', 'iES-LM', 'dif.', 'p'))
        for metrica, nome in (('rmse_Vp', 'RMSE Vp'), ('cob_Vp', 'cobertura Vp'),
                              ('env_Vp', 'envelope Vp')):
            print(_linha_resumo(nome, pares, metrica))

    _resumo_cruzada(linhas)

    print('\n=== Custo: avaliacoes do modelo direto, ao longo de N_e ===')
    plano = no_eixo(linhas, 'ne')
    for v in _valores(linhas, EIXO['ne']):
        sub = _com(plano, ne=v)
        print('  ne=%3d   ES-MDA %.1f   iES-LM %.1f'
              % (v, np.mean([l['aval_mda'] for l in sub]),
                 np.mean([l['aval_lm'] for l in sub])))


def _matriz_cruzada(linhas, cz, metrica):
    """Diferenca pareada media e p-valor em cada celula de uma grade cruzada."""
    plano = no_eixo(linhas, cz.linha, cz.coluna)
    # So os valores declarados na grade: pontos dos eixos simples tambem caem
    # neste plano, mas nao formam linhas ou colunas completas da matriz.
    decl_l = set(cz.valores_linha) | set(cz.rapido_linha)
    decl_c = set(cz.valores_coluna) | set(cz.rapido_coluna)
    vals_l = sorted({l[cz.linha] for l in plano if l[cz.linha] in decl_l})
    vals_c = sorted({l[cz.coluna] for l in plano if l[cz.coluna] in decl_c})
    dif = np.full((len(vals_l), len(vals_c)), np.nan)
    p = np.full_like(dif, np.nan)
    for i, a in enumerate(vals_l):
        for j, b in enumerate(vals_c):
            sub = _com(plano, **{cz.linha: a, cz.coluna: b})
            if sub:
                r = _pareado(sub, metrica)
                dif[i, j], p[i, j] = r[4], r[5]
    return vals_l, vals_c, dif, p


def _resumo_cruzada(linhas):
    for cz in CRUZADAS:
        vals_l, vals_c, _, _ = _matriz_cruzada(linhas, cz, 'rmse_Vp')
        if len(vals_l) < 2 or len(vals_c) < 2:
            continue
        el, ec = EIXO[cz.linha], EIXO[cz.coluna]
        for metrica, nome, fmt in (('rmse_Vp', 'RMSE de Vp', '%+.4f'),
                                   ('cob_Vp', 'cobertura de Vp', '%+.3f')):
            _, _, dif, p = _matriz_cruzada(linhas, cz, metrica)
            print('\n=== Grade cruzada %s x %s — diferenca pareada em %s (LM - MDA) ==='
                  % (el.titulo, ec.titulo, nome))
            print('%12s' % (cz.linha + ' \\ ' + cz.coluna)
                  + ''.join('%11s' % ec.rotulo(v) for v in vals_c))
            for i, a in enumerate(vals_l):
                celulas = ''.join('%10s%s' % (fmt % dif[i, j], _marca(p[i, j]))
                                  for j in range(len(vals_c)))
                print('%12s %s' % (el.rotulo(a), celulas))


# ---------------------------------------------------------------- figuras ---

def _painel_eixo(ax, linhas, eixo, metrica, rotulo_y):
    """Media e desvio entre sementes, para os dois metodos, ao longo de um eixo."""
    plano = no_eixo(linhas, eixo.campo)
    valores = _valores(linhas, eixo)
    x = valores if eixo.campo != 'alvo' else list(range(len(valores)))
    for sufixo, cor, nome in (('mda', COR_MDA, 'ES-MDA'), ('lm', COR_LM, 'iES-LM')):
        m, s = [], []
        for v in valores:
            vals = [l['%s_%s' % (metrica, sufixo)] for l in _com(plano, **{eixo.campo: v})]
            m.append(np.mean(vals))
            s.append(np.std(vals))
        ax.errorbar(x, m, yerr=s, marker='o', capsize=3, color=cor, label=nome)
    if eixo.log_x:
        ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([eixo.rotulo(v) for v in valores])
    ax.minorticks_off()
    ax.set_xlabel(eixo.titulo)
    ax.set_ylabel(rotulo_y)
    ax.grid(alpha=0.3)


def _figuras(linhas):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    n_sem = len({l['semente'] for l in linhas})

    # Figura 1: os dois eixos do objetivo especifico, em qualidade e calibracao
    fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
    for col, campo in enumerate(('ne', 'snr')):
        _painel_eixo(eixos[0, col], linhas, EIXO[campo], 'rmse_Vp', r'RMSE de $V_p$ (km/s)')
        _painel_eixo(eixos[1, col], linhas, EIXO[campo], 'cob_Vp', r'Cobertura de $V_p$')
        eixos[1, col].axhline(0.8, color='k', ls=':', lw=1.2, label='calibração ideal (0,8)')
    eixos[0, 0].set_title('Qualidade × tamanho do conjunto')
    eixos[0, 1].set_title('Qualidade × nível de ruído')
    eixos[1, 0].set_title('Calibração × tamanho do conjunto')
    eixos[1, 1].set_title('Calibração × nível de ruído')
    eixos[0, 0].legend(fontsize=8)
    eixos[1, 0].legend(fontsize=8)
    fig.suptitle('Sensibilidade: média e desvio entre %d sementes' % n_sem, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_eixos.png'), dpi=150)
    plt.close(fig)

    # Figura 2: diferenca pareada, que e o que sustenta (ou nao) a comparacao
    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, campo in zip(eixos, ('ne', 'snr', 'alvo')):
        eixo = EIXO[campo]
        plano = no_eixo(linhas, campo)
        valores = _valores(linhas, eixo)
        dados_cx = [[l['rmse_Vp_lm'] - l['rmse_Vp_mda'] for l in _com(plano, **{campo: v})]
                    for v in valores]
        cx = ax.boxplot(dados_cx, tick_labels=[eixo.rotulo(v) for v in valores],
                        patch_artist=True, widths=0.55)
        for caixa in cx['boxes']:
            caixa.set_facecolor(COR_LM)
            caixa.set_alpha(0.35)
        for mediana in cx['medians']:
            mediana.set_color(COR_LM)
        ax.axhline(0, color='k', ls='--', lw=1.0)
        ax.set_xlabel(eixo.titulo)
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel('RMSE iES-LM − RMSE ES-MDA')
    fig.suptitle('Diferença pareada: abaixo de zero, o iES-LM é melhor naquela semente',
                 y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_pareada.png'), dpi=150)
    plt.close(fig)

    # Figura 3: custo contra qualidade, ao longo de N_e
    fig, ax = plt.subplots(figsize=(6.5, 5))
    plano = no_eixo(linhas, 'ne')
    for sufixo, cor, nome in (('mda', COR_MDA, 'ES-MDA'), ('lm', COR_LM, 'iES-LM')):
        ax.scatter([l['aval_%s' % sufixo] for l in plano],
                   [l['rmse_Vp_%s' % sufixo] for l in plano],
                   s=[0.12 * l['ne'] for l in plano], color=cor, alpha=0.45,
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
    _figura_novos_eixos(linhas)
    _figura_cruzada(linhas)

    print('figuras salvas em %s' % PASTA_FIGURAS)


def _figura_assimilacoes(linhas):
    """
    FIGURA ASSIMILACOES
    Qualidade e calibracao em funcao do custo: a curva do ES-MDA ao variar
    N_a, e o iES-LM como um ponto. E a comparacao justa, no mesmo eixo de
    custo, que a figura de custo ao longo de N_e nao faz.
    """
    plano = no_eixo(linhas, 'na')
    nas = _valores(linhas, EIXO['na'])
    if len(nas) < 2:
        return

    base = _com(plano, na=REF.n_assimilacoes)
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metrica, rotulo in ((eixos[0], 'rmse_Vp', r'RMSE de $V_p$ (km/s)'),
                                (eixos[1], 'cob_Vp', r'Cobertura de $V_p$')):
        x = [na + 1 for na in nas]
        m = [np.mean([l['%s_mda' % metrica] for l in _com(plano, na=na)]) for na in nas]
        d = [np.std([l['%s_mda' % metrica] for l in _com(plano, na=na)]) for na in nas]
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


def _figura_novos_eixos(linhas):
    """Qualidade e calibracao ao longo dos eixos acrescentados na Etapa 3."""
    campos = [c for c in ('fator_ruido', 'gamma0', 'vies_prior', 'angulo_max')
              if len(_valores(linhas, EIXO[c])) > 1]
    if not campos:
        return

    fig, eixos = plt.subplots(2, len(campos), figsize=(4.2 * len(campos), 8),
                              squeeze=False)
    for col, campo in enumerate(campos):
        _painel_eixo(eixos[0, col], linhas, EIXO[campo], 'rmse_Vp', r'RMSE de $V_p$ (km/s)')
        _painel_eixo(eixos[1, col], linhas, EIXO[campo], 'cob_Vp', r'Cobertura de $V_p$')
        eixos[1, col].axhline(0.8, color='k', ls=':', lw=1.2)
        eixos[0, col].set_title(EIXO[campo].titulo)
        # marca o valor de referencia
        ref_v = REF_CAMPOS[campo]
        vals = _valores(linhas, EIXO[campo])
        x_ref = ref_v if campo != 'alvo' else vals.index(ref_v)
        for linha in (0, 1):
            eixos[linha, col].axvline(x_ref, color='0.6', ls='--', lw=0.9)
    eixos[0, 0].legend(fontsize=8)
    fig.suptitle('Novos eixos: média e desvio entre %d sementes '
                 '(tracejado vertical = valor de referência)'
                 % len({l['semente'] for l in linhas}), y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_novos_eixos.png'), dpi=150)
    plt.close(fig)


def _figura_cruzada(linhas):
    """
    FIGURA CRUZADA
    Mapas de calor da diferenca pareada (iES-LM menos ES-MDA) sobre cada
    grade cruzada. Asteriscos marcam diferencas significativas.
    """
    for cz in CRUZADAS:
        vals_l, vals_c, _, _ = _matriz_cruzada(linhas, cz, 'rmse_Vp')
        if len(vals_l) < 2 or len(vals_c) < 2:
            continue
        el, ec = EIXO[cz.linha], EIXO[cz.coluna]

        fig, eixos = plt.subplots(1, 2, figsize=(13, 4.8))
        for ax, metrica, titulo, fmt, sentido, cmap in (
                (eixos[0], 'rmse_Vp', r'$\Delta$ RMSE de $V_p$', '%+.4f',
                 'negativo = iES-LM mais preciso', 'RdBu_r'),
                (eixos[1], 'cob_Vp', r'$\Delta$ cobertura de $V_p$', '%+.3f',
                 'positivo = iES-LM mais calibrado', 'RdBu')):
            _, _, dif, p = _matriz_cruzada(linhas, cz, metrica)
            lim = np.nanmax(np.abs(dif))
            im = ax.imshow(dif, cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
            for i in range(len(vals_l)):
                for j in range(len(vals_c)):
                    ax.text(j, i, (fmt % dif[i, j]) + _marca(p[i, j]).strip(),
                            ha='center', va='center', fontsize=8)
            ax.set_xticks(range(len(vals_c)))
            ax.set_xticklabels([ec.rotulo(v) for v in vals_c])
            ax.set_yticks(range(len(vals_l)))
            ax.set_yticklabels([el.rotulo(v) for v in vals_l])
            ax.set_xlabel(ec.titulo)
            ax.set_ylabel(el.titulo)
            ax.set_title('%s (iES-LM − ES-MDA)\n%s' % (titulo, sentido), fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle('Grade cruzada %s × %s: diferença pareada média; '
                     '* = significativa a 5%%' % (el.titulo, ec.titulo), y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(PASTA_FIGURAS, 'sensibilidade_cruzada_%s.png' % cz.nome),
                    dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    if '--relatorio' in sys.argv:
        relatorio()
    else:
        executa(rapido='--rapido' in sys.argv)
