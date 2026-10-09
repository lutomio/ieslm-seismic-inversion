#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento do TCC 2: iES-LM x ES-MDA na inversao sismica elastica.

Estima simultaneamente os perfis de Vp, Vs e densidade a partir dos tracos
sismicos de tres angulos de incidencia. Os dois metodos partem do MESMO
conjunto a priori, recebem os MESMOS dados e usam a MESMA semente, de modo
que a diferenca venha do algoritmo e nao do sorteio.

    ES-MDA : EnsembleSmootherMDA da SeReMpy, sem modificacao, com sequencia
             DECRESCENTE de fatores de inflacao satisfazendo sum(1/alpha)=1.
    iES-LM : implementacao propria (ieslm.py), com alpha adaptativo. O nucleo
             e o mesmo usado no caso acustico: so muda a funcao g.

Uso:
    python experimento_elastico.py
"""

import os
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
import dados
import forward_elastico as fe
import ieslm
import metricas as mt
from ieslm import fator_lm as mt_fator
import prior
import referencia as ref
from SeReMpy.Inversion import EnsembleSmootherMDA

PASTA_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figuras')

PASTA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

# As entradas do experimento (tamanho do conjunto, SNR, numero de
# assimilacoes, gamma inicial, ...) ficam em config.py. Para mudar uma delas:
#     cfg = dataclasses.replace(config.PADRAO, snr=20.0)
#     executa(cfg)

COR_MDA, COR_LM, COR_PRIOR = 'tab:blue', 'tab:red', 'tab:gray'

# Largura (s) da janela ampliada nos paineis a posteriori. Na escala do perfil
# completo a diferenca entre os metodos e da ordem de 3% da largura do eixo e
# fica invisivel. A POSICAO da janela nao e fixa: janela_divergencia a escolhe
# onde os metodos mais divergem, para que a figura acompanhe a configuracao.
LARGURA_ZOOM = 0.012

# Formatos em que cada figura e gravada: PNG para consulta rapida, PDF
# (vetorial) para o texto em LaTeX, que nao perde nitidez ao ampliar.
FORMATOS = ('png', 'pdf')


def carrega_cenario(cfg=None, modelo=None):
    """
    CARREGA CENARIO
    Monta modelo direto, ruido, conjunto a priori e limites.

    Parameters
    ----------
    cfg : config.Configuracao, optional
        Entradas da execucao. Por omissao, config.PADRAO.
    modelo : dict, optional
        Modelo de referencia explicito, no formato de dados.carrega_dados.
        Tem precedencia sobre cfg.alvo; serve para inverter um modelo
        construido a mao, fora dos alvos nomeados.

    Returns
    -------
    dict
        O cenario. A chave 'cfg' guarda a configuracao, que roda_esmda e
        roda_ieslm leem para saber como rodar.
    """
    if isinstance(cfg, dict):
        # Antes da Configuracao, o primeiro argumento era o modelo de referencia.
        raise TypeError('o primeiro argumento agora e a configuracao; para passar '
                        'um modelo de referencia use modelo=...')
    cfg = config.PADRAO if cfg is None else cfg
    if modelo is None:
        modelo = ref.alvo_por_nome(cfg.alvo, cfg.semente)
    d = dados.carrega_dados() if modelo is None else modelo
    Time, dt = d['Time'], d['dt']
    Vp, Vs, Rho = d['Vp'], d['Vs'], d['Rho']

    verdadeiro = fe.empilha(Vp, Vs, Rho)
    g, _ = fe.monta_forward(Time, dt, freq=cfg.freq_wavelet,
                            ntw=cfg.amostras_wavelet, theta=cfg.angulos)

    # Os tracos de data5seis.dat NAO sao usados como observacao. Eles foram
    # gerados por este mesmo operador, sem ruido; inverte-los configuraria
    # "inverse crime". O dado observado e produzido a partir do poco, com
    # ruido de SNR controlada.
    rng_ruido = np.random.default_rng(cfg.semente)
    d_obs, C_D = fe.adiciona_ruido(g(verdadeiro), cfg.snr, rng_ruido)

    # A priori: tendencia de baixa frequencia + realizacoes correlacionadas
    tendencias = np.hstack([
        prior.tendencia_suave(x, ordem=cfg.ordem_tendencia, corte=cfg.corte_tendencia)
        for x in (Vp, Vs, Rho)
    ])
    # Vies opcional: a tendencia (baixa frequencia) do a priori deslocada.
    # Com vies_prior = 0 o fator e exatamente 1 e nada muda.
    tendencias = tendencias * (1.0 + cfg.vies_prior)
    sigma0 = np.cov(np.hstack([Vp, Vs, Rho]).T)
    conjunto = prior.conjunto_prior_multivariado(
        tendencias, cfg.ne, dt, sigma0,
        comprimento_correlacao=cfg.comprimento_correlacao * dt,
        rng=np.random.default_rng(cfg.semente),
    )

    # Limites fisicos por propriedade, com folga sobre a faixa observada
    nm = Vp.shape[0]
    lo = np.vstack([np.full((nm, 1), cfg.folga_inferior * float(x.min()))
                    for x in (Vp, Vs, Rho)])
    hi = np.vstack([np.full((nm, 1), cfg.folga_superior * float(x.max()))
                    for x in (Vp, Vs, Rho)])

    return dict(Time=Time, dt=dt, verdadeiro=verdadeiro, g=g, d_obs=d_obs,
                C_D=C_D, prior=conjunto, limites=(lo, hi), nm=nm, cfg=cfg)


def roda_esmda(c):
    """ES-MDA com sequencia decrescente de fatores de inflacao."""
    cfg = c['cfg']
    # EnsembleSmootherMDA sorteia com o gerador global do numpy
    np.random.seed(cfg.semente)
    C_D_inv = np.linalg.inv(c['C_D'])
    alphas = mt.sequencia_alpha_esmda(cfg.n_assimilacoes, razao=cfg.razao_alpha)

    M = c['prior'].copy()
    G = c['g'](M)
    n_aval = 1
    historico = [ieslm.desajuste_medio(c['d_obs'], G, C_D_inv)]

    for a in alphas:
        M, _ = EnsembleSmootherMDA(M, c['d_obs'], G, a, c['C_D'])
        M = np.clip(M, c['limites'][0], c['limites'][1])
        G = c['g'](M)
        n_aval += 1
        historico.append(ieslm.desajuste_medio(c['d_obs'], G, C_D_inv))

    return M, historico, n_aval, alphas


def roda_ieslm(c):
    """iES-LM com regularizacao adaptativa e parada no nivel do ruido."""
    cfg = c['cfg']
    return ieslm.ieslm(
        prior=c['prior'], d_obs=c['d_obs'], g=c['g'], C_D=c['C_D'],
        gamma0=cfg.gamma0, max_iter=cfg.max_iter, eta1=cfg.eta1, eta2=cfg.eta2,
        limites=c['limites'], fator_ruido=cfg.fator_ruido,
        rng=np.random.default_rng(cfg.semente),
    )


def roda_ieslm_sem_parada(c):
    """
    RODA IES-LM SEM PARADA
    O mesmo iES-LM com a parada por nivel de ruido (Eq. 43) desligada.

    Serve de referencia nas figuras: mostra o que a Eq. 43 evita (o
    sobreajuste ao ruido e, mais adiante, o passo que diverge). Se a
    configuracao ja tem a Eq. 43 desligada, devolve None: a execucao
    principal ja e a trajetoria sem parada.
    """
    cfg = c['cfg']
    if cfg.fator_ruido is None:
        return None
    return roda_ieslm(dict(c, cfg=replace(cfg, fator_ruido=None)))


def janela_divergencia(Time, A, B, verdadeiro, largura=LARGURA_ZOOM):
    """
    JANELA DIVERGENCIA
    Janela de tempo em que as medias de dois conjuntos mais divergem.

    Em cada amostra soma-se, sobre Vp, Vs e densidade, a diferenca absoluta
    entre as medias dos dois conjuntos, dividida pelo desvio padrao da
    propriedade no modelo de referencia (para que as tres pesem igual apesar
    das unidades diferentes). A janela escolhida e a de maior media movel
    dessa soma.

    Parameters
    ----------
    Time : array (nm,) ou (nm, 1)
    A, B : array (3*nm, ne)
        Conjuntos empilhados (fe.empilha).
    verdadeiro : array (3*nm, 1)
    largura : float
        Largura da janela, em segundos.

    Returns
    -------
    tuple (inicio, fim)
        Tempos da primeira e da ultima amostra da janela.
    """
    t = np.asarray(Time, dtype=float).ravel()
    dt = t[1] - t[0]
    n = min(len(t), int(round(largura / dt)) + 1)  # amostras na janela

    diferenca = np.zeros(len(t))
    for a, b, v in zip(fe.desempilha(A), fe.desempilha(B), fe.desempilha(verdadeiro)):
        escala = float(np.std(v)) or 1.0
        diferenca += np.abs(a.mean(axis=1) - b.mean(axis=1)) / escala

    media_movel = np.convolve(diferenca, np.ones(n) / n, mode='valid')
    k = int(np.argmax(media_movel))
    return float(t[k]), float(t[k + n - 1])


def salva_figura(fig, nome, pasta=None):
    """
    SALVA FIGURA
    Grava a figura em cada formato de FORMATOS (nome sem extensao) e a fecha.
    Por omissao grava em PASTA_FIGURAS; o estudo de sensibilidade tambem usa.
    """
    pasta = PASTA_FIGURAS if pasta is None else pasta
    for formato in FORMATOS:
        fig.savefig(os.path.join(pasta, '%s.%s' % (nome, formato)), dpi=150)
    plt.close(fig)


def executa(cfg=None, modelo=None):
    """
    EXECUTA
    Roda os dois metodos, imprime a tabela, gera as figuras e grava a
    configuracao usada em resultados/experimento_elastico_config.json.

    Parameters
    ----------
    cfg : config.Configuracao, optional
        Entradas da execucao. Por omissao, config.PADRAO.
    modelo : dict, optional
        Modelo de referencia explicito (ver carrega_cenario).
    """
    c = carrega_cenario(cfg, modelo)
    cfg = c['cfg']

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    config.salva(cfg, os.path.join(PASTA_RESULTADOS, 'experimento_elastico_config.json'),
                 modelo_externo=modelo is not None)

    print('Inversao sismica elastica - iES-LM x ES-MDA')
    print('conjunto: %d membros | %d amostras | 3 propriedades | %d angulos'
          % (cfg.ne, c['nm'], len(cfg.angulos)))
    print('ruido adicionado: SNR = %.0f  (desvio = %.2e)'
          % (cfg.snr, np.sqrt(c['C_D'][0, 0])))
    print()

    Z_mda, hist_mda, aval_mda, alphas = roda_esmda(c)

    res_lm = roda_ieslm(c)
    Z_lm = res_lm.conjunto
    res_sem = roda_ieslm_sem_parada(c)

    _tabela(c, Z_mda, hist_mda, aval_mda, res_lm, Z_lm, alphas, res_sem)
    _figuras(c, Z_mda, hist_mda, res_lm, Z_lm, res_sem)

    return dict(c=c, Z_mda=Z_mda, Z_lm=Z_lm, hist_mda=hist_mda, res_lm=res_lm,
                aval_mda=aval_mda, res_sem_parada=res_sem)


def _tabela(c, Z_mda, hist_mda, aval_mda, res_lm, Z_lm, alphas, res_sem=None):
    verd = fe.desempilha(c['verdadeiro'])
    prio = fe.desempilha(c['prior'])
    mda = fe.desempilha(Z_mda)
    lm = fe.desempilha(Z_lm)

    print('%-22s %10s %10s %10s' % ('', 'a priori', 'ES-MDA', 'iES-LM'))
    print('-' * 56)
    for i, nome in enumerate(fe.NOMES_PROPRIEDADES):
        print('%-22s %10.4f %10.4f %10.4f' % (
            'RMSE ' + nome, mt.rmse(prio[i], verd[i]),
            mt.rmse(mda[i], verd[i]), mt.rmse(lm[i], verd[i])))
    print()
    for i, nome in enumerate(fe.NOMES_PROPRIEDADES):
        print('%-22s %10.2f %10.2f %10.2f' % (
            'cobertura ' + nome, mt.taxa_cobertura(prio[i], verd[i]),
            mt.taxa_cobertura(mda[i], verd[i]), mt.taxa_cobertura(lm[i], verd[i])))
    print()
    for i, nome in enumerate(fe.NOMES_PROPRIEDADES):
        print('%-22s %10.4f %10.4f %10.4f' % (
            'largura P10-P90 ' + nome, mt.largura_envelope(prio[i]),
            mt.largura_envelope(mda[i]), mt.largura_envelope(lm[i])))
    print()
    for i, nome in enumerate(fe.NOMES_PROPRIEDADES):
        k_mda = mt.teste_ks(mda[i], verd[i])
        k_lm = mt.teste_ks(lm[i], verd[i])
        print('%-22s      ----   %5.3f (p=%.3f)  %5.3f (p=%.3f)' % (
            'KS ' + nome, k_mda[0], k_mda[1], k_lm[0], k_lm[1]))
    print()
    print('%-22s %10s %10.2e %10.2e' % ('desajuste final', '-',
                                        hist_mda[-1], min(res_lm.desajuste)))
    print('%-22s %10s %10d %10d' % ('avaliacoes de g', '-', aval_mda,
                                    res_lm.n_avaliacoes))
    print()
    print('ES-MDA alpha (decrescente):', ' '.join('%.3f' % a for a in alphas))
    print('iES-LM alpha (adaptativo) :', ' '.join('%.2e' % a for a in res_lm.alpha))
    print('iES-LM rho mediano        :', ' '.join('%.3f' % r for r in res_lm.rho_mediano))
    print('iES-LM parada             :', res_lm.motivo_parada)

    if res_sem is not None:
        sem = fe.desempilha(res_sem.conjunto)
        print()
        print('Referencia: iES-LM sem a parada da Eq. 43 (%d avaliacoes, melhor '
              'conjunto na iteracao %d)' % (res_sem.n_avaliacoes, res_sem.iteracao_final))
        print('  RMSE                    :', '  '.join(
            '%s %.4f' % (nome, mt.rmse(sem[i], verd[i]))
            for i, nome in enumerate(fe.NOMES_PROPRIEDADES)))
        print('  desajuste minimo        : %.2e' % min(res_sem.desajuste))


def _painel(ax, Time, verd, conjuntos, titulo, xlabel, zoom=None):
    """Um painel de propriedade: verdadeiro + envelopes dos metodos."""
    for conj, cor, rotulo in conjuntos:
        p10, p90 = mt.envelope(conj)
        ax.fill_betweenx(Time.ravel(), p10, p90, color=cor, alpha=0.30, lw=0)
        ax.plot(conj.mean(axis=1), Time, color=cor, lw=1.6, label=rotulo)
    ax.plot(verd, Time, 'k', lw=1.8, label='modelo de referência')
    ax.set_ylim(Time.max(), Time.min())
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.grid(alpha=0.3)

    # eixo apertado em torno do que importa: referência e envelopes dos métodos
    lo = min([verd.min()] + [mt.envelope(c)[0].min() for c, _, _ in conjuntos])
    hi = max([verd.max()] + [mt.envelope(c)[1].max() for c, _, _ in conjuntos])
    folga = 0.05 * (hi - lo)
    ax.set_xlim(lo - folga, hi + folga)

    if zoom:
        # marca a janela detalhada na figura seguinte, sem cobrir o perfil
        ax.axhspan(zoom[0], zoom[1], color='0.35', alpha=0.10, zorder=0)
        ax.axhline(zoom[0], color='0.45', lw=0.7, ls=':')
        ax.axhline(zoom[1], color='0.45', lw=0.7, ls=':')


def _painel_detalhe(ax, Time, verd, conjuntos, titulo, xlabel, zoom):
    """
    PAINEL DETALHE
    Amplia uma janela de tempo. Na escala do perfil completo a diferenca entre
    os metodos e da ordem de 3% da largura do eixo; aqui ela fica legivel.
    """
    t = Time.ravel()
    dentro = (t >= zoom[0]) & (t <= zoom[1])
    v = np.asarray(verd).ravel()[dentro]

    for conj, cor, rotulo in conjuntos:
        p10, p90 = mt.envelope(conj)
        ax.fill_betweenx(t[dentro], p10[dentro], p90[dentro], color=cor, alpha=0.30, lw=0)
        ax.plot(conj.mean(axis=1)[dentro], t[dentro], color=cor, lw=1.8, label=rotulo)
    ax.plot(v, t[dentro], 'k', lw=2.0, label='modelo de referência')
    ax.set_ylim(zoom[1], zoom[0])
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.grid(alpha=0.3)

    lo = min([v.min()] + [mt.envelope(c)[0][dentro].min() for c, _, _ in conjuntos])
    hi = max([v.max()] + [mt.envelope(c)[1][dentro].max() for c, _, _ in conjuntos])
    folga = 0.06 * (hi - lo)
    ax.set_xlim(lo - folga, hi + folga)


def _virgula(x, casas=3):
    """Numero com virgula decimal, para rotulos em portugues."""
    return ('%.*f' % (casas, x)).replace('.', ',')


def _valor_legivel(y):
    """Valor fora de escala escrito de forma curta: -2158 ou 8·10^10."""
    sinal = '\u2212' if y < 0 else ''  # sinal de menos tipografico
    if abs(y) < 1e4:
        return sinal + '%.0f' % abs(y)
    expoente = int(np.floor(np.log10(abs(y))))
    return r'%s$%.0f\cdot10^{%d}$' % (sinal, abs(y) / 10 ** expoente, expoente)


def _dentro(y, lo, hi):
    """Copia de y com NaN fora de [lo, hi]: a linha e interrompida ali."""
    y = np.array(y, dtype=float)
    y[(y < lo) | (y > hi)] = np.nan
    return y


def _marca_fora(ax, x, y, lo, hi, cor, texto):
    """
    MARCA FORA
    Pontos fora de [lo, hi] viram uma seta na borda do eixo com o valor
    escrito, em vez de esticar o eixo e achatar o resto da curva.
    """
    for xi, yi in zip(x, y):
        if lo <= yi <= hi:
            continue
        borda, marcador = (lo, 'v') if yi < lo else (hi, '^')
        ax.plot(xi, borda, marker=marcador, color=cor, ms=9, clip_on=False, zorder=5)
        ax.annotate(texto % _valor_legivel(yi), xy=(xi, borda),
                    xytext=(-10, 14 if yi < lo else -14), textcoords='offset points',
                    ha='right', va='bottom' if yi < lo else 'top', fontsize=8, color=cor,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))


def _iteracao_divergente(res):
    """Primeira iteracao em que a mediana de rho_j e negativa, ou None."""
    for j, r in enumerate(res.rho_mediano):
        if r < 0:
            return j + 1
    return None


def _figuras(c, Z_mda, hist_mda, res_lm, Z_lm, res_sem=None):
    cfg = c['cfg']
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    Time = c['Time']
    verd = fe.desempilha(c['verdadeiro'])
    prio = fe.desempilha(c['prior'])
    mda = fe.desempilha(Z_mda)
    lm = fe.desempilha(Z_lm)
    rotulos = [r'$V_p$ (km/s)', r'$V_s$ (km/s)', r'$\rho$ (g/cm$^3$)']
    titulos = ['Velocidade compressional', 'Velocidade cisalhante', 'Densidade']
    janela = janela_divergencia(Time, Z_mda, Z_lm, c['verdadeiro'])

    # Figura 1: a priori (contexto) e os dois metodos sobrepostos
    fig, eixos = plt.subplots(2, 3, figsize=(13, 10), sharey=True)
    for i in range(3):
        _painel(eixos[0, i], Time, verd[i],
                [(prio[i], COR_PRIOR, 'média a priori')],
                titulos[i] + ' — a priori', rotulos[i])
        _painel(eixos[1, i], Time, verd[i],
                [(mda[i], COR_MDA, 'ES-MDA'), (lm[i], COR_LM, 'iES-LM')],
                titulos[i] + ' — a posteriori', rotulos[i], zoom=janela)
    eixos[0, 0].set_ylabel('Tempo (s)')
    eixos[1, 0].set_ylabel('Tempo (s)')
    eixos[0, 0].legend(loc='lower right', fontsize=8)
    eixos[1, 0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Propriedades elásticas estimadas (envelopes P10–P90)', y=0.98)
    fig.tight_layout()
    salva_figura(fig, 'elastico_perfis')

    # Figura 2: ampliação da janela de maior divergência
    fig, eixos = plt.subplots(1, 3, figsize=(13, 5.4), sharey=True)
    for i, ax in enumerate(eixos):
        _painel_detalhe(ax, Time, verd[i],
                        [(mda[i], COR_MDA, 'ES-MDA'), (lm[i], COR_LM, 'iES-LM')],
                        titulos[i], rotulos[i], janela)
    eixos[0].set_ylabel('Tempo (s)')
    eixos[0].legend(loc='best', fontsize=9)
    fig.suptitle('Detalhe da janela %.3f–%.3f s: onde os métodos mais divergem'
                 % janela, y=0.98)
    fig.tight_layout()
    salva_figura(fig, 'elastico_detalhe')

    # Figura 3: erro de estimativa, onde a diferenca entre os metodos aparece
    # (nos perfis ela fica invisivel: e da ordem de 3% da largura do eixo)
    fig, eixos = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    for i, ax in enumerate(eixos):
        maior = 0.0  # maior erro absoluto, para um eixo simetrico em torno de zero
        for conj, cor, rot in [(mda[i], COR_MDA, 'ES-MDA'), (lm[i], COR_LM, 'iES-LM')]:
            erro = conj.mean(axis=1) - verd[i].ravel()
            maior = max(maior, float(np.abs(erro).max()))
            ax.plot(erro, Time, color=cor, lw=1.3, label=rot)
        ax.axvline(0, color='k', lw=1.0, ls='--')
        ax.axhspan(janela[0], janela[1], color='0.35', alpha=0.10, zorder=0)
        ax.set_xlim(-1.08 * maior, 1.08 * maior)
        ax.set_ylim(Time.max(), Time.min())
        ax.set_xlabel('Erro em ' + rotulos[i])
        ax.set_title(titulos[i])
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel('Tempo (s)')
    eixos[0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Erro de estimativa (média do conjunto − referência)', y=0.98)
    fig.tight_layout()
    salva_figura(fig, 'elastico_erro')

    # Figura 4: razão de ganho — o mecanismo que rege a adaptação de alfa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    rhos = res_lm.rho_por_membro
    it = np.arange(1, len(rhos) + 1)
    divergente = _iteracao_divergente(res_sem) if res_sem is not None else None

    partes = ax1.violinplot(rhos, positions=it, widths=0.5, showextrema=True,
                            showmedians=True)
    for corpo in partes['bodies']:
        corpo.set_facecolor(COR_LM)
        corpo.set_alpha(0.45)
    for chave in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        if chave in partes:
            partes[chave].set_color(COR_LM)
    ax1.axhline(1.0, color='k', ls='--', lw=1.0,
                label=r'$\rho=1$: linearização exata')

    # Limites pela faixa conjunta das curvas, sem os valores explosivos:
    # estes ficam marcados na borda (_marca_fora).
    visiveis = list(np.concatenate(rhos))
    if res_sem is not None:
        it_sem = np.arange(1, len(res_sem.rho_mediano) + 1)
        visiveis += [r for r in res_sem.rho_mediano if 0 <= r <= 2]
    folga = max(0.08 * np.ptp(visiveis + [1.0]), 2e-3)
    lo_rho = min(visiveis + [1.0]) - folga
    hi_rho = max(visiveis + [1.0]) + folga
    if res_sem is not None:
        ax1.plot(it_sem, _dentro(res_sem.rho_mediano, lo_rho, hi_rho), 'o--',
                 color=COR_LM, lw=1.2, ms=4, mfc='white', label='mediana sem a parada da Eq. 43')
        _marca_fora(ax1, it_sem, res_sem.rho_mediano, lo_rho, hi_rho, COR_LM,
                    r'$\rho \approx$ %s')
    ax1.set_ylim(lo_rho, hi_rho)
    ax1.set_xticks(np.arange(1, max(len(rhos), len(res_sem.rho_mediano)
                                    if res_sem is not None else 0) + 1))
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel(r'Razão de ganho $\rho_j$')
    ax1.set_title(r'Distribuição de $\rho_j$ entre os %d membros' % cfg.ne)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc='upper left')

    fator = [np.median(mt_fator(r)) for r in rhos]
    ax2.axhspan(1.0 / 3, 2.0, color='0.85', alpha=0.5, zorder=0)
    ax2.plot(it, fator, 's-', color=COR_LM, lw=1.8, label='fator aplicado (mediana)')
    if res_sem is not None:
        fator_sem = [np.median(mt_fator(r)) for r in res_sem.rho_por_membro]
        ax2.plot(it_sem, _dentro(fator_sem, 0.2, 2.15), 'o--',
                 color=COR_LM, lw=1.2, ms=4, mfc='white', label='sem a parada da Eq. 43')
        _marca_fora(ax2, it_sem, fator_sem, 0.2, 2.15, COR_LM,
                    'fator %s,\nmas $\\alpha$ não cresce (Eq. 41)')
    ax2.axhline(1 / 3, color='k', ls=':', lw=1.2,
                label=r'piso da regra: $\gamma$ dividido por 3')
    ax2.axhline(1.0, color='0.4', ls='--', lw=1.0, label=r'fator 1: $\gamma$ mantido')
    ax2.axhline(2.0, color='0.4', ls='-.', lw=1.0, label=r'fator 2: $\gamma$ dobrado')
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_ylim(0.2, 2.15)
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel(r'Fator aplicado a $\gamma$')
    ax2.set_title(r'Eq. 40: fator $\max(1/3,\ 1-(2\rho_j-1)^3)$')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc='center right')

    no_piso = all(abs(f - 1 / 3) < 1e-9 for f in fator)
    titulo = 'Mecanismo de adaptação: '
    titulo += ('com a parada, a linearização é fiel e a regra opera no piso'
               if no_piso else 'razão de ganho e fator aplicado a $\\gamma$')
    if divergente is not None:
        titulo += '\nsem a parada, o passo diverge na iteração %d ($\\rho < 0$)' % divergente
    fig.suptitle(titulo, y=0.99)
    fig.tight_layout()
    salva_figura(fig, 'elastico_ganho')

    # Figura 5: residuos sismicos por angulo
    n_ang = len(cfg.angulos)
    fig, eixos = plt.subplots(1, n_ang, figsize=(4.4 * n_ang, 4.6), sharey=True,
                              squeeze=False)
    eixos = eixos[0]
    obs = fe.separa_angulos(c['d_obs'], n_ang)
    for i, ax in enumerate(eixos):
        nome = fe.nome_angulo(cfg.angulos[i], i, n_ang)
        for M, cor, rot in [(Z_mda, COR_MDA, 'ES-MDA'), (Z_lm, COR_LM, 'iES-LM')]:
            pred = fe.separa_angulos(c['g'](M), n_ang)[i].mean(axis=1)
            ax.plot(obs[i].ravel() - pred, np.arange(len(pred)), color=cor, lw=1.2,
                    label=rot)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title('Resíduo sísmico — %s' % nome)
        ax.set_xlabel('Observado − previsto')
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel('Amostra sísmica')
    eixos[0].invert_yaxis()
    eixos[0].legend(fontsize=8)
    fig.tight_layout()
    salva_figura(fig, 'elastico_residuos')

    # Figura 6: convergencia e trajetoria da regularizacao
    def rmse_vp(M):
        return _virgula(mt.rmse(fe.desempilha(M)[0], verd[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.semilogy(range(len(hist_mda)), hist_mda, 'o-', color=COR_MDA,
                 label='ES-MDA (%d assimilações) — RMSE $V_p$ %s'
                 % (len(hist_mda) - 1, rmse_vp(Z_mda)))
    ax1.semilogy(range(len(res_lm.desajuste)), res_lm.desajuste, 's-', color=COR_LM,
                 label='iES-LM (%d iterações) — RMSE $V_p$ %s'
                 % (len(res_lm.desajuste) - 1, rmse_vp(Z_lm)))
    if res_sem is not None:
        ax1.semilogy(range(len(res_sem.desajuste)), res_sem.desajuste, 'o--',
                     color=COR_LM, lw=1.2, ms=4, mfc='white',
                     label='iES-LM sem a parada da Eq. 43 — RMSE $V_p$ %s'
                     % rmse_vp(res_sem.conjunto))
        k = res_sem.iteracao_final
        ax1.plot(k, res_sem.desajuste[k], '*', color=COR_LM, ms=13, mfc='none',
                 label='conjunto devolvido sem a parada (menor $\\bar{O}$)')
        if divergente is not None:
            ax1.annotate('passo divergente\n($\\rho_j < 0$)',
                         xy=(divergente, res_sem.desajuste[divergente]),
                         xytext=(-12, 0), textcoords='offset points', ha='right',
                         va='center', fontsize=8, color=COR_LM)
    ax1.axhline(0.5, color='k', ls=':', lw=1.2, label='nível do ruído ($\\bar{O}=0{,}5$)')
    ax1.axhspan(1e-3, 0.5, color='0.35', alpha=0.08, zorder=0)
    ax1.set_ylim(bottom=0.6 * min(min(hist_mda), min(res_lm.desajuste),
                                  min(res_sem.desajuste) if res_sem is not None else 1.0))
    ax1.text(0.98, 0.03, 'abaixo do ruído: sobreajuste', transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=8, color='0.35')
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel(r'Desajuste médio $\bar{O}$')
    ax1.set_title('Convergência')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5)

    ax2.semilogy(range(len(res_lm.alpha)), res_lm.alpha, 's-', color=COR_LM,
                 label=r'$\alpha^i$ adaptativo (iES-LM)')
    if res_sem is not None:
        ax2.semilogy(range(len(res_sem.alpha)), res_sem.alpha, 'o--', color=COR_LM,
                     lw=1.2, ms=4, mfc='white', label='iES-LM sem a parada da Eq. 43')
    alphas = mt.sequencia_alpha_esmda(cfg.n_assimilacoes, razao=cfg.razao_alpha)
    ax2.semilogy(range(1, len(alphas) + 1), alphas, 'o--', color=COR_MDA,
                 label=r'$\alpha_l$ decrescente (ES-MDA)')
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel(r'Regularização $\alpha$')
    ax2.set_title('Regularização do passo: predefinida × adaptativa')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    salva_figura(fig, 'elastico_convergencia')

    print('\nfiguras salvas em %s (%s)' % (PASTA_FIGURAS, ', '.join(FORMATOS)))


if __name__ == '__main__':
    executa()
