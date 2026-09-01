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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import dados
import forward_elastico as fe
import ieslm
import metricas as mt
import prior
from SeReMpy.Inversion import EnsembleSmootherMDA

PASTA_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figuras')

NE = 200           # tamanho do conjunto
NITER_MDA = 4      # assimilacoes do ES-MDA
MAX_ITER_LM = 10   # iteracoes maximas do iES-LM
SNR = 10.0         # razao sinal-ruido usada para contaminar o dado
FATOR_RUIDO = 4.0  # criterio de parada da Eq. 43
SEMENTE = 42

COR_MDA, COR_LM, COR_PRIOR = 'tab:blue', 'tab:red', 'tab:gray'


def carrega_cenario():
    """Monta dados, modelo direto, ruido, conjunto a priori e limites."""
    dl = np.loadtxt(os.path.join(dados.DATA_DIR, 'data5log.dat'))
    ds = np.loadtxt(os.path.join(dados.DATA_DIR, 'data5seis.dat'))

    Time = dl[:, 3:4]
    Vp, Vs, Rho = dl[:, 4:5], dl[:, 5:6], dl[:, 6:7]
    dt = float(ds[1, 0] - ds[0, 0])

    verdadeiro = fe.empilha(Vp, Vs, Rho)
    g, _ = fe.monta_forward(Time, dt)

    # O dado do pacote e livre de ruido e foi gerado por este mesmo operador;
    # sem contaminacao o experimento incorreria em "inverse crime".
    rng_ruido = np.random.default_rng(SEMENTE)
    d_obs, C_D = fe.adiciona_ruido(g(verdadeiro), SNR, rng_ruido)

    # A priori: tendencia de baixa frequencia + realizacoes correlacionadas
    tendencias = np.hstack([prior.tendencia_suave(x) for x in (Vp, Vs, Rho)])
    sigma0 = np.cov(np.hstack([Vp, Vs, Rho]).T)
    conjunto = prior.conjunto_prior_multivariado(
        tendencias, NE, dt, sigma0, rng=np.random.default_rng(SEMENTE)
    )

    # Limites fisicos por propriedade, com folga sobre a faixa observada
    nm = Vp.shape[0]
    lo = np.vstack([np.full((nm, 1), 0.7 * float(x.min())) for x in (Vp, Vs, Rho)])
    hi = np.vstack([np.full((nm, 1), 1.3 * float(x.max())) for x in (Vp, Vs, Rho)])

    return dict(Time=Time, dt=dt, verdadeiro=verdadeiro, g=g, d_obs=d_obs,
                C_D=C_D, prior=conjunto, limites=(lo, hi), nm=nm)


def roda_esmda(c):
    """ES-MDA com sequencia decrescente de fatores de inflacao."""
    np.random.seed(SEMENTE)
    C_D_inv = np.linalg.inv(c['C_D'])
    alphas = mt.sequencia_alpha_esmda(NITER_MDA)

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


def executa():
    c = carrega_cenario()

    print('Inversao sismica elastica - iES-LM x ES-MDA')
    print('conjunto: %d membros | %d amostras | 3 propriedades | 3 angulos'
          % (NE, c['nm']))
    print('ruido adicionado: SNR = %.0f  (desvio = %.2e)'
          % (SNR, np.sqrt(c['C_D'][0, 0])))
    print()

    Z_mda, hist_mda, aval_mda, alphas = roda_esmda(c)

    res_lm = ieslm.ieslm(
        prior=c['prior'], d_obs=c['d_obs'], g=c['g'], C_D=c['C_D'],
        gamma0=1.0, max_iter=MAX_ITER_LM, eta1=1e-4, eta2=1e-2,
        limites=c['limites'], fator_ruido=FATOR_RUIDO,
        rng=np.random.default_rng(SEMENTE),
    )
    Z_lm = res_lm.conjunto

    _tabela(c, Z_mda, hist_mda, aval_mda, res_lm, Z_lm, alphas)
    _figuras(c, Z_mda, hist_mda, res_lm, Z_lm)

    return dict(c=c, Z_mda=Z_mda, Z_lm=Z_lm, hist_mda=hist_mda, res_lm=res_lm,
                aval_mda=aval_mda)


def _tabela(c, Z_mda, hist_mda, aval_mda, res_lm, Z_lm, alphas):
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


def _painel(ax, Time, verd, conjuntos, titulo, xlabel):
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


def _figuras(c, Z_mda, hist_mda, res_lm, Z_lm):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    Time = c['Time']
    verd = fe.desempilha(c['verdadeiro'])
    prio = fe.desempilha(c['prior'])
    mda = fe.desempilha(Z_mda)
    lm = fe.desempilha(Z_lm)
    rotulos = [r'$V_p$ (km/s)', r'$V_s$ (km/s)', r'$\rho$ (g/cm$^3$)']
    titulos = ['Velocidade compressional', 'Velocidade cisalhante', 'Densidade']

    # Figura 1: a priori (contexto) e os dois metodos sobrepostos
    fig, eixos = plt.subplots(2, 3, figsize=(13, 10), sharey=True)
    for i in range(3):
        _painel(eixos[0, i], Time, verd[i],
                [(prio[i], COR_PRIOR, 'média a priori')],
                titulos[i] + ' — a priori', rotulos[i])
        _painel(eixos[1, i], Time, verd[i],
                [(mda[i], COR_MDA, 'ES-MDA'), (lm[i], COR_LM, 'iES-LM')],
                titulos[i] + ' — a posteriori', rotulos[i])
    eixos[0, 0].set_ylabel('Tempo (s)')
    eixos[1, 0].set_ylabel('Tempo (s)')
    eixos[0, 0].legend(loc='lower right', fontsize=8)
    eixos[1, 0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Propriedades elásticas estimadas (envelopes P10–P90)', y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'elastico_perfis.png'), dpi=150)
    plt.close(fig)

    # Figura 2: residuos sismicos por angulo
    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    obs = fe.separa_angulos(c['d_obs'])
    for i, (ax, nome) in enumerate(zip(eixos, fe.NOMES_ANGULOS)):
        for M, cor, rot in [(Z_mda, COR_MDA, 'ES-MDA'), (Z_lm, COR_LM, 'iES-LM')]:
            pred = fe.separa_angulos(c['g'](M))[i].mean(axis=1)
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
    fig.savefig(os.path.join(PASTA_FIGURAS, 'elastico_residuos.png'), dpi=150)
    plt.close(fig)

    # Figura 3: convergencia e trajetoria da regularizacao
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.semilogy(range(len(hist_mda)), hist_mda, 'o-', color=COR_MDA,
                 label='ES-MDA (%d assimilações)' % (len(hist_mda) - 1))
    ax1.semilogy(range(len(res_lm.desajuste)), res_lm.desajuste, 's-', color=COR_LM,
                 label='iES-LM (%d iterações)' % (len(res_lm.desajuste) - 1))
    ax1.axhline(0.5, color='k', ls=':', lw=1.2, label='nível do ruído ($\\bar{O}=0{,}5$)')
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel(r'Desajuste médio $\bar{O}$')
    ax1.set_title('Convergência')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.semilogy(range(len(res_lm.alpha)), res_lm.alpha, 's-', color=COR_LM,
                 label=r'$\alpha^i$ adaptativo (iES-LM)')
    alphas = mt.sequencia_alpha_esmda(NITER_MDA)
    ax2.semilogy(range(1, len(alphas) + 1), alphas, 'o--', color=COR_MDA,
                 label=r'$\alpha_l$ decrescente (ES-MDA)')
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel(r'Regularização $\alpha$')
    ax2.set_title('Regularização do passo: predefinida × adaptativa')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'elastico_convergencia.png'), dpi=150)
    plt.close(fig)

    print('\nfiguras salvas em %s' % PASTA_FIGURAS)


if __name__ == '__main__':
    executa()
