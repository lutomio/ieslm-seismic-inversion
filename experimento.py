#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento do TCC 2: iES-LM (regularizacao adaptativa) x ES-MDA (fixa).

Os dois metodos partem do MESMO conjunto a priori, recebem os MESMOS dados e
usam a MESMA semente, de modo que a diferenca nos resultados venha do
algoritmo e nao do sorteio.

    ES-MDA  : EnsembleSmootherMDA da SeReMpy, sem modificacao, com fatores de
              inflacao alpha_i fixos (alpha_i = niter, somando 1/alpha_i = 1).
    iES-LM  : implementacao propria (ieslm.py), com alpha^i adaptativo por
              regiao de confianca.

Uso:
    python experimento.py

Gera as figuras em figuras/ e imprime a tabela de metricas.
"""

import os

import matplotlib
matplotlib.use('Agg')  # sem interface grafica: roda em qualquer ambiente
import matplotlib.pyplot as plt
import numpy as np

import dados
import forward
import ieslm
import prior
from SeReMpy.Inversion import EnsembleSmootherMDA

PASTA_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figuras')

# Parametros do experimento
NE = 200              # tamanho do conjunto
NITER_MDA = 4         # numero de assimilacoes do ES-MDA
MAX_ITER_LM = 10      # numero maximo de iteracoes do iES-LM
VAR_ERRO = 1e-4       # variancia do erro de medicao
FATOR_RUIDO = 4.0     # Eq. 43: para quando R^i < 4*nd, evitando sobreajuste
SEMENTE = 42


def rmse(X, Z):
    """Erro quadratico medio da media do conjunto contra o perfil verdadeiro."""
    return float(np.sqrt(np.mean((X.mean(axis=1, keepdims=True) - Z) ** 2)))


def largura_envelope(X):
    """Largura media do intervalo P10-P90 do conjunto: espalhamento posterior."""
    p10, p90 = np.percentile(X, [10, 90], axis=1)

    return float(np.mean(p90 - p10))


def roda_esmda(conjunto, d_obs, g, C_D, niter, limites, rng):
    """
    RODA ESMDA
    ES-MDA com fatores de inflacao fixos, usando o nucleo da SeReMpy.

    A EnsembleSmootherMDA sorteia internamente com o gerador global do numpy,
    entao a semente e fixada aqui via np.random.seed para reprodutibilidade.

    Returns
    -------
    conjunto : array_like
        Conjunto final.
    desajuste : list of float
        Historico do desajuste medio (mesma metrica do iES-LM, Eq. 39).
    n_avaliacoes : int
    """
    np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))

    C_D_inv = np.linalg.inv(C_D)
    alpha = float(niter)  # soma de 1/alpha_i = 1

    M = conjunto.copy()
    G = g(M)
    n_aval = 1
    historico = [ieslm.desajuste_medio(d_obs, G, C_D_inv)]

    for _ in range(niter):
        M, _ = EnsembleSmootherMDA(M, d_obs, G, alpha, C_D)
        M = np.clip(M, limites[0], limites[1])
        G = g(M)
        n_aval += 1
        historico.append(ieslm.desajuste_medio(d_obs, G, C_D_inv))

    return M, historico, n_aval


def executa():
    d = dados.carrega_dados()
    Z, Time = d['Z'], d['Time']
    nm, nd = Z.shape[0], d['Snear'].shape[0]

    g, _ = forward.monta_forward(nm, d['dt'])
    C_D = VAR_ERRO * np.eye(nd)
    limites = prior.limites_fisicos(Z)

    # Conjunto a priori compartilhado pelos dois metodos
    tendencia = prior.tendencia_suave(Z)
    conjunto_inicial = prior.conjunto_prior(
        tendencia, NE, d['dt'], rng=np.random.default_rng(SEMENTE)
    )

    print('Inversao sismica acustica 1D - iES-LM x ES-MDA')
    print('conjunto: %d membros | %d amostras de poco | %d amostras sismicas'
          % (NE, nm, nd))
    print()

    # ---- ES-MDA (linha de base, regularizacao fixa) ----
    Z_mda, hist_mda, aval_mda = roda_esmda(
        conjunto_inicial, d['Snear'], g, C_D, NITER_MDA, limites,
        np.random.default_rng(SEMENTE),
    )

    # ---- iES-LM (regularizacao adaptativa) ----
    res_lm = ieslm.ieslm(
        prior=conjunto_inicial,
        d_obs=d['Snear'],
        g=g,
        C_D=C_D,
        gamma0=1.0,
        max_iter=MAX_ITER_LM,
        eta1=1e-4,
        eta2=1e-2,
        limites=limites,
        fator_ruido=FATOR_RUIDO,
        rng=np.random.default_rng(SEMENTE),
    )
    Z_lm = res_lm.conjunto

    # ---- Metricas ----
    print('%-22s %10s %10s' % ('', 'ES-MDA', 'iES-LM'))
    print('%-22s %10.4f %10.4f' % ('RMSE vs. poco', rmse(Z_mda, Z), rmse(Z_lm, Z)))
    print('%-22s %10.4f %10.4f' % ('RMSE a priori',
                                   rmse(conjunto_inicial, Z), rmse(conjunto_inicial, Z)))
    print('%-22s %10.2e %10.2e' % ('desajuste final',
                                   hist_mda[-1], min(res_lm.desajuste)))
    print('%-22s %10.4f %10.4f' % ('largura P10-P90',
                                   largura_envelope(Z_mda), largura_envelope(Z_lm)))
    print('%-22s %10d %10d' % ('avaliacoes de g', aval_mda, res_lm.n_avaliacoes))
    print('%-22s %10s %10d' % ('iteracoes', NITER_MDA, len(res_lm.desajuste) - 1))
    print()
    print('iES-LM: parada por %s (melhor iteracao: %d)'
          % (res_lm.motivo_parada, res_lm.iteracao_final))
    print('alpha adaptativo:', ' '.join('%.2e' % a for a in res_lm.alpha))
    print('rho mediano     :', ' '.join('%.3f' % r for r in res_lm.rho_mediano))

    _figuras(Z, Time, conjunto_inicial, Z_mda, Z_lm, hist_mda, res_lm)

    return {
        'Z_mda': Z_mda, 'Z_lm': Z_lm, 'prior': conjunto_inicial,
        'hist_mda': hist_mda, 'res_lm': res_lm, 'Z': Z,
    }


def _perfil(ax, conjunto, Time, Z, cor, titulo, rotulo):
    """Painel com o espalhamento do conjunto, a media e o perfil verdadeiro."""
    p10, p90 = np.percentile(conjunto, [10, 90], axis=1)
    ax.fill_betweenx(Time.ravel(), p10, p90, color=cor, alpha=0.25, label='P10-P90')
    ax.plot(conjunto.mean(axis=1), Time, color=cor, lw=2, label=rotulo)
    ax.plot(Z, Time, 'k', lw=1.5, label='poco (referencia)')
    ax.set_ylim(Time.max(), Time.min())
    ax.set_xlabel('Impedancia acustica')
    ax.set_title(titulo)
    ax.grid(alpha=0.3)


def _figuras(Z, Time, conjunto_inicial, Z_mda, Z_lm, hist_mda, res_lm):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)

    # Figura 1: perfis a priori e a posteriori dos dois metodos
    fig, eixos = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    _perfil(eixos[0], conjunto_inicial, Time, Z, 'tab:gray', 'A priori', 'media')
    _perfil(eixos[1], Z_mda, Time, Z, 'tab:blue',
            'ES-MDA (regularizacao fixa)', 'media')
    _perfil(eixos[2], Z_lm, Time, Z, 'tab:red',
            'iES-LM (regularizacao adaptativa)', 'media')
    eixos[0].set_ylabel('Tempo (s)')
    eixos[0].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'perfis.png'), dpi=150)
    plt.close(fig)

    # Figura 2: convergencia do desajuste e trajetoria de alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.semilogy(range(len(hist_mda)), hist_mda, 'o-', color='tab:blue',
                 label='ES-MDA (%d assimilacoes)' % (len(hist_mda) - 1))
    ax1.semilogy(range(len(res_lm.desajuste)), res_lm.desajuste, 's-',
                 color='tab:red', label='iES-LM (%d iteracoes)'
                 % (len(res_lm.desajuste) - 1))
    ax1.axvline(res_lm.iteracao_final, color='tab:red', ls=':', lw=1,
                label='melhor iteracao do iES-LM')
    ax1.set_xlabel('Iteracao')
    ax1.set_ylabel(r'Desajuste medio $\bar{O}$')
    ax1.set_title('Convergencia')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.semilogy(range(len(res_lm.alpha)), res_lm.alpha, 's-', color='tab:red',
                 label=r'$\alpha^i$ adaptativo (iES-LM)')
    ax2.axhline(len(hist_mda) - 1, color='tab:blue', ls='--',
                label=r'$\alpha_i$ fixo (ES-MDA)')
    ax2.set_xlabel('Iteracao')
    ax2.set_ylabel(r'Regularizacao $\alpha$')
    ax2.set_title('Regularizacao do passo: fixa x adaptativa')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIGURAS, 'convergencia.png'), dpi=150)
    plt.close(fig)

    print('\nfiguras salvas em %s' % PASTA_FIGURAS)


if __name__ == '__main__':
    executa()
