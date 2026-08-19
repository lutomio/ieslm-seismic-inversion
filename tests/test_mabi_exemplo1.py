# -*- coding: utf-8 -*-
"""
Validacao do nucleo contra o Exemplo 1 de Ma e Bi (2019), Secao 5.1.

Este e o portao de qualidade da implementacao: um benchmark publicado, com
numeros explicitos no artigo. Modelo linear de um parametro

    (y_m)_i = m * t_i + eps_i,    eps_i ~ N(0, sigma^2),  sigma^2 = 4

com t = [1, 2, 3] e observacoes y = [3.012, 10.744, 14.072]. A solucao de
maxima verossimilhanca tem forma fechada (Eq. 46):

    m_MLE = sum(t_i * y_i) / sum(t_i^2) = 66.716 / 14 = 4.76543

O artigo obtem, com 5 iteracoes: 4.76552 (Ne=10), 4.76528 (Ne=100) e
4.7654 (Ne=500) - Fig. 1.

Observacao conceitual: o iES-LM resolve um problema de MAXIMA
VEROSSIMILHANCA (Eq. 10, sem termo de prior), e nao de posterior bayesiano.
Por isso o oraculo correto aqui e o MLE analitico, e nao a media a
posteriori - esta ultima seria o alvo do ES-MDA.
"""

import numpy as np
import pytest

import ieslm


T = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
Y = np.array([3.012, 10.744, 14.072]).reshape(-1, 1)
VARIANCIA = 4.0
C_D = VARIANCIA * np.eye(3)

MLE_ANALITICO = float(np.sum(T * Y) / np.sum(T ** 2))


def modelo_linear(M):
    """g(m) = m * t, para um conjunto M de forma (1, ne) -> (3, ne)."""
    return T @ M


def conjunto_inicial(ne, semente):
    """Conjunto a priori uniforme em [1, 10], como no artigo."""
    rng = np.random.default_rng(semente)
    return rng.uniform(1.0, 10.0, size=(1, ne))


def roda(ne, semente=0, max_iter=5):
    """Roda o iES-LM no exemplo, sem parada antecipada (eta1 = eta2 = 0)."""
    return ieslm.ieslm(
        prior=conjunto_inicial(ne, semente),
        d_obs=Y,
        g=modelo_linear,
        C_D=C_D,
        gamma0=1.0,
        max_iter=max_iter,
        eta1=0.0,
        eta2=0.0,
        rng=np.random.default_rng(semente + 1000),
    )


def test_mle_analitico_confere_com_o_artigo():
    """Sanidade do proprio benchmark: 66.716 / 14 = 4.76543."""
    assert MLE_ANALITICO == pytest.approx(4.76543, abs=1e-5)


@pytest.mark.parametrize('ne', [10, 100, 500])
def test_converge_para_o_mle_publicado(ne):
    """
    ★ TESTE ANCORA: se este falhar, o nucleo do iES-LM esta errado.

    A media do conjunto final tem que cair sobre o MLE analitico, como nas
    tres colunas da Fig. 1 do artigo.
    """
    res = roda(ne)
    media = float(np.mean(res.conjunto))

    assert media == pytest.approx(MLE_ANALITICO, abs=1e-2)


def test_rho_vale_um_no_caso_linear():
    """
    ★ Com modelo linear a linearizacao do iES-LM e EXATA: as covariancias do
    conjunto satisfazem C_MD = C_MM A^T e C_DD = A C_MM A^T sem aproximacao.
    Logo a reducao prevista do objetivo iguala a reducao real e rho = 1.

    Este teste e o que confirma o fator 1/2 da Eq. 38 (ver Notes em
    ieslm.ieslm): sem ele, rho daria sistematicamente diferente de 1.
    """
    res = roda(ne=200)

    assert len(res.rho_mediano) > 0
    for rho in res.rho_mediano:
        assert rho == pytest.approx(1.0, abs=1e-6)


def test_desajuste_diminui():
    """O ajuste aos dados tem que melhorar em relacao ao conjunto a priori."""
    res = roda(ne=100)

    assert res.desajuste[-1] < res.desajuste[0]
    assert min(res.desajuste) == pytest.approx(res.desajuste[res.iteracao_final])


def test_conjunto_colapsa_como_o_artigo_descreve():
    """
    Secao 5.1: "after 5 iterations, all ensembles converge to a very narrow
    range", porque o iES-LM resolve MLE e nao amostra o posterior.
    """
    res = roda(ne=100)

    espalhamento_inicial = float(np.std(conjunto_inicial(100, 0)))
    espalhamento_final = float(np.std(res.conjunto))

    assert espalhamento_final < espalhamento_inicial / 10.0


def test_alpha_nunca_cresce():
    """Eq. 41: alpha^{i+1} = min(alpha^i, mediana(alpha_j))."""
    res = roda(ne=100)

    for anterior, atual in zip(res.alpha, res.alpha[1:]):
        assert atual <= anterior + 1e-12


def test_reprodutibilidade_por_semente():
    """Mesma semente tem que dar exatamente o mesmo resultado."""
    a = roda(ne=50, semente=7)
    b = roda(ne=50, semente=7)

    np.testing.assert_array_equal(a.conjunto, b.conjunto)
    assert a.desajuste == b.desajuste


def test_sementes_diferentes_dao_resultados_diferentes():
    a = roda(ne=50, semente=7)
    b = roda(ne=50, semente=8)

    assert not np.array_equal(a.conjunto, b.conjunto)


def test_conta_avaliacoes_do_modelo_direto():
    """1 avaliacao inicial + 1 por iteracao."""
    res = roda(ne=20, max_iter=5)

    assert res.n_avaliacoes == 1 + 5
