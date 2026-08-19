# -*- coding: utf-8 -*-
"""
Testes de unidade das regras do iES-LM.

Valores conferiveis a mao a partir das equacoes de Ma e Bi (2019).
"""

import numpy as np
import pytest

import ieslm


# --------------------------------------------------------------------------
# Eq. 40: fator max(1/3, 1 - (2*rho - 1)^3)
# --------------------------------------------------------------------------

def test_fator_lm_passo_perfeito():
    """rho = 1 -> 1 - 1^3 = 0 -> max(1/3, 0) = 1/3 (reduz o amortecimento)."""
    assert ieslm.fator_lm(1.0) == pytest.approx(1.0 / 3.0)


def test_fator_lm_passo_neutro():
    """rho = 0.5 -> 1 - 0^3 = 1 -> gamma nao muda."""
    assert ieslm.fator_lm(0.5) == pytest.approx(1.0)


def test_fator_lm_passo_ruim():
    """rho = 0 -> 1 - (-1)^3 = 2 -> dobra o amortecimento."""
    assert ieslm.fator_lm(0.0) == pytest.approx(2.0)


def test_fator_lm_nunca_reduz_abaixo_de_um_terco():
    """O max(1/3, .) limita o quanto gamma pode cair de uma vez."""
    assert ieslm.fator_lm(5.0) == pytest.approx(1.0 / 3.0)
    assert ieslm.fator_lm(1e6) == pytest.approx(1.0 / 3.0)


def test_fator_lm_cresce_quando_o_passo_piora():
    """rho negativo (objetivo aumentou) -> amortecimento ainda maior."""
    assert ieslm.fator_lm(-1.0) > 2.0


def test_fator_lm_vetorizado():
    fatores = ieslm.fator_lm([1.0, 0.5, 0.0])
    np.testing.assert_allclose(fatores, [1.0 / 3.0, 1.0, 2.0])


# --------------------------------------------------------------------------
# Eq. 39: desajuste medio normalizado
# --------------------------------------------------------------------------

def test_desajuste_medio_normaliza_por_2nd():
    """Com C_D = I e residuo unitario, O_barra = (nd * 1) / (2*nd) = 0.5."""
    nd, ne = 4, 3
    d_obs = np.zeros((nd, 1))
    G = np.ones((nd, ne))

    assert ieslm.desajuste_medio(d_obs, G, np.eye(nd)) == pytest.approx(0.5)


def test_desajuste_medio_zero_no_ajuste_perfeito():
    d_obs = np.array([[1.0], [2.0]])
    G = np.tile(d_obs, (1, 5))

    assert ieslm.desajuste_medio(d_obs, G, np.eye(2)) == pytest.approx(0.0)


def test_desajuste_medio_pondera_pela_covariancia():
    """Erro maior esperado (C_D maior) tem que reduzir o desajuste."""
    d_obs = np.zeros((3, 1))
    G = np.ones((3, 2))

    apertado = ieslm.desajuste_medio(d_obs, G, np.linalg.inv(np.eye(3)))
    frouxo = ieslm.desajuste_medio(d_obs, G, np.linalg.inv(4.0 * np.eye(3)))

    assert frouxo == pytest.approx(apertado / 4.0)


# --------------------------------------------------------------------------
# Eq. 42: desajuste absoluto (base do criterio de parada por ruido)
# --------------------------------------------------------------------------

def test_desajuste_absoluto_relaciona_se_com_o_normalizado():
    """R = 2 * nd * O_barra, por construcao das Eqs. 39 e 42."""
    rng = np.random.default_rng(0)
    nd = 6
    d_obs = rng.standard_normal((nd, 1))
    G = rng.standard_normal((nd, 10))
    C_D_inv = np.eye(nd)

    R = ieslm.desajuste_absoluto(d_obs, G, C_D_inv)
    O_barra = ieslm.desajuste_medio(d_obs, G, C_D_inv)

    assert R == pytest.approx(2.0 * nd * O_barra)


def test_desajuste_absoluto_no_nivel_de_ruido():
    """
    Se o residuo for do tamanho do desvio esperado, R ~ nd - bem abaixo do
    limiar 4*nd da Eq. 43, que e portanto uma parada folgada e nao apertada.
    """
    rng = np.random.default_rng(1)
    nd, ne = 200, 50
    C_D = 0.25 * np.eye(nd)
    residuo = np.sqrt(0.25) * rng.standard_normal((nd, ne))

    R = ieslm.desajuste_absoluto(np.zeros((nd, 1)), -residuo, np.linalg.inv(C_D))

    assert R == pytest.approx(nd, rel=0.15)
    assert R < 4 * nd


# --------------------------------------------------------------------------
# Eq. 34: objetivo por membro (dado perturbado)
# --------------------------------------------------------------------------

def test_objetivo_por_membro_conferido_a_mao():
    """O_j = 0.5 * ||r_j||^2 com C_D = I."""
    d_pert = np.array([[1.0, 0.0], [0.0, 2.0]])
    G = np.zeros((2, 2))

    obj = ieslm.objetivo_por_membro(d_pert, G, np.eye(2))

    np.testing.assert_allclose(obj, [0.5, 2.0])


def test_objetivo_por_membro_tem_um_valor_por_membro():
    ne = 7
    d_pert = np.zeros((3, ne))
    G = np.ones((3, ne))

    assert ieslm.objetivo_por_membro(d_pert, G, np.eye(3)).shape == (ne,)


# --------------------------------------------------------------------------
# Eqs. 30-31: covariancias pelo conjunto
# --------------------------------------------------------------------------

def test_covariancias_formatos_e_simetria():
    rng = np.random.default_rng(0)
    nm, nd, ne = 5, 3, 40
    M = rng.standard_normal((nm, ne))
    G = rng.standard_normal((nd, ne))

    C_MD, C_DD = ieslm.covariancias(M, G)

    assert C_MD.shape == (nm, nd)
    assert C_DD.shape == (nd, nd)
    np.testing.assert_allclose(C_DD, C_DD.T, atol=1e-12)


def test_covariancia_dos_dados_bate_com_numpy():
    rng = np.random.default_rng(1)
    G = rng.standard_normal((3, 50))
    _, C_DD = ieslm.covariancias(np.zeros((2, 50)), G)

    np.testing.assert_allclose(C_DD, np.cov(G), atol=1e-12)


def test_covariancia_cruzada_captura_relacao_linear():
    """Com G = A M, deve valer C_MD = C_MM A^T."""
    rng = np.random.default_rng(2)
    M = rng.standard_normal((4, 200))
    A = rng.standard_normal((3, 4))
    G = A @ M

    C_MD, _ = ieslm.covariancias(M, G)
    C_MM, _ = ieslm.covariancias(M, M)

    np.testing.assert_allclose(C_MD, C_MM @ A.T, atol=1e-10)


# --------------------------------------------------------------------------
# Perturbacao da observacao: xi ~ N(0, alpha * C_D)
# --------------------------------------------------------------------------

def test_perturbacao_tem_a_covariancia_pedida():
    """Covariancia amostral do ruido tem que se aproximar de alpha * C_D."""
    rng = np.random.default_rng(42)
    C_D = np.diag([1.0, 4.0, 0.25])
    alpha = 3.0
    d_obs = np.zeros((3, 1))

    d_pert = ieslm.perturba_observacao(
        d_obs, alpha, ieslm._raiz_covariancia(C_D), 200000, rng
    )

    # atol cobre os zeros fora da diagonal, onde tolerancia relativa nao serve
    np.testing.assert_allclose(np.cov(d_pert), alpha * C_D, rtol=0.05, atol=0.05)


def test_perturbacao_e_centrada_na_observacao():
    rng = np.random.default_rng(3)
    d_obs = np.array([[5.0], [-2.0]])
    C_D = np.eye(2)

    d_pert = ieslm.perturba_observacao(
        d_obs, 1.0, ieslm._raiz_covariancia(C_D), 100000, rng
    )

    np.testing.assert_allclose(d_pert.mean(axis=1, keepdims=True), d_obs, atol=0.02)


def test_perturbacao_funciona_com_covariancia_correlacionada():
    """Cholesky precisa dar ruido correlacionado, nao so escalado."""
    rng = np.random.default_rng(4)
    C_D = np.array([[1.0, 0.8], [0.8, 1.0]])

    d_pert = ieslm.perturba_observacao(
        np.zeros((2, 1)), 1.0, ieslm._raiz_covariancia(C_D), 200000, rng
    )

    np.testing.assert_allclose(np.cov(d_pert), C_D, atol=0.02)
