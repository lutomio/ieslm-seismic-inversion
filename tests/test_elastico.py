# -*- coding: utf-8 -*-
"""Testes do modelo direto elastico e das metricas de comparacao."""

import numpy as np
import pytest

import dados
import forward_elastico as fe
import metricas as mt
import prior


@pytest.fixture(scope='module')
def poco():
    d = dados.carrega_dados()
    return {
        'Time': d['Time'], 'Vp': d['Vp'], 'Vs': d['Vs'], 'Rho': d['Rho'],
        'dt': d['dt'],
        'obs': np.vstack([d['Snear'], d['Smid'], d['Sfar']]),
    }


# ---------------------------------------------------------------- forward ---

def test_empilha_desempilha_sao_inversos(poco):
    M = fe.empilha(poco['Vp'], poco['Vs'], poco['Rho'])
    Vp, Vs, Rho = fe.desempilha(M)

    assert M.shape == (3 * poco['Vp'].shape[0], 1)
    np.testing.assert_array_equal(Vp, poco['Vp'])
    np.testing.assert_array_equal(Vs, poco['Vs'])
    np.testing.assert_array_equal(Rho, poco['Rho'])


def test_forward_elastico_formatos(poco):
    g, _ = fe.monta_forward(poco['Time'], poco['dt'])
    nm = poco['Vp'].shape[0]
    M = fe.empilha(poco['Vp'], poco['Vs'], poco['Rho'])

    assert g(M).shape == (3 * (nm - 1), 1)
    assert g(np.tile(M, (1, 4))).shape == (3 * (nm - 1), 4)


def test_forward_elastico_reproduz_o_dado_do_pacote(poco):
    """
    O traco distribuido com a SeReMpy foi gerado por este mesmo operador a
    partir do poco, entao a reproducao deve ser exata. E por isso que o
    experimento adiciona ruido: sem ele, incorreria em inverse crime.
    """
    g, _ = fe.monta_forward(poco['Time'], poco['dt'])
    M = fe.empilha(poco['Vp'], poco['Vs'], poco['Rho'])

    assert np.abs(g(M) - poco['obs']).max() < 1e-6


def test_separa_angulos(poco):
    g, _ = fe.monta_forward(poco['Time'], poco['dt'])
    d = g(fe.empilha(poco['Vp'], poco['Vs'], poco['Rho']))
    blocos = fe.separa_angulos(d)

    assert len(blocos) == 3
    assert all(b.shape[0] == d.shape[0] // 3 for b in blocos)
    np.testing.assert_allclose(np.vstack(blocos), d)


def test_ruido_tem_a_intensidade_pedida():
    rng = np.random.default_rng(0)
    limpo = rng.standard_normal((3000, 1))

    ruidoso, C_D = fe.adiciona_ruido(limpo, snr=10.0, rng=rng)

    sigma_esperado = limpo.std() / 10.0
    assert (ruidoso - limpo).std() == pytest.approx(sigma_esperado, rel=0.1)
    assert np.sqrt(C_D[0, 0]) == pytest.approx(sigma_esperado, rel=1e-9)


# ---------------------------------------------------------------- prior ---

def test_prior_multivariado_formato_e_tendencia(poco):
    tend = np.hstack([prior.tendencia_suave(poco[k]) for k in ('Vp', 'Vs', 'Rho')])
    sigma0 = np.cov(np.hstack([poco['Vp'], poco['Vs'], poco['Rho']]).T)
    p = prior.conjunto_prior_multivariado(
        tend, 800, poco['dt'], sigma0, rng=np.random.default_rng(0)
    )

    nm = poco['Vp'].shape[0]
    assert p.shape == (3 * nm, 800)
    for i, bloco in enumerate(fe.desempilha(p)):
        np.testing.assert_allclose(bloco.mean(axis=1), tend[:, i], atol=0.05)


def test_prior_multivariado_preserva_correlacao_entre_propriedades(poco):
    """A covariancia entre Vp, Vs e rho do conjunto deve seguir sigma0."""
    tend = np.hstack([prior.tendencia_suave(poco[k]) for k in ('Vp', 'Vs', 'Rho')])
    sigma0 = np.cov(np.hstack([poco['Vp'], poco['Vs'], poco['Rho']]).T)
    p = prior.conjunto_prior_multivariado(
        tend, 4000, poco['dt'], sigma0, rng=np.random.default_rng(1)
    )

    Vp, Vs, Rho = fe.desempilha(p)
    k = 40  # uma amostra qualquer
    amostral = np.cov(np.vstack([Vp[k], Vs[k], Rho[k]]))

    np.testing.assert_allclose(amostral, sigma0, rtol=0.15, atol=0.005)


# ---------------------------------------------------------------- metricas ---

def test_cobertura_ideal_e_oitenta_por_cento():
    """
    Conjunto bem calibrado cobre ~80% com envelope P10-P90.

    Para isso o modelo verdadeiro precisa ser mais um sorteio da distribuicao
    que o conjunto representa - e nao o centro dela, caso em que a cobertura
    seria de 100% e nao mediria calibracao alguma.
    """
    rng = np.random.default_rng(0)
    media = rng.standard_normal((4000, 1))
    verdadeiro = media + rng.standard_normal((4000, 1))
    conjunto = media + rng.standard_normal((4000, 400))

    assert mt.taxa_cobertura(conjunto, verdadeiro) == pytest.approx(0.8, abs=0.02)


def test_cobertura_detecta_conjunto_colapsado():
    """Envelope estreito demais em torno do valor errado: cobertura ~0."""
    verdadeiro = np.ones((100, 1))
    colapsado = np.full((100, 50), 5.0) + 1e-6 * np.random.default_rng(0).standard_normal((100, 50))

    assert mt.taxa_cobertura(colapsado, verdadeiro) == 0.0


def test_largura_envelope():
    """Distribuicao uniforme em [0,1]: P10-P90 vale 0,8."""
    conjunto = np.random.default_rng(0).uniform(0, 1, size=(50, 20000))

    assert mt.largura_envelope(conjunto) == pytest.approx(0.8, abs=0.01)


def test_ks_nao_distingue_amostras_da_mesma_distribuicao():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 30))
    b = rng.standard_normal((200, 1))

    stat, p = mt.teste_ks(a, b)
    assert p > 0.05


def test_ks_distingue_distribuicoes_diferentes():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 30))
    b = rng.standard_normal((200, 1)) + 5.0

    stat, p = mt.teste_ks(a, b)
    assert p < 1e-6
    assert stat > 0.9


def test_rmse_zero_no_ajuste_perfeito():
    v = np.arange(10.0).reshape(-1, 1)

    assert mt.rmse(np.tile(v, (1, 7)), v) == pytest.approx(0.0)


@pytest.mark.parametrize('na', [2, 4, 6, 8])
def test_sequencia_alpha_satisfaz_a_condicao_de_consistencia(na):
    """Emerick e Reynolds (2013): sum(1/alpha_l) = 1."""
    a = mt.sequencia_alpha_esmda(na)

    assert len(a) == na
    assert np.sum(1.0 / a) == pytest.approx(1.0)


def test_sequencia_alpha_e_decrescente():
    a = mt.sequencia_alpha_esmda(4)

    assert np.all(np.diff(a) < 0)
