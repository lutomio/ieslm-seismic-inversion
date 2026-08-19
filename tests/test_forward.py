# -*- coding: utf-8 -*-
"""Testes do modelo direto acustico (forward.py)."""

import numpy as np
import pytest

import dados
import forward
from SeReMpy.Inversion import DifferentialMatrix, RickerWavelet


@pytest.fixture(scope='module')
def d():
    return dados.carrega_dados()


@pytest.fixture(scope='module')
def op(d):
    """Operador montado nas dimensoes reais do dado."""
    nm = d['Z'].shape[0]
    wavelet, _ = RickerWavelet(45, d['dt'], 64)
    D, W = forward.operador_acustico(nm, wavelet)
    return D, W


def test_matriz_diferencial_faz_diferenca_progressiva():
    """D @ x tem que dar x[k+1] - x[k] - conferivel a mao."""
    D = DifferentialMatrix(5, 1)
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0]).reshape(-1, 1)

    esperado = np.array([1.0, 2.0, 4.0, 8.0]).reshape(-1, 1)
    np.testing.assert_allclose(np.dot(D, x), esperado)


def test_formatos(d, op):
    """Um modelo entra (nm,1) e sai (nd,1); um conjunto (nm,ne) sai (nd,ne)."""
    D, W = op
    nm = d['Z'].shape[0]
    nd = nm - 1

    assert forward.modelo_direto(d['Z'], D, W).shape == (nd, 1)

    conjunto = np.tile(d['Z'], (1, 7))
    assert forward.modelo_direto(conjunto, D, W).shape == (nd, 7)


def test_conjunto_bate_com_laco_membro_a_membro(d, op):
    """A versao vetorizada precisa dar o mesmo que iterar membro a membro."""
    D, W = op
    rng = np.random.default_rng(0)
    conjunto = d['Z'] * np.exp(0.05 * rng.standard_normal((d['Z'].shape[0], 5)))

    vetorizado = forward.modelo_direto(conjunto, D, W)
    for j in range(conjunto.shape[1]):
        um = forward.modelo_direto(conjunto[:, [j]], D, W)
        np.testing.assert_allclose(vetorizado[:, [j]], um, atol=1e-12)


def test_refletividade_conferida_a_mao():
    """R_k = 0.5*(ln Z_{k+1} - ln Z_k) em um caso pequeno."""
    D = DifferentialMatrix(4, 1)
    Z = np.array([2.0, 4.0, 4.0, 1.0]).reshape(-1, 1)

    esperado = 0.5 * np.array(
        [np.log(4.0 / 2.0), 0.0, np.log(1.0 / 4.0)]
    ).reshape(-1, 1)
    np.testing.assert_allclose(forward.refletividade(Z, D), esperado)


def test_wavelet_grande_demais_da_erro_claro():
    """Sem a checagem, WaveletMatrix falharia com erro de broadcast obscuro."""
    wavelet, _ = RickerWavelet(45, 0.001, 64)
    with pytest.raises(ValueError, match='nao cabe'):
        forward.operador_acustico(10, wavelet)


def test_impedancia_negativa_e_rejeitada(op):
    D, _ = op
    Z = np.array([[5.0], [-1.0], [5.0]])
    with pytest.raises(ValueError, match='positiva'):
        forward.refletividade(Z, D)


def test_reproduz_o_dado_sismico_real(d, op):
    """
    Teste fisico: a sismica sintetica gerada a partir do poco verdadeiro tem
    que reproduzir o traco observado. Se este teste cair, o modelo direto
    (wavelet, convencao de sinal, alinhamento) esta errado.
    """
    D, W = op
    sintetico = forward.modelo_direto(d['Z'], D, W)

    correlacao = np.corrcoef(sintetico.ravel(), d['Snear'].ravel())[0, 1]
    assert correlacao > 0.99


def test_modelo_e_nao_linear_em_Z(d, op):
    """
    A comparacao do TCC so faz sentido num problema nao-linear: se g fosse
    linear, ES-MDA e iES-LM convergiriam para a mesma solucao.
    """
    D, W = op
    rng = np.random.default_rng(1)
    Z1 = d['Z'] * np.exp(0.05 * rng.standard_normal(d['Z'].shape))
    Z2 = d['Z'] * np.exp(0.05 * rng.standard_normal(d['Z'].shape))
    a, b = 0.3, 0.7

    g = lambda Z: forward.modelo_direto(Z, D, W)
    desvio = np.abs(g(a * Z1 + b * Z2) - (a * g(Z1) + b * g(Z2))).max()

    assert desvio / np.abs(g(Z1)).max() > 1e-4


def test_tempo_sismico(d):
    """Uma amostra a menos que o poco, nos pontos medios."""
    ts = forward.tempo_sismico(d['Time'])

    assert ts.shape == (d['Time'].shape[0] - 1, 1)
    np.testing.assert_allclose(ts, d['TimeSeis'], atol=1e-6)


def test_monta_forward_equivale_a_montagem_manual(d, op):
    D, W = op
    g, _ = forward.monta_forward(d['Z'].shape[0], d['dt'])

    np.testing.assert_allclose(g(d['Z']), forward.modelo_direto(d['Z'], D, W))
