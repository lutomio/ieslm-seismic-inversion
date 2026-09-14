# -*- coding: utf-8 -*-
"""Testes de infraestrutura: a SeReMpy e importavel e os dados sao os esperados."""

import numpy as np
import pytest

import dados


def test_serempy_importavel():
    """O ajuste de sys.path em dados.py torna a SeReMpy utilizavel."""
    from SeReMpy.Inversion import RickerWavelet, DifferentialMatrix, WaveletMatrix

    assert callable(RickerWavelet)
    assert callable(DifferentialMatrix)
    assert callable(WaveletMatrix)


def test_dimensoes_dos_dados():
    """A sismica tem uma amostra a menos que o poco (uma por interface)."""
    d = dados.carrega_dados()

    nm = d['Z'].shape[0]
    nd = d['Snear'].shape[0]

    assert nm == 99
    assert nd == 98
    assert nd == nm - 1


def test_passo_de_tempo():
    d = dados.carrega_dados()
    assert d['dt'] == pytest.approx(0.001, abs=1e-9)


def test_impedancia_e_positiva_e_fisica():
    """Z = Vp*Rho precisa ser positivo (o modelo direto usa log(Z))."""
    d = dados.carrega_dados()

    assert np.all(d['Z'] > 0)
    assert d['Z'].shape == (99, 1)
    # faixa fisica plausivel para este dado sintetico
    assert 5.0 < d['Z'].min() < d['Z'].max() < 15.0


def test_devolve_as_propriedades_e_os_tres_angulos():
    """Uma unica porta de entrada serve aos casos acustico e elastico."""
    d = dados.carrega_dados()

    for chave in ('Vp', 'Vs', 'Rho', 'Z', 'Time'):
        assert d[chave].shape == (99, 1)
    for chave in ('Snear', 'Smid', 'Sfar', 'TimeSeis'):
        assert d[chave].shape == (98, 1)


def test_impedancia_e_produto_de_vp_e_rho():
    d = dados.carrega_dados()
    np.testing.assert_allclose(d['Z'], d['Vp'] * d['Rho'])


def test_tempo_sismico_e_ponto_medio_do_tempo_do_poco():
    """
    Justifica tirar dt do poco: as duas malhas sao consistentes, e a sismica
    fica no ponto medio de cada par de amostras (uma por interface).
    """
    d = dados.carrega_dados()
    meio = 0.5 * (d['Time'][:-1] + d['Time'][1:])

    np.testing.assert_allclose(d['TimeSeis'], meio, atol=1e-9)


def test_propriedades_elasticas_sao_fisicamente_plausiveis():
    """Vs < Vp e densidades de rocha: pega indices de coluna trocados."""
    d = dados.carrega_dados()

    assert np.all(d['Vs'] < d['Vp'])
    assert 3.0 < d['Vp'].min() < d['Vp'].max() < 6.0
    assert 1.5 < d['Vs'].min() < d['Vs'].max() < 3.5
    assert 2.0 < d['Rho'].min() < d['Rho'].max() < 3.0
