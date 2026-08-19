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
