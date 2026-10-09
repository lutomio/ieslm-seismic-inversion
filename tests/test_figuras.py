# -*- coding: utf-8 -*-
"""Testes das figuras: janela automatica, trajetoria sem parada e exportacao."""

import contextlib
import dataclasses
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import config
import experimento_elastico as X
import forward_elastico as fe
import sensibilidade as sens

NM = 100  # amostras por propriedade nos casos sinteticos
TEMPO = np.round(np.arange(NM) * 0.001, 6).reshape(-1, 1)


def _conjuntos(desvio_vp=None, desvio_rho=None, ne=5):
    """Dois conjuntos iguais, exceto por desvios plantados na media."""
    rng = np.random.default_rng(0)
    verd = fe.empilha(*(rng.normal(1.0, 0.1, (NM, 1)) for _ in range(3)))
    A = np.repeat(verd, ne, axis=1)
    B = A.copy()
    for prop, desvio in ((0, desvio_vp), (2, desvio_rho)):
        if desvio is not None:
            inicio, fim, valor = desvio
            B[prop * NM + inicio:prop * NM + fim + 1, :] += valor
    return A, B, verd


# ---------------------------------------------------- janela de divergencia ---

def test_janela_encontra_a_divergencia_plantada():
    A, B, verd = _conjuntos(desvio_vp=(40, 52, 0.5))

    assert X.janela_divergencia(TEMPO, A, B, verd, largura=0.012) == (0.040, 0.052)


def test_janela_tem_a_largura_pedida():
    A, B, verd = _conjuntos(desvio_vp=(70, 75, 0.5))
    inicio, fim = X.janela_divergencia(TEMPO, A, B, verd, largura=0.020)

    assert fim - inicio == pytest.approx(0.020)
    assert inicio <= 0.070 and fim >= 0.075


def test_janela_maior_que_o_perfil_cobre_tudo():
    A, B, verd = _conjuntos(desvio_vp=(10, 12, 0.5))

    assert X.janela_divergencia(TEMPO, A, B, verd, largura=1.0) == (0.0, 0.099)


def test_janela_pondera_cada_propriedade_pela_sua_escala():
    """O mesmo desvio absoluto pesa mais na propriedade de menor variacao."""
    A, B, verd = _conjuntos(desvio_vp=(10, 22, 0.3), desvio_rho=(60, 72, 0.3))
    verd[:NM] *= 10.0  # Vp com desvio padrao dez vezes maior

    inicio, _ = X.janela_divergencia(TEMPO, A, B, verd, largura=0.012)
    assert inicio == 0.060


def test_janela_do_cenario_padrao_e_a_que_era_fixa():
    """A janela escolhida a mao antes desta etapa era 1,810-1,822 s."""
    c = X.carrega_cenario(config.PADRAO)
    Z_mda = X.roda_esmda(c)[0]
    Z_lm = X.roda_ieslm(c).conjunto
    inicio, fim = X.janela_divergencia(c['Time'], Z_mda, Z_lm, c['verdadeiro'])

    assert abs(inicio - 1.810) <= 0.002 and abs(fim - 1.822) <= 0.002


# ------------------------------------------------- trajetoria sem parada ---

@pytest.fixture(scope='module')
def cenario():
    return X.carrega_cenario(dataclasses.replace(config.PADRAO, ne=40, semente=3))


def test_sem_parada_e_none_quando_a_eq43_ja_esta_desligada(cenario):
    c = dict(cenario, cfg=dataclasses.replace(cenario['cfg'], fator_ruido=None))

    assert X.roda_ieslm_sem_parada(c) is None


def test_sem_parada_repete_o_inicio_e_vai_alem(cenario):
    """Mesma semente: as iteracoes em comum sao identicas."""
    com = X.roda_ieslm(cenario)
    sem = X.roda_ieslm_sem_parada(cenario)

    n = len(com.desajuste)
    np.testing.assert_array_equal(sem.desajuste[:n], com.desajuste)
    assert sem.n_avaliacoes >= com.n_avaliacoes
    assert cenario['cfg'].fator_ruido is not None  # o cenario nao foi alterado


# --------------------------------------------------------------- auxiliares ---

def test_valor_legivel():
    assert X._valor_legivel(-2157.9) == '−2158'
    assert X._valor_legivel(12.0) == '12'
    assert '10^{10}' in X._valor_legivel(8.2e10)


def test_dentro_interrompe_a_linha_fora_dos_limites():
    y = X._dentro([0.5, 3.0, -1.0, 1.0], 0.0, 2.0)

    assert np.isnan(y[1]) and np.isnan(y[2])
    assert y[0] == 0.5 and y[3] == 1.0


def test_salva_figura_grava_png_e_pdf_e_fecha(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    X.salva_figura(fig, 'teste', str(tmp_path))

    for formato in X.FORMATOS:
        assert (tmp_path / ('teste.' + formato)).stat().st_size > 0
    assert not plt.fignum_exists(fig.number)


# ------------------------------------------------------- execucao completa ---

def _executa_pequeno(cfg, pasta):
    """Roda o experimento completo sem imprimir, gravando tudo em pasta."""
    antes = X.PASTA_FIGURAS, X.PASTA_RESULTADOS
    X.PASTA_FIGURAS = X.PASTA_RESULTADOS = pasta
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return X.executa(cfg)
    finally:
        X.PASTA_FIGURAS, X.PASTA_RESULTADOS = antes


ELASTICO = ('elastico_perfis', 'elastico_detalhe', 'elastico_erro', 'elastico_ganho',
            'elastico_residuos', 'elastico_convergencia')


@pytest.mark.parametrize('fator_ruido', [4.0, None])
def test_executa_gera_todas_as_figuras_nos_dois_formatos(tmp_path, fator_ruido):
    cfg = dataclasses.replace(config.PADRAO, ne=30, fator_ruido=fator_ruido)
    r = _executa_pequeno(cfg, str(tmp_path))

    for nome in ELASTICO:
        for formato in X.FORMATOS:
            assert (tmp_path / ('%s.%s' % (nome, formato))).exists()
    assert (r['res_sem_parada'] is None) == (fator_ruido is None)


def test_figuras_de_sensibilidade_nos_dois_formatos(tmp_path, monkeypatch):
    csv = os.path.join(sens.PASTA_RESULTADOS, 'sensibilidade.csv')
    if not os.path.exists(csv):
        pytest.skip('resultados/sensibilidade.csv ainda nao foi gerado')
    monkeypatch.setattr(sens, 'PASTA_FIGURAS', str(tmp_path))
    sens._figuras(sens.carrega_csv(csv))

    nomes = {p.stem for p in tmp_path.iterdir()}
    for nome in nomes:
        for formato in X.FORMATOS:
            assert (tmp_path / ('%s.%s' % (nome, formato))).exists()
    assert {'sensibilidade_eixos', 'sensibilidade_custo'} <= nomes
    assert {'sensibilidade_cruzada_%s' % cz.nome for cz in sens.CRUZADAS} <= nomes
