# -*- coding: utf-8 -*-
"""
Teste de regressao do experimento comparativo.

Nao verifica uma formula, e sim que o resultado que vai para o TCC continua
valendo: se uma mudanca futura degradar a inversao, isto falha.
"""

import numpy as np
import pytest

import experimento


@pytest.fixture(scope='module')
def r():
    return experimento.executa()


def test_ambos_os_metodos_melhoram_o_a_priori(r):
    erro_prior = experimento.rmse(r['prior'], r['Z'])

    assert experimento.rmse(r['Z_mda'], r['Z']) < erro_prior
    assert experimento.rmse(r['Z_lm'], r['Z']) < erro_prior


def test_qualidade_da_inversao_do_ieslm(r):
    """Limiar de regressao: RMSE do iES-LM contra o poco."""
    assert experimento.rmse(r['Z_lm'], r['Z']) < 0.55


def test_ieslm_e_competitivo_com_o_esmda(r):
    """
    O resultado do TCC: com a parada no nivel do ruido, o iES-LM alcanca
    qualidade comparavel a do ES-MDA.
    """
    erro_lm = experimento.rmse(r['Z_lm'], r['Z'])
    erro_mda = experimento.rmse(r['Z_mda'], r['Z'])

    assert erro_lm <= erro_mda * 1.05


def test_ieslm_custa_menos_avaliacoes_do_modelo_direto(r):
    """
    A regularizacao adaptativa chega ao nivel do ruido em menos passos que as
    quatro assimilacoes fixas do ES-MDA.
    """
    assert r['res_lm'].n_avaliacoes < 1 + experimento.NITER_MDA


def test_ieslm_preserva_incerteza(r):
    """O conjunto final nao pode ter colapsado."""
    largura_lm = experimento.largura_envelope(r['Z_lm'])
    largura_mda = experimento.largura_envelope(r['Z_mda'])

    assert largura_lm > 0.5 * largura_mda


def test_regularizacao_adaptativa_de_fato_variou(r):
    """
    O ponto central do TCC: alpha do iES-LM muda ao longo das iteracoes,
    enquanto o do ES-MDA seria constante.
    """
    alphas = r['res_lm'].alpha

    assert len(set(alphas)) > 1
    assert min(alphas) < max(alphas) / 10.0


def test_figuras_foram_geradas(r):
    import os

    for nome in ('perfis.png', 'convergencia.png'):
        caminho = os.path.join(experimento.PASTA_FIGURAS, nome)
        assert os.path.exists(caminho)
        assert os.path.getsize(caminho) > 1000
