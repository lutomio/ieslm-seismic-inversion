# -*- coding: utf-8 -*-
"""Testes da configuracao centralizada."""

import dataclasses
import os

import numpy as np
import pytest

import config
import experimento_elastico as X
import forward_elastico as fe
from config import Configuracao, PADRAO


# ---------------------------------------------------------------- valores ---

def test_padrao_reproduz_as_entradas_que_existiam_antes():
    """Os valores antes espalhados pelo codigo, agora num so lugar."""
    assert PADRAO.ne == 200
    assert PADRAO.semente == 42
    assert PADRAO.snr == 10.0
    assert PADRAO.freq_wavelet == 45.0
    assert PADRAO.amostras_wavelet == 64
    assert PADRAO.angulos == (15.0, 30.0, 45.0)
    assert PADRAO.comprimento_correlacao == 5.0
    assert (PADRAO.ordem_tendencia, PADRAO.corte_tendencia) == (3, 0.04)
    assert (PADRAO.folga_inferior, PADRAO.folga_superior) == (0.7, 1.3)
    assert (PADRAO.n_assimilacoes, PADRAO.razao_alpha) == (4, 2.0)
    assert (PADRAO.gamma0, PADRAO.max_iter) == (1.0, 10)
    assert (PADRAO.eta1, PADRAO.eta2, PADRAO.fator_ruido) == (1e-4, 1e-2, 4.0)
    assert PADRAO.alvo == 'poço'


def test_angulos_padrao_coincidem_com_os_do_modelo_direto():
    np.testing.assert_allclose(PADRAO.angulos, fe.ANGULOS)


# ------------------------------------------------------------ imutabilidade ---

def test_configuracao_e_imutavel():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PADRAO.snr = 20.0


def test_replace_muda_so_o_campo_pedido():
    nova = dataclasses.replace(PADRAO, snr=20.0)

    assert nova.snr == 20.0
    assert PADRAO.snr == 10.0  # o padrao nao foi tocado
    for campo in dataclasses.fields(Configuracao):
        if campo.name != 'snr':
            assert getattr(nova, campo.name) == getattr(PADRAO, campo.name)


def test_angulos_em_lista_viram_tupla():
    cfg = Configuracao(angulos=[10, 20])

    assert cfg.angulos == (10.0, 20.0)
    assert hash(cfg)  # imutavel de fato, portanto utilizavel como chave


# ---------------------------------------------------------------- validacao ---

@pytest.mark.parametrize('campo, valor, trecho', [
    ('ne', 1, 'ne precisa'),
    ('snr', 0, 'snr precisa'),
    ('angulos', (), 'ao menos um angulo'),
    ('angulos', (15, 95), r'\[0, 90\)'),
    ('comprimento_correlacao', -1, 'comprimento_correlacao'),
    ('corte_tendencia', 1.5, 'corte_tendencia'),
    ('folga_inferior', 1.2, 'folga_inferior'),
    ('folga_superior', 0.9, 'folga_superior'),
    ('n_assimilacoes', 0, 'n_assimilacoes'),
    ('gamma0', 0, 'gamma0'),
    ('max_iter', 0, 'max_iter'),
    ('fator_ruido', -4, 'fator_ruido'),
    ('alvo', 'marte', 'alvo desconhecido'),
])
def test_valores_invalidos_sao_rejeitados(campo, valor, trecho):
    with pytest.raises(ValueError, match=trecho):
        dataclasses.replace(PADRAO, **{campo: valor})


def test_relata_todos_os_erros_de_uma_vez():
    with pytest.raises(ValueError) as erro:
        dataclasses.replace(PADRAO, ne=1, snr=-1)

    assert 'ne precisa' in str(erro.value) and 'snr precisa' in str(erro.value)


def test_fator_ruido_none_desliga_a_parada():
    assert dataclasses.replace(PADRAO, fator_ruido=None).fator_ruido is None


# --------------------------------------------------------------------- json ---

def test_ida_e_volta_pelo_json(tmp_path):
    cfg = dataclasses.replace(PADRAO, snr=7.5, angulos=(10, 40), fator_ruido=None,
                              alvo='6 camadas')
    caminho = os.path.join(tmp_path, 'cfg.json')

    config.salva(cfg, caminho, observacao='teste')

    assert config.carrega(caminho) == cfg


def test_json_antigo_sem_campos_novos_continua_legivel():
    """Campos ausentes assumem o padrao."""
    assert config.de_dict({'snr': 20.0}) == dataclasses.replace(PADRAO, snr=20.0)


def test_campo_desconhecido_no_json_e_rejeitado():
    with pytest.raises(ValueError, match='desconhecidos'):
        config.de_dict({'snrr': 20.0})


# --------------------------------------------- o experimento respeita a cfg ---

def test_cenario_guarda_a_configuracao():
    c = X.carrega_cenario()

    assert c['cfg'] == PADRAO


def test_ne_define_o_tamanho_do_conjunto():
    c = X.carrega_cenario(dataclasses.replace(PADRAO, ne=30))

    assert c['prior'].shape[1] == 30


def test_snr_define_a_intensidade_do_ruido():
    fraco = X.carrega_cenario(dataclasses.replace(PADRAO, snr=50.0))
    forte = X.carrega_cenario(dataclasses.replace(PADRAO, snr=2.0))

    assert forte['C_D'][0, 0] == pytest.approx(fraco['C_D'][0, 0] * 25 ** 2)


def test_angulos_definem_o_tamanho_do_dado():
    c = X.carrega_cenario(dataclasses.replace(PADRAO, angulos=(10, 25)))

    assert c['d_obs'].shape == (2 * (99 - 1), 1)


def test_folgas_definem_os_limites_fisicos():
    c = X.carrega_cenario(dataclasses.replace(PADRAO, folga_inferior=0.5,
                                              folga_superior=2.0))
    Vp = fe.desempilha(c['verdadeiro'])[0]

    assert c['limites'][0][0, 0] == pytest.approx(0.5 * Vp.min())
    assert c['limites'][1][0, 0] == pytest.approx(2.0 * Vp.max())


def test_alvo_define_o_modelo_de_referencia():
    poco = X.carrega_cenario()
    camadas = X.carrega_cenario(dataclasses.replace(PADRAO, alvo='6 camadas'))

    assert not np.array_equal(poco['verdadeiro'], camadas['verdadeiro'])


def test_n_assimilacoes_define_o_custo_do_esmda():
    c = X.carrega_cenario(dataclasses.replace(PADRAO, ne=20, n_assimilacoes=2))
    _, _, n_aval, alphas = X.roda_esmda(c)

    assert n_aval == 2 + 1
    assert np.sum(1.0 / alphas) == pytest.approx(1.0)


def test_max_iter_limita_o_ieslm():
    c = X.carrega_cenario(dataclasses.replace(PADRAO, ne=20, max_iter=1,
                                              fator_ruido=None, eta1=0, eta2=0))
    res = X.roda_ieslm(c)

    assert len(res.desajuste) - 1 == 1


def test_mesma_configuracao_mesmo_resultado():
    cfg = dataclasses.replace(PADRAO, ne=20)
    a = X.roda_ieslm(X.carrega_cenario(cfg))
    b = X.roda_ieslm(X.carrega_cenario(cfg))

    np.testing.assert_array_equal(a.conjunto, b.conjunto)


def test_chamada_antiga_com_modelo_posicional_falha_com_mensagem_clara():
    """Antes da Configuracao, carrega_cenario(modelo) era a forma de uso."""
    import dados
    with pytest.raises(TypeError, match='modelo='):
        X.carrega_cenario(dados.carrega_dados())
