# -*- coding: utf-8 -*-
"""Testes dos modelos de referencia sinteticos."""

import numpy as np
import pytest

import config
import dados
import experimento_elastico as X
import forward_elastico as fe
import referencia as ref


def modelo(**kwargs):
    kwargs.setdefault('rng', np.random.default_rng(0))
    return ref.modelo_em_camadas(**kwargs)


# ------------------------------------------------------------------ formato ---

def test_tem_as_mesmas_chaves_do_poco_real():
    """Os dois devem ser intercambiaveis pelo experimento."""
    sintetico = modelo()
    real = dados.carrega_dados()

    assert set(sintetico) <= set(real)
    for chave in ('Time', 'Vp', 'Vs', 'Rho', 'Z'):
        assert sintetico[chave].shape == (99, 1)


def test_tempo_uniforme_e_coerente_com_dt():
    m = modelo(nm=60, dt=0.002, t0=1.5)

    assert m['Time'][0, 0] == pytest.approx(1.5)
    np.testing.assert_allclose(np.diff(m['Time'], axis=0), 0.002)
    assert m['dt'] == pytest.approx(0.002)


# ------------------------------------------------------------------- fisica ---

def test_propriedades_sao_fisicamente_plausiveis():
    m = modelo()

    assert np.all(m['Vp'] > 0) and np.all(m['Vs'] > 0) and np.all(m['Rho'] > 0)
    assert np.all(m['Vs'] < m['Vp'])
    np.testing.assert_allclose(m['Z'], m['Vp'] * m['Rho'])


def test_relacoes_calibradas_sao_respeitadas():
    """Vs vem da razao Vp/Vs e rho da relacao de Gardner."""
    m = modelo()

    np.testing.assert_allclose(m['Vp'] / m['Vs'], ref.RAZAO_VP_VS)
    np.testing.assert_allclose(m['Rho'], ref.GARDNER_A * m['Vp'] ** 0.25)


def test_fica_na_mesma_escala_do_poco_real():
    """Sem isso, o conjunto a priori e os limites fisicos ficariam desajustados."""
    m = modelo()
    real = dados.carrega_dados()

    for chave in ('Vp', 'Vs', 'Rho'):
        assert abs(m[chave].mean() - real[chave].mean()) < 0.25 * real[chave].mean()


# ------------------------------------------------------------------ camadas ---

@pytest.mark.parametrize('n', [2, 4, 6, 10])
def test_produz_o_numero_de_camadas_pedido(n):
    m = modelo(n_camadas=n)

    assert len(np.unique(m['Vp'])) == n


def test_respeita_a_espessura_minima():
    """Camadas finas demais nao seriam resolvidas pela wavelet."""
    m = modelo(n_camadas=10, espessura_minima=6)
    bordas = np.flatnonzero(np.diff(np.squeeze(m['Vp'])) != 0)
    espessuras = np.diff(np.concatenate([[-1], bordas, [len(m['Vp']) - 1]]))

    assert espessuras.min() >= 6


def test_camadas_demais_nao_cabem():
    with pytest.raises(ValueError, match='nao cabem'):
        modelo(nm=30, n_camadas=10, espessura_minima=6)


def test_contraste_controla_a_variacao():
    fraco = modelo(contraste=0.02, n_camadas=8)
    forte = modelo(contraste=0.12, n_camadas=8)

    assert forte['Vp'].std() > 3 * fraco['Vp'].std()


# ----------------------------------------------------------- reprodutibilidade ---

def test_mesma_semente_mesmo_modelo():
    a = ref.modelo_em_camadas(rng=np.random.default_rng(7))
    b = ref.modelo_em_camadas(rng=np.random.default_rng(7))

    np.testing.assert_array_equal(a['Vp'], b['Vp'])


def test_sementes_diferentes_modelos_diferentes():
    a = ref.modelo_em_camadas(rng=np.random.default_rng(7))
    b = ref.modelo_em_camadas(rng=np.random.default_rng(8))

    assert not np.array_equal(a['Vp'], b['Vp'])


# ----------------------------------------------------------------- suaviza ---

def test_suaviza_reduz_a_aspereza_preservando_a_escala():
    m = modelo()
    s = ref.suaviza(m)
    aspereza = lambda x: float(np.std(np.diff(x, axis=0)))

    assert aspereza(s['Vp']) < aspereza(m['Vp']) / 2
    assert abs(s['Vp'].mean() - m['Vp'].mean()) < 0.05 * m['Vp'].mean()


def test_suaviza_preserva_as_relacoes_fisicas():
    s = ref.suaviza(modelo())

    np.testing.assert_allclose(s['Vp'] / s['Vs'], ref.RAZAO_VP_VS, rtol=1e-6)
    np.testing.assert_allclose(s['Z'], s['Vp'] * s['Rho'])


# --------------------------------------------------------------- integracao ---

def test_o_experimento_aceita_um_modelo_sintetico():
    """A observacao passa a ser gerada a partir do modelo sintetico."""
    m = modelo()
    c = X.carrega_cenario(modelo=m)

    np.testing.assert_array_equal(c['verdadeiro'], fe.empilha(m['Vp'], m['Vs'], m['Rho']))
    assert c['d_obs'].shape == (3 * (99 - 1), 1)
    assert c['prior'].shape == (3 * 99, config.PADRAO.ne)


def test_sem_modelo_usa_o_poco_do_pacote():
    real = dados.carrega_dados()
    c = X.carrega_cenario()

    np.testing.assert_array_equal(c['verdadeiro'],
                                  fe.empilha(real['Vp'], real['Vs'], real['Rho']))


def test_alvo_por_nome_devolve_modelos_validos():
    for nome in ref.ALVOS:
        m = ref.alvo_por_nome(nome, semente=0)
        if nome == 'poço':
            assert m is None  # sinaliza para usar o perfil do pacote
        else:
            assert np.all(m['Vs'] < m['Vp'])
            assert m['Vp'].shape == (99, 1)


def test_alvo_desconhecido_e_rejeitado():
    with pytest.raises(ValueError, match='desconhecido'):
        ref.alvo_por_nome('inexistente', semente=0)


def test_alvo_sintetico_independe_da_semente_do_prior():
    """O alvo usa 1000 + semente, para nao coincidir com o sorteio do a priori."""
    a = ref.alvo_por_nome('6 camadas', semente=0)
    b = ref.modelo_em_camadas(n_camadas=6, rng=np.random.default_rng(1000))

    np.testing.assert_array_equal(a['Vp'], b['Vp'])
