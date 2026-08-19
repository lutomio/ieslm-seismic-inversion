# -*- coding: utf-8 -*-
"""
Testes de integracao: o iES-LM rodando no problema sismico real.

Usam conjuntos pequenos para serem rapidos; o experimento do TCC usa
conjuntos maiores.
"""

import numpy as np
import pytest

import dados
import forward
import ieslm
import prior


@pytest.fixture(scope='module')
def cenario():
    """Monta uma vez o problema completo: dados, modelo direto e a priori."""
    d = dados.carrega_dados()
    nm = d['Z'].shape[0]
    nd = d['Snear'].shape[0]

    g, _ = forward.monta_forward(nm, d['dt'])
    tendencia = prior.tendencia_suave(d['Z'])

    return {
        'd': d,
        'nm': nm,
        'nd': nd,
        'g': g,
        'tendencia': tendencia,
        'C_D': 1e-4 * np.eye(nd),
        'limites': prior.limites_fisicos(d['Z']),
    }


def roda(cenario, ne=60, semente=0, **kwargs):
    p = prior.conjunto_prior(
        cenario['tendencia'], ne, cenario['d']['dt'],
        rng=np.random.default_rng(semente),
    )
    opcoes = dict(max_iter=10, eta1=0.0, eta2=0.0, limites=cenario['limites'])
    opcoes.update(kwargs)

    return ieslm.ieslm(
        prior=p,
        d_obs=cenario['d']['Snear'],
        g=cenario['g'],
        C_D=cenario['C_D'],
        rng=np.random.default_rng(semente + 500),
        **opcoes,
    )


# --------------------------------------------------------------------------
# Conjunto a priori
# --------------------------------------------------------------------------

def test_prior_e_positivo_e_tem_o_formato_certo(cenario):
    p = prior.conjunto_prior(
        cenario['tendencia'], 30, cenario['d']['dt'],
        rng=np.random.default_rng(0),
    )

    assert p.shape == (cenario['nm'], 30)
    assert np.all(p > 0)


def test_prior_e_centrado_na_tendencia(cenario):
    p = prior.conjunto_prior(
        cenario['tendencia'], 2000, cenario['d']['dt'],
        rng=np.random.default_rng(0),
    )

    razao = p.mean(axis=1, keepdims=True) / cenario['tendencia']
    np.testing.assert_allclose(razao, 1.0, atol=0.02)


def test_tendencia_e_mais_suave_que_o_perfil(cenario):
    """A tendencia nao pode conter os detalhes que a inversao deve recuperar."""
    Z = cenario['d']['Z']
    aspereza = lambda x: float(np.std(np.diff(x, axis=0)))

    assert aspereza(cenario['tendencia']) < aspereza(Z) / 2.0


# --------------------------------------------------------------------------
# Comportamento do algoritmo
# --------------------------------------------------------------------------

def test_inverte_o_dado_sismico(cenario):
    """O desajuste tem que cair de forma substancial em relacao ao a priori."""
    res = roda(cenario)

    assert res.desajuste[-1] < res.desajuste[0]
    assert min(res.desajuste) < res.desajuste[0] / 10.0


def test_devolve_o_conjunto_de_menor_desajuste(cenario):
    """
    Secao 4.3: o iES-LM sempre atualiza o conjunto, mas reporta ao final a
    iteracao de MENOR desajuste medio - nao necessariamente a ultima.
    """
    res = roda(cenario)

    assert res.desajuste[res.iteracao_final] == pytest.approx(min(res.desajuste))


def test_alpha_nunca_cresce(cenario):
    """Eq. 41: alpha^{i+1} = min(alpha^i, mediana(alpha_j))."""
    res = roda(cenario)

    for anterior, atual in zip(res.alpha, res.alpha[1:]):
        assert atual <= anterior + 1e-12


def test_resultado_respeita_os_limites_fisicos(cenario):
    res = roda(cenario)
    minimo, maximo = cenario['limites']

    assert res.conjunto.min() >= minimo - 1e-9
    assert res.conjunto.max() <= maximo + 1e-9


def test_aproxima_o_perfil_verdadeiro(cenario):
    """A media a posteriori tem que ficar mais perto do poco que o a priori."""
    Z = cenario['d']['Z']
    res = roda(cenario)

    erro = lambda X: float(np.sqrt(np.mean((X.mean(axis=1, keepdims=True) - Z) ** 2)))
    p = prior.conjunto_prior(
        cenario['tendencia'], 60, cenario['d']['dt'],
        rng=np.random.default_rng(0),
    )

    assert erro(res.conjunto) < erro(p)


def test_reprodutibilidade(cenario):
    a = roda(cenario, semente=3)
    b = roda(cenario, semente=3)

    np.testing.assert_array_equal(a.conjunto, b.conjunto)


# --------------------------------------------------------------------------
# Criterios de parada
# --------------------------------------------------------------------------

def test_para_por_numero_maximo_de_iteracoes(cenario):
    res = roda(cenario, max_iter=3, eta1=0.0, eta2=0.0)

    assert res.motivo_parada == 'numero maximo de iteracoes'
    assert len(res.desajuste) == 4  # inicial + 3 iteracoes


def test_para_por_eta1(cenario):
    """Tolerancia enorme no desajuste encerra logo na primeira comparacao."""
    res = roda(cenario, eta1=1e9, eta2=0.0)

    assert res.motivo_parada == 'variacao do desajuste abaixo de eta1'


def test_para_por_eta2(cenario):
    """Tolerancia enorme na variacao dos parametros tambem encerra cedo."""
    res = roda(cenario, eta1=0.0, eta2=1e9)

    assert res.motivo_parada == 'variacao dos parametros abaixo de eta2'


def test_conta_avaliacoes_do_modelo_direto(cenario):
    res = roda(cenario, max_iter=4, eta1=0.0, eta2=0.0)

    assert res.n_avaliacoes == 1 + 4


# --------------------------------------------------------------------------
# Eq. 43: parada no nivel do ruido, contra sobreajuste (Secao 4.5)
# --------------------------------------------------------------------------

def test_criterio_de_ruido_encerra_no_limiar(cenario):
    res = roda(cenario, max_iter=20, fator_ruido=4.0)

    assert res.motivo_parada == 'desajuste no nivel do ruido (Eq. 43)'


def test_criterio_de_ruido_evita_o_colapso_do_conjunto(cenario):
    """
    Sem a Eq. 43, o iES-LM continua reduzindo o desajuste ate ajustar o
    proprio ruido: o conjunto colapsa e a incerteza a posteriori desaparece.
    Este teste registra o efeito que motiva o criterio.
    """
    largura = lambda X: float(np.mean(np.diff(np.percentile(X, [10, 90], axis=1), axis=0)))

    sem_criterio = roda(cenario, max_iter=20, eta1=0.0, eta2=0.0)
    com_criterio = roda(cenario, max_iter=20, eta1=0.0, eta2=0.0, fator_ruido=4.0)

    assert largura(com_criterio.conjunto) > 5 * largura(sem_criterio.conjunto)


def test_criterio_de_ruido_melhora_a_estimativa(cenario):
    """Parar no nivel do ruido tem que aproximar mais do poco que sobreajustar."""
    Z = cenario['d']['Z']
    erro = lambda X: float(np.sqrt(np.mean((X.mean(axis=1, keepdims=True) - Z) ** 2)))

    sem_criterio = roda(cenario, max_iter=20, eta1=0.0, eta2=0.0)
    com_criterio = roda(cenario, max_iter=20, eta1=0.0, eta2=0.0, fator_ruido=4.0)

    assert erro(com_criterio.conjunto) < erro(sem_criterio.conjunto)
