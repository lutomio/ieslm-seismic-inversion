# -*- coding: utf-8 -*-
"""Testes do estudo de sensibilidade."""

import numpy as np
import pytest

import config
import sensibilidade as sens


def test_grade_varia_um_eixo_por_vez():
    """Cada configuracao difere do ponto de referencia em no maximo um fator."""
    configs, _ = sens.grade()
    referencia = (config.PADRAO.ne, config.PADRAO.snr, 'poço', config.PADRAO.n_assimilacoes)

    for c in configs:
        diferencas = sum(1 for a, b in zip(c, referencia) if a != b)
        assert diferencas <= 1


def test_grade_nao_repete_a_configuracao_central():
    configs, _ = sens.grade()

    assert len(configs) == len(set(configs))
    assert (config.PADRAO.ne, config.PADRAO.snr, 'poço',
            config.PADRAO.n_assimilacoes) in configs


def test_avalia_devolve_as_metricas_dos_dois_metodos():
    linha = sens.avalia(ne=20, snr=10.0, alvo='poço', semente=0)

    for sufixo in ('mda', 'lm'):
        for nome in ('Vp', 'Vs', 'rho'):
            assert 0 < linha['rmse_%s_%s' % (nome, sufixo)] < 10
            assert 0 <= linha['cob_%s_%s' % (nome, sufixo)] <= 1
    assert linha['ne'] == 20 and linha['alvo'] == 'poço'


def test_avalia_e_reprodutivel():
    a = sens.avalia(ne=20, snr=10.0, alvo='poço', semente=3)
    b = sens.avalia(ne=20, snr=10.0, alvo='poço', semente=3)

    assert a == b


def test_comparacao_pareada_detecta_ausencia_de_diferenca():
    """Metodos identicos nao podem produzir diferenca significativa."""
    sub = [{'x_mda': v, 'x_lm': v} for v in (1.0, 2.0, 3.0, 4.0)]
    _, _, _, _, dif, p = sens._pareado(sub, 'x')

    assert dif == 0.0 and p == 1.0


def test_comparacao_pareada_detecta_diferenca_sistematica():
    sub = [{'x_mda': v, 'x_lm': v + 0.5} for v in np.linspace(1, 5, 12)]
    _, _, _, _, dif, p = sens._pareado(sub, 'x')

    assert dif == pytest.approx(0.5)
    assert p < 0.05


# ---------------------------------------------------------- custo igualado ---

def test_grade_inclui_o_eixo_de_assimilacoes():
    configs, _ = sens.grade()
    nas = {c[3] for c in configs}

    assert set(sens.ASSIMILACOES) <= nas


def test_custo_do_esmda_e_na_mais_um():
    for na in (1, 3):
        linha = sens.avalia(ne=20, snr=10.0, alvo='poço', semente=0, na=na)
        assert linha['aval_mda'] == na + 1
        assert linha['na'] == na


def test_ieslm_nao_depende_do_numero_de_assimilacoes():
    """O iES-LM e o mesmo em todo o eixo de N_a: so o ES-MDA muda."""
    a = sens.avalia(ne=20, snr=10.0, alvo='poço', semente=1, na=1)
    b = sens.avalia(ne=20, snr=10.0, alvo='poço', semente=1, na=6)

    campos_lm = [k for k in a if k.endswith('_lm')]
    assert campos_lm
    assert all(a[k] == b[k] for k in campos_lm)
    assert a['rmse_Vp_mda'] != b['rmse_Vp_mda']


def test_cache_do_ieslm_da_o_mesmo_resultado_que_rodar_direto():
    import dataclasses
    import experimento_elastico as X
    import forward_elastico as fe
    import metricas as mt

    linha = sens.avalia(ne=20, snr=10.0, alvo='poço', semente=2, na=2)
    cfg = dataclasses.replace(config.PADRAO, ne=20, semente=2)
    c = X.carrega_cenario(cfg)
    res = X.roda_ieslm(c)
    direto = mt.rmse(fe.desempilha(res.conjunto)[0], fe.desempilha(c['verdadeiro'])[0])

    assert linha['rmse_Vp_lm'] == direto
    assert linha['aval_lm'] == res.n_avaliacoes


def _linha(semente, na, aval_lm):
    return {'ne': config.PADRAO.ne, 'snr': config.PADRAO.snr, 'alvo': 'poço',
            'na': na, 'semente': semente, 'aval_mda': na + 1, 'aval_lm': aval_lm}


def test_mesmo_custo_escolhe_o_na_com_o_mesmo_numero_de_avaliacoes():
    linhas = [_linha(0, na, 3) for na in (1, 2, 4)] + \
             [_linha(1, na, 4) for na in (1, 2, 3, 4)]
    pares = sens.mesmo_custo(linhas)

    assert [(p['semente'], p['na']) for p in pares] == [(0, 2), (1, 3)]
    assert all(p['aval_mda'] == p['aval_lm'] for p in pares)


def test_mesmo_custo_descarta_sementes_sem_na_correspondente():
    linhas = [_linha(0, na, 5) for na in (1, 2)]  # precisaria de N_a = 4

    assert sens.mesmo_custo(linhas) == []
