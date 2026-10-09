# -*- coding: utf-8 -*-
"""Testes do estudo de sensibilidade."""

import dataclasses

import numpy as np
import pytest

import config
import experimento_elastico as X
import forward_elastico as fe
import metricas as mt
import sensibilidade as sens

REF = config.PADRAO


def cfg(**campos):
    """Configuracao pequena e rapida para os testes."""
    campos.setdefault('ne', 20)
    campos.setdefault('semente', 0)
    return dataclasses.replace(REF, **campos)


# ------------------------------------------------------------------ grade ---

def test_cada_eixo_contem_o_valor_de_referencia():
    for eixo in sens.EIXOS:
        assert sens.REF_CAMPOS[eixo.campo] in eixo.valores
        assert sens.REF_CAMPOS[eixo.campo] in eixo.rapido


def test_grade_varia_um_eixo_por_vez_exceto_nas_grades_cruzadas():
    pares = [{cz.linha, cz.coluna} for cz in sens.CRUZADAS]
    configs, _ = sens.grade()
    for c in configs:
        v = sens.campos_de(c)
        livres = {k for k in sens.CAMPOS if v[k] != sens.REF_CAMPOS[k]}
        assert len(livres) <= 1 or livres in pares


def test_grade_nao_repete_configuracoes():
    configs, _ = sens.grade()

    assert len(configs) == len(set(configs))
    assert dataclasses.replace(REF, semente=0) in configs


def test_grade_contem_todos_os_valores_de_todos_os_eixos():
    configs, _ = sens.grade()
    presentes = {k: {sens.campos_de(c)[k] for c in configs} for k in sens.CAMPOS}

    for eixo in sens.EIXOS:
        assert set(eixo.valores) <= presentes[eixo.campo]


@pytest.mark.parametrize('cz', sens.CRUZADAS, ids=lambda cz: cz.nome)
def test_grades_cruzadas_completas(cz):
    configs, _ = sens.grade()
    pares = {(sens.campos_de(c)[cz.linha], sens.campos_de(c)[cz.coluna]) for c in configs}

    for a in cz.valores_linha:
        for b in cz.valores_coluna:
            assert (a, b) in pares


def test_valores_das_grades_cruzadas_pertencem_aos_eixos():
    for cz in sens.CRUZADAS:
        assert set(cz.valores_linha) <= set(sens.EIXO[cz.linha].valores)
        assert set(cz.valores_coluna) <= set(sens.EIXO[cz.coluna].valores)


def test_grade_rapida_e_menor():
    completa, n_c = sens.grade()
    rapida, n_r = sens.grade(rapido=True)

    assert len(rapida) < len(completa) and n_r < n_c


def test_eixo_de_angulos_produz_tres_angulos_a_partir_de_15_graus():
    assert sens._angulos(60.0) == (15.0, 37.5, 60.0)
    assert sens._angulos(45.0) == REF.angulos


# ----------------------------------------------------------------- filtro ---

def _linha(**campos):
    linha = dict(sens.REF_CAMPOS)
    linha.update(campos)
    return linha


def test_no_eixo_exclui_linhas_de_outros_eixos():
    linhas = [_linha(), _linha(ne=50), _linha(snr=5.0), _linha(gamma0=0.5),
              _linha(ne=50, snr=5.0)]

    assert len(sens.no_eixo(linhas, 'ne')) == 2          # referencia e ne=50
    assert len(sens.no_eixo(linhas, 'gamma0')) == 2      # referencia e gamma0=0.5
    assert len(sens.no_eixo(linhas, 'ne', 'snr')) == 4   # tudo menos gamma0


def test_no_eixo_sem_campos_devolve_so_a_referencia():
    linhas = [_linha(), _linha(ne=50), _linha(vies_prior=-0.05)]

    assert sens.no_eixo(linhas) == [_linha()]


# ----------------------------------------------------------------- avalia ---

def test_avalia_registra_o_valor_de_cada_eixo():
    linha = sens.avalia(cfg(gamma0=0.5, vies_prior=-0.02, angulos=sens._angulos(60.0)))

    assert linha['gamma0'] == 0.5
    assert linha['vies_prior'] == -0.02
    assert linha['angulo_max'] == 60.0
    for sufixo in ('mda', 'lm'):
        for nome in ('Vp', 'Vs', 'rho'):
            assert 0 < linha['rmse_%s_%s' % (nome, sufixo)] < 10
            assert 0 <= linha['cob_%s_%s' % (nome, sufixo)] <= 1


def test_avalia_e_reprodutivel():
    assert sens.avalia(cfg(semente=3)) == sens.avalia(cfg(semente=3))


def test_custo_do_esmda_e_na_mais_um():
    for na in (1, 3):
        linha = sens.avalia(cfg(n_assimilacoes=na))
        assert linha['aval_mda'] == na + 1
        assert linha['na'] == na


def test_ieslm_nao_depende_do_numero_de_assimilacoes():
    """O iES-LM e o mesmo em todo o eixo de N_a: so o ES-MDA muda."""
    a = sens.avalia(cfg(semente=1, n_assimilacoes=1))
    b = sens.avalia(cfg(semente=1, n_assimilacoes=6))

    campos_lm = [k for k in a if k.endswith('_lm')]
    assert campos_lm
    assert all(a[k] == b[k] for k in campos_lm)
    assert a['rmse_Vp_mda'] != b['rmse_Vp_mda']


def test_cache_do_ieslm_da_o_mesmo_resultado_que_rodar_direto():
    linha = sens.avalia(cfg(semente=2, n_assimilacoes=2))
    c = X.carrega_cenario(cfg(semente=2))
    res = X.roda_ieslm(c)
    direto = mt.rmse(fe.desempilha(res.conjunto)[0], fe.desempilha(c['verdadeiro'])[0])

    assert linha['rmse_Vp_lm'] == direto
    assert linha['aval_lm'] == res.n_avaliacoes


def test_fator_ruido_menor_faz_o_ieslm_iterar_mais():
    """Limiar mais apertado na Eq. 43 = parada mais tardia."""
    frouxo = sens.avalia(cfg(fator_ruido=8.0, semente=4))
    apertado = sens.avalia(cfg(fator_ruido=0.5, semente=4))

    assert apertado['iter_lm'] >= frouxo['iter_lm']


# ------------------------------------------------------------ mesmo custo ---

def _par(semente, na, aval_lm):
    return _linha(na=na, semente=semente, aval_mda=na + 1, aval_lm=aval_lm)


def test_mesmo_custo_escolhe_o_na_com_o_mesmo_numero_de_avaliacoes():
    linhas = [_par(0, na, 3) for na in (1, 2, 4)] + \
             [_par(1, na, 4) for na in (1, 2, 3, 4)]
    pares = sens.mesmo_custo(linhas)

    assert [(p['semente'], p['na']) for p in pares] == [(0, 2), (1, 3)]
    assert all(p['aval_mda'] == p['aval_lm'] for p in pares)


def test_mesmo_custo_descarta_sementes_sem_na_correspondente():
    linhas = [_par(0, na, 5) for na in (1, 2)]  # precisaria de N_a = 4

    assert sens.mesmo_custo(linhas) == []


def test_mesmo_custo_ignora_linhas_de_outros_eixos():
    """Uma linha de outro eixo com o N_a certo nao pode virar par."""
    linhas = [_par(0, 4, 3), dict(_par(0, 2, 3), gamma0=0.5)]

    assert sens.mesmo_custo(linhas) == []


# ---------------------------------------------------------------- pareado ---

def test_comparacao_pareada_detecta_ausencia_de_diferenca():
    sub = [{'x_mda': v, 'x_lm': v} for v in (1.0, 2.0, 3.0, 4.0)]
    _, _, _, _, dif, p = sens._pareado(sub, 'x')

    assert dif == 0.0 and p == 1.0


def test_comparacao_pareada_detecta_diferenca_sistematica():
    sub = [{'x_mda': v, 'x_lm': v + 0.5} for v in np.linspace(1, 5, 12)]
    _, _, _, _, dif, p = sens._pareado(sub, 'x')

    assert dif == pytest.approx(0.5)
    assert p < 0.05


def test_matriz_cruzada_usa_so_os_valores_declarados():
    """O ponto ne=25 do eixo de N_e cai no plano, mas nao e linha da matriz."""
    cz = sens.CRUZADAS[0]
    linhas = []
    celulas = [(25, 10.0)] + [(a, b) for a in cz.valores_linha for b in cz.valores_coluna]
    for a, b in celulas:
        for semente in range(3):
            linhas.append(_linha(**{cz.linha: a, cz.coluna: b, 'semente': semente,
                                    'rmse_Vp_mda': 1.0 + semente,
                                    'rmse_Vp_lm': 1.0 + semente}))
    vals_l, vals_c, dif, _ = sens._matriz_cruzada(linhas, cz, 'rmse_Vp')

    assert vals_l == list(cz.valores_linha)
    assert vals_c == list(cz.valores_coluna)
    assert not np.isnan(dif).any()


def test_csv_ida_e_volta_preserva_valores_e_tipos(tmp_path, monkeypatch):
    linhas = [sens.avalia(cfg(semente=s)) for s in range(2)]
    monkeypatch.setattr(sens, 'PASTA_RESULTADOS', str(tmp_path))
    sens.salva(linhas)

    lidas = sens.carrega_csv(str(tmp_path / 'sensibilidade.csv'))
    assert lidas == linhas
