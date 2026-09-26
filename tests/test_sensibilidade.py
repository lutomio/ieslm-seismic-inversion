# -*- coding: utf-8 -*-
"""Testes do estudo de sensibilidade."""

import numpy as np
import pytest

import experimento_elastico as X
import sensibilidade as sens


def test_grade_varia_um_eixo_por_vez():
    """Cada configuracao difere do ponto de referencia em no maximo um fator."""
    configs, _ = sens.grade()
    referencia = (X.NE, X.SNR, 'poço')

    for c in configs:
        diferencas = sum(1 for a, b in zip(c, referencia) if a != b)
        assert diferencas <= 1


def test_grade_nao_repete_a_configuracao_central():
    configs, _ = sens.grade()

    assert len(configs) == len(set(configs))
    assert (X.NE, X.SNR, 'poço') in configs


def test_constroi_alvo_devolve_modelos_validos():
    for nome in sens.ALVOS:
        m = sens.constroi_alvo(nome, semente=0)
        if nome == 'poço':
            assert m is None  # sinaliza para usar o perfil do pacote
        else:
            assert np.all(m['Vs'] < m['Vp'])
            assert m['Vp'].shape == (99, 1)


def test_alvo_desconhecido_e_rejeitado():
    with pytest.raises(ValueError, match='desconhecido'):
        sens.constroi_alvo('inexistente', semente=0)


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
