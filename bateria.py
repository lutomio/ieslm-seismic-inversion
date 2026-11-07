#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bateria de testes: iES-LM x ES-MDA em cenarios escolhidos a dedo.

O estudo de sensibilidade (sensibilidade.py) varia um fator por vez a partir
de config.PADRAO. Esta bateria parte de outro ponto: as MESMAS entradas do
driver de inversao sismica da SeReMpy (Examples/ESSeisInversionDriver.py), e
dali muda uma entrada por teste, escolhendo as que tem potencial para
favorecer ou prejudicar o iES-LM.

Entradas do driver da SeReMpy reproduzidas no teste principal (T01):

    conjunto              N_e = 500
    ES-MDA                N_a = 4, alpha constante = 4 (sum 1/alpha = 1)
    erro do dado          variancia 1e-4 (C_D = 1e-4 I)
    wavelet               Ricker de 45 Hz, 64 amostras
    angulos               15, 30 e 45 graus
    a priori              Butterworth de ordem 3, corte 0,04; correlacao 5 dt
    limites fisicos       nenhum (o driver nao trunca o conjunto)

A unica diferenca deliberada e o dado observado: o driver inverte os tracos
de data3seis.dat (identicos aos de data5seis.dat), gerados pelo proprio
operador e sem ruido -- "inverse crime". No T01 o dado e gerado do poco com
ruido de variancia exatamente 1e-4, coerente com a C_D do driver. O T02
reproduz o driver ao pe da letra, com os tracos do arquivo, para comparacao.

Cada teste roda N_SEMENTES sementes pareadas (os dois metodos recebem o mesmo
cenario em cada semente) e compara as metricas com Wilcoxon. As figuras
saem da primeira semente e vao para figuras/testes/<codigo>_<nome>/, com o
codigo do teste no nome do arquivo e no canto da figura.

Uso:
    python bateria.py                # todos os testes
    python bateria.py T01 T05        # so os testes pedidos
    python bateria.py --relatorio    # refaz resumo e figura-resumo do CSV
"""

import csv
import os
import sys
import time
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats

import config
import dados
import experimento_elastico as X
import forward_elastico as fe
import metricas as mt

PASTA_FIGURAS = os.path.join(X.PASTA_FIGURAS, 'testes')
PASTA_RESULTADOS = os.path.join(X.PASTA_RESULTADOS, 'bateria')
CSV = os.path.join(PASTA_RESULTADOS, 'bateria.csv')

N_SEMENTES = 20
SEMENTE0 = config.PADRAO.semente

# Variancia do erro do dado no driver da SeReMpy (varerr = 10 ** -4)
VARIANCIA_SEREMPY = 1e-4


def _snr_serempy():
    """
    SNR que produz ruido de variancia VARIANCIA_SEREMPY sobre a sismica do poco.

    fe.adiciona_ruido define o desvio do ruido como std(sinal) / SNR; com
    este valor o desvio sai exatamente sqrt(1e-4) = 0,01 (SNR ~ 3,4).
    """
    c = X.carrega_cenario(config.PADRAO)
    return float(np.std(c['g'](c['verdadeiro']))) / np.sqrt(VARIANCIA_SEREMPY)


SEREMPY = replace(config.PADRAO, ne=500, snr=_snr_serempy(),
                  n_assimilacoes=4, razao_alpha=1.0)


@dataclass(frozen=True)
class Teste:
    """
    TESTE
    Um cenario da bateria.

    Attributes
    ----------
    codigo : str
        Identificador curto (T01, T02, ...), usado nos nomes dos arquivos.
    nome : str
        Nome legivel por maquina, tambem nos nomes dos arquivos.
    descricao : str
        O que muda em relacao ao T01 e por que pode mudar o resultado.
    cfg : config.Configuracao
    limites : bool
        Trunca o conjunto nos limites fisicos. O driver da SeReMpy nao trunca.
    dado_original : bool
        Usa os tracos de data5seis.dat, sem ruido, como dado observado.
    """
    codigo: str
    nome: str
    descricao: str
    cfg: config.Configuracao
    limites: bool = False
    dado_original: bool = False

    @property
    def id(self):
        return '%s_%s' % (self.codigo, self.nome)


TESTES = (
    Teste('T01', 'serempy', 'entradas do driver da SeReMpy (principal)', SEREMPY),
    Teste('T02', 'serempy_dado_original',
          'dado de data5seis.dat sem ruído, como no driver (inverse crime)',
          SEREMPY, dado_original=True),
    Teste('T03', 'mesmo_custo', 'ES-MDA com N_a = 1: mesmo custo do iES-LM (2 avaliações)',
          replace(SEREMPY, n_assimilacoes=1)),
    Teste('T04', 'alpha_decrescente', 'ES-MDA com α decrescente (razão 2)',
          replace(SEREMPY, razao_alpha=2.0)),
    Teste('T05', 'esmda_na10', 'ES-MDA com N_a = 10 (5,5× o custo do iES-LM)',
          replace(SEREMPY, n_assimilacoes=10)),
    Teste('T06', 'snr10', 'menos ruído: SNR 10', replace(SEREMPY, snr=10.0)),
    Teste('T07', 'snr50', 'pouco ruído: SNR 50', replace(SEREMPY, snr=50.0)),
    Teste('T08', 'ne50', 'conjunto pequeno: N_e = 50', replace(SEREMPY, ne=50)),
    Teste('T09', 'gamma0_025', 'γ inicial 0,25 (Exemplo 2 de Ma e Bi)',
          replace(SEREMPY, gamma0=0.25)),
    Teste('T10', 'sem_eq43', 'iES-LM sem a parada no nível do ruído (Eq. 43)',
          replace(SEREMPY, fator_ruido=None)),
    Teste('T11', 'um_angulo', 'um ângulo só (30°)', replace(SEREMPY, angulos=(30.0,))),
    Teste('T12', 'angulo60', 'ângulos 15°, 37,5° e 60°',
          replace(SEREMPY, angulos=(15.0, 37.5, 60.0))),
    Teste('T13', 'wavelet25', 'wavelet de 25 Hz (menor resolução)',
          replace(SEREMPY, freq_wavelet=25.0)),
    Teste('T14', 'vies_prior', 'tendência do a priori 8% abaixo da verdadeira',
          replace(SEREMPY, vies_prior=-0.08)),
    Teste('T15', 'alvo_suavizado', 'alvo sintético suavizado',
          replace(SEREMPY, alvo='suavizado')),
    Teste('T16', 'alvo_6camadas', 'alvo sintético de 6 camadas',
          replace(SEREMPY, alvo='6 camadas')),
    Teste('T17', 'com_limites', 'com limites físicos (0,7 e 1,3 da faixa do poço)',
          SEREMPY, limites=True),
)
POR_CODIGO = {t.codigo: t for t in TESTES}


# ---------------------------------------------------------------- cenario ---

def _dado_original():
    """Tracos near, mid e far de data5seis.dat, empilhados como em fe."""
    d = dados.carrega_dados()
    return np.vstack([d['Snear'], d['Smid'], d['Sfar']])


def cenario(teste, semente):
    """
    CENARIO
    O cenario de X.carrega_cenario, ajustado ao que o teste pede.

    Sem limites fisicos, os limites viram (-inf, inf): o np.clip dos dois
    metodos passa a nao fazer nada. Com o dado original, o dado observado e
    trocado pelos tracos do arquivo e a C_D pela do driver.
    """
    c = X.carrega_cenario(replace(teste.cfg, semente=semente))
    if not teste.limites:
        c['limites'] = (-np.inf, np.inf)
    if teste.dado_original:
        d = _dado_original()
        c['d_obs'] = d
        c['C_D'] = VARIANCIA_SEREMPY * np.eye(d.shape[0])
    return c


# ---------------------------------------------------------------- metricas ---

def _metricas(M, verd, sufixo):
    linha = {}
    for i, nome in enumerate(fe.NOMES_PROPRIEDADES):
        prop = fe.desempilha(M)[i]
        cob = mt.taxa_cobertura(prop, verd[i])
        linha['rmse_%s_%s' % (nome, sufixo)] = mt.rmse(prop, verd[i])
        linha['cob_%s_%s' % (nome, sufixo)] = cob
        linha['cal_%s_%s' % (nome, sufixo)] = float(mt.erro_calibracao(cob))
        linha['env_%s_%s' % (nome, sufixo)] = mt.largura_envelope(prop)
    return linha


def roda(teste, semente, figuras=False):
    """Roda os dois metodos em uma semente; com figuras=True, tambem as gera."""
    c = cenario(teste, semente)
    M_mda, hist_mda, aval_mda, _ = X.roda_esmda(c)
    res_lm = X.roda_ieslm(c)
    verd = fe.desempilha(c['verdadeiro'])

    linha = {'teste': teste.codigo, 'nome': teste.nome, 'semente': semente,
             'aval_mda': aval_mda, 'aval_lm': res_lm.n_avaliacoes,
             'iter_lm': len(res_lm.desajuste) - 1,
             'desajuste_mda': hist_mda[-1], 'desajuste_lm': min(res_lm.desajuste),
             'parada_lm': res_lm.motivo_parada}
    linha.update(_metricas(M_mda, verd, 'mda'))
    linha.update(_metricas(res_lm.conjunto, verd, 'lm'))

    if figuras:
        res_sem = X.roda_ieslm_sem_parada(c)
        extras = []
        for s in range(semente + 1, semente + X.N_SEMENTES_RESIDUO):
            ce = cenario(teste, s)
            extras.append((ce, X.roda_esmda(ce)[0], X.roda_ieslm(ce).conjunto))
        pasta = os.path.join(PASTA_FIGURAS, teste.id)
        rotulo = '%s · %s — %s' % (teste.codigo, teste.nome, teste.descricao)
        X._figuras(c, M_mda, hist_mda, res_lm, res_lm.conjunto, res_sem, extras,
                   pasta=pasta, prefixo=teste.id + '__', rotulo=rotulo)
    return linha


def executa(testes=TESTES):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    linhas = carrega_csv() if os.path.exists(CSV) else []
    pedidos = {t.codigo for t in testes}
    linhas = [l for l in linhas if l['teste'] not in pedidos]  # refaz os pedidos

    for t in testes:
        inicio = time.time()
        config.salva(t.cfg, os.path.join(PASTA_RESULTADOS, t.id + '.json'),
                     teste=t.codigo, descricao=t.descricao, limites=t.limites,
                     dado_original=t.dado_original, n_sementes=N_SEMENTES)
        for k in range(N_SEMENTES):
            linhas.append(roda(t, SEMENTE0 + k, figuras=(k == 0)))
        print('%s %-24s %5.1f s' % (t.codigo, t.nome, time.time() - inicio), flush=True)

    linhas.sort(key=lambda l: (l['teste'], l['semente']))
    salva_csv(linhas)
    relatorio(linhas)


def salva_csv(linhas):
    with open(CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def carrega_csv():
    with open(CSV, encoding='utf-8') as f:
        linhas = list(csv.DictReader(f))
    for l in linhas:
        for k, v in l.items():
            if k in ('teste', 'nome', 'parada_lm'):
                continue
            l[k] = float(v)
    return linhas


# ------------------------------------------------------------- estatistica ---

def pareado(sub, metrica):
    """Medias, diferenca media (iES-LM - ES-MDA), IC 95% da diferenca e p de Wilcoxon."""
    a = np.array([l['%s_mda' % metrica] for l in sub])
    b = np.array([l['%s_lm' % metrica] for l in sub])
    dif = b - a
    p = 1.0 if np.allclose(dif, 0) else float(stats.wilcoxon(a, b).pvalue)
    meia = stats.t.ppf(0.975, len(dif) - 1) * dif.std(ddof=1) / np.sqrt(len(dif)) \
        if len(dif) > 1 else 0.0
    return a.mean(), b.mean(), dif.mean(), meia, p


def veredito(dif, p, nivel=0.05):
    """Menor e melhor para RMSE e erro de calibracao."""
    if p >= nivel:
        return 'empate'
    return 'iES-LM' if dif < 0 else 'ES-MDA'


# Na figura-resumo, um teste cuja diferenca (com o IC) passa de FORA_DE_ESCALA
# vezes a mediana dos testes vira uma seta na borda do eixo.
FORA_DE_ESCALA = 10

METRICAS = (('rmse_Vp', 'RMSE Vp'), ('rmse_Vs', 'RMSE Vs'), ('rmse_rho', 'RMSE rho'),
            ('cal_Vp', '|cob Vp - 0,8|'))


def relatorio(linhas=None):
    linhas = carrega_csv() if linhas is None else linhas
    codigos = sorted({l['teste'] for l in linhas})
    texto = ['Bateria de testes: iES-LM x ES-MDA (%d sementes pareadas por teste; '
             'Wilcoxon, nivel 0,05; menor e melhor)' % N_SEMENTES, '']
    cab = '%-4s %-22s %5s %5s' % ('', 'teste', 'aval', 'aval')
    sub_cab = '%-4s %-22s %5s %5s' % ('', '', 'MDA', 'LM')
    for _, rot in METRICAS:
        cab += ' | %-29s' % rot
        sub_cab += ' | %7s %7s %6s %-7s' % ('MDA', 'LM', 'p', '')
    cab += ' | %-11s' % 'cob Vp'
    sub_cab += ' | %5s %5s' % ('MDA', 'LM')
    texto += [cab, sub_cab, '-' * len(sub_cab)]

    for cod in codigos:
        sub = [l for l in linhas if l['teste'] == cod]
        s = '%-4s %-22s %5.1f %5.1f' % (cod, sub[0]['nome'][:22],
                                        np.mean([l['aval_mda'] for l in sub]),
                                        np.mean([l['aval_lm'] for l in sub]))
        for met, _ in METRICAS:
            a, b, dif, _, p = pareado(sub, met)
            s += ' | %7.4f %7.4f %6.3f %-7s' % (a, b, p, veredito(dif, p))
        s += ' | %5.2f %5.2f' % (np.mean([l['cob_Vp_mda'] for l in sub]),
                                 np.mean([l['cob_Vp_lm'] for l in sub]))
        texto.append(s)

    texto += ['', 'Descricao dos testes:']
    texto += ['  %s %-22s %s' % (t.codigo, t.nome, t.descricao) for t in TESTES
              if t.codigo in codigos]
    saida = '\n'.join(texto)
    print(saida)
    with open(os.path.join(PASTA_RESULTADOS, 'bateria_resumo.txt'), 'w',
              encoding='utf-8') as f:
        f.write(saida + '\n')
    _figura_resumo(linhas, codigos)


def _figura_resumo(linhas, codigos):
    """
    Diferenca pareada (iES-LM - ES-MDA) por teste, com IC de 95%.
    Abaixo de zero o iES-LM e melhor; ponto cheio = significativo (p < 0,05).
    """
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    paineis = (('rmse_Vp', r'$\Delta$ RMSE de $V_p$ (km/s)'),
               ('rmse_rho', r'$\Delta$ RMSE de $\rho$ (g/cm$^3$)'),
               ('cal_Vp', r'$\Delta\,|$cobertura de $V_p - 0{,}8|$'))
    nomes = ['%s %s' % (c, POR_CODIGO[c].nome if c in POR_CODIGO else '') for c in codigos]
    y = np.arange(len(codigos))[::-1]

    fig, eixos = plt.subplots(1, len(paineis), figsize=(13, 0.42 * len(codigos) + 1.8),
                              sharey=True)
    for ax, (met, rot) in zip(eixos, paineis):
        pontos = []
        for yi, cod in zip(y, codigos):
            _, _, dif, meia, p = pareado([l for l in linhas if l['teste'] == cod], met)
            pontos.append((yi, dif, meia, veredito(dif, p)))

        # Um teste muito fora da escala dos demais (o T10, sem a Eq. 43) achataria
        # todos os outros: ele vira uma seta na borda, com o valor escrito.
        extensao = np.array([abs(d) + m for _, d, m, _ in pontos])
        fora = extensao > FORA_DE_ESCALA * np.median(extensao)
        lim = 1.12 * max(extensao[~fora].max(), 1e-12)

        for (yi, dif, meia, v), longe in zip(pontos, fora):
            cor = {'iES-LM': X.COR_LM, 'ES-MDA': X.COR_MDA}.get(v, '0.55')
            if longe:
                borda = np.sign(dif) * lim
                ax.plot(borda, yi, '>' if dif > 0 else '<', ms=10, color=cor,
                        clip_on=False, zorder=5)
                ax.annotate(('%+.2f' % dif).replace('.', ','), xy=(borda, yi),
                            xytext=(-12 if dif > 0 else 12, 0), textcoords='offset points',
                            ha='right' if dif > 0 else 'left', va='center', fontsize=8,
                            color='0.2')
                continue
            ax.plot([dif - meia, dif + meia], [yi, yi], color=cor, lw=2,
                    solid_capstyle='round')
            ax.plot(dif, yi, 'o', ms=8, color=cor, mfc=cor if v != 'empate' else 'white',
                    mew=2)
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
        ax.axvline(0, color='0.2', lw=1)
        ax.set_xlabel(rot)
        ax.grid(axis='x', alpha=0.3)
        ax.set_title('← iES-LM melhor  |  ES-MDA melhor →', fontsize=9, color='0.3')
    eixos[0].set_yticks(y)
    eixos[0].set_yticklabels(nomes, fontsize=9)
    legenda = [plt.Line2D([], [], marker='o', ls='', color=X.COR_LM, ms=8,
                          label='iES-LM melhor (p < 0,05)'),
               plt.Line2D([], [], marker='o', ls='', color=X.COR_MDA, ms=8,
                          label='ES-MDA melhor (p < 0,05)'),
               plt.Line2D([], [], marker='o', ls='', color='0.55', mfc='white', mew=2, ms=8,
                          label='empate (p ≥ 0,05)')]
    fig.legend(handles=legenda, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Bateria de testes: diferença pareada iES-LM − ES-MDA '
                 '(%d sementes, IC 95%%)' % N_SEMENTES, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    X.salva_figura(fig, 'resumo_bateria', PASTA_FIGURAS)


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--relatorio' in args:
        relatorio()
    else:
        executa([POR_CODIGO[a] for a in args] if args else TESTES)
