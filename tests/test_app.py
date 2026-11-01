"""Teste do app Streamlit: abre sem erro, mostra a previsão da loja, o orçamento reage ao percentual e o
monitoramento compara as semanas do teste com o SLA da ficha; a aba de cenários soma a rede e confere o teste."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def abrir():
    return AppTest.from_file(APP, default_timeout=60).run()


def euros(texto):
    return float(texto.replace('€', '').replace('.', '').strip())


def metrica(app, rotulo):
    return next(m for m in app.metric if m.label == rotulo)


def test_app_abre_e_mostra_a_previsao():
    app = abrir()
    assert not app.exception
    assert app.metric[0].label == 'Vendas previstas (6 semanas)'
    minimo, previsto, maximo = euros(app.metric[1].value), euros(app.metric[0].value), euros(app.metric[2].value)
    assert 0 < minimo < previsto < maximo


def test_trocar_a_loja_muda_a_previsao():
    app = abrir()
    antes = app.metric[0].value
    app.selectbox[0].set_value(app.selectbox[0].options[5]).run()
    assert not app.exception
    assert app.metric[0].value != antes


def test_orcamento_acompanha_o_percentual():
    app = abrir()
    verba_5 = euros(metrica(app, 'Verba total da rede').value)
    app.slider[0].set_value(10).run()
    assert abs(euros(metrica(app, 'Verba total da rede').value) / verba_5 - 2) < 0.01
    app.toggle[0].set_value(True).run()   # orçamento conservador usa o pior cenário
    assert euros(metrica(app, 'Verba total da rede').value) < verba_5 * 2


def test_cenarios_da_rede_e_conferencia_no_teste():
    app = abrir()
    pior, provavel, melhor = (euros(metrica(app, f'{nome} da rede').value)
                              for nome in ['Pior cenário', 'Cenário provável', 'Melhor cenário'])
    assert 0 < pior < provavel < melhor
    partes = [float(metrica(app, rotulo).value.rstrip('%').replace(',', '.'))
              for rotulo in ['Total real entre os cenários', 'Abaixo do pior cenário', 'Acima do melhor cenário']]
    assert abs(sum(partes) - 100) < 0.2


def test_monitoramento_mostra_as_6_semanas_e_o_drift():
    app = abrir()
    assert metrica(app, 'Semanas fora do SLA').value.endswith('de 6')
    assert len(app.dataframe) == 5   # 2 de cenários + orçamento + semanas + drift
    app.selectbox[1].select_index(1).run()   # uma loja em vez da rede inteira
    assert not app.exception
    assert metrica(app, 'Semanas fora do SLA').value.endswith('de 6')
