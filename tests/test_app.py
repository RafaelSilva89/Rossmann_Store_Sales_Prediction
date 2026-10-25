"""Teste do app Streamlit: abre sem erro, mostra a previsão da loja e o orçamento reage ao percentual."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def abrir():
    return AppTest.from_file(APP, default_timeout=60).run()


def euros(texto):
    return float(texto.replace('€', '').replace('.', '').strip())


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
    app = abrir()   # métricas da aba de reforma: índices 3 (verba total) e 4 (verba média)
    verba_5 = euros(app.metric[3].value)
    app.slider[0].set_value(10).run()
    assert abs(euros(app.metric[3].value) / verba_5 - 2) < 0.01
    app.toggle[0].set_value(True).run()   # orçamento conservador usa o mínimo provável
    assert euros(app.metric[3].value) < verba_5 * 2
