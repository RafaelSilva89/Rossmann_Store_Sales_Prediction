# 🏪 Previsão de Vendas da Rossmann (6 semanas por loja)

**Quanto cada loja vai vender nas próximas 6 semanas? Um modelo treinado com 844 mil dias de venda de 1.115 drogarias responde loja a loja, e o CFO usa a resposta para definir a verba de reforma.**

<div align="center">
<img src="Imagens/Rossmann.jpg" width="450px">
</div>

[![Testes](https://github.com/RafaelSilva89/Rossmann_Store_Sales_Prediction/actions/workflows/testes.yml/badge.svg)](https://github.com/RafaelSilva89/Rossmann_Store_Sales_Prediction/actions/workflows/testes.yml)

**Python · Pandas · Scikit-learn · XGBoost · LightGBM · SciPy · Pytest · GitHub Actions · Streamlit · CRISP-DM**

| 📊 Destaques | |
|---|---|
| Base | **1.017.209 registros** de venda diária de **1.115 lojas** da rede alemã Rossmann (Kaggle), de jan/2013 a jul/2015, em **euros** |
| Modelo | O total de 6 semanas de cada loja erra **3,7%** em média, contra **6,0%** de quem usa só a média da loja. O erro do dia é de **€ 614** (MAPE de **9,2%**), contra **€ 933** (**13,3%**) da versão original nas mesmas semanas |
| O que faz vender | **Promoção** (**+38,8%** na venda do dia), **segunda-feira** e **dezembro** (**+24%**). A chegada de um **concorrente** tira **~6%** da venda |
| A v2 corrigiu | A EDA original **somava** as vendas em vez de tirar a média, e **6 conclusões** se invertem (as vendas **sobem** a cada ano, em vez de cair). O "modelo da média" da v1 **usava as respostas** do teste |
| Entrega | Notebook CRISP-DM, **14 testes automatizados** (GitHub Actions) e **app Streamlit** com a previsão de cada loja e o orçamento de reforma da rede |

---

## 📑 Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo](#2-objetivo)
3. [Base de dados](#3-base-de-dados)
4. [Estrutura do projeto](#4-estrutura-do-projeto)
5. [Como executar](#5-como-executar)
6. [Etapas do projeto (CRISP-DM)](#6-etapas-do-projeto-crisp-dm)
7. [Principais descobertas da análise exploratória](#7-principais-descobertas-da-análise-exploratória)
8. [Construção do modelo](#8-construção-do-modelo)
9. [Resultado do modelo](#9-resultado-do-modelo)
10. [App interativo (Streamlit)](#10-app-interativo-streamlit)
11. [O que mudou da v1 para a v2](#11-o-que-mudou-da-v1-para-a-v2)
12. [Insights](#12-insights)
13. [Limites](#13-limites)
14. [Próximos passos](#14-próximos-passos)

---

## 1. Contexto do problema

<img src="Imagens/rossmann_store.jpg" width="200px" align="right">

A **Rossmann** é uma das maiores redes de drogarias da Europa. Neste cenário (baseado numa competição do Kaggle), o **CFO** quer reformar as lojas e vai reservar para isso uma **parcela do faturamento das próximas 6 semanas** de cada uma. Por isso, pediu aos gerentes uma previsão de vendas loja a loja.

| Se a previsão fica... | O que acontece |
|---|---|
| **Alta demais** | A loja recebe uma verba de reforma que o caixa dela não sustenta |
| **Baixa demais** | A reforma é adiada ou fica menor do que a loja poderia pagar |

Hoje cada gerente prevê do seu jeito. Um modelo treinado com o histórico de **todas** as lojas dá uma previsão **objetiva, igual para todas e com o erro medido**.

## 2. Objetivo

> **Perguntas de negócio:**
> 1. **Quanto cada loja vai vender** nas próximas 6 semanas? Com que **margem de erro**?
> 2. **O que faz uma loja vender mais:** promoção, dia da semana, época do ano, tipo de loja, sortimento, concorrência?
> 3. Em quais **lojas e situações** a previsão é mais confiável?
> 4. Quanto a rede pode reservar para **reforma**, loja a loja?

## 3. Base de dados

A base é a competição [*Rossmann Store Sales*](https://www.kaggle.com/c/rossmann-store-sales/data) (Kaggle), em `Datasets/`:

| Arquivo | Conteúdo |
|---|---|
| `train.csv` | Venda diária de cada loja, de 01/01/2013 a 31/07/2015 (base do modelo) |
| `store.csv` | Cadastro das 1.115 lojas: tipo, sortimento, concorrência e Promo2 |
| `test.csv` | As **6 semanas seguintes** (01/08 a 17/09/2015) de 856 lojas, **sem as vendas**: é o que o modelo prevê |

| Coluna (traduzida) | Descrição |
|---|---|
| `vendas` | 🎯 **Alvo**: venda do dia, em euros |
| `loja`, `data`, `dia_semana` | Loja e calendário |
| `aberta`, `promo`, `feriado`, `feriado_escolar` | Se a loja abriu, se estava em promoção e se era feriado |
| `tipo_loja`, `sortimento` | Modelo da loja (a, b, c, d) e variedade de produtos (básico, extra, estendido) |
| `distancia_concorrente`, `concorrente_desde_...` | Distância e data de abertura do concorrente mais próximo |
| `promo2`, `promo2_desde_...`, `promo2_meses` | Promoção contínua que reinicia em meses fixos |
| `clientes` | ⚠️ Clientes do dia: só conhecido **depois** do dia, **fica fora do modelo** (vazamento) |

## 4. Estrutura do projeto

```
Rossmann_Store_Sales_Prediction/
├── Datasets/                           # train.csv, store.csv e test.csv (Kaggle)
├── Imagens/                            # gráficos gerados pelos notebooks
├── Model/
│   ├── modelo_rossmann_v2.pkl          # modelo de produção + faixa de erro
│   ├── modelo_rossmann_v2.json         # ficha do modelo (colunas, métricas, hiperparâmetros, avisos)
│   └── model_rossmann.pkl              # modelo da versão original
├── docs/
│   └── decisoes_analiticas.md          # registro das decisões do projeto (DA-01 a DA-16)
├── reports/
│   ├── comparacao_modelos.csv          # métricas dos 8 modelos (validação e validação cruzada)
│   ├── erro_por_tipo_de_loja.csv       # tradução do erro por tipo de loja
│   ├── erro_por_sortimento.csv         # tradução do erro por sortimento
│   └── previsoes_6_semanas.csv         # previsão dia a dia de ago-set/2015 (usada pelo app)
├── Notebooks/
│   ├── Rossmann_Store_Sales_Prediction.ipynb        # versão original
│   └── Rossmann_Store_Sales_Prediction - v2.ipynb   # versão reorganizada (CRISP-DM)
├── src/
│   ├── config.py                       # caminhos, semente, colunas e datas de corte
│   ├── dados.py                        # leitura, tradução, variáveis criadas e separação no tempo
│   ├── modelos.py                      # médias da loja (dentro do Pipeline) e modelos de referência
│   ├── validacao.py                    # validação cruzada no tempo (janelas de 6 semanas)
│   ├── estatistica.py                  # testes das hipóteses (com tamanho do efeito) e regra do IQR
│   ├── selecao.py                      # seleção de variáveis (Random Forest + Informação Mútua)
│   ├── avaliacao.py                    # ml_error, erro do total por loja, faixa de erro, tradução do erro e ficha
│   └── graficos.py                     # gráficos e formatação em português
├── tests/
│   ├── test_funcoes.py                 # 11 testes das funções de src/
│   └── test_app.py                     # 3 testes do app (AppTest do Streamlit)
├── Parameter/, rossmann/, rossmann-telegram-api/   # escalas, API e bot do Telegram da versão original (histórico)
├── .github/workflows/testes.yml        # roda os testes a cada push (GitHub Actions)
├── app.py                              # app Streamlit
├── pytest.ini
├── requirements.txt
└── README.md
```

## 5. Como executar

```bash
# 1. Clonar o repositório
git clone https://github.com/RafaelSilva89/Rossmann_Store_Sales_Prediction.git
cd Rossmann_Store_Sales_Prediction

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Rodar os testes
pytest

# 4. Abrir o notebook e executar todas as células (cerca de 1 hora; gera o modelo e as previsões usados pelo app)
jupyter notebook "Notebooks/Rossmann_Store_Sales_Prediction - v2.ipynb"

# 5. Abrir o app
streamlit run app.py
```

## 6. Etapas do projeto (CRISP-DM)

| Fase | O que foi feito |
|---|---|
| **01. Entendimento do negócio** | Definição das 4 perguntas e do público (CFO e gerentes) |
| **02. Preparação dos dados** | Junção de vendas e cadastro, tradução, nulos tratados pelo que significam, a lacuna das 180 lojas, outliers pela regra do IQR, retirada dos clientes (vazamento) e EDA com **12 hipóteses** (médias, teste estatístico e tamanho do efeito) |
| **03. Modelagem** | Separação no tempo, **validação cruzada no tempo** (5 janelas de 6 semanas), seleção de variáveis, 8 modelos (uma seção cada), ajuste fino, faixa de erro do total, calibração e teste anti-vazamento |
| **04. Avaliação** | Erro no teste (as mesmas semanas da v1), total da rede e por loja, lojas mais difíceis, **tradução do erro** por tipo de loja, sortimento e dia da semana, e variáveis mais importantes |
| **05. Implantação** | Modelo de produção com ficha, previsão real de ago–set/2015 e app Streamlit |

## 7. Principais descobertas da análise exploratória

### 🔎 Antes de tudo: entender o que a base diz

- **Somar engana.** A versão original comparava grupos pela soma das vendas. Mas 2015 vai só até julho, só 33 lojas abrem aos domingos e o sortimento "extra" tem 9 lojas: a soma desses grupos é pequena mesmo que cada loja venda muito. A v2 compara a **venda média por dia de loja aberta**.
- **180 lojas somem por 6 meses** (jul–dez/2014). Sem perceber isso, 2014 parece um ano ruim.

<div align="center">
<img src="Imagens/Lojas por dia.png" width="700px">
</div>

- Os **clientes do dia** explicam a venda quase inteira (correlação de **0,83**), mas só são conhecidos depois do dia: ficam fora do modelo.
- As vendas muito altas (**3,6%** dos dias, pela regra do IQR) são **reais**: promoções, Natal e lojas grandes. Ficam na base.

### As hipóteses

| # | Hipótese | Conclusão da v1 | Conclusão da v2 | Evidência |
|---|---|---|---|---|
| **H1** | Promoção vende mais | Verdadeira, +12% | ✅ Verdadeira | **+38,8%** (€ 5.930 → € 8.229), em **99,9%** das lojas |
| **H2** | Sortimento maior vende mais | Falsa | ⚠️ Em parte | O **extra** vende mais (**€ 8.558**), mas são só 9 lojas |
| **H3** | Concorrente perto vende menos | Falsa | ❌ Falsa | Sem relação (rho **−0,03**) |
| **H4** | O tipo de loja muda a venda | Tipo a vende mais | ⚠️ Só o tipo b | Tipo b: **€ 10.112**; a, c e d: ~€ 6.900 |
| **H5** | Vende mais a cada ano | Falsa (cai) | ✅ Verdadeira | **+3,2%** e **+3,5%** ao ano, nas mesmas lojas |
| **H6** | Vende mais no fim do ano | Falsa | ✅ Verdadeira | Dezembro: **€ 8.609** (**+24%**) |
| **H7** | Vende mais depois do dia 10 | Verdadeira | ❌ Falsa | Vende mais **antes**: € 7.287 contra € 6.800 |
| **H8** | Vende menos no fim de semana | Verdadeira | ✅ Verdadeira | Sábado: **€ 5.875** contra **€ 7.173** nos dias úteis |
| **H9** | Vende menos no feriado escolar | Verdadeira | ❌ Falsa | **+0,8%** na mesma loja |
| **H10** | Promo2 vende mais | Falsa | ❌ Falsa | As lojas da Promo2 vendem **menos** (€ 6.551 contra € 7.338) |
| **H11** | Concorrente novo reduz a venda | (não testada) | ✅ Verdadeira | **−5,9%** nos 3 meses seguintes, em **83%** das lojas |
| **H12** | Feriado vende mais | Falsa | ❌ Falsa | Nas mesmas lojas, sem diferença (p = 0,33) |

<div align="center">
<img src="Imagens/H1 Promocao.png" width="700px">
</div>

> 🧪 **Testes estatísticos:** cada hipótese tem um teste e uma medida de **tamanho do efeito**: Wilcoxon pareado (a mesma loja em duas situações: promoção, ano, feriado escolar, concorrente novo, feriado), Kruskal-Wallis + épsilon² (sortimento, tipo, mês), Mann-Whitney + correlação de postos (dia do mês, fim de semana, Promo2) e Spearman (distância). As características da loja são testadas com uma linha por loja, não por dia. Com 844 mil dias, quase tudo dá p-valor minúsculo: o que importa é o tamanho do efeito, não o p.

## 8. Construção do modelo

| Etapa | Como foi feito |
|---|---|
| **Variáveis** | 18 informações conhecidas **antes** do dia: loja, calendário, promoção, feriados e cadastro (tipo, sortimento, concorrência, Promo2). Os clientes ficam de fora |
| **Médias da loja** | O número da loja vira a **venda média** dela (geral, por dia da semana e com/sem promoção), calculada **só no treino**, dentro do `Pipeline` |
| **Divisão** | No tempo: treino até 07/05/2015, validação de 6 semanas e **teste nas mesmas 6 semanas da v1** (19/06 a 31/07/2015), aberto uma única vez |
| **Validação cruzada** | **5 janelas de 6 semanas**, sempre treinando no passado; é ela que escolhe o algoritmo |
| **Algoritmos** | Média da loja (referência), planilha (loja × dia × promoção), Regressão Linear, Lasso, Random Forest, XGBoost, LightGBM e HistGradientBoosting, com o **log da venda** como alvo e métricas em euros |
| **Escolha** | **XGBoost**: menor MAPE na validação cruzada (**10,6% ± 1,3%**) e 1º lugar em todas as métricas, empatado com o LightGBM e o HGB dentro do desvio |
| **Ajuste fino** | `RandomizedSearchCV` com a validação cruzada no tempo: erro do total na validação de **4,4%** para **3,9%** |
| **Anti-vazamento** | Com os clientes do dia, o MAPE "cairia" de 9,5% para **5,3%**; o "modelo da média" da v1 acertava o total com **0,0%** de erro porque usava as respostas |

<div align="center">
<img src="Imagens/Comparacao dos modelos.png" width="850px">
</div>

## 9. Resultado do modelo

| Teste (19/06 a 31/07/2015, 1.115 lojas) | Modelo final | Planilha | Média da loja | Versão original |
|---|---|---|---|---|
| **Erro do total de 6 semanas por loja** | **3,7%** | 6,1% | 6,0% | não medido |
| Erro do dia (MAE) | **€ 614** | € 739 | € 1.440 | € 933 |
| Erro do dia (MAPE) | **9,2%** | 10,7% | 22,8% | 13,3% |
| R² | **0,912** | 0,869 | 0,604 | — |

- Somando a rede inteira, o modelo previu **€ 291,5 milhões** para uma venda real de **€ 289,6 milhões** (**+0,68%**). Só **5,2%** das lojas têm o total errado em mais de 10%.
- O erro do total vai de **3,4%** (lojas tipo d) a **4,7%** (tipo b). O **sábado** é o dia mais difícil (MAPE de **12,6%**).
- Os erros grandes vêm de **eventos**: a loja 909, a pior, ficou fechada cerca de duas semanas em julho e voltou vendendo bem acima do normal.
- 🎯 **A faixa de erro cumpriu o prometido:** medida em **5 períodos** de 6 semanas antes do teste, ela diz que o total real fica entre **95,4% e 108,4%** do previsto em 80% das lojas. No teste, cobriu **78,9%** das lojas. Se tivesse sido medida só no último período (em que o modelo errava para baixo), cobriria apenas **60,4%**.

<div align="center">
<img src="Imagens/Previsto x observado.png" width="850px">
</div>

**O que mais pesa na previsão** (importância por permutação): a **loja** (pelas suas médias), a **promoção**, o **dia da semana** e o **dia do mês**. Num dia típico, a promoção soma **+20%**, a segunda-feira **+12%** e a semana antes do Natal **+50%**.

## 10. App interativo (Streamlit)

O app tem 3 abas:
1. **🏪 Loja:** escolha uma loja e veja o **total previsto** de 01/08 a 17/09/2015, o **mínimo e o máximo prováveis**, o cadastro da loja e a venda prevista de cada dia.
2. **🧾 Orçamento de reforma:** escolha a **parcela do faturamento** destinada à reforma e veja a verba de cada loja e da rede (ex.: com 5%, cerca de **€ 12 milhões** para 856 lojas), com a opção conservadora pelo mínimo provável. A tabela pode ser baixada em CSV.
3. **ℹ️ Sobre o modelo:** erro do total e do dia no teste, comparação com a média da loja e os avisos da ficha do modelo.

```bash
streamlit run app.py
```

> ℹ️ O app usa `reports/previsoes_6_semanas.csv` e `Model/modelo_rossmann_v2.json`, gerados ao executar o notebook até a Fase 05.

> 📱 A versão original entregava a previsão por um **bot no Telegram**, com uma API no Heroku. O plano gratuito do Heroku acabou em 2022 e o bot parou; as pastas `rossmann/` e `rossmann-telegram-api/` ficam como histórico.

## 11. O que mudou da v1 para a v2

A versão original foi mantida em `Notebooks/` (e o bot em `rossmann-telegram-api/`) para mostrar a evolução.

| # | Versão original (v1) | Problema | Versão v2 |
|---|---|---|---|
| 1 | EDA com **somas** das vendas | Mistura quanto cada loja vende com quantos dias e lojas o grupo tem: "as vendas caem", "domingo vende menos", "extra vende pouco" | **Médias** por dia de loja aberta e comparação da mesma loja: 6 conclusões se invertem |
| 2 | Hipóteses só com gráficos | Não separa efeito real de acaso | **12 hipóteses** com teste estatístico e tamanho do efeito |
| 3 | Lacuna das 180 lojas não percebida | 2014 parecia um ano de queda | Lacuna documentada; anos comparados nas mesmas lojas e meses |
| 4 | "Modelo da média" calculado nas semanas do teste | Usava as respostas (erro do total de 0,0%) | Média da loja e planilha treinadas só no passado |
| 5 | Escalas ajustadas na base toda; o arquivo errado salvo para o mês e o ano | Vazamento e produção transformando os dados errado | Tudo dentro de um `Pipeline`, ajustado só no treino |
| 6 | Distância nula → média; datas nulas → 0 | Inventava concorrentes e um "ano 0" | Nulos tratados pelo significado, com indicadores |
| 7 | XGBoost com `colsample_bytee` (ignorado) e modelo final com hiperparâmetros não testados | Ajuste não reproduzível | `RandomizedSearchCV` com validação cruzada no tempo e sementes fixas |
| 8 | README diz que o Random Forest venceu; a produção usa o XGBoost | Documentação contraditória | Escolha registrada pela validação cruzada (DA-08) |
| 9 | Cenários = total ± erro de **um dia** | Faixa de ±0,3%, que cobre 5% das lojas; "loja 286 com erro de 4%" e MAPE de 44% | Faixa relativa do total de 6 semanas, medida em 5 períodos: cobriu **78,9%** das lojas no teste (meta: 80%) |
| 10 | Sem erro do total por loja | Não respondia à pergunta do CFO | **Erro do total de 6 semanas** como métrica do negócio: **3,7%** |
| 11 | Prophet com todas as lojas numa série só | Misturava 1.115 lojas diferentes | Prophet retirado, com o motivo registrado |
| 12 | Vendas em R$ | A base é alemã | **Euros** |
| 13 | Bot no Telegram (Heroku) | Parou com o fim do plano gratuito | App Streamlit com **14 testes** no GitHub Actions |
| 14 | Resultado no teste: MAE de € 933, MAPE de 13,3% | — | MAE de **€ 614**, MAPE de **9,2%** nas mesmas semanas |

## 12. Insights

1. **Compare médias, não somas:** a soma inverteu 6 conclusões da versão original.
2. **Conte quantas unidades aparecem em cada período** antes de comparar: 180 lojas sumiram por 6 meses.
3. **A promoção é a maior alavanca da rede:** +38,8% na venda do dia, não +12%.
4. **Um concorrente novo custa ~6% da venda**, e a loja não recupera em 6 meses. A distância de um concorrente antigo não diz nada.
5. **As lojas tipo b são diferentes:** quase 2 mil clientes por dia e gasto de € 5 por cliente, como lojas de passagem.
6. **Para orçamento, o total é o número certo:** o dia erra 9,2%, o total de 6 semanas erra 3,7%.
7. **A régua de comparação também precisa ser testada:** a "média da loja" da v1 acertava o total com 0,0% de erro porque usava as respostas.
8. **A incerteza depende do período em que foi medida:** medida num período só, a faixa de 80% cobriria 60% no teste; medida em 5 períodos, cobriu 79%.
9. **Os erros grandes vêm de eventos** (reformas, reinaugurações): vale revisar essas lojas à mão.

## 13. Limites

- **A faixa ainda é um pouco otimista para baixo:** **18,5%** das lojas venderam menos que o mínimo provável (o esperado era 10%), e nas lojas menores a cobertura cai para **71,3%**. A faixa é a mesma para todas as lojas.
- **Dados de 2013 a 2015:** para usar hoje, é preciso retreinar com dados recentes.
- **Prevê só dias abertos:** o calendário de abertura precisa ser informado.
- **Eventos fora do padrão não são previstos:** reinaugurações, reformas e um concorrente abrindo durante as 6 semanas.
- **Lojas novas** recebem a média da rede.
- **Correlação não é causa:** a Promo2 está ligada a vendas menores, mas a base não diz se ela atrapalha ou se foi dada às lojas mais fracas.

## 14. Próximos passos

1. **Faixa por tamanho de loja** (ou intervalos conformes condicionais), porque as lojas menores ficaram com cobertura de 71,3%.
2. **Variável de tendência por loja** (crescimento recente), para reduzir a troca de viés entre períodos.
3. **Marcar eventos** (períodos longos de loja fechada) e avisar no app as lojas com reforma recente.
4. **Busca de hiperparâmetros maior** e comparação com o LightGBM ajustado, 3× mais rápido.
5. **Dados externos:** clima e feriados escolares por estado.
6. **Publicar o app** no Streamlit Community Cloud e gravar um GIF para o README.
