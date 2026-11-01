# Registro de decisões analíticas

Este documento registra as principais decisões do projeto, no formato
**Contexto → Decisão → Alternativas consideradas → Consequência → Onde está**.
Cada decisão tem um código (DA-XX) para ser citada em commits e pull requests.

Os números vêm das saídas do notebook [`Notebooks/Rossmann_Store_Sales_Prediction - v2.ipynb`](../Notebooks/).

| Código | Decisão |
|---|---|
| [DA-01](#da-01--médias-em-vez-de-somas-na-eda) | Médias em vez de somas na EDA |
| [DA-02](#da-02--características-da-loja-testadas-no-nível-da-loja) | Características da loja testadas no nível da loja |
| [DA-03](#da-03--lacuna-das-180-lojas-documentada) | Lacuna das 180 lojas documentada |
| [DA-04](#da-04--modelo-só-com-os-dias-de-loja-aberta-e-com-venda) | Modelo só com os dias de loja aberta e com venda |
| [DA-05](#da-05--nulos-tratados-pelo-que-significam) | Nulos tratados pelo que significam |
| [DA-06](#da-06--retirada-dos-clientes-do-dia) | Retirada dos clientes do dia |
| [DA-07](#da-07--separação-no-tempo-com-o-teste-da-v1) | Separação no tempo, com o teste da v1 |
| [DA-08](#da-08--validação-cruzada-no-tempo-decide-o-algoritmo) | Validação cruzada no tempo decide o algoritmo |
| [DA-09](#da-09--médias-da-loja-dentro-do-pipeline) | Médias da loja dentro do `Pipeline` |
| [DA-10](#da-10--alvo-em-log-métricas-em-euros) | Alvo em log, métricas em euros |
| [DA-11](#da-11--réguas-de-comparação-honestas) | Réguas de comparação honestas |
| [DA-12](#da-12--erro-do-total-de-6-semanas-como-métrica-do-negócio) | Erro do total de 6 semanas como métrica do negócio |
| [DA-13](#da-13--xgboost-ajustado-com-busca-enxuta) | XGBoost ajustado, com busca enxuta |
| [DA-14](#da-14--faixa-relativa-do-total-medida-nas-5-janelas) | Faixa relativa do total, medida nas 5 janelas |
| [DA-15](#da-15--modelo-de-produção-retreinado-com-todos-os-dias) | Modelo de produção retreinado com todos os dias |
| [DA-16](#da-16--app-streamlit-no-lugar-do-bot-do-telegram) | App Streamlit no lugar do bot do Telegram |
| [DA-17](#da-17--grid-x-random-x-bayesiana-random-search-mantido) | Grid x Random x Bayesiana: Random Search mantido |
| [DA-18](#da-18--shap-calculado-pelo-próprio-xgboost-por-grupo-de-variáveis) | SHAP calculado pelo próprio XGBoost, por grupo de variáveis |
| [DA-19](#da-19--quando-retreinar-backtest-sla-semanal-e-drift) | Quando retreinar: backtest, SLA semanal e drift |

---

## DA-01 — Médias em vez de somas na EDA

- **Contexto:** a versão original comparava grupos pela **soma** das vendas. Só que os grupos têm tamanhos muito diferentes: 2015 vai só até julho, apenas 33 lojas abrem aos domingos, o sortimento "extra" tem 9 lojas e há 21 dias depois do dia 10 contra 10 antes. A soma mistura quanto cada loja vende com quantos dias e lojas o grupo tem.
- **Decisão:** comparar sempre a **venda média por dia de loja aberta** e, quando possível, a **mesma loja** em duas situações (Wilcoxon pareado). A soma aparece ao lado, como na v1, para mostrar a diferença.
- **Alternativas consideradas:** somas (versão original); somas normalizadas pelo número de lojas (equivalente à média, mas mais difícil de explicar).
- **Consequência:** 6 das 11 hipóteses testadas pela v1 mudam de conclusão. A venda média sobe todo ano (€ 6.815 → € 7.026 → € 7.088) em vez de cair; dezembro é o melhor mês (+24%); as lojas vendem mais **antes** do dia 10 (€ 7.287 contra € 6.800); o sortimento extra é o que mais vende por loja (€ 8.558); e a promoção soma +38,8%, não +12%.
- **Onde está:** item 3.1 (regras da EDA) e H1 a H12 da Fase 02; resumo no item 3.4.

## DA-02 — Características da loja testadas no nível da loja

- **Contexto:** sortimento, tipo de loja, distância do concorrente e Promo2 são características da **loja**, que se repetem em centenas de dias. Testar com 844 mil dias daria uma falsa precisão (p-valores minúsculos para qualquer diferença).
- **Decisão:** nesses testes, cada loja entra **uma vez**, com a sua venda média (1.115 valores). Os efeitos do dia (promoção, dia da semana, feriado escolar) são testados com a mesma loja nas duas situações.
- **Alternativas consideradas:** testes com todos os dias; modelos mistos com efeito aleatório por loja (mais corretos, mas mais difíceis de explicar a um leigo).
- **Consequência:** os efeitos das características da loja aparecem como de fato são: pequenos (épsilon² de 0,028 no sortimento e 0,011 no tipo; rho de −0,03 na distância).
- **Onde está:** `por_loja` no item 3.1; H2, H3, H4 e H10.

## DA-03 — Lacuna das 180 lojas documentada

- **Contexto:** 180 lojas não têm nenhum registro entre 01/07/2014 e 31/12/2014 (184 dias). A v1 não percebeu, e a soma de 2014 perdia o 2º semestre dessas lojas.
- **Decisão:** documentar a lacuna (gráfico de lojas por dia) e comparar os anos só com as **935 lojas sem lacuna** e os **mesmos meses** (janeiro a junho). Para o modelo, os dias que existem são usados normalmente.
- **Alternativas consideradas:** retirar as 180 lojas (o modelo deixaria de prever 16% da rede); preencher os 6 meses com estimativas (inventaria dados).
- **Consequência:** o crescimento real aparece: +3,2% de 2013 para 2014 e +3,5% de 2014 para 2015 (mediana por loja). A 1ª janela da validação cruzada (Natal de 2014) tem menos linhas por causa da lacuna.
- **Onde está:** item 2.3 da Fase 02 e H5; `lojas_lacuna` no notebook.

## DA-04 — Modelo só com os dias de loja aberta e com venda

- **Contexto:** 17,0% dos registros são dias de loja fechada (venda sempre zero, 81,7% deles aos domingos), e 54 dias têm a loja aberta com venda zero. O CFO sabe com antecedência quais dias cada loja abre.
- **Decisão:** treinar e avaliar só com os **844.338 dias** em que a loja abriu e vendeu. Na previsão, os dias fechados recebem venda zero antes da soma.
- **Alternativas consideradas:** incluir os dias fechados (o modelo gastaria esforço aprendendo uma regra conhecida, e o MAPE ficaria indefinido com venda zero).
- **Consequência:** o modelo se concentra no que é incerto. A ficha do modelo avisa que os dias fechados precisam ser zerados.
- **Onde está:** `filtrar_dias_com_venda` em `src/dados.py`; item 2.5 da Fase 02; Fase 05, item 2.

## DA-05 — Nulos tratados pelo que significam

- **Contexto:** as datas da Promo2 faltam em 544 lojas, que são exatamente as que não participam; a data de abertura do concorrente falta em 354; a distância falta em 3. A v1 preenchia a distância com a média (5,4 km) e as datas com 0, usadas cruas como "ano 0".
- **Decisão:** distância nula vira **100 km** com o indicador `sem_concorrente`; criar `meses_concorrente` (0 sem data, com o indicador `concorrente_data_desconhecida`), `semanas_promo2` e `promo2_mes_ativo` (0 para quem não participa).
- **Alternativas consideradas:** média ou mediana (inventa um concorrente a uma distância comum); retirar as lojas com nulo (perderia metade da rede pela Promo2).
- **Consequência:** 0 nulos nas variáveis do modelo, sem inventar valores. Na confirmação do item 4.3, as variáveis criadas baixam o erro do total de 5,6% para 4,8%.
- **Onde está:** `criar_variaveis` em `src/dados.py`; item 2.4 da Fase 02; item 4.3 da Fase 03.

## DA-06 — Retirada dos clientes do dia

- **Contexto:** o número de clientes tem correlação de 0,832 com a venda, mas só é conhecido depois que o dia acaba. O arquivo das 6 semanas a prever (`test.csv`) nem tem essa coluna.
- **Decisão:** `clientes` fica fora do modelo (`PROIBIDAS`), e continua só na EDA.
- **Alternativas consideradas:** prever os clientes primeiro e usá-los como entrada (dois modelos encadeados, com o erro do primeiro somado ao do segundo).
- **Consequência:** com os clientes, o MAPE "cairia" de 9,5% para 5,3% (R² de 0,968), um resultado impossível de repetir no uso real. O teste anti-vazamento mostra isso.
- **Onde está:** `PROIBIDAS` em `src/config.py`; seção 🛡️ da Fase 02; item 9 da Fase 03.

## DA-07 — Separação no tempo, com o teste da v1

- **Contexto:** com dados no tempo, uma separação aleatória colocaria dias de julho de 2015 no treino, e o modelo "conheceria o futuro" do teste. O padrão do portfólio é 60/20/20.
- **Decisão:** treino até 07/05/2015, validação de 08/05 a 18/06/2015 (6 semanas) e teste de 19/06 a 31/07/2015, **as mesmas 6 semanas da versão original**. O teste é aberto uma única vez.
- **Alternativas consideradas:** 60/20/20 por data (validação e teste de vários meses, longe do uso real de 6 semanas); separação aleatória (vazamento do futuro).
- **Consequência:** o resultado pode ser comparado diretamente com o da v1 (MAE de € 933 contra € 614). A validação ficou com vendas mais altas (€ 7.286) que o teste (€ 6.995), e é um dos motivos para medir a faixa de erro em vários períodos (DA-14).
- **Onde está:** `separar_periodos` em `src/dados.py`; `INICIO_VALIDACAO` e `INICIO_TESTE` em `src/config.py`; item 4.1 da Fase 03.

## DA-08 — Validação cruzada no tempo decide o algoritmo

- **Contexto:** uma validação de 6 semanas é uma única amostra do futuro, e os três modelos de árvores em sequência ficaram a menos de 0,1 ponto de MAPE entre si nela.
- **Decisão:** em treino + validação, 5 janelas de 6 semanas, sempre treinando com tudo o que vem antes (janela crescente). O algoritmo do ajuste fino é o de menor **MAPE médio** nas 5 janelas.
- **Alternativas consideradas:** `KFold` aleatório (treinaria no futuro); `TimeSeriesSplit` do scikit-learn (as janelas não seguem semanas de calendário).
- **Consequência:** XGBoost escolhido (MAPE de 10,6% ± 1,3% e erro do total de 4,8% ± 0,9%), empatado com o HGB e o LightGBM dentro do desvio. A validação cruzada mostrou que as réguas oscilam muito mais (planilha: total de 9,2% ± 5,3%), sobretudo no Natal.
- **Onde está:** `dobras_temporais` em `src/validacao.py`; itens 4.1 e 5 (Validação cruzada) da Fase 03.

## DA-09 — Médias da loja dentro do `Pipeline`

- **Contexto:** o que mais diferencia as lojas é o tamanho de cada uma, e o número da loja não tem ordem. As médias de venda da loja resolvem isso, mas usam o alvo: calculadas com a base toda, vazariam a validação e o teste.
- **Decisão:** o transformador `MediasDaLoja` calcula, no `fit` (só com o treino de cada dobra), a venda média da loja no geral, por dia da semana e com/sem promoção, e troca a coluna `loja` por essas médias. Loja sem histórico recebe a média da rede.
- **Alternativas consideradas:** número da loja como número (versão original); one-hot com 1.115 colunas (pesado na memória de 8 GB); médias calculadas fora do `Pipeline` (vazamento na validação cruzada).
- **Consequência:** na confirmação do item 4.3, as médias baixam o erro do total de 7,0% (número da loja) para 5,6%. A loja é a variável mais importante do modelo final (+€ 2.036 na permutação).
- **Onde está:** `MediasDaLoja` em `src/modelos.py`; `montar_modelo` no item 3 da Fase 03.

## DA-10 — Alvo em log, métricas em euros

- **Contexto:** a venda tem cauda longa (assimetria de 1,59; −0,11 em log). Em euros, as lojas grandes dominariam o treino.
- **Decisão:** treinar com `log(1 + venda)` via `TransformedTargetRegressor`, que devolve a previsão em euros. Todas as métricas são calculadas em euros.
- **Alternativas consideradas:** alvo em euros (testado no protótipo, com MAPE um pouco maior e o mesmo erro do total); raiz quadrada.
- **Consequência:** um erro de 10% pesa o mesmo numa loja pequena e numa grande, coerente com o MAPE e com o erro do total por loja.
- **Onde está:** `montar_modelo` no item 3 da Fase 03.

## DA-11 — Réguas de comparação honestas

- **Contexto:** o "modelo da média" da v1 calculava a média de cada loja **nas próprias semanas avaliadas**. Com isso, o erro do total é 0,0% por construção: a média do período × o número de dias é o total real.
- **Decisão:** duas réguas treinadas só com o passado: a **média da loja** (referência) e a **média da loja × dia da semana × promoção** (o que uma planilha faria). A conta da v1 é reproduzida ao lado, para mostrar a diferença.
- **Alternativas consideradas:** só a média geral da rede (régua fraca demais: ignora o tamanho das lojas).
- **Consequência:** no teste, o modelo final erra o total em 3,7%, contra 6,0% da média da loja e 6,1% da planilha. A planilha erra pouco no dia (10,7%), mas não acompanha o crescimento das vendas.
- **Onde está:** `MediaPorGrupo` em `src/modelos.py`; seções "Média da loja" e "Planilha" do item 5 da Fase 03; Fase 04, item 1.

## DA-12 — Erro do total de 6 semanas como métrica do negócio

- **Contexto:** o CFO decide a verba de reforma com o **total de 6 semanas** de cada loja, não com a venda de um dia. A v1 só media o erro diário.
- **Decisão:** além do padrão `ml_error` (MAE, MAPE, RMSE, R², MPE), medir o **erro do total por loja**: em média, quanto o total previsto de uma loja erra o total real.
- **Alternativas consideradas:** só as métricas diárias (não respondem à pergunta do CFO); erro do total da rede inteira (esconde os erros de cada loja, que se compensam: +0,68% no teste).
- **Consequência:** essa métrica entra na comparação, na validação cruzada e no score geral. Ela separa modelos que empatam no dia: o Random Forest e o HGB têm o mesmo MAPE na validação (9,8%), mas erros do total de 6,2% e 5,0%.
- **Onde está:** `totais_por_loja` e `erro_total_6_semanas` em `src/avaliacao.py`.

## DA-13 — XGBoost ajustado, com busca enxuta

- **Contexto:** a máquina tem 8 GB de RAM e 2 núcleos; cada XGBoost leva cerca de 70 a 110 s para treinar. A v1 tinha um parâmetro digitado errado (`colsample_bytee`, ignorado) e um modelo final com hiperparâmetros fora dos testados.
- **Decisão:** `RandomizedSearchCV` com 8 combinações, avaliadas em 3 janelas de 6 semanas dentro do treino (sem tocar na validação), escolhendo pelo MAPE. O ajustado é conferido na validação e o final (`clone`) treina em treino + validação.
- **Alternativas consideradas:** busca maior (horas de execução na máquina disponível); LightGBM, empatado na validação cruzada e 3× mais rápido.
- **Consequência:** 500 árvores, profundidade 10, taxa 0,05 e 70% das colunas. Na validação, o erro do total cai de 4,4% para 3,9%; no teste, fica em 3,7% (MAPE de 9,2%).
- **Onde está:** item 7.2 da Fase 03; `hiperparametros` na ficha do modelo. A comparação com Grid e Bayesiana está na DA-17.

## DA-14 — Faixa relativa do total, medida nas 5 janelas

- **Contexto:** a v1 dava como cenários "total ± erro de um dia", uma faixa de ±0,3% que cobre só 5,1% das lojas. A loja 286 aparecia com "erro de 4%" e MAPE de 44%. O viés do modelo muda de um período para outro: a mediana de real ÷ previsto vai de 97,4% a 104,4% entre as janelas da validação cruzada.
- **Decisão:** faixa **relativa** do total de 6 semanas, entre os percentis 10 e 90 da razão real ÷ previsto nas **5 janelas** da validação cruzada (5.396 totais de loja; em cada janela, o modelo ajustado treina no passado e prevê as 6 semanas seguintes): de **95,4% a 108,4%** do previsto. A cobertura é medida no teste, e a faixa **não é recalculada** com o teste.
- **Alternativas consideradas:** faixa em euros (não acompanha o tamanho da loja); faixa só na validação (97,7% a 108,1%, que herda o viés de um único período); recalcular a faixa no teste (usaria o gabarito); intervalos conformes por tamanho de loja (ficou como próximo passo).
- **Consequência:** no teste, a faixa cobriu **78,9%** das lojas (meta: 80%); a faixa de uma janela só teria coberto **60,4%**. Ainda há mais lojas abaixo (18,5%) que acima (2,6%), porque o modelo superestimou no teste (MPE de +2,3%), e as lojas menores têm cobertura de 71,3%. Custa 4 treinos a mais do modelo ajustado (cerca de 7 minutos).
- **Uso para o negócio:** a faixa vira os 3 cenários de cada loja: **pior** (previsto × limite inferior), **provável** (o previsto) e **melhor** (previsto × limite superior). A tabela por loja no teste mostra também o total real, o MAE e o MAPE do dia, no formato da tabela de cenários da v1, mas com a faixa do total.
- **Onde está:** `faixa_relativa`, `cobertura_faixa` e `cenarios_por_loja` em `src/avaliacao.py`; itens 8.2 e 8.3 e "Cenários por loja" da Fase 03; `reports/cenarios_lojas_teste.csv` e `reports/cenarios_6_semanas.csv`; aba **📊 Cenários** do app; `cobertura_faixa_teste` na ficha.

## DA-15 — Modelo de produção retreinado com todos os dias

- **Contexto:** o modelo final foi treinado até 18/06/2015 para ser avaliado no teste. Para prever agosto e setembro de 2015, as 6 semanas mais recentes são as mais úteis.
- **Decisão:** um `clone` do modelo final, retreinado com os 844.338 dias até 31/07/2015, salvo em `Model/modelo_rossmann_v2.pkl` com a faixa e a lista de variáveis, mais a ficha em JSON.
- **Alternativas consideradas:** usar o modelo final como está (perderia as semanas de junho e julho).
- **Consequência:** a previsão de 01/08 a 17/09/2015 (856 lojas, 35.104 dias abertos) soma € 240,1 milhões e fica em `reports/previsoes_6_semanas.csv`. As métricas e a faixa valem pela avaliação feita no teste.
- **Onde está:** Fase 05 do notebook; `Model/`; `reports/previsoes_6_semanas.csv`.

## DA-16 — App Streamlit no lugar do bot do Telegram

- **Contexto:** a v1 entregava a previsão por um bot no Telegram com uma API no Heroku, cujo plano gratuito acabou em 2022. O Prophet da v1 foi treinado com todas as lojas misturadas numa série só.
- **Decisão:** um app Streamlit que lê as previsões já calculadas, com 3 abas (loja, orçamento de reforma e sobre o modelo); depois ganhou a aba de monitoramento (DA-19) e a aba de cenários (DA-14). O Prophet sai da comparação. As pastas `rossmann/` e `rossmann-telegram-api/` ficam como histórico.
- **Alternativas consideradas:** reativar a API em outro serviço (manutenção de servidor para um projeto de portfólio); um Prophet por loja (1.115 modelos, sem as variáveis do cadastro).
- **Consequência:** o CFO vê o total de cada loja com a faixa, simula a parcela do faturamento para reforma e baixa a tabela em CSV. O app tem 5 testes automatizados (`AppTest`).
- **Onde está:** `app.py` (igual à última célula do notebook); `tests/test_app.py`.

## DA-17 — Grid x Random x Bayesiana: Random Search mantido

- **Contexto:** a DA-13 escolheu os hiperparâmetros com Random Search. Era preciso saber se outro método de busca encontraria um modelo melhor, com um treino caro (cerca de 140 s por XGBoost).
- **Decisão:** comparar Grid Search (profundidade x taxa de aprendizado, 4 combinações), Random Search (6 hiperparâmetros, 8 sorteios) e Bayesiana com Optuna (8 tentativas, amostrador TPE e poda pela mediana) nas **mesmas 3 janelas de 6 semanas** do treino, pelo MAPE. Regra definida antes da busca: o Random Search só é trocado se outro método ganhar por mais que o desvio entre as janelas.
- **Alternativas consideradas:** a grade completa do espaço do Random Search (216 combinações x 3 janelas, ~25 horas); mais tentativas na Optuna (mais horas de execução, para diferenças menores que o desvio).
- **Consequência:** Random Search 9,88% ± 0,97%, Optuna 9,95% ± 1,16% e Grid 10,02% ± 1,12% (padrão: 10,47%). O Random Search ficou, e o modelo e as métricas de teste não mudaram. A Optuna podou 3 das 8 tentativas (18 modelos em vez de 24). A busca toda custa cerca de 2h20.
- **Onde está:** itens 7.1 a 7.5 da Fase 03; `objetivo_cv_com_poda` e `historico_optuna` em `src/otimizacao.py`; `metodo_de_busca` na ficha do modelo.

## DA-18 — SHAP calculado pelo próprio XGBoost, por grupo de variáveis

- **Contexto:** a importância por permutação diz o que pesa no modelo como um todo, mas não explica a previsão de uma loja. O CFO precisa saber por que uma loja recebeu aquele valor.
- **Decisão:** valores SHAP exatos do XGBoost (`pred_contribs`), na escala do log da venda: cada contribuição multiplica a venda, e exp(contribuição) − 1 é o efeito em %. As colunas do one-hot e as 3 médias da loja são somadas em 8 grupos de negócio. A eficiência é conferida: valor base + contribuições = log da previsão.
- **Alternativas consideradas:** a biblioteca `shap` (mais uma dependência, para o mesmo cálculo das árvores); SHAP numa amostra de dias (mais rápido, ficou como próximo passo).
- **Consequência:** diferença máxima de 2,5e-05 na conferência. O tamanho da loja domina (31,7% de efeito médio), depois calendário (5,1%) e promoção (4,9%); a ordem bate com a permutação (correlação de postos de 0,74). O SHAP divide o crédito entre variáveis ligadas: parte do efeito da promoção fica na média da loja com promoção. O efeito por loja vai para a aba Loja do app. O cálculo exato em todos os dias leva cerca de 2h20.
- **Onde está:** item 3 da Fase 04 e item 2 da Fase 05; `src/explicacao.py`; `reports/shap_lojas.csv`.

## DA-19 — Quando retreinar: backtest, SLA semanal e drift

- **Contexto:** o modelo prevê 6 semanas por vez. Era preciso decidir de quanto em quanto tempo retreiná-lo e como saber, em produção, que ele deixou de funcionar.
- **Decisão:** backtest nas 5 janelas da validação cruzada com 3 estratégias (retreino a cada 6 semanas com todo o histórico, retreino com os últimos 12 meses e modelo sem retreino). SLA semanal pelo percentil 90 das 30 semanas da validação cruzada, para o MAPE da semana e o erro do total da rede. Drift por PSI e K-S, comparando as entradas e a previsão do período novo com as das 5 janelas, sem precisar da venda real. Variáveis de calendário ficam fora do drift, porque mudam por construção.
- **Alternativas consideradas:** SLA pelo erro do teste (usaria o gabarito para definir a régua); retreino só quando o SLA estourar (o erro sobe aos poucos, antes de estourar).
- **Consequência:** retreinar a cada 6 semanas dá MAPE médio de 10,25% e erro do total de 4,43%, contra 10,86% e 5,38% sem retreino; os últimos 12 meses não ajudam (10,53%). SLA: MAPE da semana até 12,9% e erro do total da rede até ±8,1%; as 6 semanas do teste ficaram dentro. Só o feriado escolar mudou (PSI de 0,22, férias de julho); a previsão ficou estável (PSI de 0,006).
- **Onde está:** "3 - Monitoramento em produção" da Fase 05; `src/monitoramento.py`; bloco `monitoramento` da ficha; aba "📡 Monitoramento" do app.
