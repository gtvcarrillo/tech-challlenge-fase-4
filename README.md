# Contexto do Projeto
Modelo preditivo de Deep Learning com **Long Short-Term Memory (LSTM)** para prever o **preço de fechamento (Close)** do ativo **ITUB4.SA** a partir de dados históricos do Yahoo Finance.  
Além do modelo base, foi aplicada uma **correção por BIAS (offset)** (calibração aditiva) para reduzir uma tendência sistemática de sub/superestimação nas previsões.  
Também foi incluída uma **projeção até 31/12/2026 por cenários**, usando distribuição histórica de **log-retornos** (mais adequada para horizontes longos do que a previsão recursiva determinística).

# Arquitetura do Projeto
> Neste repositório, o foco implementado é o **desenvolvimento do modelo e avaliação** em notebook.  
> Os itens de **deploy em API, Docker e monitoramento** ficam como próximos passos para cumprir integralmente o Tech Challenge.

**Fluxo atual (implementado):**
- **Yahoo Finance (yfinance)**: fonte de dados para preços históricos.
- **Notebook de Desenvolvimento (LSTM)**: coleta, pré-processamento, normalização, janelamento, treino, validação, teste com dados recentes, correção por BIAS, export de previsões.
- **Export CSV**: tabela final com previsões calibradas (BIAS) e projeção por cenários até 2026.

**Fluxo alvo (recomendado para fechar os requisitos de deploy):**
- **API (Flask/FastAPI)** servindo o modelo treinado.
- **Docker** para empacotamento.
- **Monitoramento** (logs, latência, uso de recursos).

# Desenvolvimento do Modelo
O desenvolvimento do modelo foi feito no notebook:
- `./notebooks/Desenvolvimento_Modelo_ITUB4_final_simples_com_teste_bias_final.ipynb`

## 1) Coleta dos Dados
Os dados históricos do ativo **ITUB4.SA** são coletados do **Yahoo Finance** via `yfinance` (ou carregados a partir de um CSV previamente gerado com yfinance).  
O dataset é ordenado por data e utilizado apenas o preço de fechamento (**Close**).

## 2) Exploração dos Dados (EDA)
Nesta etapa é exibido um gráfico de linha do preço de fechamento e estatísticas descritivas para entender tendência e variabilidade.

<img width="833" height="387" alt="image" src="https://github.com/user-attachments/assets/40d377f6-25d6-4891-8005-8cb9b110d07e" />

## 3) Normalização
Antes de alimentar o modelo LSTM, os valores de fechamento são normalizados com **MinMaxScaler (0–1)**.  
Para evitar vazamento de informação, o scaler é ajustado **somente no período de desenvolvimento (DEV)**.

## 4) Preparação dos Dados (Janelamento)
A série é convertida em sequências com **janela de 60 dias** (window_size = 60):
- **X**: janela deslizante de 60 fechamentos normalizados
- **y**: fechamento do próximo dia

## 5) Treino e Validação (até o cutoff)
Para respeitar o requisito de avaliação em dados não vistos, o notebook separa:
- **DEV (desenvolvimento)**: dados **antes** do cutoff `2025-09-01` (treino + validação)
- **RECENTE (teste final)**: dados **a partir** do cutoff `2025-09-01`

No DEV, é aplicado split temporal (ex.: 80/20) para:
- **Treino**: ajuste dos pesos do modelo
- **Validação**: avaliação e calibração do BIAS

## 6) Construção do Modelo LSTM
O modelo utiliza LSTM para capturar padrões temporais na série, com camada(s) recorrente(s) e saída `Dense(1)` para prever um único valor (próximo fechamento).

## 7) Avaliação e Correção por BIAS (offset)
Além das métricas **RAW** (previsão direta), é aplicada uma **calibração por BIAS** para corrigir um erro sistemático.

### Como o BIAS é calculado

O **BIAS** é um *offset* (correção aditiva) calculado como a **média do erro** no conjunto de **validação** (DEV):

- Erro por amostra: `e_i = y_i - ŷ_i`
- BIAS: `BIAS = (1/n) * Σ (y_i - ŷ_i)`

A previsão calibrada (com BIAS) é:

- `ŷ_cal_i = ŷ_i + BIAS`

Onde:
- `y_i` = valor real (fechamento)
- `ŷ_i` = previsão do modelo (RAW)
- `n` = número de amostras na validação

**Importante:** o BIAS é estimado **somente na validação (DEV)** e aplicado depois no teste recente, evitando *data leakage*.

### Resultados (validação DEV)
O notebook imprime métricas MAE/RMSE/MAPE para:
- **RAW** (sem calibração) = MAE: 0.7645 | RMSE: 0.9120 | MAPE: 2.1669%
- **BIAS** (com calibração) =  MAE: 0.4391 | RMSE: 0.5484 | MAPE: 1.2722%

<img width="822" height="367" alt="image" src="https://github.com/user-attachments/assets/34388876-d090-4b01-b057-a50d151414ef" />

## 8) Teste com Dados Recentes (>= 2025-09-01)
O período recente é usado como **teste fora da amostra** (holdout).  
O notebook compara:
- **LSTM RAW** = MAE: 0.9318 | RMSE: 0.9999 | MAPE: 2.4418%
- **LSTM + BIAS** = MAE: 0.3349 | RMSE: 0.4156 | MAPE: 0.8754%
- **Baseline Persistência** (amanhã = hoje) = MAE: 0.2865 | RMSE: 0.3591 | MAPE: 0.7515%

### Resultado observado
No exemplo executado, a correção por BIAS reduziu significativamente o erro no teste recente (RAW → BIAS), evidenciando ganho de performance.

<img width="828" height="354" alt="image" src="https://github.com/user-attachments/assets/4c051a9f-5ead-4829-8130-0a982bcaebe4" />


![Teste recente - Real vs Previsto](./images/teste_recente_real_vs_previsto.png)

> Observação: mesmo com o ganho do BIAS, o baseline de persistência pode permanecer competitivo para horizontes muito curtos — isso é esperado em séries financeiras.

## 9) Export da Tabela de Previsões (com BIAS)
O notebook exporta uma tabela final (CSV) contendo **apenas a previsão calibrada**:
- `predicoes_itub4_bias.csv`
  - `Date`
  - `Fechamento_Previsto`

# Projeção até 2026 (cenários)
Para horizontes longos (meses/anos), uma previsão multi-step determinística com LSTM tende a acumular erro e divergir.  
Por isso, foi incluída uma projeção até **31/12/2026** baseada em **cenários de log-retornos** (pessimista/mediana/otimista) usando drift (μ) e volatilidade (σ) estimados do histórico.

<img width="828" height="347" alt="image" src="https://github.com/user-attachments/assets/2579690c-95fc-4ae0-bcb3-29b5d2aaa3c6" />


Export gerado:
- `projecao_itub4_cenarios_ate_2026.csv`
  - `Date`, `Pessimista`, `Mediana`, `Otimista`

# Checklist de Requisitos do Tech Challenge (Fase 4)
O enunciado pede: coleta/preprocessamento, LSTM, treinamento e avaliação com MAE/RMSE/MAPE, salvamento do modelo, deploy em API, e monitoramento. fileciteturn3file0L16-L56

- [x] Coleta e pré-processamento (Yahoo Finance / yfinance) fileciteturn3file0L22-L33
- [x] Desenvolvimento do modelo LSTM (treino + ajuste de hiperparâmetros) fileciteturn3file0L34-L39
- [x] Avaliação com métricas (MAE, RMSE, MAPE) fileciteturn3file0L40-L43
- [ ] Salvamento e exportação do modelo (ex.: `.keras`/SavedModel + scaler) fileciteturn3file0L44-L47
- [ ] Deploy do modelo em API (Flask/FastAPI) fileciteturn3file0L48-L52
- [ ] Monitoramento (tempo de resposta, recursos, logs) fileciteturn3file0L53-L56

**Entregáveis** citados no enunciado incluem código + documentação, scripts/Docker, link de API e vídeo. fileciteturn3file2L24-L30

# Como Executar
1. Crie um ambiente Python (recomendado: `conda`).
2. Instale dependências principais:
   - `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `yfinance`, `tensorflow`
3. Rode o notebook de desenvolvimento em `./notebooks/`.

# Próximos Passos (para completar o desafio)
- Salvar `model` + `scaler` (SavedModel/`.keras` + `joblib`/pickle).
- Criar API (Flask/FastAPI) com endpoint que recebe histórico (últimos 60 fechamentos) e retorna previsão.
- Empacotar com Docker e adicionar monitoramento (logs e métricas de latência/uso).
