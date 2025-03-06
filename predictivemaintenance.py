import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings('ignore')

# Função para limpar os dados
def clean_data(df):
    filtered_df = df[(df['PARAMETER'] == 'Z1MaxCurrent') | (df['PARAMETER'] == 'Z1GrindingWheelLot')]
    filtered_df.loc[filtered_df['PARAMETER'] == 'Z1GrindingWheelLot', 'VALUE'] = filtered_df.loc[filtered_df['PARAMETER'] == 'Z1GrindingWheelLot', 'VALUE'].astype(str)
    aggregated_df = filtered_df.groupby(['DATETIME', 'PARAMETER'])['VALUE'].first().reset_index()
    cleaned_df = aggregated_df.pivot(index='DATETIME', columns='PARAMETER', values='VALUE').reset_index()
    return cleaned_df

# Caminho para o arquivo Excel
file_path = 'C:/Users/luis_/OneDrive/Ambiente de Trabalho/dados_amkorr.xlsx'

# Leitura e limpeza dos dados de cada planilha
excel_data = pd.ExcelFile(file_path)
cleaned_dataframes = [clean_data(pd.read_excel(file_path, sheet_name=sheet)) for sheet in excel_data.sheet_names]

# Concatenar todos os dados limpos em um único DataFrame
df = pd.concat(cleaned_dataframes, ignore_index=True)

# Salvar dados limpos em Excel
df.to_excel('C:/Users/luis_/OneDrive/Ambiente de Trabalho/cleaned_data.xlsx', index=False)

# Limpeza e pré-processamento do DataFrame
df['DATETIME'] = pd.to_datetime(df['DATETIME'].str.strip(), errors='coerce')
df['Z1MaxCurrent'] = pd.to_numeric(df['Z1MaxCurrent'], errors='coerce')
df['Z1GrindingWheelLot'] = df['Z1GrindingWheelLot'].astype('category')
df = df.dropna(subset=['DATETIME', 'Z1MaxCurrent', 'Z1GrindingWheelLot'])
df = df.sort_values(by='DATETIME')

# Adicionar recursos temporais
df['TimeIndex'] = (df['DATETIME'] - df['DATETIME'].min()).dt.total_seconds() / 60
df['TimeIndex'] = df['TimeIndex'].astype(int)
df['Hour'] = df['DATETIME'].dt.hour
df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
df['Month'] = df['DATETIME'].dt.month

# Codificação one-hot para 'Z1GrindingWheelLot'
df = pd.get_dummies(df, columns=['Z1GrindingWheelLot'])

# Salvar dados pré-processados em Excel
df.to_excel('C:/Users/luis_/OneDrive/Ambiente de Trabalho/preprocessed_data.xlsx', index=False)

# Preparar dados para detecção de anomalias
features = [col for col in df.columns if col not in ['DATETIME', 'Z1MaxCurrent', 'needs_maintenance']]
X = df[features]

# Padronização das features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Detectar tendência usando regressão linear
reg = LinearRegression()
df['Trend'] = reg.fit(df[['TimeIndex']], df['Z1MaxCurrent']).predict(df[['TimeIndex']])

# Calcular média e desvio padrão para Z1MaxCurrent
mean_current = df['Z1MaxCurrent'].mean()
std_current = df['Z1MaxCurrent'].std()

# Inicializar a coluna de manutenção com 0
df['needs_maintenance'] = 0

# Definir limites superior e inferior para anomalias
threshold_upper = mean_current + 2 * std_current
threshold_lower = mean_current - 2 * std_current

# Detecção de anomalias e necessidade de manutenção
consecutive_anomalies = 0
required_consecutive_anomalies = 3

for i in range(1, len(df)):
    current_value = df.iloc[i]['Z1MaxCurrent']
    trend_diff = df.iloc[i]['Trend'] - df.iloc[i-1]['Trend']

    if (current_value > threshold_upper or
        current_value < threshold_lower or
        abs(trend_diff) > std_current * 0.5 or
        current_value == 0):
        consecutive_anomalies += 1
    else:
        consecutive_anomalies = 0

    if consecutive_anomalies >= required_consecutive_anomalies:
        df.iloc[i - consecutive_anomalies + 1: i + 1, df.columns.get_loc('needs_maintenance')] = 1

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, df['Z1MaxCurrent'], test_size=0.2, random_state=42)

# Função para avaliar modelos
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2, y_pred

# Função para calcular a precisão
def calculate_precision(y_true, y_pred, tolerance=0.10):
    accuracy = np.mean(np.abs((y_true - y_pred) / y_true) <= tolerance)
    return accuracy

# Modelos preditivos
models = {
    'Naive Bayes': BayesianRidge(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Hidden Markov Model': GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000, random_state=42)
}

# Avaliar todos os modelos
model_metrics = {}
for model_name, model in models.items():
    try:
        if model_name == 'Hidden Markov Model':
            # Preparar dados para HMM
            hmm_X_train = np.column_stack([X_train['TimeIndex'], y_train])
            model.fit(hmm_X_train)
            hmm_X_test = np.column_stack([X_test['TimeIndex']])
            y_pred, _ = model.sample(len(X_test))
            y_pred = y_pred[:, 1]
        else:
            mse, mae, rmse, r2, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        precision = calculate_precision(y_test, y_pred)
        model_metrics[model_name] = {
            'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Precision': precision
        }
        print(f'\n{model_name}:')
        print(f"  MSE: {mse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.2f}")
        print(f"  Precision: {precision:.2%}")
    except Exception as e:
        print(f'\n{model_name} failed with error: {e}')

# Selecionar o melhor modelo com base na precisão e R²
best_model_name = max(model_metrics, key=lambda x: (model_metrics[x]['Precision'], model_metrics[x]['R²']))
print(f'\nThe best model is: {best_model_name}')

# Implementar o modelo selecionado
if best_model_name == 'Hidden Markov Model':
    best_model = models[best_model_name]
    hmm_X_train = np.column_stack([X_train['TimeIndex'], y_train])
    best_model.fit(hmm_X_train)
    hmm_X_test = np.column_stack([X_test['TimeIndex']])
    y_pred, _ = best_model.sample(len(X_test))
    y_pred = y_pred[:, 1]
else:
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

# Validar o modelo
results_df = pd.DataFrame({
    'DATETIME': df.loc[X_test.index, 'DATETIME'],
    'True_Value': y_test.values,
    'Predicted_Value': y_pred,
    'Needs_Maintenance': df.loc[X_test.index, 'needs_maintenance'].values
})

# Adicionar colunas de limites superiores e inferiores
results_df['Threshold_Upper'] = threshold_upper
results_df['Threshold_Lower'] = threshold_lower

# Adicionar uma coluna "Status de Manutenção"
def maintenance_status(row):
    if row['Needs_Maintenance'] == 1:
        return "IMMEDIATE MAINTENANCE REQUIRED"
    elif row['True_Value'] > row['Threshold_Upper'] or row['True_Value'] < row['Threshold_Lower']:
        return "MAINTENANCE ALERT"
    else:
        return "NORMAL"

results_df['Maintenance_Status'] = results_df.apply(maintenance_status, axis=1)

# Reordenar colunas para melhor clareza
results_df = results_df[['DATETIME', 'True_Value', 'Predicted_Value', 'Threshold_Upper', 'Threshold_Lower', 'Needs_Maintenance', 'Maintenance_Status']]

# Salvar resultados em um arquivo Excel ordenado cronologicamente
results_df.to_excel('C:/Users/luis_/OneDrive/Ambiente de Trabalho/processed_data_with_predictions.xlsx', index=False)

# Exibir os primeiros resultados e os limites de manutenção
results_df.head()

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import dash_bootstrap_components as dbc

# Supondo que df e results_df já foram processados anteriormente
# Iniciar a aplicação Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Dashboard de Manutenção Preditiva"

app.layout = dbc.Container(fluid=True, children=[
    # Cabeçalho da Dashboard
    dbc.Row(
        dbc.Col(
            html.H1("Dashboard de Manutenção Preditiva", 
                    className="text-center", 
                    style={'color': '#ffffff', 'margin-bottom': '20px'}),
            width=12
        )
    ),

    # Descrição da Dashboard
    dbc.Row(
        dbc.Col(
            html.P(
                "Esta Dashboard fornece uma visão geral da manutenção preditiva, permitindo acompanhar a "
                "tendência dos valores de Corrente, identificar a necessidade de manutenção, comparar valores reais "
                "e previstos, e visualizar a distribuição dos dados. Utilize o seletor de intervalo de datas no canto inferior esquerdo "
                "para explorar os dados em diferentes períodos.",
                className="text-center", 
                style={'font-size': '16px', 'color': '#cfcfcf'}
            ),
            width=12
        )
    ),

    # Conteúdo da Dashboard
    dbc.Row(
        dbc.Col(
            dbc.Tabs(id="tabs", active_tab='tab-7', children=[
                dbc.Tab(label='Gráfico de Tendência', tab_id='tab-1', children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='trend-graph')
                        ]),
                        className="shadow-sm border-light"
                    )
                ]),
                dbc.Tab(label='Necessidade de Manutenção', tab_id='tab-2', children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='maintenance-graph')
                        ]),
                        className="shadow-sm border-light"
                    )
                ]),
                dbc.Tab(label='Comparação de Valores Reais e Previstos', tab_id='tab-3', children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='comparison-graph')
                        ]),
                        className="shadow-sm border-light"
                    )
                ]),
                dbc.Tab(label='Previsão de Necessidade de Manutenção', tab_id='tab-4', children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='prediction-maintenance-graph')
                        ]),
                        className="shadow-sm border-light"
                    )
                ]),
                dbc.Tab(label='Distribuição de Corrente', tab_id='tab-5', children=[
                    dbc.Tabs([
                        dbc.Tab(label='Histograma', tab_id='dist-histogram', children=[
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(id='distribution-graph')
                                ]),
                                className="shadow-sm border-light"
                            )
                        ]),
                        dbc.Tab(label='Boxplot', tab_id='dist-boxplot', children=[
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(id='boxplot-distribution')
                                ]),
                                className="shadow-sm border-light"
                            )
                        ]),
                    ], className="mb-4", style={'font-size': '14px'}),
                ]),
                dbc.Tab(label='Boxplots por Período', tab_id='tab-6', children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='boxplot-hour'),
                            dcc.Graph(id='boxplot-day'),
                            dcc.Graph(id='boxplot-month')
                        ]),
                        className="shadow-sm border-light"
                    )
                ]),
dbc.Tab(label='Avaliação dos Modelos', tab_id='tab-7', children=[
    dbc.Card(
        dbc.CardBody([
            html.H4(
                "Avaliação dos Modelos", 
                className="text-center", 
                style={'color': '#ffffff', 'margin-bottom': '20px', 'font-family': 'Roboto', 'font-weight': 'bold'}
            ),
            dash_table.DataTable(
                id='model-metrics-table',
                columns=[
                    {"name": "Modelo", "id": "Modelo"},
                    {"name": "MSE", "id": "MSE"},
                    {"name": "MAE", "id": "MAE"},
                    {"name": "RMSE", "id": "RMSE"},
                    {"name": "R²", "id": "R2"},
                    {"name": "Precisão", "id": "Precision"},
                ],
                data=[
                    {"Modelo": k, 
                    "MSE": f"{v['MSE']:.2e}",  # MSE em notação científica
                    "MAE": f"{v['MAE']:.2f}",  # MAE com 2 casas decimais
                    "RMSE": f"{v['RMSE']:.2f}",  # RMSE com 2 casas decimais
                    "R2": f"{v['R²']:.2f}",  # R² com 2 casas decimais
                    "Precision": f"{v['Precision']:.2%}"}  # Precisão em porcentagem
                    for k, v in model_metrics.items()
                ],
                style_table={'overflowX': 'auto', 'minWidth': '100%'},
                style_cell={
                    'textAlign': 'center',
                    'backgroundColor': '#1b1b1b',  # Cor de fundo mais escura
                    'color': '#e0e0e0',  # Texto em cinza claro
                    'padding': '12px',
                    'font-family': 'Roboto'
                },
                style_header={
                    'backgroundColor': '#222222',  # Cabeçalho mais escuro
                    'fontWeight': 'bold',
                    'color': '#ffffff',
                    'border-bottom': '2px solid #444444',
                },
                style_data={
                    'border': '1px solid #444444',  # Bordas mais suaves
                    'font-size': '14px',
                    'font-family': 'Roboto'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Modelo'}, 'textAlign': 'left', 'padding-left': '20px'},
                ],
                style_data_conditional=[
                    # Gradiente de cores para MSE, MAE e RMSE
                    {
                        'if': {'column_id': 'MSE', 'filter_query': '{MSE} <= 1e6'},
                        'backgroundColor': '#4CAF50',  # Subtle green for lower MSE values
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'MSE', 'filter_query': '{MSE} > 1e6'},
                        'backgroundColor': '#F44336',  # Subtle red for higher MSE values
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'MAE', 'filter_query': '{MAE} <= 400'},
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'MAE', 'filter_query': '{MAE} > 400'},
                        'backgroundColor': '#F44336',
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'RMSE', 'filter_query': '{RMSE} <= 800'},
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'RMSE', 'filter_query': '{RMSE} > 800'},
                        'backgroundColor': '#F44336',
                        'color': '#ffffff',
                    },
                    # Gradiente de cores para R² e Precisão
                    {
                        'if': {'column_id': 'R2', 'filter_query': '{R2} >= 0.9'},
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'R2', 'filter_query': '{R2} < 0.9'},
                        'backgroundColor': '#F44336',
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'R2', 'filter_query': '{R2} < 0.9'},
                        'backgroundColor': '#F44336',
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'Precision', 'filter_query': '{Precision} >= 0.9'},
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                    },
                    {
                        'if': {'column_id': 'Precision', 'filter_query': '{Precision} < 0.9'},
                        'backgroundColor': '#F44336',
                        'color': '#ffffff',
                    },
                    # Destaque para o melhor modelo
                    {
                        'if': {'filter_query': '{Modelo} = "Random Forest"'},
                        'backgroundColor': '#43A047',
                        'color': '#ffffff',
                        'border': '2px solid #FFD700',  # Subtle gold border for emphasis
                    },
                ]
            ),
            html.Br(),
            html.P(
                f"O melhor modelo é: {best_model_name}. Este modelo foi selecionado com base na melhor combinação de precisão ({model_metrics[best_model_name]['Precision']:.2%}) e R² ({model_metrics[best_model_name]['R²']:.2f}).",
                className="text-center",
                style={'color': '#e0e0e0', 'font-size': '16px', 'font-family': 'Roboto', 'margin-top': '20px'}
            ),
            html.P(
                "Métricas Explicadas:",
                className="text-center",
                style={'color': '#ffffff', 'font-family': 'Roboto', 'font-weight': 'bold'}
            ),
            html.Ul(
                children=[
                    html.Li(
                        "MSE (Erro Quadrático Médio): Mede a média dos quadrados dos erros, ou seja, a diferença média ao quadrado entre os valores previstos e os valores reais.",
                        style={'color': '#e0e0e0', 'font-family': 'Roboto'}
                    ),
                    html.Li(
                        "MAE (Erro Médio Absoluto): Mede a média dos erros absolutos, ou seja, a média das diferenças absolutas entre os valores previstos e os valores reais.",
                        style={'color': '#e0e0e0', 'font-family': 'Roboto'}
                    ),
                    html.Li(
                        "RMSE (Raiz do Erro Quadrático Médio): Mede a raiz quadrada da média dos erros quadráticos, oferecendo uma métrica que penaliza mais os grandes erros.",
                        style={'color': '#e0e0e0', 'font-family': 'Roboto'}
                    ),
                    html.Li(
                        "R²: Indica a proporção da variância dos dados que é explicada pelo modelo.",
                        style={'color': '#e0e0e0', 'font-family': 'Roboto'}
                    ),
                    html.Li(
                        "Precisão: Mede a proximidade dos valores previstos com os valores reais, dentro de uma tolerância de 10%.",
                        style={'color': '#e0e0e0', 'font-family': 'Roboto'}
                    ),
                ],
                style={'color': '#e0e0e0'}
            )
        ]),
        className="shadow-sm border-0 bg-dark",  # Darker background for cohesion
        style={'background-color': '#181818'}  
    )
]),
], className="mb-4", style={'font-size': '14px'}),
width=12
)
),
# Seletor de Data
dbc.Row(
    dbc.Col(
        html.Div(
            children=[
                html.Button("Escolha um intervalo de tempo", id='open-date-picker', className="btn btn-outline-primary btn-sm", style={'position': 'fixed', 'bottom': '16px', 'left': '25px', 'opacity': '0.2'}),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Escolha um intervalo de tempo"),
                        dbc.ModalBody(
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=results_df['DATETIME'].min(),
                                max_date_allowed=results_df['DATETIME'].max(),
                                start_date=results_df['DATETIME'].min(),
                                end_date=results_df['DATETIME'].max(),
                                display_format='YYYY-MM-DD',
                                style={'font-size': '14px', 'color': '#000000'}
                            )
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Fechar", id="close-date-picker", className="ml-auto")
                        ),
                    ],
                    id="date-picker-modal",
                    size="lg",
                    is_open=False,
                ),
            ]
        ),
        width=5
    )
)
])
# Função para filtrar DataFrame pelo intervalo de datas selecionado
def filter_df_by_date_range(df, start_date, end_date):
    mask = (df['DATETIME'] >= start_date) & (df['DATETIME'] <= end_date)
    return df.loc[mask]

# Callbacks para atualizar os gráficos com base na seleção de data
@app.callback(
    Output('trend-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_trend_graph(start_date, end_date):
    filtered_df = filter_df_by_date_range(df, start_date, end_date)
    fig = px.line(filtered_df, x='DATETIME', y='Z1MaxCurrent', title='Tendência Geral dos Valores de Z1MaxCurrent')
    fig.update_layout(xaxis_title='Data e Hora', yaxis_title='Corrente', template='plotly_dark',legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output('maintenance-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_maintenance_graph(start_date, end_date):
    filtered_df = filter_df_by_date_range(df, start_date, end_date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['DATETIME'], y=filtered_df['Z1MaxCurrent'], mode='lines', name='Valores Reais'))
    fig.add_trace(go.Scatter(x=filtered_df['DATETIME'], y=[threshold_upper]*len(filtered_df), mode='lines', name='Limite Superior', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=filtered_df['DATETIME'], y=[threshold_lower]*len(filtered_df), mode='lines', name='Limite Inferior', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=filtered_df[filtered_df['needs_maintenance'] == 1]['DATETIME'], 
                             y=filtered_df[filtered_df['needs_maintenance'] == 1]['Z1MaxCurrent'], 
                             mode='markers', name='Manutenção Necessária', marker=dict(color='red', size=8)))
    fig.update_layout(
        template='plotly_dark', 
        title='Necessidade de Manutenção', 
        xaxis=dict(title='Data e Hora'),
        yaxis=dict(title='Corrente Z1Max'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig

@app.callback(
    Output('comparison-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_comparison_graph(start_date, end_date):
    filtered_df = filter_df_by_date_range(results_df, start_date, end_date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df.head(20)['DATETIME'], y=filtered_df.head(20)['True_Value'], mode='lines', name='Valor Real'))
    fig.add_trace(go.Scatter(x=filtered_df.head(20)['DATETIME'], y=filtered_df.head(20)['Predicted_Value'], mode='lines', name='Valor Previsto'))
    fig.update_layout(title='Comparação de Valores Reais e Previstos', xaxis_title='Data e Hora', yaxis_title='Corrente', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output('prediction-maintenance-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_prediction_maintenance_graph(start_date, end_date):
    filtered_df = filter_df_by_date_range(results_df, start_date, end_date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['DATETIME'], y=filtered_df['Predicted_Value'], mode='lines', name='Valor Previsto'))
    fig.add_trace(go.Scatter(x=filtered_df[filtered_df['Needs_Maintenance'] == 1]['DATETIME'], 
                             y=filtered_df[filtered_df['Needs_Maintenance'] == 1]['Predicted_Value'], 
                             mode='markers', name='Manutenção Necessária', marker=dict(color='red', size=8)))
    fig.update_layout(title='Previsão de Necessidade de Manutenção', xaxis_title='Data e Hora', yaxis_title='Corrente', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output('distribution-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_distribution_graph(start_date, end_date):
    filtered_df = filter_df_by_date_range(df, start_date, end_date)
    fig = px.histogram(filtered_df, x='Z1MaxCurrent', title='Distribuição dos Valores de Z1MaxCurrent')
    fig.update_layout(xaxis_title='Corrente', yaxis_title='Contagem', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output('boxplot-distribution', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_boxplot_distribution(start_date, end_date):
    filtered_df = filter_df_by_date_range(df, start_date, end_date)
    fig = px.box(filtered_df, y='Z1MaxCurrent', title='Boxplot da Distribuição dos Valores de Z1MaxCurrent')
    fig.update_layout(xaxis_title='Corrente', yaxis_title='Valor', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output('boxplot-hour', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_boxplot_hour(start_date, end_date):
    filtered_df = filter_df_by_date_range(df, start_date, end_date)
    filtered_df['Hour'] = filtered_df['DATETIME'].dt.hour
    fig = px.box(filtered_df, x='Hour', y='Z1MaxCurrent', title='Boxplot de Z1MaxCurrent por Hora')
    fig.update_layout(xaxis_title='Hora', yaxis_title='Corrente', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output('boxplot-day', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_boxplot_day(start_date, end_date):
    filtered_df = filter_df_by_date_range(df, start_date, end_date)
    filtered_df['Day'] = filtered_df['DATETIME'].dt.day
    fig = px.box(filtered_df, x='Day', y='Z1MaxCurrent', title='Boxplot de Z1MaxCurrent por Dia')
    fig.update_layout(xaxis_title='Dia', yaxis_title='Corrente', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output('boxplot-month', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_boxplot_month(start_date, end_date):
    filtered_df = filter_df_by_date_range(df, start_date, end_date)
    filtered_df['Month'] = filtered_df['DATETIME'].dt.month
    fig = px.box(filtered_df, x='Month', y='Z1MaxCurrent', title='Boxplot de Z1MaxCurrent por Mês')
    fig.update_layout(xaxis_title='Mês', yaxis_title='Corrente', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40))
    return fig

@app.callback(
    Output("date-picker-modal", "is_open"),
    [Input("open-date-picker", "n_clicks"), Input("close-date-picker", "n_clicks")],
    [dash.dependencies.State("date-picker-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
