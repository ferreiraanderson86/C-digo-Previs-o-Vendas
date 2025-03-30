import pandas as pd
import matplotlib.pyplot as plt
import locale
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração de localidade para datas em português
locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')

# Função para avaliação dos modelos
def evaluate_models(y_true, y_pred_arima, y_pred_prophet):
    """
    Avalia os modelos ARIMA e Prophet com base em métricas de erro e visualiza os resíduos.

    Parâmetros:
        y_true (array-like): Valores reais do período de teste.
        y_pred_arima (array-like): Previsões feitas pelo modelo ARIMA.
        y_pred_prophet (array-like): Previsões feitas pelo modelo Prophet.

    Retorna:
        dict: Um dicionário contendo as métricas de erro (MAE, RMSE, MAPE, WMAPE) para cada modelo.
    """

    # Cálculo das métricas para ARIMA
    mae_arima = mean_absolute_error(y_true, y_pred_arima)
    rmse_arima = mean_squared_error(y_true, y_pred_arima, squared=False)
    mape_arima = (abs((y_true - y_pred_arima) / y_true).mean()) * 100  # em percentual
    wmape_arima = mae_arima / y_true.mean()

    # Cálculo das métricas para Prophet
    mae_prophet = mean_absolute_error(y_true, y_pred_prophet)
    rmse_prophet = mean_squared_error(y_true, y_pred_prophet, squared=False)
    mape_prophet = (abs((y_true - y_pred_prophet) / y_true).mean()) * 100  # em percentual
    wmape_prophet = mae_prophet / y_true.mean()

    # Impressão dos resultados
    print("ARIMA - MAE:", mae_arima, "RMSE:", rmse_arima, "MAPE:", mape_arima, "WMAPE:", wmape_arima)
    print("Prophet - MAE:", mae_prophet, "RMSE:", rmse_prophet, "MAPE:", mape_prophet, "WMAPE:", wmape_prophet)

    # Estrutura para armazenar as métricas
    metrics = {
        "ARIMA": {
            "MAE": mae_arima,
            "RMSE": rmse_arima,
            "MAPE": mape_arima,
            "WMAPE": wmape_arima
        },
        "Prophet": {
            "MAE": mae_prophet,
            "RMSE": rmse_prophet,
            "MAPE": mape_prophet,
            "WMAPE": wmape_prophet
        }
    }

    # Cálculo dos resíduos para ambos os modelos
    residuals_arima = y_true - y_pred_arima
    residuals_prophet = y_true - y_pred_prophet

    # Visualização dos resíduos
    plt.figure(figsize=(12, 6))

    # Resíduos ARIMA
    plt.subplot(1, 2, 1)
    plt.plot(residuals_arima, marker='o')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Resíduos ARIMA")
    plt.xlabel("Período")
    plt.ylabel("Resíduo")

    # Resíduos Prophet
    plt.subplot(1, 2, 2)
    plt.plot(residuals_prophet, marker='o')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Resíduos Prophet")
    plt.xlabel("Período")
    plt.ylabel("Resíduo")

    plt.tight_layout()
    plt.show()

    return metrics


# Leitura e preparação dos dados
df = pd.read_csv('C:/Users/ander/Downloads/df_case_1_-_500_linhas.csv')

# Transformação para formato longo
df_long = df.melt(
    id_vars=['CATEGORIA', 'PRODUTO', 'FORNECEDOR', 'COMPRADOR', 'CENTROS DE DISTRIBUICAO', 'DESCRICAO', 'QTD DE CAIXAS'],
    value_vars=['MAR22', 'ABR22', 'MAI22', 'JUN22', 'JUL22', 'AGO22', 'SET22', 'OUT22', 'NOV22', 'DEZ22',
                'JAN23', 'FEV23', 'MAR23', 'ABR23', 'MAI23', 'JUN23', 'JUL23', 'AGO23', 'SET23', 'OUT23',
                'NOV23', 'DEZ23', 'JAN24', 'FEV24', 'MAR24'],
    var_name='Date',
    value_name='Sales'
)

# Convertendo as datas para o formato datetime e ajustando os valores de vendas
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%b%y', errors='coerce')
df_long['Sales'] = pd.to_numeric(df_long['Sales'], errors='coerce')  # Caso existam valores não numéricos

# Visualização da série temporal
plt.figure(figsize=(10, 6))
plt.plot(df_long.groupby('Date')['Sales'].sum(), marker='o')
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Verificação de estacionaridade com o teste de Dickey-Fuller
adf_test = adfuller(df_long['Sales'].dropna())
print('ADF Statistic:', adf_test[0])
print('p-value:', adf_test[1])

# Modelagem com ARIMA
arima_model = ARIMA(df_long['Sales'].dropna(), order=(1, 1, 1))
arima_results = arima_model.fit()
arima_predictions = arima_results.forecast(steps=12)
print("ARIMA Forecast:", arima_predictions)

# Modelagem com Prophet
prophet_df = df_long[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=12, freq='M')
forecast = prophet_model.predict(future)

# Visualização da previsão com Prophet
prophet_model.plot(forecast)
plt.title("Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Separando os dados reais (últimos 12 meses) para cálculo das métricas
y_true = df_long['Sales'][-12:].values
y_pred_arima = arima_predictions[:12].values
y_pred_prophet = forecast['yhat'][-12:].values

# Avaliação das métricas e visualização dos resíduos
metrics = evaluate_models(y_true, y_pred_arima, y_pred_prophet)
print("Metrics:", metrics)
