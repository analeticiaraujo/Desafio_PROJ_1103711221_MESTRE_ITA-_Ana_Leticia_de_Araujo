# ==============================================================================
# SCRIPT PYTHON OTIMIZADO - PIPELINE DE PREVISÃO DE CUSTOS AZURE
# AUTORA: Ana Leticia de Araujo
# DESCRIÇÃO: Versão refatorada do pipeline de dados e ML, aplicando
# princípios de Clean Code para modularidade e legibilidade.
# ==============================================================================

# --- ETAPA 0: IMPORTAÇÃO DE BIBLIOTECAS E CONFIGURAÇÃO ---
import requests
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Optional

# Centraliza todas as configurações e "strings mágicas" em um único lugar
CONFIG = {
    "api_url": "https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Virtual Machines' and (armRegionName eq 'brazilsoutheast' or armRegionName eq 'eastus') and priceType eq 'Consumption'",
    "columns_to_select": ['armSkuName', 'retailPrice', 'armRegionName', 'vcpu', 'memoria_gb'],
    "column_rename_map": {
        'armSkuName': 'tipo_instancia',
        'retailPrice': 'preco_hora',
        'armRegionName': 'regiao'
    },
    "region_translation_map": {
        'eastus': 'Leste dos EUA',
        'brazilsoutheast': 'Sudeste do Brasil'
    },
    "exploratory_plot_path": "analise_exploratoria_vcpu_por_regiao.png",
    "dashboard_plot_path": "dashboard_performance_modelo.png",
    "results_csv_path": "dados_tratados_e_resultados.csv",
    "random_state": 42,
    "test_size": 0.2
}


# --- ETAPA A: DEFINIÇÃO DE FUNÇÕES MODULARES ---

def fetch_data_from_api(api_url: str) -> Optional[pd.DataFrame]:
    """Busca e pagina os dados da API da Azure, retornando um DataFrame."""
    print("--- ETAPA 1: INICIANDO COLETA DE DADOS DA API DO AZURE ---")
    all_items = []
    page_number = 1
    
    while api_url:
        print(f"Buscando dados da página {page_number}...")
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            all_items.extend(data.get('Items', []))
            api_url = data.get('NextPageLink')
            page_number += 1
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar a API: {e}")
            return None
            
    if not all_items:
        print("Nenhum dado foi retornado pela API.")
        return None
            
    print(f"SUCESSO: {len(all_items)} registros de preços coletados.")
    return pd.DataFrame(all_items)

def get_specs_from_name(sku_name: str) -> Tuple[Optional[int], Optional[float]]:
    """Extrai vCPU e Memória do nome da instância (armSkuName)."""
    memory_ratio_map = {'B': 2, 'D': 4, 'E': 8, 'F': 2, 'M': 8, 'DC': 4}
    match = re.search(r'Standard_([A-Z]+)(\d+)', sku_name)
    if not match:
        return None, None
    family, vcpu = match.group(1), int(match.group(2))
    ratio = memory_ratio_map.get(family, 4)
    memory = float(vcpu * ratio)
    return vcpu, memory

def clean_and_feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Aplica a limpeza, engenharia de features e transformações no DataFrame."""
    print("\n--- ETAPA 2: INICIANDO LIMPEZA E ENGENHARIA DE FEATURES ---")
    
    # Filtragem
    query_filter = (
        (df['unitOfMeasure'] == '1 Hour') &
        (df['armSkuName'].str.contains('Standard_D|Standard_B|Standard_E', na=False)) &
        (~df['meterName'].str.contains('Spot', na=False)) &
        (~df['meterName'].str.contains('Low Priority', na=False)) &
        (~df['productName'].str.contains('Promo', na=False))
    )
    df_clean = df[query_filter].copy()

    # Engenharia de Features
    specs_df = df_clean['armSkuName'].apply(get_specs_from_name).apply(pd.Series)
    df_clean['vcpu'], df_clean['memoria_gb'] = specs_df[0], specs_df[1]

    # Seleção, Renomeação e Limpeza Final
    df_final = df_clean[CONFIG["columns_to_select"]].rename(columns=CONFIG["column_rename_map"]).dropna().copy()
    
    if df_final.empty:
        print("\nFATAL: O DataFrame final está vazio após a limpeza.")
        return None
        
    # Tradução de Nomes
    df_final['regiao'] = df_final['regiao'].map(CONFIG["region_translation_map"])
    print(f"Dataset limpo com {len(df_final)} instâncias.")
    print("Contagem de regiões:\n", df_final['regiao'].value_counts())
    return df_final

def generate_exploratory_plot(df: pd.DataFrame):
    """Gera e salva o gráfico de análise exploratória."""
    print("\n--- ETAPA 2.5: GERANDO ANÁLISE EXPLORATÓRIA VISUAL ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    g = sns.lmplot(data=df, x='vcpu', y='preco_hora', hue='regiao', x_estimator=np.mean, height=7, aspect=1.5, palette='viridis')
    g.fig.suptitle('Tendência de Custo Médio por Quantidade de vCPUs', fontsize=18, y=1.03, weight='bold')
    g.set_axis_labels('Quantidade de vCPUs', 'Custo Médio por Hora (USD)', fontsize=14)
    g.legend.set_title("Região")
    plt.savefig(CONFIG["exploratory_plot_path"], dpi=300)
    plt.close()
    print(f"Gráfico exploratório salvo em: {CONFIG['exploratory_plot_path']}")

def train_model(df: pd.DataFrame) -> Tuple[object, pd.DataFrame, pd.DataFrame, np.ndarray, float, float]:
    """Prepara os dados, treina o modelo e retorna os resultados."""
    print("\n--- ETAPAS 3 & 4: PREPARANDO DADOS E TREINANDO O MODELO ---")
    df_processed = pd.get_dummies(df, columns=['regiao'], prefix='reg')
    df_processed.drop('tipo_instancia', axis=1, inplace=True)
    
    X = df_processed.drop('preco_hora', axis=1)
    y = df_processed['preco_hora']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])
    
    model = RandomForestRegressor(n_estimators=100, random_state=CONFIG["random_state"])
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return model, X_test, y_test, predictions, mse, r2

def plot_evaluation_dashboard(y_test: pd.Series, predictions: np.ndarray, mse: float, r2: float, df_final: pd.DataFrame, X_test: pd.DataFrame):
    """Gera e salva o dashboard de performance do modelo."""
    print("\n--- ETAPA 5: GERANDO DASHBOARD DE AVALIAÇÃO ---")
    plot_data = X_test.copy()
    plot_data.loc[:, 'preco_real'] = y_test
    plot_data.loc[:, 'preco_previsto'] = predictions
    plot_data.loc[:, 'residuos'] = y_test - predictions
    plot_data.loc[:, 'regiao'] = df_final.loc[X_test.index, 'regiao']

    plt.style.use('seaborn-v0_8-talk')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    fig.suptitle('Dashboard de Performance do Modelo de Regressão', fontsize=22, weight='bold', y=1.02)

    # Gráfico de Dispersão
    sns.scatterplot(data=plot_data, x='preco_real', y='preco_previsto', hue='regiao', palette='viridis', alpha=0.8, s=100, ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2, label='Previsão de custo ideal')
    ax1.set_title('Gráfico de Dispersão: Preço Real vs. Previsão', fontsize=16)
    ax1.set_xlabel('Preço Real por Hora (USD)', fontsize=14)
    ax1.set_ylabel('Preço Previsto por Hora (USD)', fontsize=14)
    ax1.legend(title='Região')
    ax1.text(0.05, 0.95, f'$R^2 = {r2:.4f}$\nMSE = {mse:.5f}', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', alpha=0.6))

    # Gráfico de Resíduos
    sns.scatterplot(data=plot_data, x='preco_previsto', y='residuos', hue='regiao', palette='viridis', alpha=0.7, s=100, legend=False, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Gráfico de Resíduos (Análise de Erros)', fontsize=16)
    ax2.set_xlabel('Valores Previstos (USD)', fontsize=14)
    ax2.set_ylabel('Resíduos (Erro: Real - Previsto)', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(CONFIG["dashboard_plot_path"], dpi=300)
    plt.close(fig)
    print(f"Dashboard salvo em: {CONFIG['dashboard_plot_path']}")

def save_results_to_csv(X_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray, df_final: pd.DataFrame):
    """Salva os resultados finais em um arquivo CSV."""
    print("\n--- ETAPA 6: SALVANDO RESULTADOS EM CSV ---")
    df_resultados = X_test.copy()
    df_resultados['preco_real'] = y_test
    df_resultados['preco_previsto'] = predictions
    df_resultados['regiao'] = df_final.loc[X_test.index, 'regiao']
    df_resultados.to_csv(CONFIG["results_csv_path"], index=False)
    print(f"Arquivo de resultados salvo em: {CONFIG['results_csv_path']}")


# --- ETAPA C: PONTO DE ENTRADA E ORQUESTRAÇÃO DO SCRIPT ---
def main():
    """Função principal que orquestra a execução do pipeline."""
    
    # 1. Coleta dos dados
    raw_df = fetch_data_from_api(CONFIG["api_url"])
    if raw_df is None:
        return

    # 2. Limpeza e Engenharia de Features
    final_df = clean_and_feature_engineer(raw_df)
    if final_df is None:
        return

    # 3. Análise Exploratória
    generate_exploratory_plot(final_df)

    # 4. Treinamento do Modelo
    model, X_test, y_test, predictions, mse, r2 = train_model(final_df)
    print(f"\nResultados da Avaliação: MSE={mse:.4f}, R²={r2:.4f}")

    # 5. Visualização da Avaliação
    plot_evaluation_dashboard(y_test, predictions, mse, r2, final_df, X_test)

    # 6. Salvamento dos Resultados
    save_results_to_csv(X_test, y_test, predictions, final_df)
    
    print("\n--- DESAFIO CONCLUÍDO COM SUCESSO ---")

if __name__ == "__main__":
    main()
