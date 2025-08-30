Desafio Técnico - Itaú/CNpq - Trilha de Dados & IA
Visão Geral do Projeto
Este projeto implementa um protótipo de um módulo de análise de custos para a nuvem Microsoft Azure, conforme proposto no desafio. O objetivo principal é criar um pipeline de dados robusto que coleta informações de preços em tempo real de uma API oficial e utiliza um modelo de Machine Learning para prever os custos de instâncias computacionais com alta precisão.

Metodologia
O pipeline foi construído seguindo as melhores práticas de engenharia de dados e ciência de dados, dividido nas seguintes etapas:

Coleta de Dados via API Oficial: Os dados de preços foram consumidos diretamente da API oficial Azure Retail Prices. O pipeline foi projetado para lidar com a paginação da API, coletando de forma automatizada todos os registros de preços On-Demand para Máquinas Virtuais em duas regiões distintas: Sudeste do Brasil (brazilsoutheast) e Leste dos EUA (eastus). Notavelmente, a análise foi adaptada para incluir dados de VMs com sistemas operacionais Linux e Windows, refletindo a disponibilidade real de produtos na API para as regiões consultadas.

Engenharia de Features e Processamento: Após a coleta, os dados passaram por um rigoroso processo de tratamento:

Limpeza e filtragem para remover modelos de preços voláteis (como Spot) e focar em instâncias de uso geral.

Extração programática das especificações (vCPU e Memória) diretamente do nome da instância (armSkuName) utilizando expressões regulares e um mapeamento inteligente de famílias (ex: 'D', 'E', 'B'). Esta abordagem elimina a necessidade de mapeamentos manuais frágeis.

A feature categórica regiao foi transformada em colunas numéricas através de One-Hot Encoding para ser utilizada pelo modelo.

Treinamento do Modelo de IA: Foi treinado um modelo de Regressão (RandomForestRegressor) para prever a coluna preco_hora com base nas características extraídas (vCPU, Memória) e na localização da instância.

Avaliação e Visualização: O modelo alcançou uma altíssima precisão, com um R² (Coeficiente de Determinação) superior a 0.99. Os resultados são apresentados em um dashboard de performance (dashboard_performance_modelo.png), que inclui a análise de precisão (Real vs. Previsto) e um gráfico de resíduos para validar a ausência de viés no modelo.

Como Executar
Pré-requisitos
Python 3.8+

Git e Git LFS (opcional, para projetos maiores)

Instalação de Dependências
As bibliotecas necessárias podem ser instaladas com o seguinte comando:

Bash

pip install pandas requests scikit-learn matplotlib seaborn nbdime
Configuração do Ambiente Git (Recomendado)
Este repositório inclui um arquivo .gitattributes para garantir a consistência do código e melhorar a visualização de alterações no Jupyter Notebook. Para uma melhor experiência ao colaborar ou visualizar o histórico (git diff), recomenda-se configurar o nbdime:

Bash

nbdime config-git --enable --global
Executando a Análise
Abra o Notebook: desafio_itau_previsao_custos_azure_ana_leticia_de_araujo.ipynb no Google Colab ou em um ambiente Jupyter local.

Conexão com a Internet: O script requer uma conexão ativa para acessar a API do Azure.

Execute as Células: Rode todas as células do notebook em sequência.

Resultados: Os artefatos (dados_tratados_e_resultados.csv e dashboard_performance_modelo.png) serão gerados no diretório de execução.

Conclusões e Próximos Passos
O protótipo demonstrou com sucesso a construção de um pipeline de dados ponta a ponta, desde a coleta de dados em tempo real até a criação de um modelo preditivo preciso. A flexibilidade do pipeline foi comprovada ao se adaptar para processar um dataset heterogêneo (multi-OS e multirregional), o que reflete um cenário de dados do mundo real.

Como próximos passos, o projeto poderia ser expandido para:

Incluir dados de outros provedores de nuvem (AWS, GCP), unificando-os em um modelo multicloud.

Expandir a função de extração de especificações para cobrir mais famílias de instâncias da Azure.

Adicionar explicitamente o sistema operacional como uma feature no modelo para quantificar seu impacto no preço.

Implantar o modelo treinado como uma API RESTful para que possa ser consultado em tempo real por outras aplicações.