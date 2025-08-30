<h1><strong>Desafio Técnico - Itaú/CNpq - Trilha de Dados & IA</strong></h1>

<hr>

<h2>💡 Visão Geral do Projeto</h2>
<p>Este projeto implementa um protótipo de um módulo de análise de custos para a nuvem <strong>Microsoft Azure</strong>, conforme proposto no desafio. O objetivo principal é criar um pipeline de dados robusto que coleta informações de preços em tempo real de uma API oficial e utiliza um modelo de Machine Learning para prever os custos de instâncias computacionais com alta precisão.</p>

<hr>

<h2>🛠️ Metodologia</h2>
<p>O pipeline foi construído seguindo as melhores práticas de engenharia de dados e ciência de dados, dividido nas seguintes etapas:</p>
<ul>
<li><strong>☁️ Coleta de Dados via API Oficial</strong>: Os dados de preços foram consumidos diretamente da API oficial <strong>Azure Retail Prices</strong>. O pipeline foi projetado para lidar com a paginação da API, coletando de forma automatizada todos os registos de preços <em>On-Demand</em> para Máquinas Virtuais em duas regiões distintas: <strong>Sudeste do Brasil (<code>brazilsoutheast</code>)</strong> e <strong>Leste dos EUA (<code>eastus</code>)</strong>. A análise foi adaptada para incluir dados de VMs com sistemas operativos Linux e Windows, refletindo a disponibilidade real de produtos na API para as regiões consultadas.</li>
<li><strong>✨ Engenharia de Features e Processamento</strong>: Após a coleta, os dados passaram por um rigoroso processo de tratamento:
<ul>
<li>Limpeza e filtragem para remover modelos de preços voláteis (como <em>Spot</em>) e focar em instâncias de uso geral.</li>
<li>Extração programática das especificações (vCPU e Memória) diretamente do nome da instância (<code>armSkuName</code>) utilizando expressões regulares e um mapeamento inteligente de famílias.</li>
<li>A <em>feature</em> categórica <code>regiao</code> foi transformada em colunas numéricas através de <em>One-Hot Encoding</em> para ser utilizada pelo modelo.</li>
</ul>
</li>
<li><strong>🧠 Treinamento do Modelo de IA</strong>: Foi treinado um modelo de Regressão (<code>RandomForestRegressor</code>) para prever a coluna <code>preco_hora</code> com base nas características extraídas (vCPU, Memória) e na localização da instância.</li>
<li><strong>📊 Avaliação e Visualização</strong>: O modelo alcançou uma altíssima precisão, com um <strong>R² (Coeficiente de Determinação) superior a 0.99</strong>. Os resultados são apresentados num dashboard de performance (<code>dashboard_performance_modelo.png</code>), que inclui a análise de precisão (Real vs. Previsto) e um gráfico de resíduos para validar a ausência de viés no modelo.</li>
</ul>

<hr>

<h2>▶️ Como Executar</h2>
<p>Existem dois métodos para executar este projeto.</p>

<h3><strong>Método 1: Ambiente Virtual Python (Execução Local)</strong></h3>
<p><em>Este método é ideal para desenvolvimento e análise interativa no Jupyter Notebook.</em></p>

<h4><strong>Pré-requisitos</strong></h4>
<ul>
<li>Python 3.12+</li>
<li>Git</li>
</ul>

<h4><strong>Passo a Passo</strong></h4>
<ol>
<li><strong>Clone o repositório</strong>:
<pre><code>git clone &lt;URL_DO_SEU_REPOSITORIO&gt;
cd &lt;NOME_DO_REPOSITORIO&gt;</code></pre>
</li>
<li><strong>Crie e ative um ambiente virtual</strong>:
<pre><code># Cria o ambiente virtual
python -m venv venv

Ativa o ambiente (Windows)
.\venv\Scripts\activate

Ativa o ambiente (Linux/macOS)
source venv/bin/activate</code></pre>

</li>
<li><strong>Instale as dependências</strong>:
<p>As bibliotecas necessárias estão listadas no ficheiro <code>requirements.txt</code>. Para instalar todas, execute:</p>
<pre><code>pip install -r requirements.txt</code></pre>
</li>
<li><strong>Execute o Notebook</strong>:
<p>Abra e execute as células do notebook <code>desafio_itau_previsao_custos_azure_ana_leticia_de_araujo.ipynb</code> num ambiente Jupyter.</p>
</li>
</ol>

<h3><strong>Método 2: Usando Docker (Reprodutibilidade Garantida)</strong></h3>
<p><em>Este método utiliza o Docker para executar o pipeline num ambiente contido e 100% consistente, ideal para automação e para garantir que o script rode da mesma forma em qualquer máquina.</em></p>

<h4><strong>Pré-requisitos</strong></h4>
<ul>
<li>Docker instalado e em execução.</li>
</ul>

<h4><strong>Passo a Passo</strong></h4>
<ol>
<li><strong>Construa a Imagem Docker</strong>:
<p>No terminal, na raiz do projeto, execute o comando para construir a imagem:</p>
<pre><code>docker build -t desafio-itau-previsao .</code></pre>
</li>
<li><strong>Execute o Pipeline dentro do Contêiner</strong>:
<p>Após a construção da imagem, execute o script <code>desafio_itau.py</code>:</p>
<pre><code>docker run --rm desafio-itau-previsao</code></pre>
<blockquote><strong>Nota Importante:</strong> Este comando iniciará o contêiner, executará o pipeline completo e, no final, removerá o contêiner. Os ficheiros de resultado (.csv e .png) serão gerados dentro do contêiner e descartados no final.</blockquote>
</li>
</ol>

<hr>

<h2>📂 Estrutura do Projeto</h2>
<ul>
<li><code>desafio_itau_previsao_custos_azure_ana_leticia_de_araujo.ipynb</code>: Notebook com a análise exploratória e explicativa.</li>
<li><code>desafio_itau.py</code>: Script Python automatizado para execução do pipeline completo.</li>
<li><code>Dockerfile</code>: Define o ambiente de execução contido e reprodutível.</li>
<li><code>requirements.txt</code>: Lista as dependências do projeto.</li>
<li><code>.gitignore</code> e <code>.gitattributes</code>: Ficheiros de configuração do Git para boas práticas de versionamento.</li>
</ul>