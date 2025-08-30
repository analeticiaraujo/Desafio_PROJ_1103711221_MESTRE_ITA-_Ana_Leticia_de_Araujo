# Usa uma imagem baseada em Debian (slim-bookworm) para maior robustez
# e compatibilidade com pacotes científicos.
FROM python:3.12-slim-bookworm

# ATUALIZAÇÃO E INSTALAÇÃO DE DEPENDÊNCIAS DE SISTEMA:
# Atualiza os pacotes de segurança e instala as ferramentas de compilação
# necessárias para pacotes como numpy/scikit-learn, tudo num único passo.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do contêiner.
WORKDIR /app

# Copia e instala as dependências Python.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os ficheiros da aplicação.
COPY . .

# Define o comando que será executado quando o contêiner iniciar.
CMD ["python", "desafio_itau.py"]

