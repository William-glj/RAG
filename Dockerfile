# ============================================================================
# Dockerfile — Sistema RAG sobre cine de ciencia ficción y miedo a la tecnología
#
# App web Streamlit (app.py) con ChromaDB + sentence-transformers.
# Pensado para Hugging Face Spaces (SDK "Docker"), y compatible con
# Render / Railway / Fly.io (usa $PORT si la plataforma lo define).
#
# Construcción local de prueba:
#   docker build -t rag-cine .
#   docker run --rm -p 8501:7860 --env-file .env rag-cine
#
# La app escucha en el puerto 7860 (el que espera Hugging Face Spaces).
# ============================================================================

FROM python:3.12-slim

# build-essential y libgomp1: permiten compilar ruedas si alguna dependencia
# (hnswlib, onnxruntime...) no tiene wheel para 3.12, y dan soporte OpenMP
# para el cálculo en CPU. Si confirmas que todo instala con wheels, puedes
# quitar build-essential para aligerar la imagen.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencias en su propia capa (mejor caché al reconstruir).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 2) Código, módulos y el índice vectorial (indice_chroma/).
#    El .dockerignore excluye .env, .venv, datos crudos, etc.
COPY . .

# 3) Pre-descarga del modelo de embeddings para que el arranque en frío sea
#    instantáneo. Comenta esta línea si prefieres una imagen más ligera y
#    aceptas que el primer arranque descargue ~470 MB.
RUN python -c "import config_rag as c; from sentence_transformers import SentenceTransformer; SentenceTransformer(c.EMBEDDING_MODEL)"

EXPOSE 7860

# $PORT permite que Render/Railway/Fly inyecten su propio puerto;
# si no está definido (Hugging Face), se usa 7860.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
