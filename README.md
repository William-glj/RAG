# Sistema RAG sobre cine de ciencia ficción y miedo a la tecnología

Proyecto de la asignatura para el **Trabajo 2: Diseño, implementación y evaluación de un sistema RAG**. El sistema responde preguntas sobre cómo el cine de ciencia ficción representa los temores sociales ante la innovación tecnológica, con la **inteligencia artificial** como foco principal.

El objetivo es demostrar empíricamente que un modelo de lenguaje que consulta una base documental propia (RAG) responde con más precisión y menos alucinaciones que el mismo modelo sin contexto externo.

## Índice

- [Descripción general](#descripción-general)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Interfaz web](#interfaz-web)
- [Despliegue](#despliegue)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Autores](#autores)

## Descripción general

El proyecto se organiza en fases, que corresponden a las del enunciado:

| Fase | Contenido | Scripts principales |
|---|---|---|
| **A. Captación de datos** | Descarga automatizada de IMDb, TMDB, críticas web, PDFs académicos y un vídeo de YouTube, seguida de limpieza y normalización. | `cargar_datos.py`, `scraping_criticas.py`, `limpiar_datos.py` |
| **B. Arquitectura RAG** | Troceado de los textos, embeddings multilingües y base vectorial ChromaDB. Incluye herramientas de prueba y un experimento de tamaño de fragmento. | `construir_indice.py`, `probar_recuperacion.py`, `comparar_chunking.py`, `rag_comun.py`, `config_rag.py` |
| **C. Evaluación** | Comparación de respuestas sin RAG y con RAG a partir de un banco de preguntas. | `sin_rag.py`, `con_rag.py`, `llm_comun.py` |
| **D. Interfaz y despliegue** | Aplicación web en Streamlit que permite preguntar en tiempo real y comparar ambos modos. | `app.py` |

### Corpus

- **11 documentos de texto:** críticas (Espinof, Roger Ebert, El Cuervo Enteradillo), análisis (CinemaNet, The Conversation), un artículo de La Vanguardia, dos tesis en PDF (UACM y UCM) y la transcripción de un vídeo de YouTube.
- **Datos estructurados:** unas 59 películas de ciencia ficción con metadatos de IMDb y TMDB, y unas 458 reseñas de usuarios de TMDB.

El corpus se genera automáticamente desde las fuentes; no se añade ningún fichero a mano.

### Flujo general

```
IMDb / TMDB / webs / PDFs / YouTube
              │
              ▼
   cargar_datos.py + scraping_criticas.py
              │
              ▼
          limpiar_datos.py  ──►  datosDescargados/limpio/
              │
              ▼
        construir_indice.py ──►  indice_chroma/   (troceado + embeddings + ChromaDB)
              │
     ┌────────┴─────────┐
     ▼                  ▼
 sin_rag.py         con_rag.py  ◄── recupera fragmentos ── indice_chroma/
     │                  │
     └────────┬─────────┘
              ▼
           app.py  (interfaz Streamlit)
```

## Estructura del proyecto

```
.
├── cargar_datos.py          # Fase A: IMDb y TMDB, y cruce entre ambos
├── scraping_criticas.py     # Fase A: críticas, PDFs y transcripción de vídeo
├── limpiar_datos.py         # Fase A: limpieza y normalización del corpus
├── construir_indice.py      # Fase B: troceado, embeddings e indexación
├── probar_recuperacion.py   # Fase B: consulta el índice sin LLM
├── comparar_chunking.py     # Fase B: experimento de tamaños de fragmento
├── rag_comun.py             # Fase B: embeddings E5, ChromaDB y filtros
├── config_rag.py            # Parámetros compartidos (chunking, modelos, preguntas)
├── llm_comun.py             # Cliente del LLM (proveedor, reintentos ante 429)
├── sin_rag.py               # Fase C: preguntas al LLM sin contexto
├── con_rag.py               # Fase C: preguntas al LLM con contexto recuperado
├── app.py                   # Fase D: interfaz web en Streamlit
├── presentacion.qmd         # Memoria técnica en Quarto
├── Tareas.md                # Enunciado del trabajo
├── requirements.txt         # Dependencias
├── .env                     # Claves de API (no se sube al repositorio)
├── datosDescargados/        # Datos crudos y limpios (generados)
├── indice_chroma/           # Índice vectorial (generado)
├── experimentos_chunking/   # Resultados de la comparativa de chunking (generado)
└── resultados_llm/          # Respuestas guardadas en CSV (generado)
```

## Requisitos

- Python 3.12 (recomendado). Python 3.15 no es compatible todavía con `onnxruntime`, una dependencia de ChromaDB.
- [uv](https://docs.astral.sh/uv/) para gestionar el entorno y las dependencias.
- Conexión a Internet en la primera ejecución, para descargar datasets y el modelo de embeddings.
- Una clave gratuita de TMDB (opcional, pero necesaria para las reseñas de usuarios).
- Acceso a un proveedor de LLM compatible con la API de OpenAI, y/o una cuenta de OpenRouter.

## Instalación

Desde la carpeta del proyecto:

```bash
uv venv --python 3.12
source .venv/bin/activate          # Linux / macOS / WSL
# .venv\Scripts\activate           # Windows (PowerShell o CMD)

uv pip install -r requirements.txt
uv pip install yt-dlp              # usado por scraping_criticas.py para el vídeo
```

En Windows con WSL, crea el entorno dentro de WSL. Un `.venv` creado desde Windows no funciona desde Linux.

## Configuración

### Variables de entorno

Crea un fichero `.env` en la raíz del proyecto:

```env
# Clave de TMDB (opcional; sin ella no se descargan las reseñas de usuarios)
TMDB_API_KEY=tu_clave_de_tmdb

# Proveedor "llm" (API compatible con OpenAI)
LLM_API_KEY=tu_clave
LLM_BASE_URL=https://tu-proveedor/v1
LLM_MODEL=nombre-exacto-del-modelo

# Proveedor "openrouter" (opcional)
OPENROUTER_API_KEY=tu_clave_de_openrouter
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
```

El fichero `.env` no debe subirse al repositorio. Está incluido en `.gitignore`.

### Parámetros del sistema

Los parámetros compartidos (tamaño de fragmento, modelo de embeddings, número de resultados, preguntas de prueba) están en `config_rag.py`. Cambiarlos ahí afecta a todos los scripts.

Valores actuales:

| Parámetro | Valor | Descripción |
|---|---|---|
| `CHUNK_SIZE` | 1500 | Tamaño máximo de fragmento en caracteres |
| `CHUNK_OVERLAP` | 225 | Solapamiento entre fragmentos (15 %) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modelo de embeddings multilingüe |
| `K_POR_DEFECTO` | 5 | Fragmentos recuperados por pregunta |

## Uso

Ejecuta los scripts en este orden la primera vez. Con `uv run` no hace falta activar el entorno.

### 1. Captación y limpieza de datos (Fase A)

```bash
uv run python cargar_datos.py
uv run python scraping_criticas.py
uv run python limpiar_datos.py
```

Los datos limpios quedan en `datosDescargados/limpio/`, y `resumen_limpieza.txt` recoge lo eliminado en cada paso.

### 2. Construcción del índice (Fase B)

```bash
uv run python construir_indice.py
```

Este comando borra el índice anterior y lo reconstruye desde cero. Para revisar cuántos fragmentos se generarían sin calcular embeddings:

```bash
uv run python construir_indice.py --solo-contar
```

### 3. Prueba de recuperación (sin LLM)

```bash
uv run python probar_recuperacion.py --demo
uv run python probar_recuperacion.py "¿Cómo se representa a HAL 9000?"
uv run python probar_recuperacion.py "miedo a la IA" --tipo resena
uv run python probar_recuperacion.py "miedo a la IA" --solo-ia
uv run python probar_recuperacion.py     # modo interactivo
```

### 4. Experimento de tamaño de fragmento

```bash
uv run python comparar_chunking.py
uv run python comparar_chunking.py --tamanos 400 800 1200 -k 4
```

Los resultados se guardan en `experimentos_chunking/`, incluido `resultados_chunking.md`, listo para incluir en la memoria.

### 5. Preguntas al LLM (Fase C)

Sin contexto:

```bash
uv run python sin_rag.py
uv run python sin_rag.py "¿Qué nota tiene Ex Machina en IMDb?"
uv run python sin_rag.py --proveedor openrouter
```

Con RAG:

```bash
uv run python con_rag.py
uv run python con_rag.py "miedo a la IA" -k 3 --solo-ia
uv run python con_rag.py --proveedor openrouter
```

Las respuestas se guardan en `resultados_llm/` (`sin_rag.csv`, `con_rag.csv`, y con sufijo `_openrouter` cuando se usa OpenRouter). Ante un error 429 (demasiadas peticiones), el sistema reintenta hasta dos veces con espera creciente y, si el proveedor sigue saturado, registra un mensaje en lugar de detenerse.

## Interfaz web

```bash
uv run streamlit run app.py
```

La aplicación se abre en `http://localhost:8501` y ofrece:

- **Selector de proveedor** (LLM configurado u OpenRouter).
- **Modos:** Con RAG, Sin RAG o Comparar ambos (respuestas en paralelo).
- **Parámetros:** número de fragmentos (k), filtro por tipo de documento y filtro de películas relevantes a IA.
- **Fragmentos recuperados:** cada respuesta con RAG muestra los fragmentos usados, con su fuente y su distancia.

## Despliegue

La aplicación está pensada para **Streamlit Community Cloud**:

1. Sube el repositorio a GitHub. Debe incluir `app.py`, `requirements.txt`, los módulos `.py` y la carpeta `indice_chroma/`.
2. En [share.streamlit.io](https://share.streamlit.io), crea una aplicación desde el repositorio, con `app.py` como fichero principal y Python 3.12.
3. En **Advanced settings → Secrets**, añade las variables de LLM y OpenRouter en formato TOML:

   ```toml
   LLM_API_KEY = "..."
   LLM_BASE_URL = "..."
   LLM_MODEL = "..."

   OPENROUTER_API_KEY = "..."
   OPENROUTER_MODEL = "..."
   ```

En la nube, `app.py` no ejecuta la construcción del índice. Por eso `indice_chroma/` debe estar en el repositorio.

## Limitaciones conocidas

- **Preguntas de prueba provisionales.** Las preguntas y palabras clave de `config_rag.py` se escribieron antes de revisar el contenido real del corpus. La métrica de acierto solo comprueba que aparezca una palabra clave en el fragmento recuperado; no mide si la respuesta es correcta.
- **Dependencia de proveedores externos.** Los límites de peticiones de los proveedores (especialmente en los modelos gratuitos de OpenRouter) pueden provocar errores 429.
- **Reseñas de IMDb no disponibles.** IMDb no ofrece API pública de reseñas y su web bloquea el acceso automatizado. Por eso las reseñas de texto proceden de TMDB.
- **Roger Ebert mediante Wayback Machine.** La crítica se obtiene de una copia archivada, porque el sitio original bloquea el acceso automatizado.
- **Tamaño del corpus.** Con unas 59 películas y 11 documentos, los resultados son orientativos y no generalizables.

## Autores

Equipo 3 (asignatura de Ingeniería de datos / sistemas inteligentes, curso 2025-2026):

- Alejandro
- Miguel
- Javier Cagigal

Repositorio: https://github.com/JavierCagigal/proy-rag
