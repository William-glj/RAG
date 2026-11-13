
1. Introducción y Objetivos
Siguiendo con el hilo conductor de la temática general elegida por vuestro equipo en el primer trabajo (deportes, cine, naturaleza, mascotas, etc.), el objetivo de esta segunda práctica es construir y desplegar un sistema RAG completo.
El sistema deberá demostrar de manera empírica su superioridad frente a un modelo de lenguaje (LLM) base sin contexto externo, mitigando las alucinaciones y aportando información especializada y actualizada sobre vuestro dominio.
2. Fases del Proyecto
Fase A: Captación Automatizada de Datos (Data Pipeline)

- Fuentes de información: Deberéis recopilar un corpus de texto enriquecido y específico sobre un subtema dentro de vuestra temática general. Podéis utilizar una combinación de fuentes públicas:
- Datasets estructurados (Kaggle, Hugging Face Datasets).
- Documentos no estructurados (PDFs de libros, manuales, artículos científicos o guías).
- Extracción web (Scraping) de webs especializadas, wikis, bases de datos de cine (IMDb), blogs de deportes, etc.
- APIs públicas (ej. APIs de datos meteorológicos, deportivas, de películas como TMDB).
- Automatización obligatoria: Todo el proceso de descarga, limpieza y preparación de los datos debe estar totalmente automatizado mediante un script en Python (evitando cualquier carga manual de ficheros).
Fase B: Arquitectura RAG y Estrategia de Chunking

- Librerías de orquestación: Se deberá seleccionar un framework de desarrollo RAG como LangChain, LlamaIndex o Haystack.
- Base de Datos Vectorial (Vector DB): Elegid e integrad una base de datos vectorial para almacenar los embeddings (ej. ChromaDB, FAISS, Qdrant, Milvus o Pinecone).
- Estrategia de Chunking (Segmentación): No basta con trocear los textos arbitrariamente. Deberéis justificar técnicamente el método elegido:
- Ejemplos de técnicas: Chunking por tamaño fijo con solapamiento (Fixed-size chunking), chunking basado en semántica (Semantic chunking), o segmentación estructurada basada en encabezados Markdown (Markdown/Recursive Character Text Splitting).
- Justificación: Explicar cómo el tamaño del chunk afecta al contexto y a la recuperación de información.
Fase C: Evaluación Comparativa (Sin RAG vs. Con RAG)

- Deberéis diseñar un banco de pruebas de al menos 5 a 10 preguntas específicas y complejas sobre vuestro dominio que un LLM genérico no sabría responder con precisión (preguntas de nicho, datos muy concretos o actualizados).
- Se mostrará de forma comparativa y documentada las respuestas obtenidas:
1. Modelo sin RAG: Demostrando las alucinaciones o la falta de conocimiento del modelo base.
2. Modelo con RAG: Demostrando cómo recupera la información correcta de vuestra base de datos vectorial y genera una respuesta precisa citando o basándose en los fragmentos recuperados.
Fase D: Interfaz de Usuario y Despliegue

- Se desarrollará una aplicación web interactiva que permita al usuario interactuar con el sistema RAG en tiempo real.
- Herramientas permitidas: Streamlit o Gradio.
- Despliegue: La aplicación deberá estar desplegada y accesible públicamente (por ejemplo, en Hugging Face Spaces, Streamlit Community Cloud o plataformas cloud equivalentes).
Fase E: Técnicas Avanzadas de RAG (Opcional / Puntuación Extra)
Para los equipos que deseen optar a la máxima calificación, se valorará la implementación de técnicas de RAG Avanzado, tales como:

- Reranking: Uso de modelos de reordenación (ej. Cohere Rerank o FlashRank) para afinar los fragmentos más relevantes antes de dárselos al LLM.
- Filtros de metadatos: Filtrar la búsqueda vectorial por categorías, fechas o etiquetas antes de realizar la recuperación semántica.
- RAG Adaptativo / Modular: Estrategias donde el sistema decide si necesita buscar información externa o si puede responder directamente, o combinación de múltiples recuperadores (Hybrid Search: búsqueda léxica BM25 + búsqueda vectorial).
3. Documentación y Entrega (Quarto y GitHub)

- Estructura del Repositorio: El código fuente completo (scripts de ingesta, configuración de la base de datos, código de la interfaz y ficheros de requisitos) se alojará en el repositorio de GitHub del equipo (dentro de una nueva carpeta específica para este trabajo, por ejemplo, /rag_project).
- Página web en Quarto: Se generará una memoria técnica interactiva utilizando Quarto. La web debe incluir:
- La motivación del tema y la procedencia de los datos.
- La explicación detallada del pipeline de automatización y la estrategia de chunking elegida.
- Los resultados de la evaluación comparativa (Sin RAG vs. Con RAG) con capturas o ejemplos claros.
- El enlace directo a la aplicación web desplegada en Streamlit/Gradio y al repositorio de GitHub.
- Extractos del código más relevante explicados de forma pedagógica.
- Publicación: La web generada con Quarto se publicará online (ej. mediante GitHub Pages o Quarto Pub) y su URL se compartirá en Loop (Teams).
4. Exposición y Criterios de Evaluación

- Fecha de exposición: 27 de noviembre
- Defensa coral y especializada: Durante la exposición en clase, cada miembro del equipo deberá responsabilizarse de un aspecto concreto del proyecto (por ejemplo: uno explica la captación y el pipeline de datos; otro la base de datos vectorial, el chunking y las librerías; otro la interfaz y el despliegue; y otro la evaluación comparativa y las técnicas avanzadas implementadas).
- Criterios de valoración:
- Robustez de la automatización en la ingesta de datos.
- Fundamentación técnica en la elección de la base de datos y la estrategia de chunking.
- Claridad en la comparativa de efectividad (demostración real de cómo el RAG soluciona las limitaciones del LLM).
- Calidad de la interfaz de usuario y éxito en su despliegue público.
- Calidad de la documentación en Quarto y estructuración del código en GitHub.
- Dominio técnico individual demostrado durante la exposición oral.
GRUPOS y TEMAS (exposiciones 27 de noviembre) 

- Grupo 1: Jose Ramon, Guillermo, Dylan, Tomas:  
- Grupo 2: Juan Perez, Carlos Rojas, Arie Rivas, Carlos Rabines
- Grupo 3: Alejandro, Miguel, Javier URL GitHub: https://github.com/JavierCagigal/proy-rag
