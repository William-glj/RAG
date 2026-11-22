# -*- coding: utf-8 -*-
"""
config_rag.py
=============

Ajustes compartidos de la Fase B (arquitectura RAG y chunking).

Todos los scripts de la fase (construir_indice.py, probar_recuperacion.py,
comparar_chunking.py) y, mas adelante, la evaluacion (Fase C) y la interfaz
(Fase D) leen sus parametros de AQUI, para que nadie cambie un valor en un
script y se olvide de los demas.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas (relativas a la carpeta donde viven los scripts)
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ / "datosDescargados"
CARPETA_LIMPIO = CARPETA_DATOS / "limpio"               # salida de limpiar_datos.py
CARPETA_CRITICAS_LIMPIO = CARPETA_LIMPIO / "criticas"

INDICE_DIR = RAIZ / "indice_chroma"                     # base vectorial persistente
EXPERIMENTOS_DIR = RAIZ / "experimentos_chunking"       # indices de la comparativa
COLECCION = "cine_scifi_ia"

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
# El corpus mezcla espanol (criticas) e ingles (resenas de TMDB, Roger Ebert),
# asi que hace falta un modelo MULTILINGUE.
#
# Por que NO paraphrase-multilingual-MiniLM-L12-v2: segun su ficha, trunca cada
# texto a 128 tokens (~500 caracteres). Con fragmentos de 1000 caracteres, la
# mitad del fragmento no se tendria en cuenta al calcular el embedding.
# multilingual-e5-small admite hasta 512 tokens y es ligero (se ejecuta en CPU).
#
# Los modelos E5 exigen un prefijo: "passage: " para los textos que se indexan
# y "query: " para las preguntas. Lo anade rag_comun.EmbeddingsE5.
# Si cambiais a un modelo sin prefijos, poned ambos prefijos a "".

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PREFIJO_DOCUMENTO = "passage: "
PREFIJO_CONSULTA = "query: "
BATCH_EMBEDDINGS = 32
LOTE_CHROMA = 500          # fragmentos por lote al insertar en Chroma

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Punto de partida (se valida con comparar_chunking.py):
#   - 1000 caracteres (~250 tokens) con solapamiento de 150 (15 %).
#   - Corte recursivo: primero por parrafo, luego por linea, frase y palabra.
#   - Las fichas de pelicula NO se trocean (un fragmento por pelicula).
#   - Las opiniones de SensaCine se separan por espectador ANTES de trocear.

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 225
SEPARADORES = ["\n\n", "\n", ". ", " ", ""]

# Antepone a cada fragmento una linea de contexto ("[Critica sobre ... ]") para
# que el embedding sepa de que pelicula/fuente habla aunque el trozo no la nombre.
ANADIR_CONTEXTO = True

K_POR_DEFECTO = 5          # fragmentos que se recuperan por pregunta

# Las opiniones de SensaCine que solo tienen puntuacion (sin comentario) no
# aportan informacion: se descartan las que tengan menos de estos caracteres
# de texto una vez quitada la cabecera "Autor | Nota | Fecha".
MIN_CARACTERES_OPINION = 40

# Tipo de documento segun la fuente (se busca la clave dentro del nombre de la
# fuente, en minusculas). Lo que no coincida se considera "critica".
# Estos tipos sirven despues como FILTRO de metadatos (Fase E).
TIPO_POR_FUENTE = {
    "sensacine": "opinion_espectadores",
    "uacm": "academico",
    "uoc": "academico",
    "vanguardia": "divulgacion",
    "revista central": "divulgacion",
}

# --------------------------------------------------------------------------- #
# Preguntas de prueba de recuperacion
# --------------------------------------------------------------------------- #
# Se usan en probar_recuperacion.py --demo y en comparar_chunking.py.
# Un fragmento recuperado cuenta como ACIERTO si su texto contiene TODAS las
# palabras de "palabras_clave". Cada palabra se busca COMPLETA y sin distinguir
# mayusculas (asi "HAL" no coincide con "hallazgo"); si termina en "*" se busca
# como prefijo ("cibern*" encuentra "cibernetica"). Ver rag_comun.contiene_palabras.
#
# OJO: estas preguntas y palabras clave son una PROPUESTA INICIAL escrita sin
# ver el contenido real de vuestros textos. Revisadlas con los datos delante
# (por ejemplo, con probar_recuperacion.py) y ajustadlas. La metrica es
# orientativa: que un fragmento contenga la palabra no garantiza que responda
# bien a la pregunta.

"""
PREGUNTAS_PRUEBA = [
    {"pregunta": "¿Cómo se representa a HAL 9000 en 2001: Una odisea del espacio?",
     "palabras_clave": ["HAL"]},
    {"pregunta": "¿Qué opinan los usuarios de TMDB sobre Ex Machina?",
     "palabras_clave": ["Ex Machina"]},
    {"pregunta": "¿Qué miedos históricos a la tecnología describe el artículo, desde la escritura hasta el 5G?",
     "palabras_clave": ["5G"]},
    {"pregunta": "¿De qué trata Upgrade y qué papel tiene el chip STEM?",
     "palabras_clave": ["STEM"]},
    {"pregunta": "¿Qué dicen los espectadores de SensaCine sobre 2001?",
     "palabras_clave": ["Nota:"]},
    {"pregunta": "¿Qué nota tiene Ex Machina en IMDb y cuántos votos?",
     "palabras_clave": ["Ex Machina", "Nota IMDb"]},
    {"pregunta": "¿Qué películas de ciencia ficción tratan sobre inteligencia artificial?",
     "palabras_clave": ["artificial intelligence"]},
    {"pregunta": "¿Qué dice la tesis sobre la cibernética en 2001?",
     "palabras_clave": ["cibern*"]},
]
"""
#Más especificas
PREGUNTAS_PRUEBA = [
    {"pregunta": "¿Los humanos somos reemplazables?",
     "palabras_clave": ["replicantes"]},
    {"pregunta": "¿Qué opinan los usuarios de TMDB sobre Ex Machina?",
     "palabras_clave": ["Ex Machina"]},
    {"pregunta": "¿La revolución fue un avance, la digitalización una mejora pero la IA es necesaria? Todos los descubrimientos mejoran o empeoran las condiciones",
     "palabras_clave": ["tecnología"]},
    {"pregunta": "GEMINI, CLAUDE CODE, OpenAI ¿Qué similitudes comparten con el chip STEM? ¿Darle el control de nuestros ordenadores es igual a entregar nuestro cuerpo?",
     "palabras_clave": ["STEM"]},
    {"pregunta": "¿Está bien no detenerse cuando ha sido suficiente, hasta dónde pueden llegar los avances tecnológicos?",
     "palabras_clave": ["moral"]},
    {"pregunta": "Humano vs máquina. Creador vs Creación ¿Quién es la herramienta entre los dos a día de hoy? Los humanos que dependemos de la tecnología o la tecnología que necesita a una persona",
     "palabras_clave": ["herramienta"]},
    {"pregunta": "Cuando existe una herramienta mejor que nosotros, ¿qué nos queda si dependemos de ella?",
     "palabras_clave": ["herramienta"]},
]
