# -*- coding: utf-8 -*-
"""
app.py
======

Interfaz web (Fase D) del sistema RAG sobre cine de ciencia ficción e IA.

Modos:
    - Con RAG:      recupera fragmentos del índice y responde con ellos.
    - Sin RAG:      pregunta directamente al LLM (línea base).
    - Comparar:     muestra ambas respuestas una al lado de la otra.

Proveedores: el configurado por defecto (LLM_*) o OpenRouter (OPENROUTER_*).

Ejecución local:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

import config_rag as cfg
from con_rag import SISTEMA as SISTEMA_CON_RAG, formatear_contexto
from llm_comun import (NOMBRES_PROVEEDOR, PROVEEDORES, VARIABLES,
                       crear_cliente, faltan_variables, preguntar)
from rag_comun import EmbeddingsE5, abrir_base, construir_filtro
from sin_rag import SISTEMA as SISTEMA_SIN_RAG

st.set_page_config(
    page_title="Cine de ciencia ficción y miedo tecnológico",
    page_icon="🎬",
    layout="wide",
)

TIPOS = ["(todos)", "critica", "opinion_espectadores", "academico",
         "divulgacion", "resena", "ficha"]


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

def cargar_secretos() -> None:
    """En Streamlit Cloud las claves están en st.secrets; las pasamos al entorno
    para que llm_comun.py las lea igual que desde el fichero .env local."""
    try:
        for variables in VARIABLES.values():
            for clave in variables:
                if clave in st.secrets:
                    os.environ[clave] = str(st.secrets[clave])
    except Exception:
        pass  # sin secrets.toml: se usará el .env local


cargar_secretos()


# --------------------------------------------------------------------------- #
# Recursos cacheados (se cargan una sola vez por proceso)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Cargando el modelo de embeddings y el índice...")
def obtener_base():
    return abrir_base(cfg.INDICE_DIR, EmbeddingsE5())


@st.cache_resource
def obtener_cliente(proveedor: str):
    """Cliente y modelo por proveedor; cada proveedor tiene su propia caché."""
    return crear_cliente(proveedor)


# --------------------------------------------------------------------------- #
# Lógica RAG
# --------------------------------------------------------------------------- #

def recuperar(base, pregunta: str, k: int, filtro) -> list[dict]:
    """Fragmentos más cercanos, en un formato sencillo para la interfaz."""
    resultados = base.similarity_search_with_score(pregunta, k=k, filter=filtro)
    return [
        {
            "titulo": doc.metadata.get("titulo", ""),
            "tipo": doc.metadata.get("tipo", ""),
            "fuente": doc.metadata.get("fuente", ""),
            "distancia": float(distancia),
            "texto": doc.page_content,
            "_doc": doc,  # para formatear el contexto del prompt
        }
        for doc, distancia in resultados
    ]


def responder_sin_rag(cliente, modelo: str, pregunta: str) -> str:
    return preguntar(cliente, modelo, SISTEMA_SIN_RAG, pregunta)


def responder_con_rag(cliente, modelo: str, base, pregunta: str, k: int,
                      filtro) -> tuple[str, list[dict]]:
    fragmentos = recuperar(base, pregunta, k, filtro)
    if not fragmentos:
        return "No hay información suficiente en la base de datos.", []
    contexto = formatear_contexto([(f["_doc"], f["distancia"]) for f in fragmentos])
    usuario = f"Fragmentos:\n\n{contexto}\n\nPregunta: {pregunta}"
    return preguntar(cliente, modelo, SISTEMA_CON_RAG, usuario), fragmentos


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #

def mostrar_fragmentos(fragmentos: list[dict]) -> None:
    with st.expander(f"Fragmentos recuperados ({len(fragmentos)})"):
        for n, f in enumerate(fragmentos, 1):
            st.markdown(f"**[{n}] {f['titulo']}** · tipo `{f['tipo']}` · "
                        f"distancia {f['distancia']:.3f}")
            st.caption(f"Fuente: {f['fuente']}")
            texto = f["texto"]
            st.text(texto[:500] + ("..." if len(texto) > 500 else ""))


def mostrar_entrada(entrada: dict) -> None:
    with st.chat_message("user"):
        st.write(entrada["pregunta"])

    with st.chat_message("assistant"):
        st.caption(entrada["etiqueta"])
        if entrada["sin"] is not None and entrada["con"] is not None:
            col_sin, col_con = st.columns(2)
            with col_sin:
                st.subheader("Sin RAG")
                st.write(entrada["sin"])
            with col_con:
                st.subheader("Con RAG")
                st.write(entrada["con"])
        elif entrada["sin"] is not None:
            st.write(entrada["sin"])
        else:
            st.write(entrada["con"])

        if entrada["fragmentos"]:
            mostrar_fragmentos(entrada["fragmentos"])


def main() -> None:
    st.title("🎬 Cine de ciencia ficción y miedo a la tecnología")
    st.caption("Sistema RAG sobre críticas, artículos, tesis, reseñas y fichas "
               "de películas. Compara las respuestas con y sin contexto.")

    st.session_state.setdefault("historial", [])

    with st.sidebar:
        st.header("Configuración")
        proveedor = st.selectbox("Proveedor del LLM", PROVEEDORES,
                                 format_func=lambda p: NOMBRES_PROVEEDOR[p])
        modo = st.radio("Modo", ["Con RAG", "Sin RAG", "Comparar ambos"], index=0)
        k = st.slider("Fragmentos a recuperar (k)", 1, 10, cfg.K_POR_DEFECTO)
        tipo = st.selectbox("Filtrar por tipo de documento", TIPOS)
        solo_ia = st.checkbox("Solo películas relevantes a IA")
        if st.button("Borrar conversación"):
            st.session_state.historial = []
            st.rerun()

    faltan = faltan_variables(proveedor)
    if faltan:
        st.error(f"Faltan variables para {NOMBRES_PROVEEDOR[proveedor]}: "
                 f"{', '.join(faltan)}. Configúralas en .env (local) o en "
                 "Secrets (Streamlit Cloud).")
        st.stop()

    if not cfg.INDICE_DIR.exists():
        st.error("No existe el índice vectorial (indice_chroma/). "
                 "Ejecuta construir_indice.py y súbelo al repositorio.")
        st.stop()

    filtro = construir_filtro(None if tipo == "(todos)" else tipo, solo_ia)

    pregunta = st.chat_input("Escribe una pregunta sobre cine de ciencia ficción e IA...")
    if pregunta:
        entrada = {"pregunta": pregunta, "etiqueta": "", "sin": None,
                   "con": None, "fragmentos": []}
        try:
            cliente, modelo = obtener_cliente(proveedor)
            entrada["etiqueta"] = f"{NOMBRES_PROVEEDOR[proveedor]} · modelo `{modelo}`"
            base = obtener_base() if modo != "Sin RAG" else None
            with st.spinner("Consultando el modelo..."):
                if modo in ("Sin RAG", "Comparar ambos"):
                    entrada["sin"] = responder_sin_rag(cliente, modelo, pregunta)
                if modo in ("Con RAG", "Comparar ambos"):
                    entrada["con"], entrada["fragmentos"] = responder_con_rag(
                        cliente, modelo, base, pregunta, k, filtro)
            st.session_state.historial.append(entrada)
        except Exception as exc:
            st.error(f"Error al consultar el modelo: {type(exc).__name__}: {exc}")

    for entrada in st.session_state.historial:
        mostrar_entrada(entrada)


main()