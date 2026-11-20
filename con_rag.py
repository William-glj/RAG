# -*- coding: utf-8 -*-
"""
con_rag.py
==========

Script 2: recupera fragmentos del índice ChromaDB y los envía al modelo Llama
como contexto. El modelo debe responder solo con esos fragmentos y citarlos.

Requisito: haber ejecutado antes construir_indice.py.

Uso:
    python con_rag.py                              # preguntas de config_rag.PREGUNTAS_PRUEBA
    python con_rag.py "¿Qué nota tiene Upgrade?"   # una pregunta concreta
    python con_rag.py "miedo a la IA" -k 3 --solo-ia
"""

from __future__ import annotations

import argparse
import sys

import config_rag as cfg
from llm_comun import crear_cliente, guardar_resultados, preguntar
from rag_comun import EmbeddingsE5, abrir_base, construir_filtro

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SISTEMA = (
    "Eres un asistente especializado en cine de ciencia ficción. Responde en español "
    "usando EXCLUSIVAMENTE la información de los fragmentos proporcionados. Cita entre "
    "corchetes los fragmentos que uses, por ejemplo [2]. Si la respuesta no aparece en "
    "los fragmentos, responde: 'No hay información suficiente en la base de datos.'"
)


def formatear_contexto(resultados) -> str:
    bloques = []
    for n, (doc, _distancia) in enumerate(resultados, 1):
        m = doc.metadata
        cabecera = (f"[{n}] tipo={m.get('tipo')} | titulo={m.get('titulo', '')} "
                    f"| fuente={m.get('fuente', '')}")
        bloques.append(f"{cabecera}\n{doc.page_content}")
    return "\n\n".join(bloques)


def responder(base, cliente, pregunta: str, k: int, filtro) -> tuple[str, str]:
    """Devuelve (respuesta, fuentes usadas)."""
    resultados = base.similarity_search_with_score(pregunta, k=k, filter=filtro)
    if not resultados:
        return "No hay información suficiente en la base de datos.", ""

    usuario = (f"Fragmentos:\n\n{formatear_contexto(resultados)}\n\n"
               f"Pregunta: {pregunta}")
    respuesta = preguntar(cliente, SISTEMA, usuario)
    fuentes = " ; ".join(
        f"[{n}] {doc.metadata.get('titulo', '')} ({doc.metadata.get('tipo')})"
        for n, (doc, _) in enumerate(resultados, 1)
    )
    return respuesta, fuentes


def main() -> None:
    ap = argparse.ArgumentParser(description="Preguntas al LLM con RAG.")
    ap.add_argument("pregunta", nargs="?", help="pregunta concreta (opcional)")
    ap.add_argument("-k", type=int, default=cfg.K_POR_DEFECTO,
                    help="nº de fragmentos a recuperar")
    ap.add_argument("--tipo", help="filtrar por tipo de documento")
    ap.add_argument("--solo-ia", action="store_true",
                    help="solo películas marcadas como relevantes a IA")
    ap.add_argument("--indice", default=cfg.INDICE_DIR)
    args = ap.parse_args()

    if not cfg.INDICE_DIR.exists():
        raise SystemExit("No existe el índice. Ejecutad antes: python construir_indice.py")

    base = abrir_base(args.indice, EmbeddingsE5())
    filtro = construir_filtro(args.tipo, args.solo_ia)
    cliente = crear_cliente()

    if args.pregunta:
        preguntas = [args.pregunta]
    else:
        preguntas = [q["pregunta"] for q in cfg.PREGUNTAS_PRUEBA]

    filas = []
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n[{i}/{len(preguntas)}] {pregunta}")
        respuesta, fuentes = responder(base, cliente, pregunta, args.k, filtro)
        print(respuesta)
        print(f"  Fuentes: {fuentes}")
        filas.append({"pregunta": pregunta, "respuesta": respuesta, "fuentes": fuentes})

    ruta = guardar_resultados(filas, "con_rag.csv")
    print(f"\nResultados guardados en {ruta}")


if __name__ == "__main__":
    main()