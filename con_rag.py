# -*- coding: utf-8 -*-
"""
con_rag.py
==========

Script 2: recupera fragmentos del índice ChromaDB y los envía al modelo como
contexto. El modelo debe responder solo con esos fragmentos y citarlos.

Requisito: haber ejecutado antes construir_indice.py.

Uso:
    python con_rag.py                                          # proveedor por defecto
    python con_rag.py --proveedor openrouter                   # usa OpenRouter
    python con_rag.py --proveedor openrouter "¿Qué nota tiene Upgrade?"
    python con_rag.py --proveedor openrouter "miedo a la IA" -k 3 --solo-ia
"""

from __future__ import annotations

import argparse
import sys
import time

import config_rag as cfg
from llm_comun import PROVEEDORES, crear_cliente, guardar_resultados, preguntar
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


def responder(base, cliente, modelo: str, pregunta: str, k: int,
              filtro) -> tuple[str, str]:
    """Devuelve (respuesta, fuentes usadas)."""
    resultados = base.similarity_search_with_score(pregunta, k=k, filter=filtro)
    if not resultados:
        return "No hay información suficiente en la base de datos.", ""

    usuario = (f"Fragmentos:\n\n{formatear_contexto(resultados)}\n\n"
               f"Pregunta: {pregunta}")
    respuesta = preguntar(cliente, modelo, SISTEMA, usuario)
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
    ap.add_argument("--proveedor", choices=PROVEEDORES, default="llm",
                    help="proveedor del LLM (por defecto: llm)")
    ap.add_argument("--indice", default=cfg.INDICE_DIR)
    args = ap.parse_args()

    if not args.indice.exists():
        raise SystemExit("No existe el índice. Ejecutad antes: python construir_indice.py")

    cliente, modelo = crear_cliente(args.proveedor)
    base = abrir_base(args.indice, EmbeddingsE5())
    filtro = construir_filtro(args.tipo, args.solo_ia)
    print(f"Proveedor: {args.proveedor} | modelo: {modelo}")

    if args.pregunta:
        preguntas = [args.pregunta]
    else:
        preguntas = [q["pregunta"] for q in cfg.PREGUNTAS_PRUEBA]

    filas = []
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n[{i}/{len(preguntas)}] {pregunta}")
        print("  Consultando al modelo...", flush=True)

        inicio = time.time()
        respuesta, fuentes = responder(base, cliente, modelo, pregunta, args.k, filtro)
        print(f"  Tiempo de respuesta: {time.time() - inicio:.1f} s")

        if respuesta:
            print(respuesta)
        else:
            print("  [aviso] el modelo devolvió una respuesta vacía. Prueba otro "
                  "modelo o aumenta MAX_TOKENS en llm_comun.py.")
        print(f"  Fuentes: {fuentes}")
        filas.append({"pregunta": pregunta, "respuesta": respuesta, "fuentes": fuentes})

    ruta = guardar_resultados(filas, "con_rag", args.proveedor)
    print(f"\nResultados guardados en {ruta}")


if __name__ == "__main__":
    main()