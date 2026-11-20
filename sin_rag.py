# -*- coding: utf-8 -*-
"""
sin_rag.py
==========

Script 1: envía preguntas al modelo SIN contexto externo.
Es la línea base para comparar con con_rag.py.

Uso:
    python sin_rag.py                                   # preguntas de config_rag
    python sin_rag.py "¿Qué nota tiene Upgrade?"        # una pregunta concreta
    python sin_rag.py --proveedor openrouter            # usa OpenRouter
"""

from __future__ import annotations

import argparse
import sys

import config_rag as cfg
from llm_comun import PROVEEDORES, crear_cliente, guardar_resultados, preguntar

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SISTEMA = (
    "Eres un experto en cine de ciencia ficción. Responde en español y de forma "
    "concisa. Si no conoces un dato concreto, dilo claramente en lugar de inventarlo."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preguntas al LLM sin RAG.")
    ap.add_argument("pregunta", nargs="?", help="pregunta concreta (opcional)")
    ap.add_argument("--proveedor", choices=PROVEEDORES, default="llm",
                    help="proveedor del LLM (por defecto: llm)")
    args = ap.parse_args()

    if args.pregunta:
        preguntas = [args.pregunta]
    else:
        preguntas = [q["pregunta"] for q in cfg.PREGUNTAS_PRUEBA]

    cliente, modelo = crear_cliente(args.proveedor)
    print(f"Proveedor: {args.proveedor} | modelo: {modelo}")

    filas = []
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n[{i}/{len(preguntas)}] {pregunta}")
        respuesta = preguntar(cliente, modelo, SISTEMA, pregunta)
        print(respuesta)
        filas.append({"pregunta": pregunta, "respuesta": respuesta})

    ruta = guardar_resultados(filas, "sin_rag", args.proveedor)
    print(f"\nResultados guardados en {ruta}")


if __name__ == "__main__":
    main()