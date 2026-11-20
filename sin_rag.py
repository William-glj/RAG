# -*- coding: utf-8 -*-
"""
sin_rag.py
==========

Script 1: envía preguntas al modelo Llama SIN contexto externo.
Es la línea base para comparar con con_rag.py.

Uso:
    python sin_rag.py                              # preguntas de config_rag.PREGUNTAS_PRUEBA
    python sin_rag.py "¿Qué nota tiene Upgrade?"   # una pregunta concreta
"""

from __future__ import annotations

import argparse
import sys

import config_rag as cfg
from llm_comun import crear_cliente, guardar_resultados, preguntar

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
    args = ap.parse_args()

    if args.pregunta:
        preguntas = [args.pregunta]
    else:
        preguntas = [q["pregunta"] for q in cfg.PREGUNTAS_PRUEBA]

    cliente = crear_cliente()
    filas = []
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n[{i}/{len(preguntas)}] {pregunta}")
        respuesta = preguntar(cliente, SISTEMA, pregunta)
        print(respuesta)
        filas.append({"pregunta": pregunta, "respuesta": respuesta})

    ruta = guardar_resultados(filas, "sin_rag.csv")
    print(f"\nResultados guardados en {ruta}")


if __name__ == "__main__":
    main()