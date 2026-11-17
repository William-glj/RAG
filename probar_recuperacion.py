# -*- coding: utf-8 -*-
"""
probar_recuperacion.py
======================

Comprueba la RECUPERACION del indice sin usar ningun modelo de lenguaje:
dada una pregunta, muestra los fragmentos mas cercanos.

Uso:
    python probar_recuperacion.py "¿Cómo se representa a HAL 9000?"
    python probar_recuperacion.py "nota de Ex Machina" -k 3
    python probar_recuperacion.py "miedo a la IA" --tipo resena
    python probar_recuperacion.py "miedo a la IA" --solo-ia   # filtro de metadatos
    python probar_recuperacion.py --demo                      # preguntas de config_rag.py
    python probar_recuperacion.py                             # modo interactivo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config_rag as cfg
from rag_comun import EmbeddingsE5, abrir_base, construir_filtro

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def mostrar(base, pregunta: str, k: int, filtro: dict | None) -> None:
    print("\n" + "=" * 70)
    print(f"PREGUNTA: {pregunta}")
    if filtro:
        print(f"FILTRO:   {filtro}")
    print("=" * 70)
    resultados = base.similarity_search_with_score(pregunta, k=k, filter=filtro)
    if not resultados:
        print("  (sin resultados)")
        return
    for pos, (doc, distancia) in enumerate(resultados, 1):
        m = doc.metadata
        print(f"\n#{pos}  distancia={distancia:.3f}  tipo={m.get('tipo')}  "
              f"titulo={m.get('titulo', '')[:60]}  fuente={m.get('fuente', '')}")
        texto = doc.page_content.replace("\n", " ")
        print(f"    {texto[:300]}{'...' if len(texto) > 300 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prueba la recuperacion del indice.")
    ap.add_argument("pregunta", nargs="?", help="pregunta a buscar")
    ap.add_argument("-k", type=int, default=cfg.K_POR_DEFECTO,
                    help="nº de fragmentos a recuperar")
    ap.add_argument("--tipo", help="filtrar por tipo: critica, opinion_espectadores, "
                    "academico, divulgacion, resena, ficha")
    ap.add_argument("--solo-ia", action="store_true",
                    help="solo peliculas marcadas como relevantes a IA")
    ap.add_argument("--demo", action="store_true",
                    help="lanza las preguntas de prueba de config_rag.py")
    ap.add_argument("--indice", type=Path, default=cfg.INDICE_DIR)
    args = ap.parse_args()

    if not args.indice.exists():
        raise SystemExit(f"No existe el índice en {args.indice}. "
                         "Ejecutad antes: python construir_indice.py")

    base = abrir_base(args.indice, EmbeddingsE5())
    filtro = construir_filtro(args.tipo, args.solo_ia)

    if args.demo:
        for q in cfg.PREGUNTAS_PRUEBA:
            mostrar(base, q["pregunta"], args.k, filtro)
    elif args.pregunta:
        mostrar(base, args.pregunta, args.k, filtro)
    else:
        print("Modo interactivo (línea vacía para salir).")
        while True:
            pregunta = input("\nPregunta> ").strip()
            if not pregunta:
                break
            mostrar(base, pregunta, args.k, filtro)


if __name__ == "__main__":
    main()
