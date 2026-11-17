# -*- coding: utf-8 -*-
"""
comparar_chunking.py
====================

Experimento que JUSTIFICA con datos el tamaño de fragmento elegido.

Para cada tamaño (por defecto 500, 1000 y 1500 caracteres, con un solape del
15 %), construye un indice independiente y lanza las MISMAS preguntas de prueba
(config_rag.PREGUNTAS_PRUEBA). Mide:

    - acierto@k : fraccion de preguntas en las que, entre los k fragmentos
                  recuperados, hay alguno que contiene todas las palabras
                  clave esperadas.
    - MRR       : media de 1/posicion del primer acierto (0 si no hay).
                  Cuanto mas alto, antes aparece el fragmento correcto.
    - nº de fragmentos y longitud media.

Salidas (en ./experimentos_chunking):
    resultados_chunking.csv          resumen por tamaño
    resultados_chunking_detalle.csv  resultado de cada pregunta y tamaño
    resultados_chunking.md           tabla lista para pegar en el Quarto

AVISO: la metrica es orientativa. Que un fragmento contenga las palabras clave
no garantiza que responda bien; revisad ademas a mano los casos dudosos con
probar_recuperacion.py.

Uso:
    python comparar_chunking.py
    python comparar_chunking.py --tamanos 400 800 1200 -k 4
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import config_rag as cfg
from construir_indice import cargar_todo, construir
from rag_comun import EmbeddingsE5, abrir_base, contiene_palabras

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluar(base, preguntas: list[dict], k: int) -> list[dict]:
    """Devuelve, por pregunta, si hay acierto y en que posicion."""
    filas = []
    for q in preguntas:
        resultados = base.similarity_search(q["pregunta"], k=k)
        posicion = 0
        for rango, doc in enumerate(resultados, 1):
            if contiene_palabras(doc.page_content, q["palabras_clave"]):
                posicion = rango
                break
        filas.append({"pregunta": q["pregunta"], "acierto": posicion > 0,
                      "posicion": posicion})
    return filas


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara tamaños de fragmento.")
    ap.add_argument("--tamanos", type=int, nargs="+", default=[500, 1000, 1500])
    ap.add_argument("--solape", type=float, default=0.15,
                    help="solape como fraccion del tamaño (0.15 = 15 %%)")
    ap.add_argument("-k", type=int, default=cfg.K_POR_DEFECTO)
    args = ap.parse_args()

    print("=" * 70)
    print("COMPARATIVA DE TAMAÑOS DE FRAGMENTO (chunking)")
    print("=" * 70)

    docs = cargar_todo()                 # se cargan una sola vez
    embeddings = EmbeddingsE5()          # el modelo se carga una sola vez
    cfg.EXPERIMENTOS_DIR.mkdir(parents=True, exist_ok=True)

    resumen, detalle = [], []
    for tam in args.tamanos:
        solape = int(tam * args.solape)
        print(f"\n##### Tamaño {tam} (solape {solape}) #####")
        carpeta = cfg.EXPERIMENTOS_DIR / f"chunk_{tam}"
        stats = construir(tam, solape, carpeta, embeddings, docs)
        base = abrir_base(carpeta, embeddings)
        filas = evaluar(base, cfg.PREGUNTAS_PRUEBA, args.k)

        aciertos = sum(f["acierto"] for f in filas)
        mrr = sum(1 / f["posicion"] for f in filas if f["posicion"]) / len(filas)
        resumen.append({
            "tamano": tam, "solape": solape,
            "fragmentos": stats["fragmentos"],
            "long_media": stats["long_media"],
            f"acierto@{args.k}": round(aciertos / len(filas), 3),
            "MRR": round(mrr, 3),
        })
        for f in filas:
            detalle.append({"tamano": tam, **f})
        print(f"  acierto@{args.k} = {aciertos}/{len(filas)} | MRR = {mrr:.3f}")
        for f in filas:
            if not f["acierto"]:
                print(f"    [sin acierto] {f['pregunta']}")

    df = pd.DataFrame(resumen)
    df.to_csv(cfg.EXPERIMENTOS_DIR / "resultados_chunking.csv", index=False,
              encoding="utf-8")
    pd.DataFrame(detalle).to_csv(
        cfg.EXPERIMENTOS_DIR / "resultados_chunking_detalle.csv", index=False,
        encoding="utf-8")

    # Tabla markdown para el Quarto.
    cols = list(df.columns)
    lineas = ["| " + " | ".join(cols) + " |",
              "|" + "|".join("---" for _ in cols) + "|"]
    for fila in df.to_dict("records"):      # to_dict conserva los enteros
        lineas.append("| " + " | ".join(str(fila[c]) for c in cols) + " |")
    md = "\n".join(lineas) + "\n"
    (cfg.EXPERIMENTOS_DIR / "resultados_chunking.md").write_text(md, encoding="utf-8")

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(md)
    print(f"Resultados guardados en {cfg.EXPERIMENTOS_DIR.relative_to(cfg.RAIZ)}")
    print("\nListo.")


if __name__ == "__main__":
    main()
