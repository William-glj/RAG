# -*- coding: utf-8 -*-
"""
construir_indice.py
===================

Fase B del proyecto RAG: de los datos limpios a una base vectorial.

Lee lo que deja la Fase A en ./datosDescargados/limpio y construye el indice:

    1. Carga tres tipos de material como documentos de LangChain:
         - Criticas, opiniones y textos academicos (limpio/criticas/*.txt)
         - Resenas de usuarios de TMDB            (limpio/tmdb_reviews.csv)
         - Fichas de pelicula, una por fila       (limpio/cruceIMDBTMDB.csv)
    2. Trocea los textos largos (RecursiveCharacterTextSplitter).
    3. Calcula los embeddings (modelo multilingue, ver config_rag.py).
    4. Los guarda en ChromaDB (./indice_chroma).

El proceso es REPETIBLE: cada ejecucion borra el indice anterior y lo
reconstruye desde cero, sin ninguna intervencion manual.

Uso:
    python construir_indice.py                  # construye el indice
    python construir_indice.py --solo-contar    # solo carga y trocea (sin embeddings)
    python construir_indice.py --chunk-size 800 --overlap 120
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

import config_rag as cfg
from rag_comun import EmbeddingsE5, abrir_base, meta_limpia

# Consola con acentos correctos en Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def _leer_csv(nombre: str) -> pd.DataFrame:
    ruta = cfg.CARPETA_LIMPIO / nombre
    if not ruta.exists():
        print(f"  [aviso] no existe {ruta.relative_to(cfg.RAIZ)}")
        return pd.DataFrame()
    return pd.read_csv(ruta, encoding="utf-8", low_memory=False)


def _vacio(valor) -> bool:
    """True si el valor es None, NaN o una cadena en blanco."""
    if valor is None:
        return True
    if isinstance(valor, str):
        return not valor.strip()
    try:
        return bool(pd.isna(valor))
    except (TypeError, ValueError):
        return False


def _texto(valor, defecto: str = "") -> str:
    return defecto if _vacio(valor) else str(valor).strip()


def _a_bool(valor) -> bool | None:
    if _vacio(valor):
        return None
    if isinstance(valor, str):
        return valor.strip().lower() in ("true", "1", "si", "sí")
    return bool(valor)


def _a_entero(valor) -> int | None:
    if _vacio(valor):
        return None
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return None


def _info_peliculas() -> dict[str, dict]:
    """imdb_id -> {anio, relevante_ia}, para enriquecer los metadatos."""
    cruce = _leer_csv("cruceIMDBTMDB.csv")
    info: dict[str, dict] = {}
    if cruce.empty:
        return info
    for _, fila in cruce.iterrows():
        imdb_id = _texto(fila.get("imdb_id"))
        if imdb_id:
            info[imdb_id] = {
                "anio": _a_entero(fila.get("anio")),
                "relevante_ia": _a_bool(fila.get("relevante_ia")),
            }
    return info


# --------------------------------------------------------------------------- #
# Carga de documentos
# --------------------------------------------------------------------------- #

def _separar_cabecera(texto: str) -> tuple[dict[str, str], str]:
    """Separa la cabecera 'Titulo/Fuente/URL/Autor/Fecha' del cuerpo."""
    cabecera, _, cuerpo = texto.partition("\n\n")
    claves = {"título": "titulo", "titulo": "titulo", "fuente": "fuente",
              "url": "url", "autor": "autor", "fecha": "fecha"}
    meta: dict[str, str] = {}
    for linea in cabecera.splitlines():
        m = re.match(r"^([^:]+):\s*(.*)$", linea.strip())
        if m and m.group(1).strip().lower() in claves:
            meta[claves[m.group(1).strip().lower()]] = m.group(2).strip()
    return meta, cuerpo.strip()


def _tipo_de_fuente(fuente: str) -> str:
    f = fuente.lower()
    for clave, tipo in cfg.TIPO_POR_FUENTE.items():
        if clave in f:
            return tipo
    return "critica"


def cargar_criticas(info: dict[str, dict]) -> list[Document]:
    """Criticas, opiniones y texto academico (uno o varios Document por .txt)."""
    print("  Cargando criticas ...")
    carpeta = cfg.CARPETA_CRITICAS_LIMPIO
    if not carpeta.exists():
        print(f"  [aviso] no existe {carpeta.relative_to(cfg.RAIZ)}")
        return []

    manifest = _leer_csv("criticas_manifest.csv")
    imdb_por_archivo: dict[str, str] = {}
    if not manifest.empty and "archivo" in manifest.columns:
        for _, fila in manifest.iterrows():
            if not _vacio(fila.get("imdb_id")):
                imdb_por_archivo[str(fila["archivo"])] = str(fila["imdb_id"])

    docs: list[Document] = []
    for f in sorted(carpeta.glob("*.txt")):
        cab, cuerpo = _separar_cabecera(f.read_text(encoding="utf-8"))
        if not cuerpo:
            continue
        fuente = cab.get("fuente", "")
        tipo = _tipo_de_fuente(fuente)
        imdb_id = imdb_por_archivo.get(f.name)
        base = {
            "tipo": tipo,
            "fuente": fuente,
            "titulo": cab.get("titulo", f.stem),
            "url": cab.get("url"),
            "autor": cab.get("autor"),
            "fecha": cab.get("fecha"),
            "archivo": f.name,
            "imdb_id": imdb_id,
        }
        if imdb_id and imdb_id in info:
            base["anio"] = info[imdb_id]["anio"]
            base["relevante_ia"] = info[imdb_id]["relevante_ia"]

        # Las opiniones de espectadores vienen juntas, separadas por '---':
        # cada espectador es un documento propio para no mezclar opiniones.
        if tipo == "opinion_espectadores":
            bloques = [b.strip() for b in re.split(r"\n\s*---\s*\n", cuerpo)
                       if b.strip()]
            # Se descartan las opiniones sin comentario (solo nota y autor).
            bloques = [b for b in bloques
                       if len(re.sub(r"^Autor:[^\n]*\n?", "", b).strip())
                       >= cfg.MIN_CARACTERES_OPINION]
        else:
            bloques = [cuerpo]
        for n, bloque in enumerate(bloques):
            meta = dict(base)
            meta["doc_id"] = f"{tipo}-{f.stem}-{n}"
            docs.append(Document(page_content=bloque, metadata=meta))
    return docs


def cargar_resenas(info: dict[str, dict]) -> list[Document]:
    """Resenas de usuarios de TMDB (una por Document)."""
    print("  Cargando resenas de TMDB ...")
    df = _leer_csv("tmdb_reviews.csv")
    docs: list[Document] = []
    if df.empty:
        return docs
    for n, (_, fila) in enumerate(df.iterrows()):
        contenido = _texto(fila.get("contenido"))
        if not contenido:
            continue
        imdb_id = _texto(fila.get("imdb_id")) or None
        meta = {
            "tipo": "resena",
            "fuente": "TMDB",
            "titulo": _texto(fila.get("titulo")),
            "imdb_id": imdb_id,
            "tmdb_id": _a_entero(fila.get("tmdb_id")),
            "autor": _texto(fila.get("autor")),
            "fecha": _texto(fila.get("fecha")),
            "rating_autor": fila.get("rating_autor"),
            "doc_id": f"resena-{n}",
        }
        if imdb_id and imdb_id in info:
            meta["anio"] = info[imdb_id]["anio"]
            meta["relevante_ia"] = info[imdb_id]["relevante_ia"]
        docs.append(Document(page_content=contenido, metadata=meta))
    return docs


def _lista_texto(valor) -> str:
    """Generos / keywords: aceptan una lista JSON ('["A", "B"]') o texto 'A,B'.

    Los datos de TMDB vienen como lista JSON y los de IMDb como texto separado
    por comas; aqui se unifican como 'A, B'.
    """
    if _vacio(valor):
        return ""
    texto = str(valor).strip()
    try:
        datos = json.loads(texto)
        if isinstance(datos, list):
            return ", ".join(str(x).strip() for x in datos if str(x).strip())
    except ValueError:
        pass
    return ", ".join(p.strip() for p in texto.split(",") if p.strip())


def cargar_fichas() -> list[Document]:
    """Una ficha de texto por pelicula (a partir de una fila del cruce)."""
    print("  Construyendo fichas de pelicula ...")
    df = _leer_csv("cruceIMDBTMDB.csv")
    docs: list[Document] = []
    if df.empty:
        return docs
    for n, (_, f) in enumerate(df.iterrows()):
        titulo = _texto(f.get("titulo"))
        if not titulo:
            continue
        anio = _a_entero(f.get("anio"))
        # El titulo aparece al principio y junto a cada nota: las preguntas del
        # tipo "¿que nota tiene X?" se parecen asi mas a la ficha de X que a
        # cualquier otra ficha corta.
        partes = [f"Ficha de la película «{titulo}» ({anio})." if anio
                  else f"Ficha de la película «{titulo}»."]
        generos = _lista_texto(f.get("generos"))
        if generos:
            partes.append(f"Géneros: {generos}.")
        if not _vacio(f.get("rating_imdb")):
            votos = _a_entero(f.get("votos_imdb"))
            extra = f" ({votos} votos)" if votos is not None else ""
            partes.append(f"Nota IMDb de {titulo}: "
                          f"{float(f['rating_imdb']):.1f}{extra}.")
        if not _vacio(f.get("rating_tmdb")):
            votos = _a_entero(f.get("votos_tmdb"))
            extra = f" ({votos} votos)" if votos is not None else ""
            partes.append(f"Nota TMDB de {titulo}: "
                          f"{float(f['rating_tmdb']):.1f}{extra}.")
        if not _vacio(f.get("tagline")):
            partes.append(f"Eslogan: {_texto(f.get('tagline'))}")
        if not _vacio(f.get("sinopsis")):
            partes.append(f"Sinopsis: {_texto(f.get('sinopsis'))}")
        kws = _lista_texto(f.get("keywords"))
        if kws:
            partes.append(f"Palabras clave: {kws}.")

        meta = {
            "tipo": "ficha",
            "fuente": "IMDb/TMDB",
            "titulo": titulo,
            "imdb_id": _texto(f.get("imdb_id")) or None,
            "tmdb_id": _a_entero(f.get("tmdb_id")),
            "anio": anio,
            "relevante_ia": _a_bool(f.get("relevante_ia")),
            "doc_id": f"ficha-{n}",
        }
        docs.append(Document(page_content=" ".join(partes), metadata=meta))
    return docs


def cargar_todo() -> list[Document]:
    """Carga los tres tipos de material."""
    print("\n[1/3] Cargando documentos de datosDescargados/limpio ...")
    if not cfg.CARPETA_LIMPIO.exists():
        raise SystemExit(
            f"No existe {cfg.CARPETA_LIMPIO}. Ejecutad antes, en este orden: "
            "cargar_datos.py, scraping_criticas.py y limpiar_datos.py."
        )
    info = _info_peliculas()
    docs = cargar_criticas(info) + cargar_resenas(info) + cargar_fichas()
    cuenta = Counter(d.metadata["tipo"] for d in docs)
    print(f"  {len(docs)} documentos: " +
          ", ".join(f"{k}={v}" for k, v in sorted(cuenta.items())))
    return docs


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def _contexto(meta: dict) -> str:
    """Linea de contexto que se antepone a cada fragmento."""
    titulo = meta.get("titulo", "")
    fuente = meta.get("fuente", "")
    tipo = meta.get("tipo")
    if tipo == "critica":
        return f"[Crítica: {titulo} — {fuente}]\n"
    if tipo == "opinion_espectadores":
        return f"[Opinión de espectadores — {titulo}]\n"
    if tipo == "academico":
        return f"[Texto académico: {titulo}]\n"
    if tipo == "divulgacion":
        return f"[Artículo divulgativo: {titulo} — {fuente}]\n"
    if tipo == "resena":
        return f"[Reseña de usuario de TMDB sobre {titulo}]\n"
    return ""


def trocear(docs: list[Document], chunk_size: int,
            chunk_overlap: int) -> list[Document]:
    """Divide los documentos largos en fragmentos y limpia los metadatos.

    - Las fichas no se trocean (ya son cortas y cada una es una unidad).
    - Un texto que cabe en `chunk_size` se queda como un unico fragmento.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=cfg.SEPARADORES,
        length_function=len,
    )
    fragmentos: list[Document] = []
    for d in docs:
        texto = d.page_content
        if d.metadata.get("tipo") == "ficha" or len(texto) <= chunk_size:
            trozos = [texto]
        else:
            trozos = splitter.split_text(texto)
        for i, trozo in enumerate(trozos):
            meta = dict(d.metadata)
            meta["chunk"] = i
            meta["n_chunks"] = len(trozos)
            meta["chunk_id"] = f"{meta['doc_id']}-{i}"
            contenido = (_contexto(meta) + trozo) if cfg.ANADIR_CONTEXTO else trozo
            fragmentos.append(Document(page_content=contenido,
                                       metadata=meta_limpia(meta)))
    return fragmentos


def estadisticas(fragmentos: list[Document]) -> dict:
    """Resumen del troceado (nº de fragmentos y longitud media por tipo)."""
    por_tipo: dict[str, list[int]] = {}
    for f in fragmentos:
        por_tipo.setdefault(f.metadata["tipo"], []).append(len(f.page_content))
    detalle = {t: {"fragmentos": len(v), "long_media": round(sum(v) / len(v))}
               for t, v in sorted(por_tipo.items())}
    todas = [len(f.page_content) for f in fragmentos]
    return {
        "fragmentos": len(fragmentos),
        "long_media": round(sum(todas) / len(todas)) if todas else 0,
        "long_max": max(todas) if todas else 0,
        "por_tipo": detalle,
    }


# --------------------------------------------------------------------------- #
# Indexacion
# --------------------------------------------------------------------------- #

def _borrar_indice(indice_dir: Path) -> None:
    """Borra un indice anterior (solo si esta dentro de la carpeta del proyecto)."""
    if not indice_dir.exists():
        return
    if cfg.RAIZ.resolve() not in indice_dir.resolve().parents:
        raise SystemExit(f"Por seguridad no se borra {indice_dir}: "
                         "esta fuera de la carpeta del proyecto.")
    shutil.rmtree(indice_dir)


def construir(chunk_size: int = cfg.CHUNK_SIZE,
              chunk_overlap: int = cfg.CHUNK_OVERLAP,
              indice_dir: Path = cfg.INDICE_DIR,
              embeddings=None,
              docs: list[Document] | None = None,
              solo_contar: bool = False) -> dict:
    """Carga, trocea e indexa. Devuelve las estadisticas del troceado."""
    if docs is None:
        docs = cargar_todo()

    print(f"\n[2/3] Troceando (tamaño={chunk_size}, solape={chunk_overlap}) ...")
    fragmentos = trocear(docs, chunk_size, chunk_overlap)
    stats = estadisticas(fragmentos)
    print(f"  {stats['fragmentos']} fragmentos | longitud media "
          f"{stats['long_media']} | máxima {stats['long_max']} caracteres")
    for tipo, d in stats["por_tipo"].items():
        print(f"    - {tipo}: {d['fragmentos']} fragmentos "
              f"(media {d['long_media']} car.)")
    if solo_contar:
        return stats

    print(f"\n[3/3] Indexando en ChromaDB ({indice_dir.name}) ...")
    inicio = time.time()
    _borrar_indice(indice_dir)
    if embeddings is None:
        embeddings = EmbeddingsE5()
    base = abrir_base(indice_dir, embeddings)

    for i in range(0, len(fragmentos), cfg.LOTE_CHROMA):
        lote = fragmentos[i:i + cfg.LOTE_CHROMA]
        base.add_documents(lote, ids=[f.metadata["chunk_id"] for f in lote])
        print(f"  {min(i + cfg.LOTE_CHROMA, len(fragmentos))}/"
              f"{len(fragmentos)} fragmentos indexados")
    stats["segundos"] = round(time.time() - inicio, 1)
    print(f"  Listo en {stats['segundos']} s.")
    return stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Construye el indice vectorial.")
    ap.add_argument("--chunk-size", type=int, default=cfg.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=cfg.CHUNK_OVERLAP)
    ap.add_argument("--indice", type=Path, default=cfg.INDICE_DIR,
                    help="carpeta del indice (por defecto indice_chroma)")
    ap.add_argument("--solo-contar", action="store_true",
                    help="solo carga y trocea, sin calcular embeddings")
    args = ap.parse_args()

    print("=" * 70)
    print("CONSTRUCCIÓN DEL ÍNDICE VECTORIAL (Fase B)")
    print("=" * 70)
    construir(args.chunk_size, args.overlap, args.indice,
              solo_contar=args.solo_contar)
    print("\nListo.")


if __name__ == "__main__":
    main()
