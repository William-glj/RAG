# -*- coding: utf-8 -*-
"""
rag_comun.py
============

Piezas compartidas por los scripts de la Fase B (y reutilizables en C y D):

  - EmbeddingsE5: envoltorio de sentence-transformers compatible con LangChain
    que anade los prefijos "passage: " / "query: " de los modelos E5.
  - abrir_base(): abre (o crea) la coleccion de ChromaDB.
  - construir_filtro(): filtros de metadatos para la busqueda.
  - meta_limpia(): deja los metadatos en tipos que Chroma acepta.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

import config_rag as cfg


class EmbeddingsE5(Embeddings):
    """Embeddings multilingues con sentence-transformers (modelo E5)."""

    def __init__(self, modelo: str = cfg.EMBEDDING_MODEL,
                 prefijo_documento: str = cfg.PREFIJO_DOCUMENTO,
                 prefijo_consulta: str = cfg.PREFIJO_CONSULTA) -> None:
        # Import diferido: cargar sentence-transformers tarda unos segundos
        # y solo hace falta cuando de verdad se calculan embeddings.
        from sentence_transformers import SentenceTransformer

        print(f"  Cargando modelo de embeddings '{modelo}' "
              f"(la primera vez se descarga) ...")
        self.modelo = SentenceTransformer(modelo)
        self.prefijo_documento = prefijo_documento
        self.prefijo_consulta = prefijo_consulta

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        textos = [self.prefijo_documento + t for t in texts]
        vectores = self.modelo.encode(
            textos,
            batch_size=cfg.BATCH_EMBEDDINGS,
            normalize_embeddings=True,
            show_progress_bar=len(textos) > 200,
        )
        return vectores.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.modelo.encode(
            self.prefijo_consulta + text, normalize_embeddings=True
        )
        return vector.tolist()


def abrir_base(indice_dir: Path, embeddings: Embeddings,
               coleccion: str = cfg.COLECCION):
    """Abre (o crea) la coleccion de ChromaDB en `indice_dir`."""
    from langchain_chroma import Chroma

    # Los embeddings estan normalizados, asi que la distancia coseno y la L2
    # ordenan los resultados igual; se pide coseno para que la "distancia"
    # que se muestra sea facil de interpretar (0 = identico).
    return Chroma(
        collection_name=coleccion,
        persist_directory=str(indice_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def construir_filtro(tipo: str | None = None,
                     solo_relevantes_ia: bool = False) -> dict | None:
    """Filtro de metadatos para Chroma (None = sin filtro)."""
    condiciones: list[dict] = []
    if tipo:
        condiciones.append({"tipo": tipo})
    if solo_relevantes_ia:
        condiciones.append({"relevante_ia": True})
    if not condiciones:
        return None
    if len(condiciones) == 1:
        return condiciones[0]
    return {"$and": condiciones}


def meta_limpia(meta: dict[str, Any]) -> dict[str, Any]:
    """Quita nulos/NaN y convierte tipos de numpy a tipos nativos.

    Chroma solo admite str, int, float y bool en los metadatos.
    """
    limpia: dict[str, Any] = {}
    for clave, valor in meta.items():
        if valor is None:
            continue
        if hasattr(valor, "item"):            # numpy -> Python
            try:
                valor = valor.item()
            except Exception:
                pass
        if isinstance(valor, float) and math.isnan(valor):
            continue
        if isinstance(valor, str) and not valor.strip():
            continue
        if not isinstance(valor, (str, int, float, bool)):
            valor = str(valor)
        limpia[clave] = valor
    return limpia


def contiene_palabras(texto: str, palabras: list[str]) -> bool:
    """True si el texto contiene TODAS las palabras clave.

    - Cada palabra se busca COMPLETA y sin distinguir mayusculas: asi "HAL" no
      coincide con "hallazgo" ni "STEM" con "sistemas".
    - Si la palabra termina en "*" se busca como prefijo: "cibern*" encuentra
      "cibernetica" y "cibernetico".
    """
    for p in palabras:
        if p.endswith("*"):
            patron = r"(?<!\w)" + re.escape(p[:-1]) + r"\w*"
        else:
            patron = r"(?<!\w)" + re.escape(p) + r"(?!\w)"
        if not re.search(patron, texto, flags=re.IGNORECASE):
            return False
    return True
