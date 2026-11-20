# -*- coding: utf-8 -*-
"""
llm_comun.py
============

Cliente del modelo Llama compartido por sin_rag.py y con_rag.py.
Usa una API compatible con OpenAI. Configuración en .env:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import config_rag as cfg

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(cfg.RAIZ / ".env")

CARPETA_RESULTADOS = cfg.RAIZ / "resultados_llm"
MAX_TOKENS = 600


def crear_cliente() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    modelo = os.getenv("LLM_MODEL")
    if not (api_key and base_url and modelo):
        raise SystemExit("Faltan LLM_API_KEY, LLM_BASE_URL o LLM_MODEL en .env")
    return OpenAI(api_key=api_key, base_url=base_url)


def preguntar(cliente: OpenAI, sistema: str, usuario: str) -> str:
    respuesta = cliente.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[{"role": "system", "content": sistema},
                  {"role": "user", "content": usuario}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    return (respuesta.choices[0].message.content or "").strip()


def guardar_resultados(filas: list[dict], nombre: str) -> Path:
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_RESULTADOS / nombre
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8")
    return ruta