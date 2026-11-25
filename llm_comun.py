# -*- coding: utf-8 -*-
"""
llm_comun.py
============

Cliente del LLM compartido por sin_rag.py, con_rag.py y app.py.

Proveedores:
    - "llm"        : LLM_API_KEY, LLM_BASE_URL y LLM_MODEL (en .env).
    - "openrouter" : OPENROUTER_API_KEY y OPENROUTER_MODEL (en .env).
                     La URL base de OpenRouter es fija.

Ante un error 429 (demasiadas peticiones) reintenta con espera creciente y, si
el proveedor sigue saturado, devuelve MENSAJE_SATURADO en lugar de fallar.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import (OpenAI, APIConnectionError, APITimeoutError,
                    InternalServerError, RateLimitError)

import config_rag as cfg

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(cfg.RAIZ / ".env")

PROVEEDORES = ("llm", "openrouter")
NOMBRES_PROVEEDOR = {
    "llm": "Proveedor configurado (LLM_*)",
    "openrouter": "OpenRouter",
}
VARIABLES = {
    "llm": ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"),
}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CARPETA_RESULTADOS = cfg.RAIZ / "resultados_llm"
MAX_TOKENS = 6000

MENSAJE_SATURADO = (
    "El proveedor del modelo no está disponible en este momento (error "
    "transitorio 429/5xx). Inténtalo de nuevo en unos minutos."
)

TIMEOUT_PETICION = 120   # segundos máximos por petición
REINTENTOS_429 = 3     # reintentos ante errores transitorios (429/5xx) antes de rendirse
ESPERA_429 = 5         # segundos de espera inicial; se duplica en cada reintento


def faltan_variables(proveedor: str) -> list[str]:
    """Variables de entorno que faltan para el proveedor indicado."""
    return [v for v in VARIABLES[proveedor] if not os.getenv(v)]



def crear_cliente(proveedor: str = "llm") -> tuple[OpenAI, str]:
    """Devuelve (cliente, nombre_del_modelo) para el proveedor indicado."""
    faltan = faltan_variables(proveedor)
    if faltan:
        raise SystemExit(f"Faltan en .env: {', '.join(faltan)}")

    if proveedor == "openrouter":
        cliente = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                         base_url=OPENROUTER_BASE_URL,
                         timeout=TIMEOUT_PETICION, max_retries=0)
        return cliente, os.environ["OPENROUTER_MODEL"]

    cliente = OpenAI(api_key=os.environ["LLM_API_KEY"],
                     base_url=os.environ["LLM_BASE_URL"],
                     timeout=TIMEOUT_PETICION, max_retries=0)
    return cliente, os.environ["LLM_MODEL"]


def preguntar(cliente: OpenAI, modelo: str, sistema: str, usuario: str) -> str:
    """Envía una petición al modelo. Ante errores transitorios (429, 5xx o de
    conexión) reintenta con espera creciente y, si el proveedor sigue caído,
    devuelve MENSAJE_SATURADO."""
    for intento in range(REINTENTOS_429 + 1):
        try:
            respuesta = cliente.chat.completions.create(
                model=modelo,
                messages=[{"role": "system", "content": sistema},
                          {"role": "user", "content": usuario}],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
            return (respuesta.choices[0].message.content or "").strip()
        except (RateLimitError, InternalServerError,
                APIConnectionError, APITimeoutError) as exc:
            if intento < REINTENTOS_429:
                espera = ESPERA_429 * 2 ** intento
                print(f"  [{getattr(exc, 'status_code', type(exc).__name__)}] "
                      f"error transitorio; reintento en {espera} s ...")
                time.sleep(espera)
    return MENSAJE_SATURADO


def guardar_resultados(filas: list[dict], nombre_base: str,
                       proveedor: str = "llm") -> Path:
    """Guarda en CSV. OpenRouter usa un sufijo para no sobrescribir el otro proveedor."""
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    sufijo = "_openrouter" if proveedor == "openrouter" else ""
    ruta = CARPETA_RESULTADOS / f"{nombre_base}{sufijo}.csv"
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8")
    return ruta