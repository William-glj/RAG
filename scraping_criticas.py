# -*- coding: utf-8 -*-
"""
scraping_criticas.py
====================

Scraping de críticas / reseñas de películas para el corpus del RAG.

Tema: el miedo al avance tecnológico y las revoluciones tecnológicas en el
cine de ciencia ficción (subtema: la inteligencia artificial).

Lee la lista de enlaces (definida aquí abajo en TARGETS) y, de cada página,
extrae ÚNICAMENTE el texto de la crítica/reseña. Guarda cada texto en un
fichero .txt dentro de ./datosDescargados/criticas/, con un nombre derivado
del título de la película o del artículo.

Cada fichero lleva una cabecera mínima (título, fuente, URL, autor y fecha
cuando están disponibles) seguida del texto, para poder rastrear la
procedencia de cada fragmento durante la indexación (provenance).

Notas:
- rogerebert.com bloquea el scraping (Cloudflare, HTTP 403). Se usa la copia
  archivada en Wayback Machine (web.archive.org) para obtener la crítica.
- El enlace "share.google/6Rqwgfbprt0uiEJ4L" es inválido (dominio inexistente:
  redirige a un PDF académico sin relación). Se omite y se avisa al final.
- El PDF del repositorio UACM se descarga en streaming (es un servidor lento)
  y su texto se extrae con pypdf.

Uso:
    python scraping_criticas.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Consola con acentos correctos en Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parent
CARPETA_CRITICAS = RAIZ / "datosDescargados" / "criticas"
CARPETA_RAW = RAIZ / "datosDescargados" / "raw"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

# Copia archivada de la crítica de Roger Ebert (el sitio directo devuelve 403).
WAYBACK_ROGEREBERT = (
    "https://web.archive.org/web/2024/"
    "https://www.rogerebert.com/reviews/ex-machina-2015"
)


# --------------------------------------------------------------------------- #
# Objetivos de scraping
# --------------------------------------------------------------------------- #
# Cada entrada: (nombre_archivo, fuente, url, tipo_de_extractor)
#   nombre_archivo -> base del nombre del fichero .txt (deriva del título).
#   fuente         -> nombre legible de la procedencia.
#   url            -> enlace a scrapear.
#   extractor      -> clave que identifica la función de extracción.

TARGETS = [
    (
        "2001 - Una odisea del espacio (Espinof)",
        "Espinof",
        "https://www.espinof.com/criticas/gran-clasico-ciencia-ficcion-tambien-cumbre-estilo-inhumano-2001-odisea-espacio-marco-antes-despues-obra-stanley-kubrick",
        "espinof",
    ),
    (
        "2001 - Una odisea del espacio (SensaCine espectadores)",
        "SensaCine",
        "https://www.sensacine.com/peliculas/pelicula-27442/criticas-espectadores/",
        "sensacine",
    ),
    (
        "2001 - Una odisea del espacio (tesis UACM)",
        "Repositorio UACM",
        "https://www.repositorioinstitucionaluacm.mx/jspui/bitstream/123456789/1186/3/Aura%20Mar%C3%ADa%20Barajas%20Gonz%C3%A1lez-CyC%20PDF.pdf",
        "pdf",
    ),
    (
        "El miedo a la tecnologia desde la escritura al 5G (La Vanguardia)",
        "La Vanguardia",
        "https://www.lavanguardia.com/historiayvida/historia-antigua/20201103/790783/miedo-tecnologia-escritura-5g.html",
        "lavanguardia",
    ),
    (
        "Miedo a la tecnologia - peliculas (Revista Central)",
        "Revista Central",
        "https://www.revistacentral.com.mx/en-corto/miedo-tecnologia-peliculas/",
        "revistacentral",
    ),
    (
        "Ex Machina (Roger Ebert)",
        "Roger Ebert (vía Wayback Machine)",
        WAYBACK_ROGEREBERT,
        "rogerebert",
    ),
    (
        "Upgrade (El Cuervo Enteradillo)",
        "El Cuervo Enteradillo",
        "https://elcuervoenteradillo.blogspot.com/2018/09/critica-upgrade.html",
        "blogspot",
    ),
]

# Enlaces que no se pudieron procesar (se informa al final).
ENLACES_PENDIENTES = [
    "https://share.google/6Rqwgfbprt0uiEJ4L  (enlace inválido: el dominio "
    "'share.google' no existe y redirige a un PDF académico sin relación; "
    "revisa si era un enlace de Google Drive y pásame la URL correcta)",
]


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def sanitizar(nombre: str) -> str:
    """Convierte un título en un nombre de fichero válido en Windows."""
    nombre = re.sub(r'[<>:"/\\|?*]', " - ", nombre)
    nombre = re.sub(r"\s+", " ", nombre).strip().strip(".")
    return nombre


def _get(url: str, timeout: int = 60, reintentos: int = 5) -> requests.Response:
    """GET con reintentos ante 429/5xx (rate-limit, p. ej. Wayback Machine)."""
    for intento in range(reintentos):
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504) and intento < reintentos - 1:
            espera = 3 * (intento + 1)
            print(f"      [{r.status_code}] reintentando en {espera}s ...")
            time.sleep(espera)
            continue
        r.raise_for_status()
    raise RuntimeError(f"No se pudo descargar {url}")


def _soup(url: str) -> BeautifulSoup:
    return BeautifulSoup(_get(url).text, "lxml")


def _texto_parrafos(contenedor) -> str:
    """Extrae el texto de los párrafos de un contenedor HTML."""
    parrafos = contenedor.find_all("p")
    if parrafos:
        return "\n\n".join(p.get_text(" ", strip=True) for p in parrafos)
    return contenedor.get_text(" ", strip=True)


def guardar(nombre_archivo: str, titulo: str, fuente: str, url: str,
            cuerpo: str, autor: str | None = None,
            fecha: str | None = None) -> Path:
    """Escribe la crítica en un .txt con cabecera de procedencia."""
    CARPETA_CRITICAS.mkdir(parents=True, exist_ok=True)
    destino = CARPETA_CRITICAS / f"{sanitizar(nombre_archivo)}.txt"

    lineas = [f"Título: {titulo}", f"Fuente: {fuente}", f"URL: {url}"]
    if autor:
        lineas.append(f"Autor: {autor}")
    if fecha:
        lineas.append(f"Fecha: {fecha}")
    cabecera = "\n".join(lineas)

    destino.write_text(cabecera + "\n\n" + cuerpo.strip() + "\n",
                       encoding="utf-8")
    return destino


# --------------------------------------------------------------------------- #
# Extractores por sitio
# --------------------------------------------------------------------------- #

def extraer_espinof(url: str):
    s = _soup(url)
    h1 = s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Sin título"
    cont = s.select_one("div.article-content")
    cuerpo = _texto_parrafos(cont) if cont else ""
    autor = None
    fecha = None
    autor_el = s.select_one("meta[name=author]") or s.select_one("a[rel=author]")
    if autor_el:
        autor = autor_el.get("content") or autor_el.get_text(strip=True)
    return titulo, cuerpo, autor, fecha


def extraer_sensacine(url: str):
    s = _soup(url)
    h1 = s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Sin título"

    bloques = []
    for card in s.select("div.review-card"):
        autor_el = card.select_one("div.meta-title")
        contenido_el = card.select_one("div.content-txt")
        if not contenido_el:
            continue
        autor = autor_el.get_text(" ", strip=True) if autor_el else "Anónimo"
        contenido = contenido_el.get_text(" ", strip=True)

        # La nota (ej. "5,0") y la fecha suelen estar en review-card-meta.
        meta_el = card.select_one("div.review-card-meta")
        nota = fecha = None
        if meta_el:
            meta = meta_el.get_text(" ", strip=True)
            m_nota = re.match(r"^([0-9]+,[0-9]|[0-9]+)", meta)
            if m_nota:
                nota = m_nota.group(1)
            m_fecha = re.search(r"Publicada el (.+)$", meta)
            if m_fecha:
                fecha = m_fecha.group(1)

        encabezado = f"Autor: {autor}"
        if nota:
            encabezado += f" | Nota: {nota}"
        if fecha:
            encabezado += f" | Fecha: {fecha}"
        bloques.append(f"{encabezado}\n{contenido}")

    cuerpo = "\n\n---\n\n".join(bloques)
    return titulo, cuerpo, None, None


def extraer_lavanguardia(url: str):
    s = _soup(url)
    h1 = s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Sin título"
    cont = s.select_one("article")
    cuerpo = _texto_parrafos(cont) if cont else ""
    return titulo, cuerpo, None, None


def extraer_revistacentral(url: str):
    s = _soup(url)
    h1 = s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Sin título"
    cont = s.select_one("article.b-article-body") or s.select_one("article")
    cuerpo = _texto_parrafos(cont) if cont else ""
    return titulo, cuerpo, None, None


def extraer_rogerebert(url: str):
    # url ya apunta a la Wayback Machine.
    s = _soup(url)
    h1 = s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Ex Machina"
    cont = s.select_one("article.review div.entry-content")
    cuerpo = _texto_parrafos(cont) if cont else ""

    autor = None
    autor_el = s.select_one("a[rel=author]") or s.select_one("div.author")
    if autor_el:
        autor = autor_el.get_text(" ", strip=True)
    return titulo, cuerpo, autor, None


def extraer_blogspot(url: str):
    s = _soup(url)
    h3 = s.select_one("h3.post-title") or s.select_one("h1.post-title")
    titulo = h3.get_text(" ", strip=True) if h3 else "Sin título"
    cont = s.select_one("div.post-body")
    cuerpo = _texto_parrafos(cont) if cont else ""
    return titulo, cuerpo, None, None


def extraer_pdf(url: str, nombre_archivo: str):
    """Descarga el PDF en streaming y extrae todo su texto con pypdf."""
    destino = CARPETA_RAW / f"{sanitizar(nombre_archivo)}.pdf"
    CARPETA_RAW.mkdir(parents=True, exist_ok=True)

    if not destino.exists():
        print(f"    Descargando PDF (servidor lento, ten paciencia) ...")
        with requests.get(url, headers=HEADERS, stream=True,
                          timeout=(30, 300)) as r:
            r.raise_for_status()
            with open(destino, "wb") as f:
                for bloque in r.iter_content(chunk_size=1024 * 1024):
                    f.write(bloque)

    lector = PdfReader(str(destino))
    paginas = [p.extract_text() or "" for p in lector.pages]
    cuerpo = "\n\n".join(paginas).strip()

    # Intentar recuperar un título razonable de los metadatos o de la 1ª página.
    titulo = None
    meta = lector.metadata or {}
    if meta.get("title"):
        titulo = str(meta["title"]).strip()
    if not titulo and paginas:
        primera = paginas[0].strip().splitlines()
        primera = [l.strip() for l in primera if l.strip()]
        titulo = primera[0] if primera else nombre_archivo
    return titulo or nombre_archivo, cuerpo, None, None


EXTRACTORES = {
    "espinof": extraer_espinof,
    "sensacine": extraer_sensacine,
    "lavanguardia": extraer_lavanguardia,
    "revistacentral": extraer_revistacentral,
    "rogerebert": extraer_rogerebert,
    "blogspot": extraer_blogspot,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 70)
    print("SCRAPING DE CRÍTICAS Y RESEÑAS (corpus RAG)")
    print("=" * 70)

    guardados: list[Path] = []
    fallos: list[tuple[str, str]] = []

    for nombre_archivo, fuente, url, tipo in TARGETS:
        print(f"\n- {nombre_archivo}")
        print(f"  {url[:80]}{'...' if len(url) > 80 else ''}")
        try:
            if tipo == "pdf":
                titulo, cuerpo, autor, fecha = extraer_pdf(url, nombre_archivo)
            else:
                titulo, cuerpo, autor, fecha = EXTRACTORES[tipo](url)

            if not cuerpo.strip():
                raise RuntimeError("no se encontró texto de crítica en la página")

            destino = guardar(nombre_archivo, titulo, fuente, url,
                              cuerpo, autor, fecha)
            guardados.append(destino)
            print(f"  OK -> {destino.relative_to(RAIZ)} "
                  f"({len(cuerpo)} caracteres)")
        except Exception as exc:
            fallos.append((nombre_archivo, f"{type(exc).__name__}: {exc}"))
            print(f"  FALLO -> {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print(f"Guardadas {len(guardados)} críticas en "
          f"'{CARPETA_CRITICAS.relative_to(RAIZ)}':")
    for g in guardados:
        print(f"  - {g.name}")

    if fallos:
        print(f"\nNo se pudieron procesar {len(fallos)} enlaces:")
        for nombre, error in fallos:
            print(f"  - {nombre}: {error}")

    if ENLACES_PENDIENTES:
        print("\nEnlaces pendientes de revisión:")
        for e in ENLACES_PENDIENTES:
            print(f"  - {e}")

    print("\nListo.")


if __name__ == "__main__":
    main()
