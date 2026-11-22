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
- Además de las críticas, se procesan los análisis de CinemaNet (Blade Runner
  y Blade Runner 2049), The Conversation y el TFG del repositorio Docta (UCM).
  Se descartan los enlaces fuera de tema (tutorial de sentiment analysis,
  artículo de negocios) y los listicles poco densos (ver ENLACES_PENDIENTES).
- Los vídeos de YouTube se transcriben descargando sus subtítulos (manuales o
  automáticos) en español con yt-dlp y convirtiéndolos a texto plano.

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

try:
    import yt_dlp
    TIENE_YTDLP = True
except ImportError:  # pragma: no cover
    yt_dlp = None
    TIENE_YTDLP = False

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

# Copia archivada de Cinéfilos Frustrados (el sitio directo cierra la conexión).
WAYBACK_CINEFILOS = (
    "https://web.archive.org/web/20240713132824/"
    "https://cinefilosfrustrados.com/critica-ex-machina-2014/"
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
    # --- Análisis de fondo (los mejores, todos sobre el miedo a la tecnología) --
    (
        "Inteligencia artificial y cine - dilemas eticos (TFG UCM)",
        "Repositorio Docta UCM",
        "https://docta.ucm.es/rest/api/core/bitstreams/c49d45b9-5f05-481e-bc55-2bc17c23bf16/content",
        "pdf",
    ),
    (
        "El futuro de la robotica en las distopias cinematograficas (The Conversation)",
        "The Conversation",
        "https://theconversation.com/el-futuro-de-la-robotica-en-las-distopias-cinematograficas-129052",
        "theconversation",
    ),
    (
        "Blade Runner (1982) - la apoteosis de lo humano (CinemaNet)",
        "CinemaNet",
        "https://www.cinemanet.info/2017/11/analisis-blade-runner-la-apoteosis-de-lo-humano/",
        "cinemanet",
    ),
    (
        "Blade Runner 2049 - la apoteosis de lo humano (CinemaNet)",
        "CinemaNet",
        "https://www.cinemanet.info/2018/01/analisis-blade-runner-2049-la-apoteosis-de-lo-humano-2a-parte/",
        "cinemanet",
    ),
    # --- Enlaces nuevos (segunda tanda, de enlacesHTML.txt) ------------------
    (
        "2001 - Una odisea del espacio (Cine y Psicología)",
        "Cine y Psicología",
        "https://www.cineypsicologia.com/2012/10/2001-odisea-en-el-espacio-mis.html",
        "blogspot",
    ),
    (
        "Ex Machina (El Antepenúltimo Mohicano)",
        "El Antepenúltimo Mohicano",
        "https://www.elantepenultimomohicano.com/2015/02/critica-ex-machina.html",
        "blogspot",
    ),
    (
        "Ex Machina (Cinéfilos Frustrados)",
        "Cinéfilos Frustrados (vía Wayback Machine)",
        WAYBACK_CINEFILOS,
        "wordpress",
    ),
    (
        "Her (El Antepenúltimo Mohicano)",
        "El Antepenúltimo Mohicano",
        "https://www.elantepenultimomohicano.com/2014/02/her-critica.html",
        "blogspot",
    ),
    (
        "Her - análisis académico (UOC open access)",
        "Repositorio UOC",
        "https://openaccess.uoc.edu/server/api/core/bitstreams/1ae95eed-60e8-4087-85fb-fd03b56ebace/content",
        "pdf",
    ),
    (
        "Upgrade (El Contraplano)",
        "El Contraplano",
        "https://elcontraplano.com/2018/08/22/upgrade-vivir-para-la-tecnologia/",
        "wordpress",
    ),
    (
        "Upgrade - implantes de inteligencia artificial (Cinemas Comics)",
        "Cinemas Comics",
        "https://www.cinemascomics.com/pelicula-de-ciencia-ficcion-sobre-implantes-de-inteligencia-artificial-que-mejoran-a-los-humanos/",
        "wordpress",
    ),
    (
        "Blade Runner - análisis filosófico (Proyecto Scio UCV)",
        "Proyecto Scio UCV",
        "https://proyectoscio.ucv.es/filosofia-y-cine/blade-runner-un-analisis-filosofico/",
        "wordpress",
    ),
]

# Metadatos conocidos de PDFs académicos cuyos metadatos embebidos no son
# fiables (p. ej. el autor del PDF de Word no es el autor del trabajo).
META_PDF = {
    "Inteligencia artificial y cine - dilemas eticos (TFG UCM)": {
        "titulo": "Inteligencia artificial y cine: dilemas éticos a través de "
                  "Metrópolis, 2001: Odisea en el espacio y Blade Runner",
        "autor": "Laura Meseguer Sánchez",
        "fecha": "2025-05-26",
    },
}

# Vídeos de YouTube a transcribir. De cada uno se descargan los subtítulos
# (manuales o, si no los hay, automáticos) en español y se convierten a texto.
# Formato: (nombre_archivo, url).
VIDEOS = [
    (
        "¿Por qué le tenemos miedo a la tecnología - hola internet ep 8 (YouTube)",
        "https://www.youtube.com/watch?v=_SF5MY9OoEQ",
    ),
    (
        "Upgrade - vídeo sobre implantes de inteligencia artificial (YouTube)",
        "https://www.youtube.com/watch?v=9TfNhfKNZmU",
    ),
    (
        "Vídeo extra 2 - miedo a la tecnología (YouTube)",
        "https://www.youtube.com/watch?v=sTRZc-HqhoQ",
    ),
    (
        "Vídeo extra 3 - miedo a la tecnología (YouTube)",
        "https://www.youtube.com/watch?v=CJoYTIud4ns",
    ),
]

# Enlaces descartados por estar fuera del tema o rotos (se documentan para la
# memoria, pero no se scrapean):
#   - share.google/6Rqwgfbprt0uiEJ4L  -> dominio inválido (enlace roto).
#   - Medium "Sentiment Analysis on IMDB Reviews" -> tutorial de ML, fuera de tema.
#   - ADEN "2001 y el mundo de los negocios" -> enfoque empresarial, fuera de tema.
#   - Revista Central "Miedo a la tecnología: películas" -> listicle de ~2K chars, poco denso.
#   - YouTube "Top 10 Películas sobre Tecnología" y "10 Películas de Ciencia Ficción..." -> listicles.
ENLACES_PENDIENTES: list[str] = []


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
            fecha: str | None = None,
            extra: dict[str, str] | None = None) -> Path:
    """Escribe la crítica en un .txt con cabecera de procedencia."""
    CARPETA_CRITICAS.mkdir(parents=True, exist_ok=True)
    destino = CARPETA_CRITICAS / f"{sanitizar(nombre_archivo)}.txt"

    lineas = [f"Título: {titulo}", f"Fuente: {fuente}", f"URL: {url}"]
    if autor:
        lineas.append(f"Autor: {autor}")
    if fecha:
        lineas.append(f"Fecha: {fecha}")
    if extra:
        for clave, valor in extra.items():
            lineas.append(f"{clave}: {valor}")
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


def _texto_articulo(contenedor) -> str:
    """Extrae párrafos, encabezados y elementos de lista en orden del documento."""
    partes = []
    for el in contenedor.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            partes.append(txt)
    return "\n\n".join(partes)


def _meta_fecha(s) -> str | None:
    """Fecha de publicación desde meta tags (OpenGraph / article / pubdate)."""
    for sel in ("meta[property='article:published_time']",
                "meta[property='og:published_time']",
                "meta[name='date']",
                "meta[name='pubdate']"):
        el = s.select_one(sel)
        if el and el.get("content"):
            valor = el.get("content", "").strip()
            # "2020-01-15..." o "20200115".
            if re.match(r"^\d{4}-\d{2}-\d{2}", valor):
                return valor[:10]
            if re.match(r"^\d{8}$", valor):
                return f"{valor[0:4]}-{valor[4:6]}-{valor[6:8]}"
    return None


def extraer_theconversation(url: str):
    """Artículo de The Conversation."""
    s = _soup(url)
    h1 = s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Sin título"
    cont = s.select_one("div.content-body") or s.select_one("article")
    if cont:
        # Quitar pies de foto y embeds promocionales que se cuelan como <p>.
        for malo in cont.select("figcaption, div.slot, div.inline-promos"):
            malo.decompose()
    cuerpo = _texto_articulo(cont) if cont else ""
    autor = None
    autor_el = s.select_one("meta[name='author']")
    if autor_el:
        autor = autor_el.get("content")
    fecha = _meta_fecha(s)
    return titulo, cuerpo, autor, fecha


def extraer_cinemanet(url: str):
    """Análisis de WordPress (CinemaNet)."""
    s = _soup(url)
    h1 = s.select_one("h1.entry-title") or s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Sin título"
    cont = (s.select_one("div.entry-content")
            or s.select_one("div.post-content")
            or s.select_one("div.td-post-content")
            or s.select_one("article"))
    if cont:
        # Quitar navegación, botones de compartir, comentarios y relacionados.
        for malo in cont.select(
                "div.addtoany_share_save_container, div.sharedaddy, "
                "div#comments, div.comment-respond, div#respond, "
                "div.related, nav, aside, footer, div.jp-relatedposts, "
                "div.wp-block-uagb-container"):
            malo.decompose()
    cuerpo = _texto_articulo(cont) if cont else ""
    autor = None
    autor_el = (s.select_one("a[rel=author]")
                or s.select_one("a[href*='/author/']"))
    if autor_el:
        autor = autor_el.get_text(" ", strip=True)
    fecha = _meta_fecha(s)
    return titulo, cuerpo, autor, fecha


def extraer_wordpress(url: str):
    """Artículo genérico de WordPress (blog/crítica con permalink de entrada)."""
    s = _soup(url)
    h1 = s.select_one("h1.entry-title") or s.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else "Sin título"
    cont = (s.select_one("div.entry-content")
            or s.select_one("div.post-content")
            or s.select_one("main.site-main")
            or s.select_one("article.hentry")
            or s.select_one("div.td-post-content")
            or s.select_one("article"))
    if cont:
        # Quitar widgets de compartir, comentarios y navegación que se cuelan.
        for malo in cont.select(
                "div.addtoany_share_save_container, div.sharedaddy, "
                "div#comments, div.comment-respond, div#respond, "
                "div.comments-area, article.comment-body, section.comments, "
                "nav, aside, footer, div.related"):
            malo.decompose()
    cuerpo = _texto_articulo(cont) if cont else ""
    autor = None
    autor_el = (s.select_one("a[rel=author]")
                or s.select_one("a[href*='/author/']"))
    if autor_el:
        autor = autor_el.get_text(" ", strip=True)
    fecha = _meta_fecha(s)
    return titulo, cuerpo, autor, fecha


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

    # Metadatos conocidos (título/author/fecha) para PDFs concretos.
    autor = fecha = None
    sobrante = META_PDF.get(nombre_archivo)
    if sobrante:
        titulo = sobrante["titulo"]
        autor = sobrante.get("autor")
        fecha = sobrante.get("fecha")

    return titulo or nombre_archivo, cuerpo, autor, fecha


EXTRACTORES = {
    "espinof": extraer_espinof,
    "sensacine": extraer_sensacine,
    "lavanguardia": extraer_lavanguardia,
    "rogerebert": extraer_rogerebert,
    "blogspot": extraer_blogspot,
    "theconversation": extraer_theconversation,
    "cinemanet": extraer_cinemanet,
    "wordpress": extraer_wordpress,
}


# --------------------------------------------------------------------------- #
# Transcripción de vídeos de YouTube
# --------------------------------------------------------------------------- #

_TAG_SUB = re.compile(r"<[^>]+>")
_SOUND_SUB = re.compile(r"\[[^\]]*\]")
_WEBVTT_META = re.compile(
    r"^(WEBVTT|Kind:|Language:|NOTE|X-TIMESTAMP-MAP|align:|position:"
    r"|line:|region:|vertical:|size:)", re.IGNORECASE)


def subtitulos_a_texto(ruta: Path) -> str:
    """Convierte un .srt/.vtt a texto plano (sin marcas de tiempo ni etiquetas)."""
    cues: list[list[str]] = []
    actual: list[str] = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        s = linea.strip()
        if not s:
            if actual:
                cues.append(actual)
                actual = []
            continue
        if "-->" in s:  # línea de tiempo
            if actual:
                cues.append(actual)
                actual = []
            continue
        if s.isdigit() and not actual:  # número de secuencia SRT
            continue
        if _WEBVTT_META.match(s):
            continue
        t = _TAG_SUB.sub("", s)
        t = _SOUND_SUB.sub("", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            actual.append(t)
    if actual:
        cues.append(actual)

    texto: list[str] = []
    for cue in cues:
        fr = " ".join(cue).strip()
        if fr and (not texto or texto[-1] != fr):
            texto.append(fr)
    return "\n".join(texto)


def transcribir_video(nombre_archivo: str, url: str):
    """Descarga subtítulos (manuales o automáticos) en español de un vídeo y
    devuelve (titulo, autor, fecha, duracion, cuerpo)."""
    if not TIENE_YTDLP:
        raise RuntimeError("yt-dlp no está instalado (pip install yt-dlp)")

    base = CARPETA_RAW / sanitizar(nombre_archivo)
    CARPETA_RAW.mkdir(parents=True, exist_ok=True)

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["es"],
        "subtitlesformat": "srt",
        "outtmpl": str(base),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    titulo = info.get("title") or nombre_archivo
    autor = info.get("uploader") or info.get("channel") or None
    fecha_raw = info.get("upload_date")
    fecha = (f"{fecha_raw[0:4]}-{fecha_raw[4:6]}-{fecha_raw[6:8]}"
             if fecha_raw and len(fecha_raw) >= 8 else None)
    duracion = info.get("duration_string")

    # Localizar el fichero de subtítulos generado por yt-dlp.
    ruta_sub = None
    for cand in (f"{base}.es.srt", f"{base}.es.vtt",
                 f"{base}.es-orig.srt", f"{base}.es-orig.vtt"):
        if Path(cand).exists():
            ruta_sub = Path(cand)
            break
    if ruta_sub is None:
        ruta_sub = next(
            (p for p in CARPETA_RAW.glob(f"{sanitizar(nombre_archivo)}.*")
             if p.suffix in (".srt", ".vtt")),
            None,
        )

    if ruta_sub is None:
        raise RuntimeError("no se pudieron descargar subtítulos en español")

    cuerpo = subtitulos_a_texto(ruta_sub)
    return titulo, autor, fecha, duracion, cuerpo


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

    # --- Vídeos de YouTube -------------------------------------------------- #
    if VIDEOS:
        print("\n" + "=" * 70)
        print("TRANSCRIPCIÓN DE VÍDEOS DE YOUTUBE (corpus RAG)")
        print("=" * 70)
        if not TIENE_YTDLP:
            print("  [aviso] yt-dlp no está instalado; se omiten los vídeos.")
        else:
            for nombre_archivo, url in VIDEOS:
                print(f"\n- {nombre_archivo}")
                print(f"  {url[:80]}{'...' if len(url) > 80 else ''}")
                try:
                    titulo, autor, fecha, duracion, cuerpo = transcribir_video(
                        nombre_archivo, url)
                    if not cuerpo.strip():
                        raise RuntimeError("la transcripción quedó vacía")
                    extra = {}
                    if duracion:
                        extra["Duración"] = duracion
                    destino = guardar(nombre_archivo, titulo, "YouTube", url,
                                      cuerpo, autor, fecha, extra)
                    guardados.append(destino)
                    print(f"  OK -> {destino.relative_to(RAIZ)} "
                          f"({len(cuerpo)} caracteres)")
                except Exception as exc:
                    fallos.append((nombre_archivo, f"{type(exc).__name__}: {exc}"))
                    print(f"  FALLO -> {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print(f"Total de ficheros guardados: {len(guardados)}")
    if fallos:
        print(f"Total de fallos: {len(fallos)}")
    print("\nListo.")


if __name__ == "__main__":
    main()
