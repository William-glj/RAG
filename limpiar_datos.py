# -*- coding: utf-8 -*-
"""
limpiar_datos.py
================

Limpieza y normalización del corpus (Fase A del proyecto RAG).

Tema del corpus:
    El miedo al avance tecnológico y las revoluciones tecnológicas en el cine
    de ciencia ficción (subtema: la inteligencia artificial).

Lee los ficheros generados por `cargar_datos.py` y `scraping_criticas.py`
(dentro de ./datosDescargados) y produce una versión limpia en
./datosDescargados/limpio, aplicando estas reglas:

    1. Se descartan NULOS / datos faltantes:
       - Películas sin nota (ni `rating_imdb` ni `rating_tmdb`).
       - Reseñas vacías o demasiado cortas (ruido para el LLM).
    2. Se descartan DUPLICADOS (filas repetidas y reseñas repetidas).
    3. Se conservan SOLO las películas con reseña de texto (reseña de TMDB
       o crítica scrapeada). Las películas que únicamente aportan metadatos
       (título, año, nota, votos) sin texto recuperable se descartan.
    4. Se PRIORIZA la relevancia a IA/miedo tecnológico (columna
       `relevante_ia`), sin descartar el resto de la ciencia ficción.
    5. Se limpian los textos de las críticas scrapeadas: se eliminan pies
       de página, suscripciones, pies de foto, números de página, portadas
       y avisos legales que se hayan colado desde las páginas/PDF.

Salidas (todas en ./datosDescargados/limpio):
    imdb_peliculas.csv       IMDb filtrado + columnas de relevancia/review.
    tmdb_peliculas.csv       TMDB filtrado + columnas de review.
    tmdb_reviews.csv         Reseñas limpias (sin vacías/cortas/duplicadas).
    cruceIMDBTMDB.csv        Fichero maestro consolidado y normalizado.
    criticas/*.txt           Críticas scrapeadas limpias (mismo nombre).
    criticas_manifest.csv    Metadatos de cada crítica limpia.
    resumen_limpieza.txt     Informe de lo eliminado (antes/después).

Uso:
    python limpiar_datos.py

Requisitos: pandas (instalado en el Python global, junto con requests,
beautifulsoup4, pypdf...; ver requirements.txt).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# Mostrar correctamente los acentos en consolas de Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ / "datosDescargados"
CARPETA_LIMPIO = CARPETA_DATOS / "limpio"
CARPETA_CRITICAS = CARPETA_DATOS / "criticas"
CARPETA_CRITICAS_LIMPIO = CARPETA_LIMPIO / "criticas"

F_IMDB = CARPETA_DATOS / "imdb_peliculas.csv"
F_TMDB = CARPETA_DATOS / "tmdb_peliculas.csv"
F_REVIEWS = CARPETA_DATOS / "tmdb_reviews.csv"
F_CRUCE = CARPETA_DATOS / "cruceIMDBTMDB.csv"

# Las reseñas más cortas que esto se descartan (son ruido: "Awesome!",
# "Boring crap.", etc., no aportan contenido al LLM).
MIN_LONGITUD_REVIEW = 80

# Términos usados para la heurística de relevancia IA cuando solo tenemos el
# título (películas que vienen únicamente de IMDb y no tienen keywords de TMDB).
TERMINOS_IA = [
    "artificial intelligence", "inteligencia artificial", "a.i.", "a i",
    "robot", "androide", "android", "cyborg", "ciborg", "cyberpunk",
    "dystopia", "distopía", "singularity", "singularidad", "automata",
    "autómata", "terminator", "blade runner", "ex machina",
    "ghost in the shell", "matrix", "i robot", "robocop", "wargames",
    "westworld", "tron", "chappie", "transcendence", "upgrade", "m3gan",
    "wall-e", "wall e", "bicentennial man", "short circuit",
    "electric dreams", "forbidden planet", "colossus", "metropolis",
    "tecnología", "technology", "robots",
]

# imdb_id de las películas con crítica scrapeada (para enlazarlas al corpus).
CRITICA_A_IMDB = {
    # 2001: Una odisea del espacio (Espinof, SensaCine y tesis UACM).
    "2001": "tt0062622",
    "odisea": "tt0062622",
    "espinof": "tt0062622",
    "sensacine": "tt0062622",
    "uacm": "tt0062622",
    # Ex Machina (Roger Ebert).
    "ex machina": "tt0470752",
    "rogerebert": "tt0470752",
    # Her (El Antepenúltimo Mohicano y UOC open access).
    "her": "tt1798709",
    # Upgrade (El Cuervo Enteradillo).
    "upgrade": "tt6499752",
    "cuervo": "tt6499752",
    # Blade Runner y Blade Runner 2049 (CinemaNet, análisis). Importa el
    # orden: "blade runner 2049" debe comprobarse antes que "blade runner".
    "blade runner 2049": "tt1856101",
    "blade runner": "tt0083658",
    "cinemanet": "tt0083658",
}

# Título correcto de la tesis (el PDF expone el encabezado de la universidad
# como título en sus metadatos).
TITULO_TESIS = (
    "2001: Odisea del espacio. Stanley Kubrick y la cibernética en el "
    "nacimiento de un nuevo hombre"
)
AUTOR_TESIS = "Aura María Barajas González"


# --------------------------------------------------------------------------- #
# Utilidades de texto
# --------------------------------------------------------------------------- #

def sin_acentos(texto: str) -> str:
    """Quita tildes/ñ para comparaciones robustas entre cadenas."""
    tabla = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    return texto.translate(tabla)


def normalizar_espacios(texto: str) -> str:
    """Colapsa espacios múltiples y saltos, dejando un texto limpio."""
    if texto is None:
        return ""
    texto = str(texto).replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# Patrones de "boilerplate" (pie de página, navegación, suscripciones, etc.)
# que se eliminan de las críticas. Se aplican sobre párrafos ya normalizados.
_BOILERPLATE = [
    r"suscr[ií]b", r"newsletter", r"bolet[ií]n", r"recibe nuestro",
    r"recibe la newsletter", r"recibe los boletines",
    r"tambi[eé]n te puede interesar", r"tambi[eé]n le puede interesar",
    r"te puede interesar", r"te recomendamos", r"art[ií]culos relacionados",
    r"^en espinof \|", r"^en xataka \|", r"^en magnet \|", r"^en genbeta \|",
    r"^en 3djuegos \|", r"^en vidaextra \|", r"^en applesfera \|",
    r"^compartir", r"^comparte esto", r"^s[ií]guenos", r"^s[ií]gueme",
    r"^s[ií]ganos", r"^comp[aá]rtelo", r"^imprimir$", r"^volver$",
    r"^m[aá]s en ", r"^m[aá]s sobre ", r"^temas?: ", r"^etiquetas?: ",
    r"^categor[ií]as?: ", r"^publicado por", r"^publicado el",
    r"^publicada el", r"no hay comentarios", r"deja un comentario",
    r"dejar un comentario", r"leer m[aá]s", r"ver m[aá]s", r"seguir leyendo",
    r"^fuente: ", r"^copyright", r"^©", r"derechos reservados",
    r"pol[ií]tica de cookies", r"pol[ií]tica de privacidad", r"aviso legal",
    r"t[eé]rminos y condiciones",
]

# Pies de foto / descripciones de imagen que entran en La Vanguardia.
_CAPTIONS_LAVANGUARDIA = [
    r"cuadro de joseph wright", r"mosaico del siglo", r"^imprenta europea",
    r"grabado de 1813", r"^pintura con", r"viaje inaugural",
    r"en los a[ñn]os noventa", r"como muestra este cuadro",
    r"obras se centran en la revoluci[oó]n", r"academia de plat[oó]n",
    r"l[ií]der de los luditas",
]

# Líneas de portada / aviso legal / índice de la tesis UACM (se eliminan).
_TESIS_FRONTMATTER = [
    r"colegio de humanidades", r"licenciatura en", r"trabajo recepcional",
    r"^presenta:", r"^director:", r"^c\. ?aura", r"^lic\. octavio",
    r"sistema bibliotecario", r"coordinaci[oó]n acad[eé]mica",
    r"restricciones de uso", r"derechos reservados",
    r"universidad aut[oó]noma de la ciudad de m[eé]xico",
    r"ley federal del derecho", r"estatuto general org[aá]nico",
    r"reglamento de titulaci[oó]n", r"organismo p[uú]blico aut[oó]nomo",
    r"la presente obra y cada uno", r"queda prohibida su reproducci[oó]n",
    r"ser[aá]n reclamados por el titular", r"sancionados conforme a la legislaci[oó]n",
    r"para fines educativos", r"uso de su contenido", r"citar esta fuente",
    r"quien haga uso de la", r"29 de enero de 2002", r"^20\d\d$",
    r"^[ií]ndice$",
    # --- TFG UCM (Word): portada, resumen y palabras clave ---
    r"universidad complutense de madrid",
    r"facultad de ciencias de la informaci[oó]n",
    r"trabajo fin de grado", r"comunicaci[oó]n audiovisual",
    r"^autora?:", r"^tutor:", r"^nota:", r"^resumen$",
    r"^palabras clave:?", r"^keywords:?", r"^abstract$",
    r"^grado en ",
]


def _compilar(patrones: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patrones]


_BOILERPLATE_RE = _compilar(_BOILERPLATE)
_CAPTIONS_LV_RE = _compilar(_CAPTIONS_LAVANGUARDIA)
_TESIS_FRONT_RE = _compilar(_TESIS_FRONTMATTER)

# Página de la tesis: "~ 1 ~", "~ 142 ~" o números de página sueltos (1-3 cifras).
_RE_PAGINA = re.compile(r"^\s*~?\s*\d{1,3}\s*~?\s*$")
# Solo marcadores con virgulilla "~ N ~" (los usa la TOC / saltos de página).
_RE_PAGINA_TILDE = re.compile(r"^\s*~\s*\d+\s*~\s*$")
# Entradas de índice con puntos guía "...." (TFG UCM generado desde Word).
_RE_DOT_LEADER = re.compile(r"\.{4,}")


def _es_boilerplate(parrafo: str) -> bool:
    return any(r.search(parrafo) for r in _BOILERPLATE_RE)


def _es_caption_lavanguardia(parrafo: str) -> bool:
    return any(r.search(parrafo) for r in _CAPTIONS_LV_RE)


def _es_frontmatter_tesis(parrafo: str) -> bool:
    return any(r.search(parrafo) for r in _TESIS_FRONT_RE)


# --------------------------------------------------------------------------- #
# Parte 1: limpieza de los CSV
# --------------------------------------------------------------------------- #

def _leer_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV del pipeline con UTF-8 y tipos sin ambigüedad."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def _imdb_id_de_critica(nombre: str) -> str | None:
    """Devuelve el imdb_id al que se refiere una crítica scrapeada (o None)."""
    n = sin_acentos(nombre).lower()
    for clave, imdb_id in CRITICA_A_IMDB.items():
        if clave in n:
            return imdb_id
    return None


def limpiar_csvs() -> dict:
    """Limpia IMDb, TMDB, reseñas y cruce. Devuelve estadísticas."""
    stats: dict[str, int] = {}

    imdb = _leer_csv(F_IMDB)
    stats["imdb_original"] = len(imdb)

    tmdb = _leer_csv(F_TMDB) if F_TMDB.exists() else pd.DataFrame()
    stats["tmdb_original"] = len(tmdb)

    reviews = _leer_csv(F_REVIEWS) if F_REVIEWS.exists() else pd.DataFrame()
    stats["reviews_original"] = len(reviews)

    cruce = _leer_csv(F_CRUCE)
    stats["cruce_original"] = len(cruce)

    # --- Conjuntos de ids con reseña de texto ---------------------------------
    if not reviews.empty:
        review_imdb_ids = set(reviews["imdb_id"].dropna().astype(str))
        review_tmdb_ids = set(reviews["tmdb_id"].dropna().astype(int))
    else:
        review_imdb_ids, review_tmdb_ids = set(), set()

    # Películas con crítica scrapeada (2001, Ex Machina, Upgrade).
    scraped_count: dict[str, int] = {}
    if CARPETA_CRITICAS.exists():
        for f in CARPETA_CRITICAS.glob("*.txt"):
            imdb_id = _imdb_id_de_critica(f.name)
            if imdb_id:
                scraped_count[imdb_id] = scraped_count.get(imdb_id, 0) + 1

    # --- Relevancia IA desde TMDB (por tmdb_id) -------------------------------
    relevante_tmdb: dict[int, bool] = {}
    if not tmdb.empty and "relevante_ia" in tmdb.columns:
        for tmdb_id, rel in zip(tmdb["tmdb_id"], tmdb["relevante_ia"]):
            if pd.notna(rel):
                relevante_tmdb[int(tmdb_id)] = bool(rel)

    def heuristica_ia(titulo: object, generos: object, keywords: object) -> bool:
        texto = " ".join(
            str(v) for v in (titulo, generos, keywords) if pd.notna(v)
        ).lower()
        return any(t in texto for t in TERMINOS_IA)

    # --- Normalizar el cruce y aplicar filtros --------------------------------
    # Tipos numéricos anulables.
    for col in ["rating_imdb", "rating_tmdb", "votos_imdb", "votos_tmdb"]:
        cruce[col] = pd.to_numeric(cruce[col], errors="coerce")

    tiene_nota = cruce["rating_imdb"].notna() | cruce["rating_tmdb"].notna()

    has_review = (
        cruce["imdb_id"].isin(review_imdb_ids)
        | cruce["imdb_id"].isin(set(scraped_count))
    )

    # Solo se conservan las películas con reseña de texto (TMDB o scrapeada).
    # Las que únicamente aportan metadatos (título/año/nota/votos) se descartan.
    conservar = tiene_nota & has_review

    master = cruce.loc[conservar].copy()
    stats["cruce_sin_nota"] = int((~tiene_nota).sum())
    stats["cruce_sin_review"] = int((tiene_nota & ~has_review).sum())

    # Columnas consolidadas.
    master["titulo_original"] = master["titulo_original_imdb"].fillna(
        master["titulo_original_tmdb"]
    )
    master["anio"] = pd.to_numeric(master["anio"], errors="coerce").astype("Int64")
    master["duracion_min"] = pd.to_numeric(
        master["duracion_tmdb"].fillna(master["duracion_imdb"]), errors="coerce"
    ).astype("Int64")
    master["generos"] = master["generos_imdb"].fillna(master["generos_tmdb"])

    # Nº de reseñas de texto por película.
    conteo_reviews = reviews.groupby("imdb_id").size() if not reviews.empty \
        else pd.Series(dtype=int)
    master["n_reviews_tmdb"] = (
        master["imdb_id"].map(conteo_reviews).fillna(0).astype(int)
    )
    master["n_criticas_scrapeadas"] = (
        master["imdb_id"].map(scraped_count).fillna(0).astype(int)
    )
    master["has_review"] = (master["n_reviews_tmdb"] + master["n_criticas_scrapeadas"]) > 0

    # Relevancia IA final + su procedencia.
    def relevancia(row):
        tmdb_id = row["tmdb_id"]
        if pd.notna(tmdb_id) and int(tmdb_id) in relevante_tmdb:
            return relevante_tmdb[int(tmdb_id)], "tmdb_keywords"
        h = heuristica_ia(row["titulo"], row["generos_imdb"], row["keywords"])
        return h, ("titulo" if h else "")

    relev = master.apply(relevancia, axis=1, result_type="expand")
    master["relevante_ia"] = relev[0].astype(bool)
    master["relevante_ia_fuente"] = relev[1]

    # Orden final: primero las relevantes a IA y más votadas.
    master = master.sort_values(
        ["relevante_ia", "votos_imdb", "votos_tmdb"],
        ascending=[False, False, False], na_position="last",
    )

    columnas_master = [
        "imdb_id", "tmdb_id", "titulo", "titulo_original", "anio",
        "duracion_min", "generos", "generos_imdb", "generos_tmdb", "keywords",
        "sinopsis", "tagline", "rating_imdb", "votos_imdb", "rating_tmdb",
        "votos_tmdb", "relevante_ia", "relevante_ia_fuente", "has_review",
        "n_reviews_tmdb", "n_criticas_scrapeadas", "origen",
    ]
    master = master[[c for c in columnas_master if c in master.columns]]

    # --- Derivar ficheros por fuente ------------------------------------------
    imdb_survive = set(master.loc[master["origen"] != "tmdb", "imdb_id"])
    imdb_limpio = imdb[imdb["imdb_id"].isin(imdb_survive)].copy()
    imdb_limpio = imdb_limpio.merge(
        master[["imdb_id", "relevante_ia", "relevante_ia_fuente", "has_review",
                "n_reviews_tmdb", "n_criticas_scrapeadas"]],
        on="imdb_id", how="left",
    )
    stats["imdb_final"] = len(imdb_limpio)

    tmdb_survive = set(master.loc[master["origen"] != "imdb", "tmdb_id"].dropna().astype(int))
    tmdb_limpio = tmdb[tmdb["tmdb_id"].isin(tmdb_survive)].copy() if not tmdb.empty \
        else tmdb.copy()
    if not tmdb.empty:
        tmdb_limpio = tmdb_limpio.merge(
            master[["tmdb_id", "has_review", "n_reviews_tmdb",
                    "n_criticas_scrapeadas"]].dropna(subset=["tmdb_id"]),
            on="tmdb_id", how="left",
        )
    stats["tmdb_final"] = len(tmdb_limpio)

    # --- Limpiar reseñas ------------------------------------------------------
    reviews_limpio = reviews.copy()
    if not reviews.empty:
        reviews_limpio["contenido"] = (
            reviews_limpio["contenido"].fillna("").astype(str)
            .map(normalizar_espacios)
        )
        reviews_limpio["autor"] = reviews_limpio["autor"].fillna("").astype(str).str.strip()
        # Quitar vacías, demasiado cortas y duplicadas.
        reviews_limpio = reviews_limpio[reviews_limpio["contenido"].str.len() >= MIN_LONGITUD_REVIEW]
        reviews_limpio = reviews_limpio.drop_duplicates(
            subset=["tmdb_id", "autor", "contenido"]
        ).drop_duplicates(subset=["tmdb_id", "contenido"])
        reviews_limpio = reviews_limpio.sort_values(["tmdb_id", "fecha"])
    stats["reviews_final"] = len(reviews_limpio)

    # --- Guardar --------------------------------------------------------------
    CARPETA_LIMPIO.mkdir(parents=True, exist_ok=True)
    imdb_limpio.to_csv(CARPETA_LIMPIO / "imdb_peliculas.csv", index=False, encoding="utf-8")
    if not tmdb.empty:
        tmdb_limpio.to_csv(CARPETA_LIMPIO / "tmdb_peliculas.csv", index=False, encoding="utf-8")
        reviews_limpio.to_csv(CARPETA_LIMPIO / "tmdb_reviews.csv", index=False, encoding="utf-8")
    master.to_csv(CARPETA_LIMPIO / "cruceIMDBTMDB.csv", index=False, encoding="utf-8")

    stats["cruce_final"] = len(master)
    stats["relevantes_ia"] = int(master["relevante_ia"].sum())
    stats["con_review"] = int(master["has_review"].sum())
    return stats


# --------------------------------------------------------------------------- #
# Parte 2: limpieza de las críticas scrapeadas
# --------------------------------------------------------------------------- #

def _parsear_cabecera(texto: str) -> tuple[dict[str, str], str]:
    """Separa la cabecera (Título/Fuente/URL/Autor/Fecha) del cuerpo."""
    lineas = texto.split("\n")
    meta: dict[str, str] = {}
    idx = 0
    campos = {"titulo": "titulo", "fuente": "fuente", "url": "url",
              "autor": "autor", "fecha": "fecha", "duracion": "duracion"}
    while idx < len(lineas):
        linea = lineas[idx].strip()
        m = re.match(r"^([^:]+):\s*(.*)$", linea)
        if m:
            clave = sin_acentos(m.group(1)).strip().lower()
            if clave in campos:
                meta[campos[clave]] = m.group(2).strip()
                idx += 1
                continue
        break
    cuerpo = "\n".join(lineas[idx:])
    return meta, cuerpo


def limpiar_texto_critica(cuerpo: str, fuente: str, es_pdf: bool = False,
                          es_tfg: bool = False) -> tuple[str, str | None]:
    """Limpia el cuerpo de una crítica y devuelve (cuerpo, autor_detectado)."""
    if es_tfg:
        return _limpiar_cuerpo_pdf_ucm(cuerpo), None
    if es_pdf:
        return _limpiar_cuerpo_pdf(cuerpo), None

    autor_detectado = None
    parrafos = [normalizar_espacios(p) for p in cuerpo.split("\n")]
    limpios: list[str] = []

    # Heurística de autor (solo La Vanguardia): línea con 2-4 palabras en
    # mayúscula inicial, sin puntuación, que es la firma del articulista.
    re_autor = re.compile(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?: [A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){1,3}$")

    for p in parrafos:
        if not p:
            continue
        if _es_boilerplate(p):
            continue

        if fuente == "La Vanguardia":
            if _es_caption_lavanguardia(p):
                continue
            if autor_detectado is None and re_autor.match(p) and len(p) < 50:
                autor_detectado = p
                continue

        limpios.append(p)

    return "\n\n".join(limpios), autor_detectado


def _reunir_lineas_pdf(texto: str) -> list[str]:
    """Reune las líneas cortadas por la extracción del PDF en párrafos.

    pypdf deja un espacio al final de las líneas que continúan en la
    siguiente (salto de línea dentro de un párrafo) y sin espacio al final
    del párrafo. Usamos ese detalle para reconstruir los párrafos.
    """
    parrafos: list[str] = []
    buf = ""
    for linea in texto.split("\n"):
        s = linea.strip()
        if not s:
            if buf:
                parrafos.append(buf)
                buf = ""
            continue
        buf = (buf + " " + s) if buf else s
        if not linea.endswith(" ") and not s.endswith("-"):
            parrafos.append(buf)
            buf = ""
    if buf:
        parrafos.append(buf)
    return parrafos


def _es_linea_toc(linea: str) -> bool:
    """True si una línea (tras "ÍNDICE") pertenece a la tabla de contenidos.

    Detecta: líneas vacías, números de página, puntos guía "...." y títulos en
    mayúsculas (las entradas del índice van en mayúsculas, mientras que el
    cuerpo empieza con prosa en minúsculas).
    """
    s = linea.strip()
    if not s:
        return True
    if _RE_PAGINA.match(s):
        return True
    if _RE_DOT_LEADER.search(s):
        return True
    letras = [c for c in s if c.isalpha()]
    if letras and sum(c.isupper() for c in letras) / len(letras) > 0.7:
        return True
    return False


def _limpiar_cuerpo_pdf(cuerpo: str) -> str:
    """Limpia el cuerpo de un PDF académico (tesis): quita portada, aviso
    legal, índice (TOC) y números de página, y reune los párrafos cortados."""
    lineas = cuerpo.split("\n")

    # 1) Recortar todo lo anterior al índice (portada + aviso legal).
    i_indice = next(
        (i for i, l in enumerate(lineas)
         if l.strip().lower() in ("índice", "indice")),
        None,
    )
    if i_indice is not None:
        lineas = lineas[i_indice:]
        # 2) Recortar la tabla de contenidos (hasta el primer salto "~ N ~").
        for i, l in enumerate(lineas):
            if _RE_PAGINA_TILDE.match(l):
                lineas = lineas[i + 1:]
                break

    # 3) Re-unir líneas cortadas en párrafos reales.
    parrafos = _reunir_lineas_pdf("\n".join(lineas))

    # 4) Quitar números de página, front matter residual y boilerplate.
    limpios: list[str] = []
    for p in parrafos:
        p = normalizar_espacios(p)
        if not p:
            continue
        if _RE_PAGINA.match(p) or _es_frontmatter_tesis(p) or _es_boilerplate(p):
            continue
        limpios.append(p)
    return "\n\n".join(limpios)


def _limpiar_cuerpo_pdf_ucm(cuerpo: str) -> str:
    """Limpia el cuerpo de un TFG (PDF generado desde Word, p. ej. Docta UCM):
    quita portada, resumen, palabras clave, índice (puntos guía) y números de
    página, y reune los párrafos cortados."""
    lineas = cuerpo.split("\n")

    # 1) Recortar portada + resumen + palabras clave: todo hasta "ÍNDICE".
    i_indice = next(
        (i for i, l in enumerate(lineas)
         if l.strip().lower() in ("índice", "indice")),
        None,
    )
    if i_indice is not None:
        lineas = lineas[i_indice + 1:]

    # 2) Quitar las entradas del índice (líneas de TOC, puntos guía, páginas).
    i = 0
    while i < len(lineas) and _es_linea_toc(lineas[i]):
        i += 1
    lineas = lineas[i:]

    # 3) Re-unir líneas cortadas en párrafos reales.
    parrafos = _reunir_lineas_pdf("\n".join(lineas))

    # 4) Quitar números de página, front matter residual y boilerplate.
    limpios: list[str] = []
    for p in parrafos:
        p = normalizar_espacios(p)
        if not p:
            continue
        if _RE_PAGINA.match(p) or _RE_DOT_LEADER.search(p):
            continue
        if _es_frontmatter_tesis(p) or _es_boilerplate(p):
            continue
        limpios.append(p)
    return "\n\n".join(limpios)


def limpiar_criticas() -> dict:
    """Limpia cada .txt de críticas y genera un manifiesto. Devuelve stats."""
    CARPETA_CRITICAS_LIMPIO.mkdir(parents=True, exist_ok=True)
    filas_manifest: list[dict] = []
    stats: dict[str, int] = {}

    if not CARPETA_CRITICAS.exists():
        return stats

    for f in sorted(CARPETA_CRITICAS.glob("*.txt")):
        texto = f.read_text(encoding="utf-8")
        meta, cuerpo = _parsear_cabecera(texto)
        fuente = meta.get("fuente", "")

        # Casos especiales.
        n = sin_acentos(f.name).lower()
        es_pdf = "uacm" in n
        es_tfg = "tfg" in n
        if es_pdf:
            meta["titulo"] = TITULO_TESIS

        cuerpo_limpio, autor_det = limpiar_texto_critica(
            cuerpo, fuente, es_pdf=es_pdf, es_tfg=es_tfg)
        if autor_det:
            meta.setdefault("autor", autor_det)
        if es_pdf and "autor" not in meta:
            meta["autor"] = AUTOR_TESIS

        # Reconstruir cabecera canónica.
        cabecera = ["Título: " + meta.get("titulo", f.stem),
                    "Fuente: " + meta.get("fuente", ""),
                    "URL: " + meta.get("url", "")]
        if meta.get("autor"):
            cabecera.append("Autor: " + meta["autor"])
        if meta.get("fecha"):
            cabecera.append("Fecha: " + meta["fecha"])
        if meta.get("duracion"):
            cabecera.append("Duración: " + meta["duracion"])
        salida = "\n".join(cabecera) + "\n\n" + cuerpo_limpio.strip() + "\n"

        destino = CARPETA_CRITICAS_LIMPIO / f.name
        destino.write_text(salida, encoding="utf-8")

        imdb_id = _imdb_id_de_critica(f.name)
        filas_manifest.append({
            "archivo": f.name,
            "titulo": meta.get("titulo", ""),
            "fuente": meta.get("fuente", ""),
            "autor": meta.get("autor", ""),
            "fecha": meta.get("fecha", ""),
            "duracion": meta.get("duracion", ""),
            "url": meta.get("url", ""),
            "imdb_id": imdb_id or "",
            "n_caracteres_limpio": len(cuerpo_limpio),
        })

    manifiesto = pd.DataFrame(filas_manifest).sort_values("archivo")
    manifiesto.to_csv(CARPETA_LIMPIO / "criticas_manifest.csv", index=False, encoding="utf-8")
    stats["criticas_limpias"] = len(filas_manifest)
    stats["criticas_chars"] = int(manifiesto["n_caracteres_limpio"].sum())
    return stats


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #

def escribir_resumen(stats_csv: dict, stats_crit: dict) -> None:
    lineas: list[str] = []
    lineas.append("=" * 70)
    lineas.append("RESUMEN DE LA LIMPIEZA DEL CORPUS (cine ciencia ficción / IA)")
    lineas.append("=" * 70)
    lineas.append("")
    lineas.append("Criterios aplicados:")
    lineas.append("  - Se conservan SOLO películas con reseña de texto (TMDB o crítica scrapeada).")
    lineas.append("  - Se descartan las películas que solo aportan metadatos (sin texto recuperable).")
    lineas.append(f"  - Se descartan reseñas vacías o < {MIN_LONGITUD_REVIEW} caracteres.")
    lineas.append("  - Se eliminan duplicados y nulos.")
    lineas.append("  - Se marca relevante_ia (prioridad, sin descartar el resto).")
    lineas.append("")
    lineas.append("--- Datos estructurados -------------------------------------------")
    lineas.append(f"  IMDb  películas: {stats_csv.get('imdb_original', 0)} -> "
                  f"{stats_csv.get('imdb_final', 0)}")
    lineas.append(f"  TMDB  películas: {stats_csv.get('tmdb_original', 0)} -> "
                  f"{stats_csv.get('tmdb_final', 0)}")
    lineas.append(f"  TMDB  reseñas:   {stats_csv.get('reviews_original', 0)} -> "
                  f"{stats_csv.get('reviews_final', 0)}")
    lineas.append(f"  Cruce registros: {stats_csv.get('cruce_original', 0)} -> "
                  f"{stats_csv.get('cruce_final', 0)}")
    lineas.append(f"    (sin nota: {stats_csv.get('cruce_sin_nota', 0)} | "
                  f"sin reseña: {stats_csv.get('cruce_sin_review', 0)})")
    lineas.append(f"  Relevantes a IA: {stats_csv.get('relevantes_ia', 0)}")
    lineas.append(f"  Con reseña de texto: {stats_csv.get('con_review', 0)}")
    lineas.append("")
    lineas.append("--- Críticas scrapeadas -------------------------------------------")
    lineas.append(f"  Críticas limpias: {stats_crit.get('criticas_limpias', 0)}")
    lineas.append(f"  Caracteres totales tras limpiar: {stats_crit.get('criticas_chars', 0)}")
    lineas.append("")
    lineas.append("Salidas en " + str(CARPETA_LIMPIO.relative_to(RAIZ)) + ".")

    resumen = "\n".join(lineas) + "\n"
    (CARPETA_LIMPIO / "resumen_limpieza.txt").write_text(resumen, encoding="utf-8")
    print(resumen)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 70)
    print("LIMPIEZA DEL CORPUS (cine ciencia ficción / IA)")
    print("=" * 70)

    print("\n[1/2] Limpiando datos estructurados (IMDb, TMDB, reseñas, cruce) ...")
    stats_csv = limpiar_csvs()

    print("\n[2/2] Limpiando críticas scrapeadas ...")
    stats_crit = limpiar_criticas()

    print()
    escribir_resumen(stats_csv, stats_crit)
    print("\nListo.")


if __name__ == "__main__":
    main()
