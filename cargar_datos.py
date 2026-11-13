# -*- coding: utf-8 -*-
"""
cargar_datos.py
===============

Pipeline automatizado de captación de datos (Fase A del proyecto RAG).

Tema del corpus:
    El miedo al avance tecnológico y las revoluciones tecnológicas en el cine
    de ciencia ficción (subtema: la inteligencia artificial).

Fuentes:
    1. IMDb (datasets oficiales no comerciales, datasets.imdbws.com)
       -> metadatos estructurados y ratings (title.basics + title.ratings).
    2. TMDB (API v3, https://www.themoviedb.org)
       -> metadatos, sinopsis (overview), palabras clave y RESEÑAS de usuarios.

Salidas (todas en ./datosDescargados):
    imdb_peliculas.csv      Películas de ciencia ficción de IMDb (títulos, años,
                            géneros, nota media y número de votos).
    tmdb_peliculas.csv      Películas de TMDB (sinopsis, géneros, keywords, nota).
    tmdb_reviews.csv        Reseñas de usuarios de TMDB para esas películas.
    cruceIMDBTMDB.csv       Fichero conjunto que enlaza ambas fuentes mediante
                            el identificador de IMDb (tconst == imdb_id).

    ./datosDescargados/raw/ Ficheros fuente sin procesar descargados de IMDb.

NOTA SOBRE LAS RESEÑAS DE IMDb:
    IMDb no ofrece una API pública y su web está protegida por AWS WAF (anti-bot),
    por lo que el scraping directo de reseñas devuelve un reto JavaScript (HTTP 202).
    La librería Cinemagoer (IMDbPY), en su versión actual, ya no incluye soporte
    para reseñas de usuarios. Por eso las reseñas de texto se obtienen de la API
    de TMDB (que sí expone reseñas de usuarios) y de IMDb se obtienen los datos
    estructurados y ratings oficiales. Ambos se cruzan en "cruceIMDBTMDB.csv".

Uso:
    python cargar_datos.py

    Para habilitar la parte de TMDB, crea una API key gratuita en
    https://www.themoviedb.org/settings/api y exponla en una variable de entorno
    TMDB_API_KEY o en un fichero .env (ver .env.example). Sin clave, el script
    descarga y procesa IMDb y avisa de cómo completar la parte de TMDB.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Mostrar correctamente los acentos en consolas de Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# Configuración general
# --------------------------------------------------------------------------- #

# Carpeta raíz del proyecto (donde vive este script) y carpeta de datos.
RAIZ = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ / "datosDescargados"
CARPETA_RAW = CARPETA_DATOS / "raw"

# Ficheros de salida.
F_IMDB_PELICULAS = CARPETA_DATOS / "imdb_peliculas.csv"
F_TMDB_PELICULAS = CARPETA_DATOS / "tmdb_peliculas.csv"
F_TMDB_REVIEWS = CARPETA_DATOS / "tmdb_reviews.csv"
F_CRUCE = CARPETA_DATOS / "cruceIMDBTMDB.csv"

# --------------------------------------------------------------------------- #
# Configuración IMDb (datasets oficiales)
# --------------------------------------------------------------------------- #

IMDB_DATASETS = {
    "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
}

# Columnas que nos interesan de title.basics (reduce memoria al leer).
COLS_BASICS = [
    "tconst",
    "titleType",
    "primaryTitle",
    "originalTitle",
    "isAdult",
    "startYear",
    "runtimeMinutes",
    "genres",
]

# Solo películas (excluye series, cortos, videojuegos, etc.) de ciencia ficción.
TITULO_TIPO = "movie"
GENERO_SCIFI_IMDB = "Sci-Fi"


# --------------------------------------------------------------------------- #
# Configuración TMDB (API v3)
# --------------------------------------------------------------------------- #

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_GENERO_SCIFI = 878  # Science Fiction

# Términos de IA/tecnología que resolvemos como keywords en TMDB.
TMDB_TERMINOS_KEYWORD = [
    "artificial intelligence",
    "robot",
    "android",
    "cyborg",
    "technology",
    "cyberpunk",
    "dystopia",
]

# Términos usados para marcar una película como "relevante" (IA / miedo tecnológico).
TERMINOS_IA = [
    "artificial intelligence",
    "a.i.",
    "robot",
    "android",
    "cyborg",
    "machine",
    "technology",
    "cyberpunk",
    "dystopia",
    "singularity",
    "automata",
    "terminator",
    "blade runner",
    "ex machina",
    "ghost in the shell",
    "matrix",
]

# Películas canónicas sobre IA / miedo a la tecnología que siempre incluimos
# aunque no aparezcan en las primeras páginas de "descubrir".
# Formato: (título normalizado, año o None).
PELICULAS_CANONICAS = [
    ("2001 a space odyssey", 1968),
    ("metropolis", 1927),
    ("blade runner", 1982),
    ("blade runner 2049", 2017),
    ("the terminator", 1984),
    ("terminator 2 judgment day", 1991),
    ("the matrix", 1999),
    ("the matrix reloaded", 2003),
    ("the matrix revolutions", 2003),
    ("the matrix resurrections", 2021),
    ("ex machina", 2015),
    ("her", 2013),
    ("a i artificial intelligence", 2001),
    ("i robot", 2004),
    ("ghost in the shell", 1995),
    ("ghost in the shell", 2017),
    ("robocop", 1987),
    ("wargames", 1983),
    ("tron", 1982),
    ("westworld", 1973),
    ("chappie", 2015),
    ("transcendence", 2014),
    ("automata", 2014),
    ("moon", 2009),
    ("wall e", 2008),
    ("the machine", 2013),
    ("colossus the forbin project", 1970),
    ("short circuit", 1986),
    ("bicentennial man", 1999),
    ("electric dreams", 1984),
    ("upgrade", 2018),
    ("m3gan", 2022),
    ("the creator", 2023),
    ("forbidden planet", 1956),
]

# Límites de la captura en TMDB (para no agotar el rate-limit gratuito).
TMDB_PAGINAS_GENERO = 5      # páginas de "descubrir" por género (20 pelis/pág).
TMDB_PAGINAS_KEYWORD = 1     # páginas de "descubrir" por keyword.
TMDB_MAX_DETALLES = 200      # máximo de películas a las que pedimos detalles.
TMDB_MAX_REVIEWS = 60        # máximo de películas a las que pedimos reseñas.

# Rate-limit TMDB: pausa entre peticiones (segundos) para no superar ~4 req/s.
TMDB_PAUSA = 0.28

# User-Agent para las descargas (buena práctica con IMDb datasets).
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Utilidades generales
# --------------------------------------------------------------------------- #

def asegurar_carpetas() -> None:
    """Crea las carpetas de salida si no existen."""
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    CARPETA_RAW.mkdir(parents=True, exist_ok=True)


def descargar_archivo(url: str, destino: Path, timeout: int = 120) -> Path:
    """Descarga un fichero con barra de progreso y lo guarda en `destino`."""
    if destino.exists():
        print(f"  [omitido] ya existe {destino.name}")
        return destino

    print(f"  Descargando {destino.name} ...")
    with requests.get(url, headers={"User-Agent": USER_AGENT},
                      stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"    {destino.name}", ncols=80,
        ) as barra:
            for bloque in r.iter_content(chunk_size=1024 * 1024):
                f.write(bloque)
                barra.update(len(bloque))
    return destino


# --------------------------------------------------------------------------- #
# Parte 1: IMDb (datasets oficiales)
# --------------------------------------------------------------------------- #

def descargar_datasets_imdb() -> dict[str, Path]:
    """Descarga los datasets oficiales de IMDb y devuelve sus rutas locales."""
    print("\n[1/4] Descargando datasets oficiales de IMDb ...")
    rutas: dict[str, Path] = {}
    for nombre, url in IMDB_DATASETS.items():
        rutas[nombre] = descargar_archivo(url, CARPETA_RAW / nombre)
    return rutas


def cargar_basics(path: Path) -> pd.DataFrame:
    """Lee title.basics.tsv.gz y devuelve solo películas de ciencia ficción."""
    print("  Leyendo title.basics.tsv.gz (puede tardar un poco) ...")
    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=COLS_BASICS,
        dtype=str,
        na_values=["\\N"],
        low_memory=False,
    )
    mascara = (
        (df["titleType"] == TITULO_TIPO)
        & df["genres"].str.contains(GENERO_SCIFI_IMDB, na=False)
        & (df["isAdult"] == "0")
    )
    df = df.loc[mascara].copy()
    # Convertir columnas numéricas a enteros anulables.
    df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce").astype("Int64")
    df["runtimeMinutes"] = pd.to_numeric(
        df["runtimeMinutes"], errors="coerce"
    ).astype("Int64")
    return df


def cargar_ratings(path: Path) -> pd.DataFrame:
    """Lee title.ratings.tsv.gz (nota media y nº de votos)."""
    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        dtype={"tconst": str, "averageRating": "float64", "numVotes": "int64"},
        low_memory=False,
    )
    return df


def procesar_imdb() -> pd.DataFrame:
    """Descarga y procesa IMDb: devuelve DataFrame de pelis Sci-Fi con rating."""
    rutas = descargar_datasets_imdb()
    pelis = cargar_basics(rutas["title.basics.tsv.gz"])
    ratings = cargar_ratings(rutas["title.ratings.tsv.gz"])

    pelis = pelis.merge(ratings, on="tconst", how="left")

    # Reordenar y renombrar para que sea legible.
    pelis = pelis.rename(columns={
        "tconst": "imdb_id",
        "primaryTitle": "titulo",
        "originalTitle": "titulo_original",
        "startYear": "anio",
        "runtimeMinutes": "duracion_min",
        "genres": "generos",
        "averageRating": "rating_imdb",
        "numVotes": "votos_imdb",
    })
    # Reasegurar tipos (la mezcla puede degradar Int64 a float64).
    pelis["anio"] = pd.to_numeric(pelis["anio"], errors="coerce").astype("Int64")
    pelis["duracion_min"] = pd.to_numeric(
        pelis["duracion_min"], errors="coerce"
    ).astype("Int64")
    pelis["votos_imdb"] = pd.to_numeric(
        pelis["votos_imdb"], errors="coerce"
    ).astype("Int64")
    pelis["rating_imdb"] = pd.to_numeric(pelis["rating_imdb"], errors="coerce")

    columnas = [
        "imdb_id", "titulo", "titulo_original", "anio", "duracion_min",
        "generos", "rating_imdb", "votos_imdb",
    ]
    pelis = pelis[columnas].sort_values(
        ["votos_imdb", "rating_imdb"], ascending=False, na_position="last"
    )
    return pelis


def guardar_imdb(pelis: pd.DataFrame) -> None:
    """Guarda las películas de IMDb en CSV."""
    pelis.to_csv(F_IMDB_PELICULAS, index=False, encoding="utf-8")
    print(f"  -> {len(pelis)} películas Sci-Fi guardadas en "
          f"{F_IMDB_PELICULAS.relative_to(RAIZ)}")


# --------------------------------------------------------------------------- #
# Parte 2: TMDB (API v3)
# --------------------------------------------------------------------------- #

def cargar_api_key_tmdb() -> str | None:
    """Devuelve la API key de TMDB desde el entorno o el fichero .env."""
    load_dotenv(RAIZ / ".env")
    return os.getenv("TMDB_API_KEY")


def tmdb_get(session: requests.Session, endpoint: str, params: dict,
             reintentos: int = 4) -> dict:
    """GET a la API de TMDB con rate-limit y reintentos ante 429/5xx."""
    url = TMDB_BASE + endpoint
    for intento in range(reintentos):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 200:
            time.sleep(TMDB_PAUSA)
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and intento < reintentos - 1:
            espera = 2 ** intento
            print(f"    [TMDB] {r.status_code} en {endpoint}; "
                  f"reintentando en {espera}s ...")
            time.sleep(espera)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Fallo al consultar {endpoint}")


def resolver_keywords_tmdb(session: requests.Session, api_key: str) -> list[int]:
    """Resuelve los términos de IA/tecnología a IDs de keyword de TMDB."""
    print("  Resolviendo keywords de IA/tecnología en TMDB ...")
    ids: list[int] = []
    for termino in TMDB_TERMINOS_KEYWORD:
        datos = tmdb_get(session, "/search/keyword",
                         {"api_key": api_key, "query": termino})
        resultados = datos.get("results", [])
        if resultados:
            mejor = next(
                (r for r in resultados
                 if r.get("name", "").strip().lower() == termino.strip().lower()),
                resultados[0],
            )
            ids.append(int(mejor["id"]))
    return list(dict.fromkeys(ids))  # sin duplicados, manteniendo orden


def descubrir_tmdb(session: requests.Session, api_key: str, extra: dict,
                   paginas: int) -> set[int]:
    """Recorre páginas de /discover/movie y devuelve un set de ids de película."""
    ids: set[int] = set()
    for pagina in range(1, paginas + 1):
        params = {
            "api_key": api_key,
            "language": "en-US",
            "sort_by": "vote_count.desc",
            "include_adult": "false",
            "page": pagina,
            **extra,
        }
        datos = tmdb_get(session, "/discover/movie", params)
        for peli in datos.get("results", []):
            ids.add(int(peli["id"]))
        if pagina >= datos.get("total_pages", 1):
            break
    return ids


def buscar_pelicula_tmdb(session: requests.Session, api_key: str,
                         titulo: str, anio: int | None) -> int | None:
    """Busca una película canónica por título (y año si se indica)."""
    params = {
        "api_key": api_key,
        "query": titulo,
        "language": "en-US",
        "include_adult": "false",
    }
    if anio:
        params["year"] = anio
    datos = tmdb_get(session, "/search/movie", params)
    resultados = datos.get("results", [])
    if not resultados:
        return None
    return int(resultados[0]["id"])


def detalles_pelicula_tmdb(session: requests.Session, api_key: str,
                           movie_id: int) -> dict:
    """Detalles de una película incluyendo keywords."""
    return tmdb_get(
        session,
        f"/movie/{movie_id}",
        {"api_key": api_key, "language": "en-US",
         "append_to_response": "keywords"},
    )


def reviews_pelicula_tmdb(session: requests.Session, api_key: str,
                          movie_id: int) -> list[dict]:
    """Reseñas de usuarios (página 1) de una película."""
    datos = tmdb_get(
        session,
        f"/movie/{movie_id}/reviews",
        {"api_key": api_key, "language": "en-US", "page": 1},
    )
    return datos.get("results", [])


def es_relevante_ia(titulo: str, keywords: list[str]) -> bool:
    """True si el título o las keywords apuntan a IA / miedo tecnológico."""
    texto = (str(titulo) + " " + " ".join(keywords)).lower()
    return any(termino in texto for termino in TERMINOS_IA)


def descargar_tmdb(api_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga películas y reseñas de TMDB. Devuelve (películas, reseñas)."""
    print("\n[2/4] Descargando datos de TMDB ...")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # 1) Recopilar ids candidatos: género Sci-Fi + keywords IA + canónicas.
    ids = descubrir_tmdb(session, api_key,
                         {"with_genres": TMDB_GENERO_SCIFI},
                         TMDB_PAGINAS_GENERO)
    print(f"  {len(ids)} películas por género Sci-Fi.")

    ids_keywords: set[int] = set()
    for kw_id in resolver_keywords_tmdb(session, api_key):
        ids_keywords |= descubrir_tmdb(session, api_key,
                                       {"with_keywords": kw_id},
                                       TMDB_PAGINAS_KEYWORD)
    ids |= ids_keywords
    print(f"  {len(ids)} películas tras añadir keywords de IA/tecnología.")

    ids_canonicas: set[int] = set()
    for titulo, anio in PELICULAS_CANONICAS:
        mid = buscar_pelicula_tmdb(session, api_key, titulo, anio)
        if mid:
            ids_canonicas.add(mid)
    ids |= ids_canonicas
    print(f"  {len(ids)} películas tras añadir las {len(ids_canonicas)} "
          f"canónicas encontradas.")

    # 2) Pedir detalles (limitado a TMDB_MAX_DETALLES).
    ids = list(ids)[:TMDB_MAX_DETALLES]
    peliculas: list[dict] = []
    for mid in tqdm(ids, desc="    Detalles TMDB", ncols=80):
        try:
            d = detalles_pelicula_tmdb(session, api_key, mid)
        except requests.HTTPError as exc:
            print(f"\n    [aviso] no se pudo obtener detalles de {mid}: {exc}")
            continue
        generos = [g["name"] for g in d.get("genres", [])]
        keywords = [k["name"] for k in (d.get("keywords") or {}).get("keywords", [])]
        peliculas.append({
            "tmdb_id": d.get("id"),
            "imdb_id": d.get("imdb_id"),
            "titulo": d.get("title"),
            "titulo_original": d.get("original_title"),
            "anio": (d.get("release_date") or "")[:4],
            "duracion_min": d.get("runtime"),
            "generos": json.dumps(generos, ensure_ascii=False),
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "sinopsis": d.get("overview"),
            "tagline": d.get("tagline"),
            "rating_tmdb": d.get("vote_average"),
            "votos_tmdb": d.get("vote_count"),
            "popularidad": d.get("popularity"),
            "relevante_ia": es_relevante_ia(d.get("title"), keywords),
        })

    df_pelis = pd.DataFrame(peliculas).drop_duplicates(subset="tmdb_id")

    # 3) Seleccionar películas a las que pedir reseñas: canónicas + relevantes
    #    + las más votadas, sin superar TMDB_MAX_REVIEWS.
    def prioridad(row):
        return (
            0 if row["tmdb_id"] in ids_canonicas else
            1 if row["relevante_ia"] else
            2
        )

    df_pelis = df_pelis.copy()
    df_pelis["_prio"] = df_pelis.apply(prioridad, axis=1)
    df_rev_cand = (
        df_pelis.sort_values(["_prio", "votos_tmdb"], ascending=[True, False])
        .head(TMDB_MAX_REVIEWS)
    )

    print(f"  Pidiendo reseñas de {len(df_rev_cand)} películas ...")
    reseñas: list[dict] = []
    for _, fila in tqdm(df_rev_cand.iterrows(), total=len(df_rev_cand),
                        desc="    Reseñas TMDB", ncols=80):
        mid = int(fila["tmdb_id"])
        try:
            for rev in reviews_pelicula_tmdb(session, api_key, mid):
                autor = rev.get("author_details") or {}
                reseñas.append({
                    "tmdb_id": mid,
                    "imdb_id": fila["imdb_id"],
                    "titulo": fila["titulo"],
                    "autor": rev.get("author"),
                    "rating_autor": autor.get("rating"),
                    "fecha": rev.get("created_at", "")[:10],
                    "contenido": rev.get("content"),
                })
        except requests.HTTPError as exc:
            print(f"\n    [aviso] no se pudieron obtener reseñas de {mid}: {exc}")

    df_reviews = pd.DataFrame(reseñas)
    df_pelis = df_pelis.drop(columns=["_prio"])

    return df_pelis, df_reviews


def guardar_tmdb(pelis: pd.DataFrame, reviews: pd.DataFrame) -> None:
    """Guarda películas y reseñas de TMDB en CSV."""
    pelis.to_csv(F_TMDB_PELICULAS, index=False, encoding="utf-8")
    reviews.to_csv(F_TMDB_REVIEWS, index=False, encoding="utf-8")
    print(f"  -> {len(pelis)} películas guardadas en "
          f"{F_TMDB_PELICULAS.relative_to(RAIZ)}")
    print(f"  -> {len(reviews)} reseñas guardadas en "
          f"{F_TMDB_REVIEWS.relative_to(RAIZ)}")


# --------------------------------------------------------------------------- #
# Parte 3: Cruce IMDB <-> TMDB
# --------------------------------------------------------------------------- #

def construir_cruce(pelis_imdb: pd.DataFrame,
                    pelis_tmdb: pd.DataFrame | None) -> pd.DataFrame:
    """Enlaza IMDb y TMDB por imdb_id/tconst y construye el fichero conjunto."""
    print("\n[3/4] Construyendo el fichero de cruce IMDB <-> TMDB ...")

    if pelis_tmdb is None or pelis_tmdb.empty:
        # Sin TMDB: el cruce se limita a la parte IMDb, con columnas TMDB vacías.
        cruce = pelis_imdb.copy()
        for col in ["tmdb_id", "titulo_tmdb", "generos_tmdb", "keywords",
                    "sinopsis", "tagline", "rating_tmdb", "votos_tmdb",
                    "duracion_tmdb", "relevante_ia"]:
            cruce[col] = pd.NA
        cruce["origen"] = "imdb"
        print("  [aviso] Sin datos de TMDB: el cruce solo contiene la parte IMDb.")
        return cruce

    imdb = pelis_imdb.rename(columns={
        "titulo": "titulo_imdb",
        "titulo_original": "titulo_original_imdb",
        "anio": "anio_imdb",
        "duracion_min": "duracion_imdb",
        "generos": "generos_imdb",
    })
    tmdb = pelis_tmdb.rename(columns={
        "titulo": "titulo_tmdb",
        "titulo_original": "titulo_original_tmdb",
        "anio": "anio_tmdb",
        "duracion_min": "duracion_tmdb",
        "generos": "generos_tmdb",
    })

    # Cruce por el identificador de IMDb (tconst == imdb_id).
    cruce = imdb.merge(tmdb, left_on="imdb_id", right_on="imdb_id",
                       how="outer", indicator=True, suffixes=("", "_tmdb"))

    # Normalizar: un único imdb_id, título y año preferidos de TMDB.
    cruce["titulo"] = cruce["titulo_tmdb"].fillna(cruce["titulo_imdb"])
    cruce["anio"] = cruce["anio_tmdb"].fillna(cruce["anio_imdb"])

    cruce["origen"] = cruce["_merge"].map({
        "both": "imdb+tmdb",
        "left_only": "imdb",
        "right_only": "tmdb",
    })
    cruce = cruce.drop(columns=["_merge"])

    columnas = [
        "imdb_id", "tmdb_id", "titulo", "titulo_imdb", "titulo_tmdb",
        "titulo_original_imdb", "titulo_original_tmdb", "anio", "anio_imdb",
        "anio_tmdb", "duracion_imdb", "duracion_tmdb", "generos_imdb",
        "generos_tmdb", "keywords", "rating_imdb", "votos_imdb", "rating_tmdb",
        "votos_tmdb", "sinopsis", "tagline", "relevante_ia", "origen",
    ]
    cruce = cruce[[c for c in columnas if c in cruce.columns]]
    cruce = cruce.sort_values(["votos_imdb", "votos_tmdb"],
                              ascending=False, na_position="last")
    return cruce


def guardar_cruce(cruce: pd.DataFrame) -> None:
    """Guarda el fichero de cruce en CSV."""
    cruce.to_csv(F_CRUCE, index=False, encoding="utf-8")
    print(f"  -> {len(cruce)} registros guardados en {F_CRUCE.relative_to(RAIZ)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Ejecuta el pipeline completo de captación de datos."""
    print("=" * 70)
    print("PIPELINE DE CAPTACIÓN DE DATOS (cine ciencia ficción / IA)")
    print("=" * 70)
    asegurar_carpetas()

    # --- IMDb (siempre disponible, sin clave) ---
    pelis_imdb = procesar_imdb()
    guardar_imdb(pelis_imdb)

    # --- TMDB (opcional, requiere API key) ---
    api_key = cargar_api_key_tmdb()
    pelis_tmdb: pd.DataFrame | None = None
    reviews_tmdb = pd.DataFrame()

    if not api_key:
        print("\n[2/4] TMDB OMITIDO: no se ha encontrado la API key.")
        print("  Para habilitarlo:")
        print("    1. Crea una cuenta y una API key gratuita en "
              "https://www.themoviedb.org/settings/api")
        print("    2. Guarda la clave en un fichero .env (ver .env.example) "
              "o en la variable de entorno TMDB_API_KEY.")
        print("    3. Vuelve a ejecutar: python cargar_datos.py")
    else:
        pelis_tmdb, reviews_tmdb = descargar_tmdb(api_key)
        guardar_tmdb(pelis_tmdb, reviews_tmdb)

    # --- Cruce ---
    cruce = construir_cruce(pelis_imdb, pelis_tmdb)
    guardar_cruce(cruce)

    # --- Resumen final ---
    print("\n[4/4] Resumen de la captación:")
    print(f"  IMDb  películas: {len(pelis_imdb)}")
    if pelis_tmdb is not None:
        print(f"  TMDB  películas: {len(pelis_tmdb)}")
        print(f"  TMDB  reseñas:   {len(reviews_tmdb)}")
    else:
        print("  TMDB  películas: 0 (pendiente de añadir la API key)")
    print(f"  Cruce registros:  {len(cruce)}")
    print(f"\nTodos los ficheros están en '{CARPETA_DATOS.relative_to(RAIZ)}'.")
    print("Listo.")


if __name__ == "__main__":
    main()
