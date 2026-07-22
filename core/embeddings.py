"""
embeddings.py — Convierte texto en vectores numéricos (embeddings).

Por qué existe este módulo:
Un ordenador no puede comparar directamente el "significado" de dos frases,
pero sí puede comparar números. Este módulo usa un modelo de IA (que corre
en tu propio ordenador, sin llamar a ninguna API externa) para convertir
cada chunk de texto en una lista de ~384 números que representan su
significado. Frases con significado parecido generan vectores parecidos,
aunque no compartan ni una sola palabra.

Este es el único módulo que "sabe" de modelos de IA de bajo nivel;
el resto del proyecto solo llama a las funciones de aquí.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Nombre del modelo que vamos a usar. "all-MiniLM-L6-v2" es un modelo
# pequeño (unos 80MB) y rápido, pensado exactamente para esto: generar
# embeddings de frases cortas/medias. No es el modelo más potente que
# existe, pero es el estándar para empezar en RAG: ligero, gratis, corre
# en CPU sin problema (no necesitas tarjeta gráfica).
NOMBRE_MODELO = "all-MiniLM-L6-v2"

# Variable a nivel de módulo para guardar el modelo ya cargado.
# Por qué: cargar el modelo desde disco tarda unos segundos. Si lo
# cargáramos cada vez que se llama a embed_texts(), la app sería lentísima.
# Cargándolo UNA vez y reutilizándolo, solo pagamos ese coste al arrancar.
_modelo: SentenceTransformer | None = None


def _obtener_modelo() -> SentenceTransformer:
    """Carga el modelo la primera vez que se necesita, y lo reutiliza después."""
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(NOMBRE_MODELO)
    return _modelo


def embed_texts(textos: list[str]) -> np.ndarray:
    """Convierte una lista de textos en su matriz de embeddings.

    Devuelve un array de numpy de forma (n_textos, 384) — una fila por
    cada texto de entrada.

    normalize_embeddings=True: ajusta cada vector para que su longitud
    sea exactamente 1. Así, comparar dos vectores por similitud coseno
    se reduce a un simple producto escalar en similitud_coseno().
    """
    modelo = _obtener_modelo()
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    return embeddings


def embed_query(texto: str) -> np.ndarray:
    """Convierte un único texto (ej. la búsqueda del usuario) en su embedding."""
    return embed_texts([texto])[0]


def similitud_coseno(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calcula qué tan parecidos son dos vectores (entre -1 y 1).

    1.0 = significan lo mismo, 0.0 = sin relación, -1.0 = opuestos.
    Como los vectores ya vienen normalizados, es solo un producto escalar.
    """
    return float(np.dot(vector_a, vector_b))


if __name__ == "__main__":
    frases = [
        "OCR extrae texto de imágenes escaneadas",
        "El reconocimiento óptico de caracteres lee texto en fotos",
        "El curry de pollo lleva leche de coco",
    ]

    print("Generando embeddings (la primera vez tarda un poco, descarga el modelo)...")
    vectores = embed_texts(frases)

    print(f"\nCada embedding tiene {vectores.shape[1]} números.\n")

    sim_ocr = similitud_coseno(vectores[0], vectores[1])
    sim_curry = similitud_coseno(vectores[0], vectores[2])

    print(f"Similitud entre las dos frases de OCR: {sim_ocr:.3f}  (debería ser ALTA, >0.5)")
    print(f"Similitud entre OCR y la frase del curry: {sim_curry:.3f}  (debería ser BAJA, <0.3)")