"""
vector_store.py — Guarda vectores y busca los más parecidos a una consulta.

Por qué existe este módulo:
Ya sabemos convertir texto en vectores (embeddings.py). Ahora necesitamos
un sitio donde guardar TODOS esos vectores, y una forma de preguntar:
"de todos los que tengo guardados, ¿cuáles son los 5 más parecidos a
este vector nuevo?". Eso es literalmente lo que hace una búsqueda
semántica, y es lo que este módulo implementa desde cero.

Nota sobre escala: aquí hacemos "búsqueda por fuerza bruta" (comparamos
la consulta contra TODOS los vectores guardados, uno por uno). Para tu
caso de uso (apuntes de una certificación, como mucho unos cientos de
chunks) esto es instantáneo. Si algún día tuvieras millones de vectores
(no es tu caso), ahí sí haría falta una librería especializada como
FAISS. Aquí lo hacemos a mano precisamente para que entiendas qué hace
esa librería por debajo, en vez de usarla como caja negra.
"""

import numpy as np


class VectorStore:
    """Almacén de vectores en memoria, con búsqueda por similitud coseno."""

    def __init__(self):
        self._ids: list[int] = []
        self._vectores: np.ndarray | None = None

    def add(self, ids: list[int], vectores: np.ndarray) -> None:
        """Añade nuevos vectores al almacén.

        ids: identificadores (normalmente el id del chunk en la BD).
        vectores: array numpy (n, 384), ya normalizados.
        """
        if len(ids) != vectores.shape[0]:
            raise ValueError("El número de ids no coincide con el número de vectores")

        self._ids.extend(ids)

        if self._vectores is None:
            self._vectores = vectores
        else:
            self._vectores = np.vstack([self._vectores, vectores])

    def search(self, vector_consulta: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Devuelve los `top_k` ids más parecidos al vector de consulta.

        Como los vectores están normalizados, la similitud coseno es el
        producto escalar. Con una sola multiplicación de matrices
        (`self._vectores @ vector_consulta`) calculamos la similitud
        contra TODOS los vectores guardados de golpe, usando código
        optimizado de numpy en vez de un bucle for en Python puro.
        """
        if self._vectores is None or len(self._ids) == 0:
            return []

        similitudes = self._vectores @ vector_consulta
        indices_ordenados = np.argsort(similitudes)[::-1][:top_k]
        resultados = [(self._ids[i], float(similitudes[i])) for i in indices_ordenados]
        return resultados

    def __len__(self) -> int:
        return len(self._ids)


if __name__ == "__main__":
    from embeddings import embed_texts, embed_query

    chunks_de_ejemplo = [
        "El OCR extrae texto legible desde imágenes o documentos escaneados.",
        "La detección de objetos devuelve coordenadas de un recuadro alrededor de cada objeto.",
        "El curry de pollo es un plato tradicional con leche de coco y especias.",
        "Azure AI Language permite analizar sentimiento y extraer entidades de un texto.",
    ]
    ids_de_ejemplo = [101, 102, 103, 104]

    print("Generando embeddings de los chunks de ejemplo...")
    vectores = embed_texts(chunks_de_ejemplo)

    store = VectorStore()
    store.add(ids_de_ejemplo, vectores)
    print(f"Vector store con {len(store)} chunks guardados.\n")

    pregunta = "¿cómo se extrae texto de una imagen escaneada?"
    vector_pregunta = embed_query(pregunta)

    resultados = store.search(vector_pregunta, top_k=2)

    print(f"Pregunta: '{pregunta}'\n")
    print("Top 2 chunks más relevantes encontrados:")
    for id_chunk, similitud in resultados:
        texto_original = chunks_de_ejemplo[ids_de_ejemplo.index(id_chunk)]
        print(f"  id={id_chunk} | similitud={similitud:.3f} | \"{texto_original}\"")