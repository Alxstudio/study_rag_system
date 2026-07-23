"""
retriever.py — Une database.py + embeddings.py + vector_store.py.

Por qué existe este módulo:
Ya tenemos las cuatro piezas sueltas: trocear texto, convertirlo en
vectores, buscar por similitud, y guardar todo en SQLite. Este módulo
es el "pegamento" que las conecta en un único flujo con sentido:

    "Dame los N chunks más relevantes de este dominio (o de toda la
    certificación) para responder a esta pregunta/tema."

No inventa lógica nueva — solo orquesta llamadas a los módulos que ya
construimos y probamos por separado.

Nota sobre rendimiento: cada llamada reconstruye el VectorStore desde
cero leyendo de la base de datos. Para tu volumen de datos (cientos de
chunks como mucho) esto tarda milisegundos. Si este proyecto creciera
mucho, ahí sí tendría sentido cachear el VectorStore en memoria — pero
añadir esa complejidad ahora sería optimizar algo que no es un problema real.
"""

from core.embeddings import embed_query
from core.vector_store import VectorStore
from database import obtener_chunks_de_dominio, obtener_chunks_de_certificacion


def _construir_vector_store(chunks: list[dict]) -> VectorStore:
    """Toma una lista de chunks (con 'id' y 'embedding' ya reconstruido)
    y los carga en un VectorStore nuevo, listo para buscar."""
    store = VectorStore()

    if not chunks:
        return store  # vector store vacío; search() ya sabe devolver [] en ese caso

    ids = [chunk["id"] for chunk in chunks]
    import numpy as np
    vectores = np.stack([chunk["embedding"] for chunk in chunks])

    store.add(ids, vectores)
    return store


def retrieve_por_dominio(pregunta: str, dominio_id: int, top_k: int = 5) -> list[dict]:
    """Busca los chunks más relevantes dentro de UN dominio concreto.

    Uso típico: el usuario ha seleccionado un dominio/tema en la app y
    pide "generar test sobre esto" — solo queremos contenido de ese
    dominio, no mezclado con otros.

    Devuelve una lista ordenada de más a menos relevante:
        [{"id": int, "texto": str, "similitud": float}, ...]
    """
    chunks = obtener_chunks_de_dominio(dominio_id)
    store = _construir_vector_store(chunks)

    vector_pregunta = embed_query(pregunta)
    resultados_busqueda = store.search(vector_pregunta, top_k=top_k)

    texto_por_id = {chunk["id"]: chunk["texto"] for chunk in chunks}

    return [
        {"id": id_chunk, "texto": texto_por_id[id_chunk], "similitud": similitud}
        for id_chunk, similitud in resultados_busqueda
    ]


def retrieve_por_certificacion(pregunta: str, certificacion_id: int, top_k: int = 5) -> list[dict]:
    """Igual que retrieve_por_dominio, pero buscando entre TODOS los
    chunks de una certificación, sin importar de qué dominio vengan.

    Uso típico: el test final simulado, que mezcla contenido de toda
    la certificación en vez de un único dominio."""
    chunks = obtener_chunks_de_certificacion(certificacion_id)
    store = _construir_vector_store(chunks)

    vector_pregunta = embed_query(pregunta)
    resultados_busqueda = store.search(vector_pregunta, top_k=top_k)

    info_por_id = {chunk["id"]: chunk for chunk in chunks}

    return [
        {
            "id": id_chunk,
            "texto": info_por_id[id_chunk]["texto"],
            "dominio_id": info_por_id[id_chunk]["dominio_id"],
            "similitud": similitud,
        }
        for id_chunk, similitud in resultados_busqueda
    ]


if __name__ == "__main__":
    from database import crear_certificacion, obtener_dominios

    cert_id = crear_certificacion("AI-901")  # devuelve el id ya existente, no duplica
    dominios = obtener_dominios(cert_id)

    if not dominios:
        print("No hay dominios guardados todavía — ejecuta primero 'python database.py'")
    else:
        dominio_id = dominios[0]["id"]
        pregunta = "¿cómo se extrae texto de una imagen?"

        print(f"Buscando chunks relevantes para: '{pregunta}'\n")
        resultados = retrieve_por_dominio(pregunta, dominio_id, top_k=3)

        if not resultados:
            print("No se encontraron chunks. ¿Has guardado contenido en este dominio?")
        else:
            for resultado in resultados:
                print(f"  similitud={resultado['similitud']:.3f} | \"{resultado['texto'][:70]}...\"")