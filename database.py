"""
database.py — Todas las funciones que hablan con la base de datos SQLite.

Por qué existe este módulo:
Hasta ahora hemos probado chunking, embeddings y vector_store con listas
sueltas en memoria, que desaparecen en cuanto cierras el programa. Este
módulo es el que hace que todo se guarde de verdad, de forma permanente,
en un archivo (data/study.db) que sigue ahí la próxima vez que abras la app.

Diseño pensado para varias certificaciones (no solo AI-901):
    certificaciones
        └── dominios (con su % de peso en el examen real)
                └── contenidos (los apuntes que tú escribes)
                        └── chunks (los trozos, cada uno con su embedding guardado)
                                └── preguntas (los tests generados sobre ese chunk)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

import numpy as np

RUTA_DB = Path(__file__).parent / "data" / "study.db"


def get_connection() -> sqlite3.Connection:
    """Abre una conexión a la base de datos.

    check_same_thread=False es necesario porque Streamlit a veces ejecuta
    el código en hilos distintos; sin este flag, sqlite3 lanzaría un error
    de seguridad al reutilizar la conexión desde otro hilo.
    """
    RUTA_DB.parent.mkdir(exist_ok=True)
    conexion = sqlite3.connect(RUTA_DB, check_same_thread=False)
    conexion.row_factory = sqlite3.Row  # permite leer filas como diccionarios
    return conexion


def init_db() -> None:
    """Crea todas las tablas si no existen todavía."""
    conexion = get_connection()
    cursor = conexion.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS certificaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dominios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            certificacion_id INTEGER NOT NULL,
            nombre TEXT NOT NULL,
            peso_examen REAL NOT NULL,
            FOREIGN KEY (certificacion_id) REFERENCES certificaciones(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contenidos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dominio_id INTEGER NOT NULL,
            titulo TEXT NOT NULL,
            texto TEXT NOT NULL,
            fecha TEXT NOT NULL,
            FOREIGN KEY (dominio_id) REFERENCES dominios(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contenido_id INTEGER NOT NULL,
            texto TEXT NOT NULL,
            indice INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (contenido_id) REFERENCES contenidos(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preguntas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            pregunta TEXT NOT NULL,
            opciones TEXT NOT NULL,
            respuesta_correcta TEXT NOT NULL,
            fallada INTEGER NOT NULL DEFAULT 0,
            fecha TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        )
    """)

    conexion.commit()
    conexion.close()


# --- Certificaciones y dominios -------------------------------------------

def crear_certificacion(nombre: str) -> int:
    """Crea una certificación nueva, o devuelve el id de la ya existente
    con ese nombre (para no duplicar si la app se reinicia)."""
    conexion = get_connection()
    cursor = conexion.cursor()

    cursor.execute("SELECT id FROM certificaciones WHERE nombre = ?", (nombre,))
    fila = cursor.fetchone()
    if fila is not None:
        conexion.close()
        return fila["id"]

    cursor.execute("INSERT INTO certificaciones (nombre) VALUES (?)", (nombre,))
    conexion.commit()
    nuevo_id = cursor.lastrowid
    conexion.close()
    return nuevo_id


def crear_dominio(certificacion_id: int, nombre: str, peso_examen: float) -> int:
    """Crea un dominio con su peso real en el examen (ej. 0.55 = 55%).
    Ese peso lo usaremos para repartir cuántas preguntas de cada dominio
    entran en el test final simulado."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute(
        "INSERT INTO dominios (certificacion_id, nombre, peso_examen) VALUES (?, ?, ?)",
        (certificacion_id, nombre, peso_examen),
    )
    conexion.commit()
    nuevo_id = cursor.lastrowid
    conexion.close()
    return nuevo_id


def obtener_dominios(certificacion_id: int) -> list[dict]:
    """Devuelve todos los dominios de una certificación."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("SELECT * FROM dominios WHERE certificacion_id = ?", (certificacion_id,))
    filas = cursor.fetchall()
    conexion.close()
    return [dict(fila) for fila in filas]


# --- Contenidos y chunks ----------------------------------------------------

def guardar_contenido(dominio_id: int, titulo: str, texto: str) -> int:
    """Guarda un apunte completo (antes de trocearlo) y devuelve su id."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute(
        "INSERT INTO contenidos (dominio_id, titulo, texto, fecha) VALUES (?, ?, ?, ?)",
        (dominio_id, titulo, texto, datetime.now().isoformat()),
    )
    conexion.commit()
    nuevo_id = cursor.lastrowid
    conexion.close()
    return nuevo_id


def guardar_chunk(contenido_id: int, texto: str, indice: int, embedding: np.ndarray) -> int:
    """Guarda un chunk ya troceado junto a su embedding.

    SQLite no tiene un tipo de dato nativo para "array de decimales".
    La columna `embedding` es BLOB (binario genérico). Convertimos el
    array de numpy a bytes puros con .astype(np.float32).tobytes() antes
    de guardarlo, y hacemos el proceso inverso al leerlo. Forzamos
    float32 (en vez de float64, el tipo por defecto de numpy) para
    ocupar la mitad de espacio sin perder precisión relevante.
    """
    embedding_bytes = embedding.astype(np.float32).tobytes()

    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute(
        "INSERT INTO chunks (contenido_id, texto, indice, embedding) VALUES (?, ?, ?, ?)",
        (contenido_id, texto, indice, embedding_bytes),
    )
    conexion.commit()
    nuevo_id = cursor.lastrowid
    conexion.close()
    return nuevo_id


def obtener_chunks_de_dominio(dominio_id: int) -> list[dict]:
    """Devuelve todos los chunks de un dominio, con su embedding ya
    reconstruido como array de numpy (no como bytes en crudo).
    Cada elemento: {"id": int, "texto": str, "embedding": np.ndarray (384,)}
    """
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("""
        SELECT chunks.id, chunks.texto, chunks.embedding
        FROM chunks
        JOIN contenidos ON chunks.contenido_id = contenidos.id
        WHERE contenidos.dominio_id = ?
    """, (dominio_id,))
    filas = cursor.fetchall()
    conexion.close()

    resultado = []
    for fila in filas:
        # np.frombuffer reconstruye el array a partir de los bytes guardados.
        # El dtype tiene que coincidir exactamente con el usado al guardar.
        embedding = np.frombuffer(fila["embedding"], dtype=np.float32)
        resultado.append({"id": fila["id"], "texto": fila["texto"], "embedding": embedding})

    return resultado


def obtener_chunks_de_certificacion(certificacion_id: int) -> list[dict]:
    """Igual que obtener_chunks_de_dominio, pero de TODOS los dominios de
    una certificación a la vez. Esta es la función que usaremos para el
    test final simulado, que mezcla contenido de toda la certificación."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("""
        SELECT chunks.id, chunks.texto, chunks.embedding, dominios.id AS dominio_id
        FROM chunks
        JOIN contenidos ON chunks.contenido_id = contenidos.id
        JOIN dominios ON contenidos.dominio_id = dominios.id
        WHERE dominios.certificacion_id = ?
    """, (certificacion_id,))
    filas = cursor.fetchall()
    conexion.close()

    resultado = []
    for fila in filas:
        embedding = np.frombuffer(fila["embedding"], dtype=np.float32)
        resultado.append({
            "id": fila["id"],
            "texto": fila["texto"],
            "embedding": embedding,
            "dominio_id": fila["dominio_id"],
        })

    return resultado


# --- Preguntas generadas -----------------------------------------------------

def guardar_pregunta(chunk_id: int, pregunta: str, opciones: list[str], respuesta_correcta: str) -> int:
    """Guarda una pregunta generada por la IA.

    `opciones` se guarda como texto JSON porque SQLite no tiene un tipo
    de columna para listas — lo serializamos con json.dumps al guardar,
    y json.loads al leerlo."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute(
        """INSERT INTO preguntas (chunk_id, pregunta, opciones, respuesta_correcta, fallada, fecha)
           VALUES (?, ?, ?, ?, 0, ?)""",
        (chunk_id, pregunta, json.dumps(opciones), respuesta_correcta, datetime.now().isoformat()),
    )
    conexion.commit()
    nuevo_id = cursor.lastrowid
    conexion.close()
    return nuevo_id


def marcar_pregunta_fallada(pregunta_id: int) -> None:
    """Marca una pregunta como fallada, para que aparezca luego en el repaso."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("UPDATE preguntas SET fallada = 1 WHERE id = ?", (pregunta_id,))
    conexion.commit()
    conexion.close()


def obtener_preguntas_falladas(certificacion_id: int) -> list[dict]:
    """Devuelve todas las preguntas marcadas como falladas de una
    certificación, para la pantalla de repaso."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("""
        SELECT preguntas.*
        FROM preguntas
        JOIN chunks ON preguntas.chunk_id = chunks.id
        JOIN contenidos ON chunks.contenido_id = contenidos.id
        JOIN dominios ON contenidos.dominio_id = dominios.id
        WHERE dominios.certificacion_id = ? AND preguntas.fallada = 1
    """, (certificacion_id,))
    filas = cursor.fetchall()
    conexion.close()

    resultado = []
    for fila in filas:
        item = dict(fila)
        item["opciones"] = json.loads(item["opciones"])
        resultado.append(item)
    return resultado


if __name__ == "__main__":
    # Prueba end-to-end: crea la certificación AI-901 con un dominio,
    # guarda un contenido, lo trocea, genera embeddings y los guarda,
    # y los vuelve a leer para comprobar que todo cuadra.
    from core.chunking import chunk_text
    from core.embeddings import embed_texts

    print("Inicializando base de datos...")
    init_db()

    cert_id = crear_certificacion("AI-901")
    dominio_id = crear_dominio(cert_id, "Implementar soluciones de IA con Foundry", peso_examen=0.575)
    print(f"Certificación id={cert_id}, dominio id={dominio_id}")

    texto_apunte = """Azure AI Vision permite analizar imágenes con capacidades de
OCR, clasificación y detección de objetos.

El OCR extrae texto legible desde imágenes o documentos escaneados."""

    contenido_id = guardar_contenido(dominio_id, titulo="Azure AI Vision - intro", texto=texto_apunte)
    print(f"Contenido guardado con id={contenido_id}")

    trozos = chunk_text(texto_apunte, tamano_maximo_palabras=30, solape_palabras=5)
    textos_trozos = [t.texto for t in trozos]
    embeddings_trozos = embed_texts(textos_trozos)

    for trozo, embedding in zip(trozos, embeddings_trozos):
        chunk_id = guardar_chunk(contenido_id, trozo.texto, trozo.indice, embedding)
        print(f"  Chunk {trozo.indice} guardado con id={chunk_id}")

    print("\nReleyendo los chunks del dominio desde la base de datos...")
    chunks_leidos = obtener_chunks_de_dominio(dominio_id)
    for chunk in chunks_leidos:
        print(f"  id={chunk['id']} | embedding shape={chunk['embedding'].shape} | texto=\"{chunk['texto'][:50]}...\"")