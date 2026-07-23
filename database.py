"""
database.py — Todas las funciones que hablan con la base de datos SQLite.

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
    RUTA_DB.parent.mkdir(exist_ok=True)
    conexion = sqlite3.connect(RUTA_DB, check_same_thread=False)
    conexion.row_factory = sqlite3.Row
    return conexion


def init_db() -> None:
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


def crear_certificacion(nombre: str) -> int:
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
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("SELECT * FROM dominios WHERE certificacion_id = ?", (certificacion_id,))
    filas = cursor.fetchall()
    conexion.close()
    return [dict(fila) for fila in filas]


def guardar_contenido(dominio_id: int, titulo: str, texto: str) -> int:
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
        embedding = np.frombuffer(fila["embedding"], dtype=np.float32)
        resultado.append({"id": fila["id"], "texto": fila["texto"], "embedding": embedding})

    return resultado


def obtener_chunks_de_certificacion(certificacion_id: int) -> list[dict]:
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


def guardar_pregunta(chunk_id: int, pregunta: str, opciones: list[str], respuesta_correcta: str) -> int:
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
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("UPDATE preguntas SET fallada = 1 WHERE id = ?", (pregunta_id,))
    conexion.commit()
    conexion.close()


def marcar_pregunta_dominada(pregunta_id: int) -> None:
    """Quita la marca de 'fallada' de una pregunta, para sacarla de la
    lista de repaso una vez que el usuario ya se la sabe bien."""
    conexion = get_connection()
    cursor = conexion.cursor()
    cursor.execute("UPDATE preguntas SET fallada = 0 WHERE id = ?", (pregunta_id,))
    conexion.commit()
    conexion.close()


def obtener_preguntas_falladas(certificacion_id: int) -> list[dict]:
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