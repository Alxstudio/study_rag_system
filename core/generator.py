"""
generator.py — Llama a la API de Groq para redactar preguntas de test.

Por qué existe este módulo:
Esta es la ÚNICA pieza de todo el proyecto que llama a un servicio
externo. Todo lo anterior (chunking, embeddings, vector_store, retriever)
es 100% tuyo y corre en tu ordenador. Aquí, y solo aquí, le pasamos a un
LLM (a través de Groq) el contexto que TU pipeline ya decidió que es
relevante, y le pedimos que redacte preguntas tipo test basándose
exclusivamente en ese contexto — el LLM no decide qué es importante,
eso ya lo hizo retriever.py.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()  # lee el archivo .env y carga GROQ_API_KEY como variable de entorno

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELO = "llama-3.3-70b-versatile"


def _construir_prompt(contexto: str, nombre_dominio: str, num_preguntas: int) -> str:
    """Construye el prompt que le enviamos al LLM.

    - Le decimos explícitamente que se base SOLO en el contexto dado,
      para reducir que invente cosas fuera de tus apuntes.
    - Pedimos JSON con una forma muy concreta, para poder parsearlo
      con json.loads() sin sorpresas.
    - Incluimos el nombre del dominio para que use terminología acorde.
    """
    return f"""Eres un generador de preguntas de examen tipo test para la certificación
que se está estudiando. Tu tarea es crear {num_preguntas} preguntas de opción
múltiple (4 opciones, solo 1 correcta) basadas EXCLUSIVAMENTE en el
siguiente contexto. No inventes información que no esté en el contexto.

Dominio del examen: {nombre_dominio}

Contexto (apuntes reales del usuario):
\"\"\"
{contexto}
\"\"\"

Devuelve ÚNICAMENTE un JSON válido con esta forma exacta, sin texto
adicional antes ni después:

{{
  "preguntas": [
    {{
      "pregunta": "texto de la pregunta",
      "opciones": ["opción A", "opción B", "opción C", "opción D"],
      "respuesta_correcta": "opción A"
    }}
  ]
}}

El valor de "respuesta_correcta" debe ser idéntico (carácter a carácter)
a una de las cadenas de "opciones"."""


def generar_preguntas(chunks_contexto: list[str], nombre_dominio: str, num_preguntas: int = 5) -> list[dict]:
    """Genera preguntas de test a partir de una lista de chunks de texto.

    chunks_contexto: lista de textos (el campo "texto" de los resultados
        de retriever.retrieve_por_dominio()).

    Devuelve: [{"pregunta": str, "opciones": list[str], "respuesta_correcta": str}, ...]

    Lanza RuntimeError si la API falla o la respuesta no es JSON válido,
    en vez de fallar en silencio — así app.py puede mostrar un error claro.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("No se encontró GROQ_API_KEY. Revisa tu archivo .env")

    contexto_unido = "\n\n---\n\n".join(chunks_contexto)
    prompt = _construir_prompt(contexto_unido, nombre_dominio, num_preguntas)

    respuesta = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODELO,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},  # fuerza JSON válido garantizado
            "temperature": 0.7,
        },
        timeout=30,
    )

    if respuesta.status_code != 200:
        raise RuntimeError(f"Error de la API de Groq ({respuesta.status_code}): {respuesta.text}")

    cuerpo = respuesta.json()
    texto_generado = cuerpo["choices"][0]["message"]["content"]

    try:
        datos = json.loads(texto_generado)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"La respuesta del LLM no era JSON válido: {texto_generado}") from error

    return datos["preguntas"]


if __name__ == "__main__":
    contexto_de_prueba = [
        "El OCR (reconocimiento óptico de caracteres) extrae texto legible desde imágenes o documentos escaneados.",
        "La detección de objetos devuelve coordenadas de un recuadro (bounding box) alrededor de cada objeto detectado en la imagen.",
    ]

    print("Generando preguntas con Groq...\n")
    preguntas = generar_preguntas(contexto_de_prueba, nombre_dominio="Azure AI Vision", num_preguntas=2)

    for i, p in enumerate(preguntas, start=1):
        print(f"{i}. {p['pregunta']}")
        for opcion in p["opciones"]:
            marca = " (correcta)" if opcion == p["respuesta_correcta"] else ""
            print(f"   - {opcion}{marca}")
        print()