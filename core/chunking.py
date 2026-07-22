"""
chunking.py — Trocea texto largo en fragmentos pequeños (chunks).

Por qué existe este módulo:
Cuando guardas un apunte de estudio, puede tener 2000 palabras hablando
de 4 subtemas distintos. Si luego buscamos "OCR en Azure AI Vision",
no queremos que la IA reciba las 2000 palabras enteras: queremos
solo el trozo de 100-150 palabras que habla específicamente de OCR.

Este módulo se encarga de esa primera fase: partir el texto en trozos
manejables ANTES de convertirlos en vectores (eso lo hace embeddings.py).
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """Representa un fragmento de texto ya troceado.

    Usamos una dataclass en vez de un diccionario suelto porque así,
    en el resto del proyecto, siempre sabemos qué forma tiene un chunk
    (texto + posición) sin tener que recordar claves de diccionario.
    """
    texto: str
    indice: int  # posición del chunk dentro del documento original (0, 1, 2...)


def _dividir_en_parrafos(texto: str) -> list[str]:
    """Divide el texto en párrafos usando líneas en blanco como separador.

    Decisión de diseño: partimos primero por párrafos (no por número fijo
    de caracteres) porque un párrafo casi siempre es una unidad de sentido
    completa. Cortar a lo bruto cada 500 caracteres podría partir una frase
    por la mitad, y eso rompe el significado cuando luego generamos el
    embedding de ese trozo.
    """
    parrafos_crudos = texto.split("\n\n")
    parrafos = [p.strip() for p in parrafos_crudos if p.strip()]
    return parrafos


def chunk_text(texto: str, tamano_maximo_palabras: int = 200, solape_palabras: int = 30) -> list[Chunk]:
    """Trocea un texto en chunks de tamaño similar, con solape entre ellos.

    Parámetros:
        texto: el apunte completo que ha escrito el usuario.
        tamano_maximo_palabras: cuántas palabras como máximo debe tener
            cada chunk. 200 es un punto de partida razonable para apuntes
            de estudio: suficiente contexto, pero no tanto como para diluir
            el significado cuando se convierta en un único vector.
        solape_palabras: cuántas palabras del final de un chunk se repiten
            al principio del siguiente.

    Por qué existe el solape:
        Imagina que una idea empieza en la última frase del chunk 3 y
        termina en la primera frase del chunk 4. Sin solape, ninguno de
        los dos chunks contendría la idea completa. Repitiendo las últimas
        ~30 palabras de un chunk al principio del siguiente, nos aseguramos
        de que las ideas que quedan "a caballo" entre dos chunks no se
        pierdan.

    Devuelve:
        Una lista de objetos Chunk, cada uno con su texto y su posición.
    """
    parrafos = _dividir_en_parrafos(texto)

    chunks: list[Chunk] = []
    palabras_chunk_actual: list[str] = []

    for parrafo in parrafos:
        palabras_parrafo = parrafo.split()

        if palabras_chunk_actual and len(palabras_chunk_actual) + len(palabras_parrafo) > tamano_maximo_palabras:
            texto_chunk = " ".join(palabras_chunk_actual)
            chunks.append(Chunk(texto=texto_chunk, indice=len(chunks)))

            palabras_solape = palabras_chunk_actual[-solape_palabras:] if solape_palabras > 0 else []
            palabras_chunk_actual = palabras_solape + palabras_parrafo
        else:
            palabras_chunk_actual.extend(palabras_parrafo)

    if palabras_chunk_actual:
        texto_chunk = " ".join(palabras_chunk_actual)
        chunks.append(Chunk(texto=texto_chunk, indice=len(chunks)))

    return chunks


if __name__ == "__main__":
    texto_de_prueba = """Azure AI Vision es un servicio que permite analizar imágenes.
Incluye capacidades de clasificación de imágenes, detección de objetos y OCR.

El OCR (reconocimiento óptico de caracteres) extrae texto legible desde
imágenes o documentos escaneados. Es útil para digitalizar formularios,
facturas o carteles.

La detección de objetos, a diferencia de la clasificación, no solo dice
qué hay en la imagen sino también dónde está, devolviendo coordenadas
de un recuadro (bounding box) alrededor de cada objeto detectado."""

    resultado = chunk_text(texto_de_prueba, tamano_maximo_palabras=40, solape_palabras=10)

    print(f"Se generaron {len(resultado)} chunks:\n")
    for chunk in resultado:
        print(f"--- Chunk {chunk.indice} ({len(chunk.texto.split())} palabras) ---")
        print(chunk.texto)
        print()