"""
app.py — Interfaz en Streamlit que conecta todo el motor RAG.
"""

import streamlit as st

from database import (
    init_db,
    crear_certificacion,
    crear_dominio,
    obtener_dominios,
    guardar_contenido,
    guardar_chunk,
    obtener_chunks_de_dominio,
    guardar_pregunta,
    marcar_pregunta_fallada,
    obtener_preguntas_falladas,
    marcar_pregunta_dominada,
)
from core.chunking import chunk_text
from core.embeddings import embed_texts
from core.generator import generar_preguntas

st.set_page_config(page_title="Study RAG System", layout="wide")

init_db()

st.title("📚 Study RAG System")
st.caption("Genera tests de certificación a partir de tus propios apuntes, con un motor RAG construido desde cero.")


if "certificacion_id" not in st.session_state:
    st.session_state.certificacion_id = None
if "dominio_id" not in st.session_state:
    st.session_state.dominio_id = None


with st.sidebar:
    st.header("Certificación activa")

    nombre_certificacion = st.text_input(
        "Nombre de la certificación",
        value="AI-901",
        help="Si ya existe, la selecciona. Si es nueva, la crea.",
    )

    if st.button("Seleccionar / crear certificación"):
        st.session_state.certificacion_id = crear_certificacion(nombre_certificacion)
        st.session_state.dominio_id = None
        st.rerun()

    if st.session_state.certificacion_id is not None:
        st.success(f"Certificación activa: {nombre_certificacion} (id={st.session_state.certificacion_id})")

        st.divider()
        st.subheader("Dominios de esta certificación")

        dominios = obtener_dominios(st.session_state.certificacion_id)

        if dominios:
            opciones_dominio = {
                f"{d['nombre']} ({d['peso_examen']*100:.0f}% del examen)": d["id"]
                for d in dominios
            }
            etiqueta_elegida = st.selectbox("Dominio activo", options=list(opciones_dominio.keys()))
            st.session_state.dominio_id = opciones_dominio[etiqueta_elegida]
        else:
            st.info("Esta certificación todavía no tiene dominios. Añade el primero abajo.")

        st.divider()
        st.subheader("Añadir nuevo dominio")

        nombre_dominio_nuevo = st.text_input("Nombre del dominio", key="input_nombre_dominio")
        peso_dominio_nuevo = st.slider(
            "Peso en el examen (%)",
            min_value=0, max_value=100, value=50,
            key="input_peso_dominio",
            help="El % que representa este dominio en el examen real. Se usará para repartir preguntas en el test final simulado.",
        )

        if st.button("Añadir dominio"):
            if nombre_dominio_nuevo.strip():
                crear_dominio(
                    st.session_state.certificacion_id,
                    nombre_dominio_nuevo.strip(),
                    peso_examen=peso_dominio_nuevo / 100,
                )
                st.rerun()
            else:
                st.warning("Escribe un nombre para el dominio antes de añadirlo.")


if st.session_state.certificacion_id is None:
    st.info("👈 Selecciona o crea una certificación en la barra lateral para empezar.")
else:
    tab_guardar, tab_test, tab_repaso = st.tabs(["📥 Guardar contenido", "📝 Generar test", "🔁 Repasar fallos"])

    with tab_guardar:
        st.subheader("Guardar un apunte nuevo")

        titulo_apunte = st.text_input("Título del apunte", placeholder="Ej. Azure AI Vision - OCR y detección de objetos")
        texto_apunte = st.text_area(
            "Contenido",
            height=250,
            placeholder="Pega aquí tu apunte de estudio. Sepáralo en párrafos (línea en blanco entre ideas) para que el troceo respete el sentido de cada una.",
        )

        if st.button("💾 Guardar y procesar"):
            if not titulo_apunte.strip() or not texto_apunte.strip():
                st.warning("Necesitas un título y contenido antes de guardar.")
            else:
                contenido_id = guardar_contenido(
                    st.session_state.dominio_id, titulo_apunte.strip(), texto_apunte.strip()
                )

                with st.spinner("Troceando el contenido..."):
                    trozos = chunk_text(texto_apunte.strip())

                with st.spinner(f"Generando embeddings de {len(trozos)} fragmentos..."):
                    textos_trozos = [t.texto for t in trozos]
                    embeddings_trozos = embed_texts(textos_trozos)

                for trozo, embedding in zip(trozos, embeddings_trozos):
                    guardar_chunk(contenido_id, trozo.texto, trozo.indice, embedding)

                st.success(f"✅ Apunte guardado y troceado en {len(trozos)} fragmentos.")
                st.rerun()

    with tab_test:
        st.subheader("Generar test sobre el dominio activo")

        nombre_dominio_activo = next(
            (d["nombre"] for d in dominios if d["id"] == st.session_state.dominio_id),
            "este dominio",
        )

        num_preguntas = st.slider("Número de preguntas", min_value=1, max_value=10, value=5)

        if st.button("🎲 Generar test"):
            chunks_del_dominio = obtener_chunks_de_dominio(st.session_state.dominio_id)

            if not chunks_del_dominio:
                st.warning("Este dominio todavía no tiene contenido guardado. Ve a la pestaña 'Guardar contenido' primero.")
            else:
                contexto = [c["texto"] for c in chunks_del_dominio]

                try:
                    with st.spinner("Generando preguntas con IA..."):
                        preguntas = generar_preguntas(contexto, nombre_dominio_activo, num_preguntas)

                    st.session_state.test_actual = {
                        "preguntas": preguntas,
                        "chunk_id_referencia": chunks_del_dominio[0]["id"],
                        "corregido": False,
                    }
                except RuntimeError as error:
                    st.error(f"No se pudo generar el test: {error}")

        if "test_actual" in st.session_state:
            test = st.session_state.test_actual
            st.divider()

            with st.form("form_test"):
                respuestas_usuario = []
                for i, pregunta in enumerate(test["preguntas"]):
                    st.markdown(f"**{i+1}. {pregunta['pregunta']}**")
                    respuesta = st.radio(
                        f"Opciones pregunta {i}",
                        options=pregunta["opciones"],
                        key=f"respuesta_{i}",
                        label_visibility="collapsed",
                        index=None,
                    )
                    respuestas_usuario.append(respuesta)

                enviado = st.form_submit_button("✅ Corregir test")

            if enviado:
                aciertos = 0
                for i, pregunta in enumerate(test["preguntas"]):
                    es_correcta = respuestas_usuario[i] == pregunta["respuesta_correcta"]
                    if es_correcta:
                        aciertos += 1

                    pregunta_id = guardar_pregunta(
                        test["chunk_id_referencia"],
                        pregunta["pregunta"],
                        pregunta["opciones"],
                        pregunta["respuesta_correcta"],
                    )
                    if not es_correcta:
                        marcar_pregunta_fallada(pregunta_id)

                st.success(f"Resultado: {aciertos}/{len(test['preguntas'])} correctas")

                for i, pregunta in enumerate(test["preguntas"]):
                    es_correcta = respuestas_usuario[i] == pregunta["respuesta_correcta"]
                    icono = "✅" if es_correcta else "❌"
                    st.write(f"{icono} **{pregunta['pregunta']}** — Correcta: {pregunta['respuesta_correcta']}")

                del st.session_state.test_actual

    with tab_repaso:
        st.subheader("Preguntas falladas de esta certificación")

        preguntas_falladas = obtener_preguntas_falladas(st.session_state.certificacion_id)

        if not preguntas_falladas:
            st.info("No tienes ninguna pregunta fallada pendiente de repasar. 🎉")
        else:
            st.caption(f"{len(preguntas_falladas)} pregunta(s) pendiente(s) de repaso")

            for pregunta in preguntas_falladas:
                with st.container(border=True):
                    st.markdown(f"**{pregunta['pregunta']}**")

                    for opcion in pregunta["opciones"]:
                        if opcion == pregunta["respuesta_correcta"]:
                            st.markdown(f"✅ {opcion}")
                        else:
                            st.markdown(f"◻️ {opcion}")

                    if st.button("✅ Ya me lo sé, quitar del repaso", key=f"dominada_{pregunta['id']}"):
                        marcar_pregunta_dominada(pregunta["id"])
                        st.rerun()