"""
app.py — Interfaz en Streamlit que conecta todo el motor RAG.

Jerarquía: certificación -> dominio (bloque) -> tema -> apuntes.
"""

import streamlit as st

from database import (
    init_db,
    crear_certificacion,
    crear_dominio,
    obtener_dominios,
    crear_tema,
    obtener_temas,
    guardar_contenido,
    guardar_chunk,
    obtener_contenidos_de_tema,
    obtener_chunks_de_tema,
    obtener_chunks_de_dominio,
    guardar_pregunta,
    marcar_pregunta_fallada,
    marcar_pregunta_dominada,
    obtener_preguntas_falladas,
    eliminar_contenido,
    eliminar_tema,
    eliminar_dominio,
)
from core.chunking import chunk_text
from core.embeddings import embed_texts
from core.generator import generar_preguntas

st.set_page_config(page_title="Study RAG System", layout="wide")

init_db()

st.title("📚 Study RAG System")
st.caption("Genera tests de certificación a partir de tus propios apuntes, con un motor RAG construido desde cero.")


# --- session_state: ahora con 3 niveles de selección ------------------------
if "certificacion_id" not in st.session_state:
    st.session_state.certificacion_id = None
if "dominio_id" not in st.session_state:
    st.session_state.dominio_id = None
if "tema_id" not in st.session_state:
    st.session_state.tema_id = None


# --- Sidebar: certificación -> dominio (bloque) -> tema ---------------------
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
        st.session_state.tema_id = None
        st.rerun()

    dominios = []

    if st.session_state.certificacion_id is not None:
        st.success(f"Certificación: {nombre_certificacion} (id={st.session_state.certificacion_id})")

        st.divider()
        st.subheader("Bloque (dominio) activo")

        dominios = obtener_dominios(st.session_state.certificacion_id)

        if dominios:
            opciones_dominio = {
                f"{d['nombre']} ({d['peso_examen']*100:.0f}% del examen)": d["id"]
                for d in dominios
            }
            etiqueta_dominio = st.selectbox("Bloque", options=list(opciones_dominio.keys()))
            st.session_state.dominio_id = opciones_dominio[etiqueta_dominio]
        else:
            st.info("Esta certificación todavía no tiene bloques. Añade el primero abajo.")
            st.session_state.dominio_id = None

        with st.expander("➕ Añadir nuevo bloque (dominio)"):
            nombre_dominio_nuevo = st.text_input("Nombre del bloque", key="input_nombre_dominio")
            peso_dominio_nuevo = st.slider(
                "Peso en el examen (%)", min_value=0, max_value=100, value=50,
                key="input_peso_dominio",
                help="El % que representa este bloque en el examen real.",
            )
            if st.button("Añadir bloque"):
                if nombre_dominio_nuevo.strip():
                    crear_dominio(
                        st.session_state.certificacion_id,
                        nombre_dominio_nuevo.strip(),
                        peso_examen=peso_dominio_nuevo / 100,
                    )
                    st.rerun()
                else:
                    st.warning("Escribe un nombre para el bloque.")

        # --- Nivel nuevo: tema dentro del bloque activo ---
        if st.session_state.dominio_id is not None:
            st.divider()
            st.subheader("Tema activo (dentro del bloque)")

            temas = obtener_temas(st.session_state.dominio_id)

            if temas:
                opciones_tema = {t["nombre"]: t["id"] for t in temas}
                etiqueta_tema = st.selectbox("Tema", options=list(opciones_tema.keys()))
                st.session_state.tema_id = opciones_tema[etiqueta_tema]
            else:
                st.info("Este bloque todavía no tiene temas. Añade el primero abajo.")
                st.session_state.tema_id = None

            with st.expander("➕ Añadir nuevo tema"):
                nombre_tema_nuevo = st.text_input("Nombre del tema", key="input_nombre_tema")
                if st.button("Añadir tema"):
                    if nombre_tema_nuevo.strip():
                        crear_tema(st.session_state.dominio_id, nombre_tema_nuevo.strip())
                        st.rerun()
                    else:
                        st.warning("Escribe un nombre para el tema.")


# --- Área principal ----------------------------------------------------------
if st.session_state.certificacion_id is None:
    st.info("👈 Selecciona o crea una certificación en la barra lateral para empezar.")
elif st.session_state.dominio_id is None:
    st.info("👈 Selecciona o crea un bloque (dominio) en la barra lateral.")
elif st.session_state.tema_id is None:
    st.info("👈 Selecciona o crea un tema dentro del bloque en la barra lateral.")
else:
    tab_guardar, tab_test, tab_gestionar, tab_repaso = st.tabs(
        ["📥 Guardar contenido", "📝 Generar test", "🗂️ Gestionar", "🔁 Repasar fallos"]
    )

    # --- Tab: guardar contenido (ahora dentro de un TEMA, no del bloque) ---
    with tab_guardar:
        st.subheader("Guardar un apunte nuevo")

        titulo_apunte = st.text_input("Título del apunte", placeholder="Ej. OCR y detección de objetos")
        texto_apunte = st.text_area(
            "Contenido", height=250,
            placeholder="Pega aquí tu apunte de estudio. Sepáralo en párrafos (línea en blanco entre ideas).",
        )

        if st.button("💾 Guardar y procesar"):
            if not titulo_apunte.strip() or not texto_apunte.strip():
                st.warning("Necesitas un título y contenido antes de guardar.")
            else:
                contenido_id = guardar_contenido(
                    st.session_state.tema_id, titulo_apunte.strip(), texto_apunte.strip()
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

    # --- Tab: generar test (elige alcance: tema o bloque entero) ---
    with tab_test:
        st.subheader("Generar test")

        temas_del_dominio = obtener_temas(st.session_state.dominio_id)
        nombre_tema_activo = next(
            (t["nombre"] for t in temas_del_dominio if t["id"] == st.session_state.tema_id),
            "este tema",
        )
        nombre_dominio_activo = next(
            (d["nombre"] for d in dominios if d["id"] == st.session_state.dominio_id),
            "este bloque",
        )

        alcance = st.radio(
            "¿Sobre qué quieres el test?",
            options=[f"Solo el tema: {nombre_tema_activo}", f"Todo el bloque: {nombre_dominio_activo}"],
        )
        num_preguntas = st.slider("Número de preguntas", min_value=1, max_value=10, value=5)

        if st.button("🎲 Generar test"):
            if alcance.startswith("Solo el tema"):
                chunks_contexto = obtener_chunks_de_tema(st.session_state.tema_id)
                nombre_para_prompt = nombre_tema_activo
            else:
                chunks_contexto = obtener_chunks_de_dominio(st.session_state.dominio_id)
                nombre_para_prompt = nombre_dominio_activo

            if not chunks_contexto:
                st.warning("No hay contenido guardado todavía en este alcance. Ve a 'Guardar contenido' primero.")
            else:
                contexto = [c["texto"] for c in chunks_contexto]
                try:
                    with st.spinner("Generando preguntas con IA..."):
                        preguntas = generar_preguntas(contexto, nombre_para_prompt, num_preguntas)

                    st.session_state.test_actual = {
                        "preguntas": preguntas,
                        "chunk_id_referencia": chunks_contexto[0]["id"],
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
                        f"Opciones pregunta {i}", options=pregunta["opciones"],
                        key=f"respuesta_{i}", label_visibility="collapsed", index=None,
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
                        test["chunk_id_referencia"], pregunta["pregunta"],
                        pregunta["opciones"], pregunta["respuesta_correcta"],
                    )
                    if not es_correcta:
                        marcar_pregunta_fallada(pregunta_id)

                st.success(f"Resultado: {aciertos}/{len(test['preguntas'])} correctas")

                for i, pregunta in enumerate(test["preguntas"]):
                    es_correcta = respuestas_usuario[i] == pregunta["respuesta_correcta"]
                    icono = "✅" if es_correcta else "❌"
                    st.write(f"{icono} **{pregunta['pregunta']}** — Correcta: {pregunta['respuesta_correcta']}")

                del st.session_state.test_actual

    # --- Tab NUEVA: gestionar (ver y borrar) ---
    with tab_gestionar:
        st.subheader(f"Apuntes guardados en: {nombre_tema_activo if 'nombre_tema_activo' in dir() else ''}")

        temas_del_dominio_gestion = obtener_temas(st.session_state.dominio_id)
        nombre_tema_activo_gestion = next(
            (t["nombre"] for t in temas_del_dominio_gestion if t["id"] == st.session_state.tema_id),
            "este tema",
        )
        st.caption(f"Tema actual: **{nombre_tema_activo_gestion}**")

        contenidos = obtener_contenidos_de_tema(st.session_state.tema_id)

        if not contenidos:
            st.info("Este tema todavía no tiene apuntes guardados.")
        else:
            for contenido in contenidos:
                with st.container(border=True):
                    col_info, col_boton = st.columns([4, 1])
                    with col_info:
                        st.markdown(f"**{contenido['titulo']}**")
                        st.caption(f"{contenido['num_chunks']} fragmentos · guardado {contenido['fecha'][:10]}")
                    with col_boton:
                        if st.button("🗑️ Borrar", key=f"borrar_contenido_{contenido['id']}"):
                            eliminar_contenido(contenido["id"])
                            st.rerun()

        st.divider()
        st.subheader("⚠️ Zona de borrado mayor")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Borrar este TEMA entero"):
                st.session_state.confirmar_borrado_tema = True
        with col_b:
            if st.button("🗑️ Borrar este BLOQUE entero"):
                st.session_state.confirmar_borrado_dominio = True

        if st.session_state.get("confirmar_borrado_tema"):
            st.warning(f"¿Seguro que quieres borrar el tema '{nombre_tema_activo_gestion}' y TODOS sus apuntes?")
            col_si, col_no = st.columns(2)
            with col_si:
                if st.button("Sí, borrar tema definitivamente"):
                    eliminar_tema(st.session_state.tema_id)
                    st.session_state.tema_id = None
                    st.session_state.confirmar_borrado_tema = False
                    st.rerun()
            with col_no:
                if st.button("Cancelar", key="cancelar_tema"):
                    st.session_state.confirmar_borrado_tema = False
                    st.rerun()

        if st.session_state.get("confirmar_borrado_dominio"):
            st.warning(f"¿Seguro que quieres borrar el bloque '{nombre_dominio_activo}' y TODOS sus temas y apuntes?")
            col_si2, col_no2 = st.columns(2)
            with col_si2:
                if st.button("Sí, borrar bloque definitivamente"):
                    eliminar_dominio(st.session_state.dominio_id)
                    st.session_state.dominio_id = None
                    st.session_state.tema_id = None
                    st.session_state.confirmar_borrado_dominio = False
                    st.rerun()
            with col_no2:
                if st.button("Cancelar", key="cancelar_dominio"):
                    st.session_state.confirmar_borrado_dominio = False
                    st.rerun()

    # --- Tab: repasar fallos (sin cambios de lógica) ---
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