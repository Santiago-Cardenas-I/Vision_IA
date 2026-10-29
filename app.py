import streamlit as st
import base64
import io
import hashlib

from PIL import Image, ImageOps
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="Vision AI | Reconocimiento de imágenes",
    page_icon="👁️",
    layout="wide"
)

# --------------------------------------------------
# DISEÑO DE LA INTERFAZ
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #F2F5FA;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14213D, #243B63);
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    background: linear-gradient(120deg, #14213D, #344B83);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(20,33,61,0.15);
}

.hero h1 {
    font-size: 36px;
    font-weight: 700;
}

.hero p {
    color: #D8E3F5;
    font-size: 15px;
}

.section-title {
    color: #192D50;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 12px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #E1E8F0;
    box-shadow: 0 4px 12px rgba(20,33,61,0.04);
}

.metric-label {
    color: #8492A6;
    font-size: 11px;
    font-weight: 600;
}

.metric-value {
    color: #243B63;
    font-size: 23px;
    font-weight: 700;
    margin-top: 8px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 17px;
    border-left: 5px solid #18B6A4;
    box-shadow: 0 5px 18px rgba(20,33,61,0.05);
}

div.stButton > button {
    background: linear-gradient(90deg, #167D9A, #18B6A4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 600;
}

div.stButton > button:hover {
    color: white;
    border: none;
    background: #126D83;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FUNCIONES
# --------------------------------------------------

def preparar_imagen(archivo):

    imagen = Image.open(io.BytesIO(archivo))
    imagen = ImageOps.exif_transpose(imagen)
    imagen = imagen.convert("RGB")

    dimensiones_originales = imagen.size

    # Reducir imágenes demasiado grandes
    imagen.thumbnail((2000, 2000))

    buffer = io.BytesIO()

    imagen.save(
        buffer,
        format="JPEG",
        quality=90
    )

    imagen_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    url_imagen = (
        f"data:image/jpeg;base64,{imagen_base64}"
    )

    return imagen, url_imagen, dimensiones_originales


def analizar_imagen(api_key, imagen_url, pregunta):

    modelo = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    mensajes = [
        SystemMessage(
            content="""
            Eres un experto en reconocimiento y análisis
            de imágenes mediante inteligencia artificial.

            Tu tarea es analizar visualmente las imágenes
            proporcionadas y responder en español.

            Debes:

            1. Identificar los objetos y elementos visibles.
            2. Describir las características relevantes.
            3. Explicar el contexto visual cuando sea posible.
            4. Reconocer texto visible cuando sea legible.
            5. Diferenciar observaciones de interpretaciones.
            6. No inventar información.
            7. Reconocer cuando no puedas identificar algo
               con suficiente certeza.

            Organiza las respuestas de manera clara.
            """
        ),

        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": pregunta
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": imagen_url
                    }
                }
            ]
        )
    ]

    respuesta = modelo.invoke(mensajes)

    return respuesta.content


# --------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 👁️ VISION AI")

    st.markdown("---")

    st.subheader("Configuración")

    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        placeholder="Ingresa tu API Key"
    )

    st.markdown("---")

    st.subheader("Instrucciones")

    st.write("""
    1. Introduce tu clave API.
    2. Carga una imagen.
    3. Observa la vista previa.
    4. Escribe una pregunta.
    5. Presiona analizar.
    """)

    st.markdown("---")

    st.caption("Motor de inteligencia artificial")

    st.success("GPT-4o mini Vision")


# --------------------------------------------------
# ENCABEZADO
# --------------------------------------------------

st.markdown("""
<div class="hero">

    <h1>👁️ Vision AI</h1>

    <p>
    Sistema inteligente de reconocimiento y análisis
    de imágenes mediante inteligencia artificial.
    </p>

    <p>
    Identifica objetos, interpreta fotografías,
    reconoce elementos visuales y responde preguntas.
    </p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CARGA DE IMAGEN
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📸 Carga tu imagen</div>',
    unsafe_allow_html=True
)

archivo = st.file_uploader(
    "Selecciona una imagen para analizar",
    type=["png", "jpg", "jpeg", "webp"]
)


# --------------------------------------------------
# PROCESAMIENTO
# --------------------------------------------------

if archivo is None:

    st.info(
        "Carga una imagen para comenzar el reconocimiento visual."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">FUNCIÓN 01</div>
        <div class="metric-value">🔍 Objetos</div>
        <p>Identificación de elementos visibles.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">FUNCIÓN 02</div>
        <div class="metric-value">🧠 Análisis</div>
        <p>Interpretación del contenido visual.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">FUNCIÓN 03</div>
        <div class="metric-value">💬 Preguntas</div>
        <p>Conversación sobre la imagen.</p>
        </div>
        """, unsafe_allow_html=True)

else:

    try:

        imagen_bytes = archivo.getvalue()

        # Identificar si cambió la imagen
        hash_imagen = hashlib.sha256(
            imagen_bytes
        ).hexdigest()

        if st.session_state.get("hash_imagen") != hash_imagen:

            st.session_state.hash_imagen = hash_imagen
            st.session_state.resultado = None

        imagen, imagen_url, dimensiones = preparar_imagen(
            imagen_bytes
        )

        ancho, alto = dimensiones

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="section-title">🖼️ Vista previa</div>',
                unsafe_allow_html=True
            )

            st.image(
                imagen,
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="section-title">📊 Información de la imagen</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">NOMBRE DEL ARCHIVO</div>
                    <div class="metric-value" style="font-size:17px">
                        {archivo.name}
                    </div>
                    <br>
                    <div class="metric-label">DIMENSIONES</div>
                    <div class="metric-value">
                        {ancho} × {alto}
                    </div>
                    <br>
                    <div class="metric-label">TAMAÑO</div>
                    <div class="metric-value">
                        {len(imagen_bytes) / 1024:.2f} KB
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown("### 💬 ¿Qué quieres saber?")

            with st.form("formulario_analisis"):

                pregunta = st.text_area(
                    "Escribe tu pregunta",
                    placeholder=(
                        "Ejemplo: ¿Qué objetos aparecen "
                        "en esta imagen? Describe sus características."
                    ),
                    height=130
                )

                analizar = st.form_submit_button(
                    "🔍 Analizar imagen",
                    use_container_width=True
                )

            if analizar:

                if not api_key:

                    st.error(
                        "Primero debes ingresar tu clave API de OpenAI."
                    )

                elif not pregunta.strip():

                    st.warning(
                        "Escribe una pregunta para analizar la imagen."
                    )

                else:

                    try:

                        with st.spinner(
                            "La inteligencia artificial está analizando la imagen..."
                        ):

                            resultado = analizar_imagen(
                                api_key,
                                imagen_url,
                                pregunta.strip()
                            )

                        st.session_state.resultado = resultado

                    except Exception as error:

                        st.error(
                            "No fue posible analizar la imagen. "
                            "Verifica tu API Key, conexión y saldo disponible."
                        )

                        # Detalle técnico para facilitar la depuración
                        st.caption(str(error))

        # --------------------------------------------------
        # RESULTADO
        # --------------------------------------------------

        if st.session_state.get("resultado"):

            st.markdown("---")

            st.markdown(
                '<div class="section-title">🧠 Resultado del reconocimiento</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-card">
                    {st.session_state.resultado}
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception:

        st.error(
            "No se pudo abrir la imagen. "
            "Comprueba que el archivo sea válido."
        )


# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#8492A6;font-size:12px">
        VISION AI · RECONOCIMIENTO DE IMÁGENES
        <br>
        Powered by OpenAI
    </div>
    """,
    unsafe_allow_html=True
)
