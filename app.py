import io
import platform
import hashlib

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="Nexo Docs | Laboratorio documental",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Identidad visual: verde petróleo, menta y marfil ──────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
      --forest: #123D3A;
      --forest-deep: #0B2928;
      --teal: #1C7770;
      --mint: #BDE9D8;
      --mint-pale: #E9F6EF;
      --paper: #F7F6F0;
      --white: #FFFFFF;
      --ink: #1E302F;
      --muted: #70817D;
      --line: #E1E8E2;
      --amber: #E6B765;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,246,240,.92); }
    .block-container { max-width: 1320px; padding-top: 1.7rem; padding-bottom: 3rem; }

    [data-testid="stSidebar"] {
      background: var(--forest-deep);
      border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #F1F7F2; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #B6C9C2; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
    [data-testid="stSidebar"] label { color: #D8E7DF !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
      background: rgba(255,255,255,.06); border: 1px dashed rgba(189,233,216,.45);
      border-radius: 14px; padding: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #EAF4EF; }

    .logo-wrap { display:flex; gap:12px; align-items:center; margin: .4rem 0 2rem; }
    .logo-symbol {
      width:43px;height:43px;border-radius:13px;background:var(--mint);
      display:flex;align-items:center;justify-content:center;color:var(--forest);
      font:700 25px 'Space Grotesk',sans-serif;
    }
    .logo-title { font:700 19px 'Space Grotesk',sans-serif; letter-spacing:-.5px; }
    .logo-caption { font-size:10px;letter-spacing:1.8px;color:#9CB9AE;margin-top:2px; }
    .sidebar-step {
      color: #D9E9E1; font-size: 11px; letter-spacing: 1.5px;
      font-weight: 700; text-transform: uppercase; margin: 15px 0 9px;
    }
    .hero {
      position:relative; overflow:hidden; background:var(--forest); color:white;
      padding:32px 34px; border-radius:24px; margin-bottom:23px;
      box-shadow:0 12px 28px rgba(18,61,58,.10);
    }
    .hero:after {
      content:""; position:absolute; width:250px;height:250px;border:1px solid rgba(189,233,216,.22);
      border-radius:50%;right:-65px;top:-100px;box-shadow:0 0 0 28px rgba(189,233,216,.04),0 0 0 58px rgba(189,233,216,.035);
    }
    .hero-kicker { color:var(--mint);font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px; }
    .hero-heading { font:700 clamp(27px,3vw,39px) 'Space Grotesk',sans-serif;letter-spacing:-1.4px;line-height:1.12;margin-bottom:11px;max-width:700px; }
    .hero-copy { color:#D0E0D9;font-size:14px;line-height:1.65;max-width:660px; }
    .hero-tag {
      display:inline-flex;align-items:center;gap:8px;margin-top:21px;padding:8px 11px;
      border-radius:8px;background:rgba(189,233,216,.12);border:1px solid rgba(189,233,216,.2);
      color:#D8F4E7;font-size:11px;font-weight:600;
    }
    .section-title { font:700 18px 'Space Grotesk',sans-serif;color:var(--forest);letter-spacing:-.4px;margin:0 0 4px; }
    .section-desc { color:var(--muted);font-size:13px;margin-bottom:15px; }
    .metric {
      background:var(--white);border:1px solid var(--line);border-radius:16px;
      padding:18px 20px;min-height:116px;box-shadow:0 4px 12px rgba(17,58,52,.025);
    }
    .metric-top { display:flex;align-items:center;justify-content:space-between;margin-bottom:12px; }
    .metric-label { color:#7C8C85;font-size:10px;letter-spacing:1.3px;font-weight:700; }
    .metric-mark { width:28px;height:28px;border-radius:9px;background:var(--mint-pale);color:var(--teal);display:flex;align-items:center;justify-content:center;font-size:14px; }
    .metric-value { color:var(--forest);font:700 25px 'Space Grotesk',sans-serif;letter-spacing:-.7px;overflow-wrap:anywhere; }
    .metric-note { color:#87968F;font-size:11px;margin-top:4px; }
    .surface {
      background:white;border:1px solid var(--line);border-radius:19px;padding:23px 24px;
      box-shadow:0 5px 16px rgba(17,58,52,.025);margin-bottom:17px;
    }
    .surface-heading { font:700 16px 'Space Grotesk',sans-serif;color:var(--forest);margin-bottom:5px; }
    .surface-copy { color:var(--muted);font-size:12px;line-height:1.55;margin-bottom:17px; }
    .empty {
      background:#FDFCF8;border:1px dashed #C8D9D0;border-radius:20px;
      text-align:center;padding:38px 24px;margin-top:18px;
    }
    .empty-mark { width:57px;height:57px;margin:0 auto 14px;display:flex;align-items:center;justify-content:center;border-radius:17px;background:var(--mint-pale);color:var(--teal);font-size:25px; }
    .empty-title { font:700 18px 'Space Grotesk',sans-serif;color:var(--forest);margin-bottom:7px; }
    .empty-copy { color:var(--muted);font-size:13px;line-height:1.65;max-width:510px;margin:auto; }
    .tip {
      padding:14px 0;border-bottom:1px solid #EDF0EB;
    }
    .tip:last-child { border-bottom:0;padding-bottom:0; }
    .tip-title { color:var(--forest);font-size:13px;font-weight:700; }
    .tip-copy { color:var(--muted);font-size:12px;line-height:1.6;margin-top:5px; }
    .answer-chip {
      display:inline-flex;align-items:center;gap:7px;background:var(--mint-pale);color:var(--teal);
      border-radius:7px;padding:7px 10px;font-size:10px;letter-spacing:1px;font-weight:800;margin:8px 0 13px;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
      border-radius:11px !important;border-color:#D9E4DC !important;background:#FDFEFC !important;
    }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
      border-color:#58A99A !important;box-shadow:0 0 0 2px rgba(28,119,112,.11) !important;
    }
    .stButton > button, .stFormSubmitButton > button {
      background:var(--forest);color:white;border:1px solid var(--forest);border-radius:10px;
      min-height:43px;font-weight:700;transition:all .15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
      background:var(--teal);border-color:var(--teal);color:white;transform:translateY(-1px);
    }
    div[data-testid="stExpander"] { background:white;border:1px solid var(--line);border-radius:12px; }
    .footer { text-align:center;color:#93A099;font-size:10px;letter-spacing:.7px;margin-top:26px; }
    @media(max-width:700px) {
      .block-container { padding:1rem .85rem 2rem; }
      .hero { padding:25px 21px;border-radius:18px; }
      .surface { padding:17px; }
      .hero-heading { letter-spacing:-.8px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "nexo_answer" not in st.session_state:
    st.session_state.nexo_answer = None
if "nexo_pdf_hash" not in st.session_state:
    st.session_state.nexo_pdf_hash = None


@st.cache_resource(show_spinner=False)
def crear_base_conocimiento(pdf_bytes: bytes, api_key: str):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)
    if not fragmentos:
        return None, len(texto), 0, len(reader.pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    base = FAISS.from_texts(fragmentos, embeddings)
    return base, len(texto), len(fragmentos), len(reader.pages)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="logo-wrap">
          <div class="logo-symbol">◈</div>
          <div><div class="logo-title">Nexo Docs</div>
          <div class="logo-caption">DOCUMENT INTELLIGENCE</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Laboratorio de documentos")
    st.caption("Configura tu conexión y carga el archivo que quieres explorar.")
    st.markdown('<div class="sidebar-step">01 / Conexión</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        placeholder="sk-••••••••••••",
        label_visibility="collapsed",
        help="Se utiliza para crear embeddings y responder consultas.",
    )
    st.markdown('<div class="sidebar-step">02 / Documento fuente</div>', unsafe_allow_html=True)
    pdf = st.file_uploader("Carga tu PDF", type=["pdf"], label_visibility="collapsed")
    if pdf:
        st.success("Documento añadido")
        st.caption(pdf.name)
    st.divider()
    st.caption("MOTOR DE ANÁLISIS")
    st.markdown("**OpenAI · FAISS · LangChain**")
    st.caption(f"Entorno Python {platform.python_version()}")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.caption("Consejo: formula preguntas específicas y contrasta las respuestas con el texto original.")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">◈ &nbsp; LABORATORIO DOCUMENTAL</div>
      <div class="hero-heading">Del documento a la comprensión.</div>
      <div class="hero-copy">Explora informes, encuentra ideas clave y resuelve dudas con un asistente que consulta el contenido de tus PDF.</div>
      <div class="hero-tag">● &nbsp; ESPACIO DE ANÁLISIS CON IA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key or pdf is None:
    st.markdown('<div class="section-title">Panel de trabajo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Tu espacio de análisis se actualizará al conectar la API y cargar un archivo.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "ESTADO DEL PROYECTO", "En espera", "Conecta la API y añade un PDF", "◷"),
        (c2, "ÍNDICE SEMÁNTICO", "No creado", "Se genera al procesar el documento", "⌘"),
        (c3, "MODO DE CONSULTA", "Pregunta y explora", "Respuestas basadas en el texto", "↗"),
    ]
    for col, label, value, note, mark in cards:
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-top"><div class="metric-label">{label}</div>'
                f'<div class="metric-mark">{mark}</div></div><div class="metric-value" style="font-size:{"19px" if len(value)>14 else "24px"}">{value}</div>'
                f'<div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )
    if not api_key:
        empty_title, empty_copy, empty_mark = "Primero, conecta tu espacio", "Introduce tu clave de OpenAI en el panel lateral para habilitar el análisis.", "⌘"
    else:
        empty_title, empty_copy, empty_mark = "Tu documento, listo para explorar", "Carga un PDF desde el panel lateral. Nexo Docs preparará su contenido para responder preguntas.", "↥"
    st.markdown(
        f'<div class="empty"><div class="empty-mark">{empty_mark}</div><div class="empty-title">{empty_title}</div>'
        f'<div class="empty-copy">{empty_copy}</div></div>',
        unsafe_allow_html=True,
    )
else:
    pdf_bytes = pdf.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.nexo_pdf_hash != pdf_hash:
        st.session_state.nexo_answer = None
        st.session_state.nexo_pdf_hash = pdf_hash

    try:
        with st.spinner("Leyendo el documento y construyendo el índice semántico…"):
            base, n_caracteres, n_fragmentos, n_paginas = crear_base_conocimiento(pdf_bytes, api_key)

        if base is None:
            st.error("No se pudo extraer texto del PDF. Si es un documento escaneado, aplica OCR antes de cargarlo.")
            st.stop()

        st.markdown('<div class="section-title">Ficha del documento</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">El índice semántico está preparado para tus preguntas.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        metrics = [
            (c1, "ARCHIVO ACTUAL", pdf.name, f"{n_paginas} páginas", "▤"),
            (c2, "TEXTO PROCESADO", f"{n_caracteres:,}", "Caracteres extraídos", "≋"),
            (c3, "FRAGMENTOS INDEXADOS", f"{n_fragmentos:,}", "Unidades de contexto", "⌘"),
        ]
        for col, label, value, note, mark in metrics:
            with col:
                size = "17px" if len(str(value)) > 22 else ("22px" if len(str(value)) > 13 else "25px")
                st.markdown(
                    f'<div class="metric"><div class="metric-top"><div class="metric-label">{label}</div>'
                    f'<div class="metric-mark">{mark}</div></div><div class="metric-value" style="font-size:{size}">{value}</div>'
                    f'<div class="metric-note">{note}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        left, right = st.columns([1.55, 1], gap="large")
        with left:
            st.markdown('<div class="surface-heading">◈ &nbsp; Consulta inteligente</div>', unsafe_allow_html=True)
            st.markdown('<div class="surface-copy">Haz una pregunta sobre el contenido. La respuesta se construye a partir de los fragmentos más relevantes del PDF.</div>', unsafe_allow_html=True)
            with st.form("nexo_question_form", clear_on_submit=False):
                pregunta = st.text_area(
                    "Pregunta",
                    placeholder="Ej.: ¿Qué recomendaciones plantea el autor y qué evidencia las respalda?",
                    height=125,
                    label_visibility="collapsed",
                )
                enviar = st.form_submit_button("Analizar documento  →", use_container_width=True)

            if enviar:
                if not pregunta.strip():
                    st.warning("Escribe una pregunta para iniciar el análisis.")
                else:
                    try:
                        with st.spinner("Buscando contexto relevante y preparando respuesta…"):
                            docs = base.similarity_search(pregunta.strip(), k=4)
                            contexto = "\n\n---\n\n".join(d.page_content for d in docs)
                            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
                            mensajes = [
                                ("system",
                                 "Responde en español usando únicamente el contexto proporcionado. "
                                 "Si la respuesta no está en el contexto, dilo claramente. "
                                 "No inventes datos y distingue hechos de inferencias."),
                                ("human", f"Contexto:\n{contexto}\n\nPregunta: {pregunta.strip()}"),
                            ]
                            respuesta = llm.invoke(mensajes)
                        st.session_state.nexo_answer = {
                            "question": pregunta.strip(),
                            "content": respuesta.content,
                            "docs": [d.page_content for d in docs],
                        }
                    except Exception:
                        st.error("No se pudo completar el análisis. Verifica tu clave de API, conexión y límites de uso.")

        with right:
            st.markdown('<div class="surface-heading">Guía de exploración</div>', unsafe_allow_html=True)
            st.markdown('<div class="surface-copy">Pequeñas estrategias para obtener resultados más útiles.</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="tip"><div class="tip-title">01 &nbsp; Pregunta con intención</div>
                <div class="tip-copy">Enfócate en una sección, argumento, cifra o conclusión específica.</div></div>
                <div class="tip"><div class="tip-title">02 &nbsp; Solicita contexto</div>
                <div class="tip-copy">Pide que la respuesta incluya las razones o los datos que la respaldan.</div></div>
                <div class="tip"><div class="tip-title">03 &nbsp; Verifica la fuente</div>
                <div class="tip-copy">Abre los fragmentos recuperados para comprobar la evidencia utilizada.</div></div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.nexo_answer:
            answer = st.session_state.nexo_answer
            st.write("")
            st.markdown('<div class="surface-heading">Resultado del análisis</div>', unsafe_allow_html=True)
            st.markdown('<div class="answer-chip">◈ &nbsp; RESPUESTA GENERADA</div>', unsafe_allow_html=True)
            st.markdown(f"**Pregunta:** {answer['question']}")
            st.markdown(answer["content"])
            with st.expander("Consultar los fragmentos que sustentan la respuesta"):
                for i, fragmento in enumerate(answer["docs"], 1):
                    st.markdown(f"**Fragmento {i}**")
                    st.text(fragmento)
                    if i < len(answer["docs"]):
                        st.divider()

    except Exception:
        st.error("No fue posible procesar el PDF. Verifica que el archivo sea válido y que la clave de OpenAI tenga acceso.")
        with st.expander("Información para diagnóstico"):
            st.caption("Consulta la consola donde ejecutas Streamlit para revisar el error técnico. No publiques claves API en registros o capturas.")

st.markdown('<div class="footer">NEXO DOCS &nbsp;·&nbsp; DOCUMENT INTELLIGENCE &nbsp;·&nbsp; REVISA LAS RESPUESTAS CONTRA LA FUENTE ORIGINAL</div>', unsafe_allow_html=True)
