import streamlit as st
import requests
import uuid
import os

# Configuration
API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Assistant Médical RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le flux des agents et le mode sombre
st.markdown("""
<style>
    .agent-flow {
        font-family: 'Courier New', monospace;
        background-color: #1E1E2E;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #A855F7;
        margin-bottom: 10px;
        font-size: 0.9em;
        color: #CDD6F4;
    }
    .agent-step {
        display: inline-block;
        margin-right: 5px;
    }
    .agent-arrow {
        color: #6C7086;
        margin: 0 5px;
    }
    section[data-testid="stSidebar"] {
        background-color: #11111B;
    }
    .stChatMessage {
        background-color: #181825;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de la session
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Fonctions API ---

def create_session():
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

def get_sessions():
    try:
        response = requests.get(f"{API_BASE}/sessions")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Erreur de connexion au backend : {e}")
    return []

def get_session_messages(session_id):
    try:
        response = requests.get(f"{API_BASE}/sessions/{session_id}/messages")
        if response.status_code == 200:
            return response.json().get("messages", [])
    except Exception:
        pass
    return []

def send_message(message):
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = str(uuid.uuid4())
    
    payload = {
        "message": message,
        "session_id": st.session_state.current_session_id
    }
    
    try:
        response = requests.post(f"{API_BASE}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur : {response.text}")
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
    return None

def delete_session(session_id):
    try:
        response = requests.delete(f"{API_BASE}/sessions/{session_id}")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Erreur lors de la suppression : {e}")
    return False

def ingest_file(uploaded_file):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    try:
        response = requests.post(f"{API_BASE}/ingest/file", files=files)
        return response
    except Exception as e:
        st.error(f"Erreur d'ingestion : {e}")
        return None

# --- Sidebar ---

with st.sidebar:
    st.title("Assistant Médical")
    
    # Section Upload
    st.subheader("📁 Indexation de Documents")
    uploaded_file = st.file_uploader("Choisir un fichier (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        if st.button("Indexer le document", use_container_width=True):
            with st.status("Indexation en cours...", expanded=True) as status:
                st.write("Extraction et découpage...")
                response = ingest_file(uploaded_file)
                if response and response.status_code == 200:
                    data = response.json()
                    status.update(label=f"✅ Indexé : {data.get('chunks', 0)} fragments", state="complete")
                    st.success(f"Document '{uploaded_file.name}' indexé avec succès !")
                else:
                    status.update(label="❌ Échec de l'indexation", state="error")
                    st.error("Une erreur est survenue lors de l'indexation.")

    st.divider()
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        if st.button("➕ Nouvelle Discussion", use_container_width=True):
            create_session()
    
    st.markdown("### Discussions Récentes")
    sessions = get_sessions()
    
    for sess in sessions:
        sid = sess.get("session_id")
        label = sess.get("last_query", "Nouvelle Discussion")[:25] + "..."
        
        col_sess, col_del = st.columns([0.85, 0.15])
        
        with col_sess:
            if sid == st.session_state.current_session_id:
                st.markdown(f"**👉 {label}**")
            else:
                if st.button(label, key=f"btn_{sid}", use_container_width=True):
                    st.session_state.current_session_id = sid
                    st.session_state.messages = get_session_messages(sid)
                    st.rerun()
        
        with col_del:
            if st.button("🗑️", key=f"del_{sid}", help="Supprimer"):
                if delete_session(sid):
                    if sid == st.session_state.current_session_id:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()

# --- Zone de Chat Principale ---

st.title("🩺 Assistant de Recherche Médicale")

if not st.session_state.current_session_id:
    st.info("Commencez une nouvelle discussion ou sélectionnez-en une dans la barre latérale.")
else:
    if not st.session_state.messages and st.session_state.current_session_id:
        st.session_state.messages = get_session_messages(st.session_state.current_session_id)

    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg.get("user_query", ""))
        
        with st.chat_message("assistant"):
            path = msg.get("agent_path", [])
            if isinstance(path, list):
                flow_html = " <span class='agent-arrow'>→</span> ".join([
                    f"<span class='agent-step'>{step.replace('_agent', '').capitalize()}</span>" 
                    for step in path
                ])
            else:
                flow_html = str(path)

            if flow_html:
                st.markdown(f"<div class='agent-flow'>{flow_html}</div>", unsafe_allow_html=True)
            
            st.write(msg.get("agent_response", ""))

    if prompt := st.chat_input("Posez votre question médicale..."):
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                response_data = send_message(prompt)
                
                if response_data:
                    answer = response_data.get("response")
                    path = response_data.get("agent_path", [])
                    
                    flow_html = " <span class='agent-arrow'>→</span> ".join([
                        f"<span class='agent-step'>{step.replace('_agent', '').capitalize()}</span>" 
                        for step in path
                    ])
                    st.markdown(f"<div class='agent-flow'>{flow_html}</div>", unsafe_allow_html=True)
                    st.write(answer)
                    
                    new_msg = {
                        "user_query": prompt,
                        "agent_response": answer,
                        "agent_path": path
                    }
                    st.session_state.messages.append(new_msg)
