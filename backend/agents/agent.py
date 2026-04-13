import os
import logging
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

from backend.tools.retrieval_tool import hybrid_search
from backend.tools.memory_tool import store_interaction, get_past_interactions
from backend.tools import arxiv_tools, wikipedia_tools, serper_tools

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION DES MODÈLES
# ============================================================

groq_smart = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

groq_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# ============================================================
# ÉTAT DU GRAPHE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    retrieved_docs: str
    external_results: str
    final_response: str
    agent_path: list[str]
    routing_decision: str

# ============================================================
# NŒUDS DU GRAPHE
# ============================================================

def orchestrator_node(state: AgentState) -> AgentState:
    """
    Nœud orchestrateur : analyse la question et décide du routing.
    - direct    : salutations, questions hors médecine
    - retrieval : questions médicales → chercher dans la base
    - tools     : actualités récentes, recherches web
    """
    question = state["question"]
    logger.info(f"[Orchestrateur] Question reçue : {question[:50]}...")

    prompt = f"""Tu es un orchestrateur médical. Analyse cette question et décide du routing.

Question : {question}

Réponds UNIQUEMENT par un de ces mots :
- "direct" : si c'est une salutation, remerciement ou question hors médecine (ex: bonjour, merci, context window, IA)
- "retrieval" : si c'est une question médicale (pathologie, traitement, médicament, symptôme, maladie, santé)
- "tools" : si c'est une actualité récente ou besoin de recherche web

Routing :"""

    response = groq_smart.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()

    if "retrieval" in decision:
        routing = "retrieval"
    elif "tools" in decision:
        routing = "tools"
    else:
        routing = "direct"

    logger.info(f"[Orchestrateur] Décision de routing : {routing}")

    return {
        **state,
        "routing_decision": routing,
        "agent_path": state.get("agent_path", []) + ["orchestrateur"]
    }


def retrieval_node(state: AgentState) -> AgentState:
    """
    Nœud de récupération RAG hybride.
    Cherche dans LanceDB + BM25 + Neo4j via hybrid_search.
    """
    question = state["question"]
    logger.info(f"[Retrieval] Recherche hybride pour : {question[:50]}...")

    try:
        results = hybrid_search(query=question, limit=5)
        logger.debug(f"[Retrieval] Type retourné : {type(results)}")

        # Cas 1 : hybrid_search retourne une string
        if isinstance(results, str):
            docs_text = results.strip()
            if len(docs_text) < 50:
                logger.info("[Retrieval] Contexte trop court (<50 chars) → ignoré")
                docs_text = ""
            else:
                logger.info(f"[Retrieval] Contexte reçu ({len(docs_text)} caractères)")

        # Cas 2 : hybrid_search retourne une liste de dicts
        elif isinstance(results, list):
            docs = []
            for i, r in enumerate(results):
                if isinstance(r, dict):
                    text = (
                        r.get("text") or
                        r.get("content") or
                        r.get("page_content") or
                        ""
                    )
                else:
                    text = str(r)

                if text and len(text.strip()) > 30:
                    docs.append(f"[Document {i+1}]\n{text.strip()}")

            docs_text = "\n\n".join(docs)
            logger.info(f"[Retrieval] {len(docs)}/{len(results)} documents utiles extraits")

        # Cas 3 : résultat None ou type inattendu
        else:
            docs_text = ""
            logger.info(f"[Retrieval] Type inattendu : {type(results)} → ignoré")

    except Exception as e:
        logger.error(f"[Retrieval] Erreur hybrid_search : {e}", exc_info=True)
        docs_text = ""

    return {
        **state,
        "retrieved_docs": docs_text,
        "agent_path": state.get("agent_path", []) + ["retrieval_agent"]
    }


def tools_node(state: AgentState) -> AgentState:
    """
    Nœud d'utilisation des outils externes.
    Appelle Wikipedia, Google Search ou arXiv selon la question.
    """
    question = state["question"]
    logger.info(f"[Tools] Recherche externe pour : {question[:50]}...")

    tools_list = arxiv_tools + wikipedia_tools + serper_tools
    groq_with_tools = groq_smart.bind_tools(tools_list)

    prompt = f"""Tu es un agent de recherche. 
Utilise les outils disponibles pour trouver des informations sur :
{question}

- search_wikipedia : pour les définitions et concepts généraux
- search_google    : pour les informations récentes
- search_arxiv     : pour les articles scientifiques"""

    try:
        response = groq_with_tools.invoke([HumanMessage(content=prompt)])
        results = response.content if response.content else "Aucun résultat trouvé."
        logger.info("[Tools] Recherche externe terminée")
    except Exception as e:
        logger.error(f"[Tools] Erreur : {e}", exc_info=True)
        results = ""

    return {
        **state,
        "external_results": results,
        "agent_path": state.get("agent_path", []) + ["tool_use_agent"]
    }


def check_retrieval_node(state: AgentState) -> AgentState:
    """
    Vérifie si les documents récupérés sont VRAIMENT pertinents.

    Problème : LanceDB retourne toujours des résultats même si la question
    est hors domaine (ex: "reinforcement learning" retourne des docs médicaux
    sans rapport). Il faut vérifier la pertinence sémantique.

    Solution : on demande à Groq si les docs correspondent à la question.
    Si non → bascule vers les outils externes (Wikipedia, Google).
    """
    retrieved_docs = state.get("retrieved_docs", "")
    question = state.get("question", "")

    # Vérification 1 : documents vides ou trop courts
    if not retrieved_docs or len(retrieved_docs.strip()) < 50:
        logger.info("[Check] Contexte insuffisant → bascule vers outils externes")
        return {**state, "routing_decision": "tools_fallback"}

    # Vérification 2 : pertinence sémantique avec Groq
    prompt = f"""Question posée : {question}

Extrait des documents récupérés :
{retrieved_docs[:600]}

Ces documents contiennent-ils des informations DIRECTEMENT utiles pour répondre à cette question ?
Réponds UNIQUEMENT par "oui" ou "non"."""

    try:
        response = groq_fast.invoke([HumanMessage(content=prompt)])
        reponse = response.content.strip().lower()
        pertinent = "oui" in reponse

        if pertinent:
            logger.info(f"[Check] Documents pertinents → synthèse")
            return {**state, "routing_decision": "summarize"}
        else:
            logger.info(f"[Check] Documents non pertinents → outils externes")
            return {**state, "routing_decision": "tools_fallback"}

    except Exception as e:
        # En cas d'erreur Groq, on continue avec les documents par défaut
        logger.warning(f"[Check] Erreur vérification pertinence : {e} → synthèse par défaut")
        return {**state, "routing_decision": "summarize"}


def summarization_node(state: AgentState) -> AgentState:
    """
    Nœud de synthèse finale.
    Combine les documents et génère une réponse médicale claire en français.
    """
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", "")
    external_results = state.get("external_results", "")

    logger.info("[Summarization] Génération de la réponse finale...")

    context_parts = []
    if retrieved_docs and retrieved_docs.strip():
        context_parts.append(f"=== Documents de la base médicale ===\n{retrieved_docs}")
    if external_results and external_results.strip():
        context_parts.append(f"=== Résultats de recherche externe ===\n{external_results}")

    if context_parts:
        context = "\n\n".join(context_parts)
        prompt = f"""Tu es un médecin expert. Réponds à la question en français 
en te basant sur les documents fournis. Sois précis, clair et structuré.
Si les documents ne suffisent pas, utilise tes connaissances générales.

{context}

Question : {question}

Réponse en français :"""
    else:
        prompt = f"""Tu es un expert. Réponds à cette question en français 
de manière précise et structurée.

Question : {question}

Réponse en français :"""

    response = groq_smart.invoke([HumanMessage(content=prompt)])
    logger.info("[Summarization] Réponse générée")

    return {
        **state,
        "final_response": response.content,
        "agent_path": state.get("agent_path", []) + ["summarization_agent"]
    }


def direct_response_node(state: AgentState) -> AgentState:
    """
    Nœud de réponse directe pour les salutations et questions hors domaine.
    """
    question = state["question"]
    logger.info("[Direct] Réponse directe...")

    prompt = f"""Tu es un assistant médical en français. 
Réponds brièvement et chaleureusement à ce message :
{question}"""

    response = groq_smart.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "final_response": response.content,
        "agent_path": state.get("agent_path", []) + ["orchestrateur_direct"]
    }

# ============================================================
# ROUTEURS
# ============================================================

def route_after_orchestrator(state: AgentState) -> str:
    """Routing après l'orchestrateur."""
    decision = state.get("routing_decision", "direct")
    if decision == "retrieval":
        return "retrieval"
    elif decision == "tools":
        return "tools"
    else:
        return "direct"


def route_after_check(state: AgentState) -> str:
    """Routing après vérification de la pertinence."""
    decision = state.get("routing_decision", "summarize")
    return "tools" if decision == "tools_fallback" else "summarize"

# ============================================================
# CONSTRUCTION DU GRAPHE
# ============================================================

builder = StateGraph(AgentState)

# Ajout des nœuds
builder.add_node("orchestrateur",       orchestrator_node)
builder.add_node("retrieval_agent",     retrieval_node)
builder.add_node("tool_use_agent",      tools_node)
builder.add_node("check_retrieval",     check_retrieval_node)
builder.add_node("summarization_agent", summarization_node)
builder.add_node("direct_response",     direct_response_node)

# Point d'entrée
builder.add_edge(START, "orchestrateur")

# Routing après orchestrateur
builder.add_conditional_edges(
    "orchestrateur",
    route_after_orchestrator,
    {
        "retrieval": "retrieval_agent",
        "tools":     "tool_use_agent",
        "direct":    "direct_response"
    }
)

# Après retrieval → vérification de pertinence
builder.add_edge("retrieval_agent", "check_retrieval")

# Routing après vérification
builder.add_conditional_edges(
    "check_retrieval",
    route_after_check,
    {
        "tools":     "tool_use_agent",
        "summarize": "summarization_agent"
    }
)

# Après tools → synthèse
builder.add_edge("tool_use_agent",      "summarization_agent")

# Fin du graphe
builder.add_edge("summarization_agent", END)
builder.add_edge("direct_response",     END)

# Compilation
graph = builder.compile()

logger.info("✓ Graphe LangGraph compilé avec succès")
logger.info("  Flux : orchestrateur → retrieval → [check pertinence] → tools/summarization → END")