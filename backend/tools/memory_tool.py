import logging
from typing import Optional
from backend.memory.persistent import store_episode, get_recent_episodes

logger = logging.getLogger(__name__)


def store_interaction(
    session_id: str,
    user_query: str,
    response: str,
    agent_path: str,
    tools_used: Optional[list] = None
) -> str:
    """Sauvegarde une interaction dans la mémoire persistante PostgreSQL."""
    try:
        episode = store_episode(
            session_id=session_id,
            user_query=user_query,
            agent_response=response,
            agent_path=agent_path,
            tools_used=tools_used or []
        )
        logger.info(f"Interaction sauvegardée pour la session {session_id}")
        return f"Interaction sauvegardée (épisode {episode.id})"
    except Exception as e:
        logger.error(f"Erreur sauvegarde : {e}")
        return f"Erreur sauvegarde : {e}"


def get_past_interactions(session_id: str, limit: int = 5) -> str:
    """Récupère les interactions passées depuis la mémoire persistante."""
    try:
        episodes = get_recent_episodes(session_id, limit)
        if not episodes:
            return "Aucune interaction passée trouvée."
        output = f"{len(episodes)} interactions passées :\n\n"
        for i, ep in enumerate(episodes, 1):
            output += f"[{i}] Question : {ep.user_query[:100]}\n"
            output += f"    Chemin : {ep.agent_path}\n\n"
        return output
    except Exception as e:
        logger.error(f"Erreur récupération : {e}")
        return f"Erreur récupération : {e}"