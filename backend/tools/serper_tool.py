import httpx
import os
import logging

logger = logging.getLogger(__name__)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_BASE_URL = "https://google.serper.dev"


def search_google(query: str, num_results: int = 5) -> dict:
    """Recherche Google via l'API Serper."""
    try:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query, "num": min(num_results, 10)}
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{SERPER_BASE_URL}/search", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        results = [
            {"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")}
            for r in data.get("organic", [])
        ]
        return {"status": "success", "query": query, "results": results}
    except Exception as e:
        return {"status": "error", "query": query, "error": str(e), "results": []}


def search_google_news(query: str, num_results: int = 5) -> dict:
    """Recherche des actualités médicales via Google News."""
    try:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query, "num": min(num_results, 10)}
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{SERPER_BASE_URL}/news", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        results = [
            {"title": r.get("title"), "url": r.get("link"),
             "snippet": r.get("snippet"), "date": r.get("date")}
            for r in data.get("news", [])
        ]
        return {"status": "success", "query": query, "results": results}
    except Exception as e:
        return {"status": "error", "query": query, "error": str(e), "results": []}


# Export des fonctions Python directement (compatible LangChain)
serper_tools = [search_google, search_google_news]