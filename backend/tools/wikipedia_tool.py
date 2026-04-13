import httpx
import logging

logger = logging.getLogger(__name__)

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
BASE_URL = "https://en.wikipedia.org/api/rest_v1"
HEADERS = {"User-Agent": "MedicalAssistantBot/1.0 (Educational; Python/httpx)"}


def search_wikipedia(query: str, max_results: int = 5) -> dict:
    """Recherche des articles Wikipedia correspondant à la requête."""
    try:
        params = {
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": min(max_results, 10),
            "format": "json", "utf8": 1
        }
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            response = client.get(SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
            results.append({
                "title": item.get("title"),
                "snippet": snippet,
                "url": f"https://en.wikipedia.org/wiki/{item.get('title','').replace(' ','_')}"
            })
        return {"status": "success", "query": query, "results": results}
    except Exception as e:
        return {"status": "error", "query": query, "error": str(e), "results": []}


def get_wikipedia_summary(title: str) -> dict:
    """Récupère le résumé d'un article Wikipedia."""
    try:
        encoded = title.replace(" ", "_")
        url = f"{BASE_URL}/page/summary/{encoded}"
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        return {
            "status": "success",
            "title": data.get("title"),
            "extract": data.get("extract"),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page")
        }
    except Exception as e:
        return {"status": "error", "title": title, "error": str(e)}


# Export des fonctions Python directement (compatible LangChain)
wikipedia_tools = [search_wikipedia, get_wikipedia_summary]