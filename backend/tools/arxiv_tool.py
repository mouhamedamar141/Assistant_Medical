import arxiv
import logging

logger = logging.getLogger(__name__)


def search_arxiv(query: str, max_results: int = 5) -> dict:
    """Recherche des articles académiques sur arXiv."""
    max_results = min(max_results, 10)
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary[:500],
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.entry_id,
            })
        return {"status": "success", "query": query, "results": results}
    except Exception as e:
        return {"status": "error", "query": query, "error": str(e), "results": []}


def get_arxiv_paper(arxiv_id: str) -> dict:
    """Récupère les détails d'un article arXiv par son identifiant."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search), None)
        if paper is None:
            return {"status": "error", "error": "Article non trouvé"}
        return {
            "status": "success",
            "result": {
                "title": paper.title,
                "abstract": paper.summary,
                "url": paper.entry_id,
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Export des fonctions Python directement (compatible LangChain)
arxiv_tools = [search_arxiv, get_arxiv_paper]