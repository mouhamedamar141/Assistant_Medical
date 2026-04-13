from backend.storage.lancedb_store import search_similar, db, TABLE_NAME
from backend.ingestion.embedder import embed_query
from backend.storage.neo4j_store import search_graph
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Fusionne plusieurs listes de résultats via Reciprocal Rank Fusion (RRF).
    Score RRF = Σ 1/(k + rang) pour chaque liste.
    """
    fused_scores = {}
    doc_map = {}

    for results in results_list:
        for rank, result in enumerate(results):
            doc_id = result.get("text", "")
            if not doc_id:
                continue
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = result
            fused_scores[doc_id] += 1.0 / (rank + k)

    # Trier par score RRF décroissant
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for doc_id, score in sorted_docs:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = score
        final_results.append(doc)

    return final_results


def hybrid_search(query: str, limit: int = 5) -> str:
    """
    Recherche hybride combinant 3 sources :
    1. LanceDB  — recherche vectorielle dense (sémantique)
    2. BM25     — recherche lexicale sparse (mots-clés)
    3. Neo4j    — graphe de connaissances médicales (entités/relations)

    Fusion dense + sparse via RRF.
    Neo4j enrichit le contexte avec les relations entre entités médicales.

    Returns:
        str : Contexte formaté pour le LLM
    """
    logger.info(f"Recherche hybride pour : '{query}'")

    # ═══════════════════════════════════════════════════
    # 1. RECHERCHE VECTORIELLE DENSE (LanceDB)
    # ═══════════════════════════════════════════════════
    vector_results = []
    try:
        query_vector = embed_query(query)
        vector_results = search_similar(query_vector, limit=limit * 2)
        logger.info(f"LanceDB : {len(vector_results)} résultats")
    except Exception as e:
        logger.error(f"Erreur LanceDB : {e}")

    # ═══════════════════════════════════════════════════
    # 2. RECHERCHE LEXICALE SPARSE (BM25)
    # ═══════════════════════════════════════════════════
    bm25_results = []
    try:
        table = db.open_table(TABLE_NAME)
        all_docs = table.to_pandas()

        if not all_docs.empty:
            tokenized_corpus = [doc.split() for doc in all_docs["text"].tolist()]
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(query.split())

            top_indices = (
                pd.Series(doc_scores)
                .sort_values(ascending=False)
                .head(limit * 2)
                .index
            )

            for idx in top_indices:
                row = all_docs.iloc[idx]
                if doc_scores[idx] > 0:
                    bm25_results.append({
                        "text":   row["text"],
                        "score":  doc_scores[idx],
                        "source": row.get("source", ""),
                        "title":  row.get("title", ""),
                        "domain": row.get("domain", ""),
                    })

            logger.info(f"BM25 : {len(bm25_results)} résultats")
    except Exception as e:
        logger.error(f"Erreur BM25 : {e}")

    # ═══════════════════════════════════════════════════
    # 3. FUSION RRF (Dense + Sparse)
    # ═══════════════════════════════════════════════════
    fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
    top_fused = fused_results[:limit]

    # ═══════════════════════════════════════════════════
    # 4. RECHERCHE GRAPHE (Neo4j)
    # ═══════════════════════════════════════════════════
    graph_results = []
    try:
        graph_results = search_graph(query, limit=10)
        logger.info(f"Neo4j : {len(graph_results)} résultats")
    except Exception as e:
        logger.error(f"Erreur Neo4j : {e}")

    # ═══════════════════════════════════════════════════
    # 5. CONSTRUCTION DU CONTEXTE FINAL
    # ═══════════════════════════════════════════════════
    output_parts = []

    # Résultats vectoriels + BM25
    if top_fused:
        output_parts.append("=== DOCUMENTS MÉDICAUX (LanceDB + BM25) ===")
        for i, r in enumerate(top_fused, 1):
            titre = r.get("title", r.get("source", "Inconnu"))
            texte = r.get("text", "")[:400]
            score = r.get("rrf_score", 0)
            output_parts.append(
                f"[{i}] Source : {titre} | Score : {score:.4f}\n{texte}"
            )
    else:
        output_parts.append("=== DOCUMENTS MÉDICAUX ===\nAucun document trouvé dans la base vectorielle.")

    # Résultats Neo4j
    if graph_results:
        output_parts.append("\n=== GRAPHE DE CONNAISSANCES MÉDICALES (Neo4j) ===")

        # Séparer les relations et les documents
        relations = [r for r in graph_results if r.get("type_entite") != "Document"]
        documents = [r for r in graph_results if r.get("type_entite") == "Document"]

        if relations:
            output_parts.append("Relations entre entités médicales :")
            for r in relations[:8]:
                output_parts.append(
                    f"  • ({r['type_entite']}) {r['nom_entite']} "
                    f"-[{r['relation']}]-> "
                    f"({r['type_lie']}) {r['nom_lie']}"
                )

        if documents:
            output_parts.append("Documents contenant des entités pertinentes :")
            for d in documents[:5]:
                output_parts.append(
                    f"  • {d['nom_entite']} → entités : {d.get('nom_lie', '')}"
                )
    else:
        output_parts.append("\n=== GRAPHE DE CONNAISSANCES (Neo4j) ===\nAucune entité médicale trouvée.")

    return "\n\n".join(output_parts)