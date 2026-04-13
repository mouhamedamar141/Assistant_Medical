from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Initialisation du driver Neo4j
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    logger.info(f"Connecté à Neo4j : {NEO4J_URI}")
except Exception as e:
    logger.error(f"Erreur connexion Neo4j : {e}")
    neo4j_driver = None

# Types d'entités médicales
ENTITY_TYPES = [
    "Document",      # Document source (PDF)
    "Maladie",       # Pathologies, maladies
    "Symptome",      # Symptômes cliniques
    "Traitement",    # Traitements, thérapies
    "Medicament",    # Médicaments, molécules
    "Examen",        # Examens, analyses
    "Organe",        # Organes, systèmes anatomiques
    "Concept",       # Concepts médicaux généraux
]


def verify_connection() -> bool:
    """Vérifie que la connexion Neo4j fonctionne."""
    if not neo4j_driver:
        return False
    try:
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            return result.single()["ok"] == 1
    except Exception as e:
        logger.error(f"Vérification Neo4j échouée : {e}")
        return False


def store_document_chunks(chunks: list[dict], source: str, title: str):
    """
    Stocke les chunks d'un document dans Neo4j avec extraction d'entités simples.
    Crée un nœud Document central lié aux entités extraites des chunks.

    Args:
        chunks: Liste de dicts avec 'text', 'embedding'
        source: Nom du fichier source
        title: Titre du document
    """
    if not neo4j_driver:
        logger.warning("Neo4j non disponible — stockage ignoré")
        return

    try:
        with neo4j_driver.session() as session:
            # Créer le nœud Document principal
            session.run(
                """
                MERGE (d:Document {source: $source})
                SET d.title = $title, d.chunk_count = $chunk_count
                """,
                source=source,
                title=title,
                chunk_count=len(chunks)
            )
            logger.info(f"Nœud Document créé : {title} ({len(chunks)} chunks)")

            # Extraire et stocker les entités médicales simples depuis les chunks
            # On prend un sample des chunks pour ne pas surcharger Neo4j
            sample_chunks = chunks[:10]
            for chunk in sample_chunks:
                texte = chunk.get("text", "")
                entites = extraire_entites_simples(texte)
                for type_entite, noms in entites.items():
                    for nom in noms:
                        if nom and len(nom) > 3:
                            session.run(
                                f"""
                                MERGE (e:{type_entite} {{name: $nom}})
                                WITH e
                                MATCH (d:Document {{source: $source}})
                                MERGE (d)-[:CONTIENT]->(e)
                                """,
                                nom=nom,
                                source=source
                            )
    except Exception as e:
        logger.error(f"Erreur stockage Neo4j : {e}")


def extraire_entites_simples(texte: str) -> dict:
    """
    Extrait des entités médicales simples par mots-clés.
    Approche légère sans NLP externe.
    """
    texte_lower = texte.lower()
    entites = {
        "Maladie": [],
        "Symptome": [],
        "Traitement": [],
        "Medicament": [],
    }

    # Dictionnaires de mots-clés médicaux
    maladies = [
        "diabète", "hypertension", "tuberculose", "paludisme", "cancer",
        "pneumonie", "infarctus", "avc", "cholera", "vih", "sida",
        "drépanocytose", "malnutrition", "sepsis", "méningite", "dengue",
        "covid", "grippe", "hépatite", "cirrhose", "insuffisance",
        "lyme", "g6pd", "sphérocytose", "anémie", "tsa"
    ]
    symptomes = [
        "fièvre", "douleur", "toux", "dyspnée", "fatigue", "nausée",
        "vomissement", "diarrhée", "céphalée", "vertiges", "œdème",
        "hypotension", "tachycardie", "pâleur", "ictère", "convulsion",
        "confusion", "frissons", "sueurs", "asthénie", "anorexie"
    ]
    traitements = [
        "chirurgie", "radiothérapie", "chimiothérapie", "dialyse",
        "transplantation", "vaccination", "antibiothérapie", "réanimation",
        "corticothérapie", "immunothérapie", "oxygénothérapie", "transfusion"
    ]
    medicaments = [
        "aspirine", "paracétamol", "ibuprofène", "amoxicilline", "métformine",
        "insuline", "amlodipine", "artémisinine", "doxycycline", "cotrimoxazole",
        "rifampicine", "isoniazide", "hydroxychloroquine", "azithromycine",
        "ciprofloxacine", "héparine", "warfarine", "methotrexate"
    ]

    for m in maladies:
        if m in texte_lower:
            entites["Maladie"].append(m.capitalize())

    for s in symptomes:
        if s in texte_lower:
            entites["Symptome"].append(s.capitalize())

    for t in traitements:
        if t in texte_lower:
            entites["Traitement"].append(t.capitalize())

    for med in medicaments:
        if med in texte_lower:
            entites["Medicament"].append(med.capitalize())

    return entites


def search_graph(query: str, limit: int = 10) -> list[dict]:
    """
    Recherche dans le graphe Neo4j les entités médicales liées à la requête.
    Retourne les documents et entités pertinents.

    Args:
        query: Requête de recherche
        limit: Nombre maximum de résultats

    Returns:
        Liste de dicts avec les relations trouvées
    """
    if not neo4j_driver:
        logger.warning("Neo4j non disponible")
        return []

    # Extraire les termes de recherche (mots > 3 caractères)
    termes = [t.strip().lower() for t in query.split() if len(t.strip()) > 3]
    if not termes:
        return []

    try:
        with neo4j_driver.session() as session:
            # Recherche 1 : entités médicales correspondant aux termes
            graph_query = """
            MATCH (e)-[r]-(related)
            WHERE any(term IN $termes WHERE toLower(e.name) CONTAINS term)
            RETURN DISTINCT
                labels(e)[0]       AS type_entite,
                e.name             AS nom_entite,
                type(r)            AS relation,
                labels(related)[0] AS type_lie,
                related.name       AS nom_lie
            LIMIT $limit
            """
            result = session.run(graph_query, termes=termes, limit=limit)
            records = list(result)

            resultats = []
            for rec in records:
                resultats.append({
                    "type_entite": rec["type_entite"],
                    "nom_entite": rec["nom_entite"],
                    "relation": rec["relation"],
                    "type_lie": rec["type_lie"],
                    "nom_lie": rec["nom_lie"],
                })

            # Recherche 2 : documents contenant ces entités
            doc_query = """
            MATCH (d:Document)-[:CONTIENT]->(e)
            WHERE any(term IN $termes WHERE toLower(e.name) CONTAINS term)
            RETURN DISTINCT d.title AS titre, d.source AS source,
                   collect(DISTINCT e.name) AS entites_trouvees
            LIMIT $limit
            """
            doc_result = session.run(doc_query, termes=termes, limit=limit)
            for rec in doc_result:
                resultats.append({
                    "type_entite": "Document",
                    "nom_entite": rec["titre"],
                    "relation": "CONTIENT",
                    "type_lie": "Entités",
                    "nom_lie": ", ".join(rec["entites_trouvees"][:5]),
                    "source": rec["source"]
                })

            logger.info(f"Neo4j : {len(resultats)} résultats pour '{query}'")
            return resultats

    except Exception as e:
        logger.error(f"Erreur recherche Neo4j : {e}")
        return []


def store_article_with_entities(article_data: dict):
    """
    Stocke un article et ses entités extraites.
    Compatible avec l'ancien code d'ingestion.
    """
    if not neo4j_driver:
        return

    title = article_data.get("title", "")
    source = article_data.get("source_url", title)

    try:
        with neo4j_driver.session() as session:
            session.run(
                "MERGE (d:Document {source: $source}) SET d.title = $title",
                source=source, title=title
            )
            for topic in article_data.get("topics", []):
                session.run(
                    """
                    MERGE (e:Concept {name: $nom})
                    WITH e MATCH (d:Document {source: $source})
                    MERGE (d)-[:CONTIENT]->(e)
                    """,
                    nom=topic, source=source
                )
        logger.info(f"Article stocké dans Neo4j : {title}")
    except Exception as e:
        logger.error(f"Erreur store_article : {e}")


def close_connection():
    """Ferme la connexion Neo4j."""
    if neo4j_driver:
        neo4j_driver.close()
        logger.info("Connexion Neo4j fermée")