import lancedb
import os
import logging
import uuid
import pandas as pd
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Paramètres LanceDB
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb_data")
TABLE_NAME = os.getenv("LANCEDB_TABLE", "medical_rag_table")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Initialisation de la connexion LanceDB
if not os.path.exists(LANCEDB_URI):
    os.makedirs(LANCEDB_URI)

db = lancedb.connect(LANCEDB_URI)

import pyarrow as pa

def init_table():
    """
    Initialise la table LanceDB si elle n'existe pas.
    """
    try:
        if TABLE_NAME not in db.table_names():
            logger.info(f"Création de la table '{TABLE_NAME}' dans {LANCEDB_URI}")
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)),
                pa.field("source", pa.string()),
                pa.field("title", pa.string()),
                pa.field("domain", pa.string()),
            ])
            return db.create_table(TABLE_NAME, schema=schema)
        else:
            logger.info(f"La table '{TABLE_NAME}' existe déjà.")
            return db.open_table(TABLE_NAME)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la table : {e}")
        raise

def store_chunks(chunks: list[dict], metadata: dict = None) -> int:
    """
    Stocke les fragments de documents dans LanceDB.
    
    Args:
        chunks: Liste de dictionnaires avec 'text' et 'embedding'
        metadata: Métadonnées additionnelles (source, title, domain)
    Returns:
        Nombre de fragments stockés
    """
    if not chunks:
        logger.warning("Aucun fragment à stocker.")
        return 0
    
    table = init_table()
    
    data_to_insert = []
    for chunk in chunks:
        record = {
            "id": str(uuid.uuid4()),
            "text": chunk.get('text', ''),
            "vector": chunk.get('embedding'),
            "source": metadata.get('source', 'unknown') if metadata else chunk.get('source', 'unknown'),
            "title": metadata.get('title', 'N/A') if metadata else chunk.get('title', 'N/A'),
            "domain": metadata.get('domain', 'medical') if metadata else chunk.get('domain', 'medical')
        }
        data_to_insert.append(record)
    
    table.add(data_to_insert)
    logger.info(f"{len(data_to_insert)} fragments stockés dans la table '{TABLE_NAME}'.")
    return len(data_to_insert)

def search_similar(query_embedding: list, limit: int = 5) -> list[dict]:
    """
    Recherche des fragments de documents similaires dans LanceDB.
    
    Args:
        query_embedding: Vecteur d'embedding de la requête
        limit: Nombre de résultats à retourner
    Returns:
        Liste de dictionnaires avec 'text', 'score', 'source', 'title', 'domain'
    """
    try:
        table = db.open_table(TABLE_NAME)
        # Recherche vectorielle (dense)
        results = table.search(query_embedding).limit(limit).to_pandas()
        
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append({
                "text": row['text'],
                "score": row.get('_distance', 0), # LanceDB retourne la distance L2 par défaut
                "source": row['source'],
                "title": row['title'],
                "domain": row['domain']
            })
        
        logger.info(f"{len(formatted_results)} fragments similaires trouvés.")
        return formatted_results
    except Exception as e:
        logger.error(f"Erreur lors de la recherche : {e}")
        return []
