# 🩺 Assistant de Recherche Médicale RAG Autonome

Une application de **RAG (Retrieval-Augmented Generation) Hybride** et **Multi-Agents** spécialisée dans le domaine médical, entièrement en français.

## 🚀 Fonctionnalités Clés

- **Stockage Vectoriel Local** : Utilisation de **LanceDB** pour une recherche sémantique rapide sans Docker.
- **RAG Hybride** : Combinaison de recherche dense (LanceDB), lexicale (BM25) et de graphe (Neo4j) avec fusion RRF.
- **Multi-Agents Intelligent** : Orchestration via Gemini 2.0 Flash Lite et Groq (Llama 3.3 70B & 3.1 8B).
- **Interface Intuitive** : Application Streamlit avec upload de documents et visualisation du flux des agents.
- **Évaluation Rigoureuse** : Notebook de comparaison des approches RAG avec métriques ROUGE, BLEU et RAGAS.

## 🛠️ Architecture des Agents

L'application utilise une hiérarchie d'agents spécialisés :
1. **Orchestrateur (Gemini)** : Point d'entrée, gère les salutations et route les requêtes.
2. **Planning (Groq Llama 3.3 70B)** : Cerveau stratégique, décide de la source de données.
3. **Retrieval (Groq Llama 3.1 8B)** : Expert en recherche hybride (LanceDB + Neo4j).
4. **Tool Use (Groq Llama 3.1 8B)** : Accès aux outils externes (arXiv, Wikipedia, Google).
5. **Summarization (Gemini)** : Synthèse finale des informations en français médical.

## 📋 Prérequis

- Python 3.10+
- Clé API Google (Gemini)
- Clé API Groq
- Instance Neo4j (optionnel pour le graphe)

## ⚙️ Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/mouhamedamar141/Autonomous_research_assistant
   cd Autonomous_research_assistant
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurer le fichier `.env` avec vos clés API.

4. Lancer le backend :
   ```bash
   uvicorn backend.api:app --reload
   ```

5. Lancer le frontend :
   ```bash
   streamlit run frontend/app.py
   ```

## 📚 Sources de Données Médicales Recommandées

Pour tester l'application, vous pouvez indexer les sources suivantes :
- [PubMed Central (PMC)](https://www.ncbi.nlm.nih.gov/pmc/) - Articles en accès libre.
- [Orphanet](https://www.orpha.net/) - Informations sur les maladies rares.
- [Vidal](https://www.vidal.fr/) - Base de données médicamenteuse.
- [Haute Autorité de Santé (HAS)](https://www.has-sante.fr/) - Recommandations de bonnes pratiques.

## 📊 Évaluation

Consultez le fichier `comparaison_rag.ipynb` pour voir l'analyse comparative des performances entre le RAG simple, hybride et multi-agents. Les résultats sont sauvegardés dans `data/resultats_comparaison.csv`.

## 📄 Licence

