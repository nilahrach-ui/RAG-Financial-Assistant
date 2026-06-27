# RAG Financial Assistant

## Description
Ce projet consiste à développer un système de question-réponse basé sur une architecture RAG (Retrieval Augmented Generation).

Le système permet d'interroger un document PDF et de générer des réponses pertinentes grâce à un modèle de langage.

## Technologies utilisées
- Python
- LangChain
- LangGraph
- ChromaDB
- HuggingFace Embeddings
- Llama 3 avec Ollama

## Fonctionnement
1. Chargement du document PDF
2. Découpage du document en plusieurs parties
3. Transformation en vecteurs avec un modèle d'embedding
4. Recherche des passages pertinents
5. Génération de la réponse avec le modèle LLM

## Installation

Installer les dépendances :

pip install -r requirements.txt

## Auteur
Nisrine Lahrach
