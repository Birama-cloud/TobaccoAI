# TobaccoAI

**Assistant documentaire sur le tabagisme au Sénégal, fondé sur l'approche RAG (Retrieval-Augmented Generation).**

TobaccoAI permet d'interroger en langage naturel des rapports officiels sur le tabagisme (enquête GATS Sénégal 2023, rapport de l'OMS…) et d'obtenir des réponses qui **citent le document et la page** d'où vient chaque information. Un tableau de bord présente aussi les indicateurs clés de l'enquête GATS 2023.

## Fonctionnalités

- **Questions-réponses sur plusieurs rapports PDF** : on dépose les rapports dans l'application, ils sont découpés, indexés, puis interrogeables.
- **Citations systématiques** : chaque réponse indique le document et le numéro de page utilisés, et les extraits sont consultables.
- **Garde-fou contre les réponses inventées** : si aucun passage n'est assez proche de la question (distance cosinus supérieure à 0,78), l'application répond qu'elle ne trouve pas l'information au lieu de générer une réponse.
- **Tableau de bord GATS 2023** : prévalence par âge, sexe et milieu de résidence, évolution 2015 → 2023. Ces chiffres sont saisis à la main depuis le rapport, avec la page source, et ne sont pas produits par l'IA.
- **Script d'évaluation** : un jeu de 18 questions (dont 3 questions pièges) mesure la récupération de la bonne page, l'exactitude des réponses et le bon refus sur les questions hors sujet.

## Fonctionnement

1. **Extraction** du texte de chaque PDF, page par page (pypdf).
2. **Découpage** en morceaux d'environ 220 mots avec 40 mots de chevauchement, sans jamais mélanger deux pages.
3. **Vectorisation** avec un modèle multilingue (`paraphrase-multilingual-MiniLM-L12-v2`, sentence-transformers), adapté aux rapports en français.
4. **Stockage** dans une base vectorielle persistante ChromaDB (distance cosinus).
5. **Recherche** des 5 passages les plus proches de la question, filtrés par le seuil de pertinence.
6. **Génération** de la réponse par `gpt-4o-mini` (OpenAI), à partir de ces seuls passages, avec citation des pages.

## Installation

Python 3.10 ou plus récent.

```
git clone https://github.com/Birama-cloud/TobaccoAI.git
cd TobaccoAI
python -m venv .venv
.venv\Scripts\activate           (Windows)
source .venv/bin/activate        (Linux, macOS)
pip install -r requirements.txt
```

## Base de connaissances

Les rapports qui forment la base de connaissances sont inclus dans le dossier `documents/` :

- **Enquête mondiale sur le tabagisme chez les adultes (GATS) – Sénégal 2023**, Agence nationale de la statistique et de la démographie (ANSD) : `documents/GATS_2023.pdf`.
- **Rapport de l'OMS sur les tendances de la prévalence du tabagisme 2000–2024 et projections 2025–2030**, © Organisation mondiale de la santé 2025, sous licence [CC BY-NC-SA 3.0 IGO](https://creativecommons.org/licenses/by-nc-sa/3.0/igo) : `documents/Rapport OMS_Tabagisme 2000_2024  projections 2025_2030.pdf`.

D'autres rapports PDF peuvent être ajoutés dans ce dossier ou déposés directement dans l'application.

## Utilisation

```
streamlit run application.py
```

1. Saisissez votre clé API OpenAI dans la barre latérale. Elle reste dans la session et n'est jamais enregistrée.
2. Cliquez sur **Indexer les nouveaux documents**.
3. Posez vos questions dans l'onglet **Assistant**, ou consultez l'onglet **Tableau de bord**.

### Évaluer le pipeline

Après avoir indexé les documents dans l'application :

```
set OPENAI_API_KEY=sk-...        (Windows)
export OPENAI_API_KEY=sk-...     (Linux, macOS)
python evaluate_rag.py
```

Le script affiche un résumé et enregistre le détail dans `eval_results.csv`.

## Limites connues

- L'extraction de texte des PDF perd une partie de la structure des tableaux : certains chiffres présents dans des tableaux sont mal retrouvés.
- Le seuil de pertinence (0,78) a été réglé à la main et reste à calibrer sur un jeu de test plus large.
- Les réponses dépendent d'une API payante (OpenAI).

## Technologies

Python · Streamlit · pypdf · sentence-transformers · ChromaDB · OpenAI (`gpt-4o-mini`) · pandas

## Auteur

Birama TOGOLA · [github.com/Birama-cloud](https://github.com/Birama-cloud)

## Licence

Code sous licence MIT, voir `LICENSE`. Les rapports du dossier `documents/` restent la propriété de leurs auteurs (ANSD, OMS) et sont redistribués à des fins non commerciales, avec leurs licences d'origine.
