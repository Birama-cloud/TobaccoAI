# TobaccoAI

**Plateforme d'information en santé publique pour le Sénégal, destinée aux chercheurs, aux étudiants et aux organisations (ONG, associations, institutions).**

Elle réunit un assistant qui interroge un corpus de documents officiels en **citant le document et la page** de chaque information (approche RAG, *Retrieval-Augmented Generation*), et un espace de données chiffrées et sourcées. Le nom définitif de la plateforme reste à choisir (`NOM_PLATEFORME` dans `config.py`).

| Thème | Dossier | État |
|---|---|---|
| Tabac | `documents/tabac/` | GATS Sénégal 2023 (ANSD) et rapport OMS indexés ; tableau de bord |
| Cancer du sein | `documents/cancer_sein/` | En attente des documents officiels |
| Cancer du col de l'utérus | `documents/cancer_col/` | En attente des documents officiels |
| Paludisme | `documents/paludisme/` | En attente des documents officiels |

## Les pages

| Page | Contenu |
|---|---|
| **Accueil** | Accroche, barre « Interroger le corpus » (ouvre l'Assistant avec la question), chiffres clés par thème, publics, thèmes |
| **Assistant** | Questions-réponses sourcées : choix du thème, filtre par type de source, extraits avec score de similarité, référence à copier, suggestions |
| **Synthèses** | Bientôt disponible |
| **Cartographie** | Bientôt disponible |
| **Espace données** | Un onglet par thème : indicateurs GATS 2023, tableau des sources, export CSV |

Les maquettes validées sont dans `docs/maquettes/`.

## Règles de fiabilité

- **Jamais de réponse sans source** : si aucun extrait ne dépasse le seuil de pertinence (distance cosinus ≤ 0,78), l'assistant le dit et propose de reformuler, sans appeler le modèle.
- **Chaque affirmation est suivie de sa source** (document, page) ; si les documents ne permettent pas de répondre, l'assistant l'écrit au lieu de deviner.
- **Les chiffres de l'Espace données et de l'Accueil ne sont pas produits par l'IA** : ils sont saisis à la main dans `indicateurs.py`, avec leur page. Un test (`tests/test_indicateurs.py`) fige les chiffres vérifiés du GATS 2023. Un thème sans document affiche « données à venir ».
- **Aucun document, chiffre ou source n'est inventé** : chaque PDF est décrit dans `documents/sources.json` à partir des informations écrites dans le document lui-même.

## Santé et données personnelles

- L'assistant **informe** à partir des sources. Il ne pose **aucun diagnostic** et ne donne **aucun conseil de traitement** (règles inscrites dans le prompt, `rag/generation.py`).
- L'encadré « Important » rappelle que les réponses peuvent contenir des erreurs, que les questions ne sont pas enregistrées, et qu'il ne faut y inclure aucune donnée permettant d'identifier une personne.
- **Les questions ne sont jamais enregistrées sur le disque** : l'historique de conversation reste en mémoire de session. Les questions sont en revanche transmises au fournisseur du modèle (OpenAI, Gemini) ; avec Ollama, tout reste sur la machine.
- Tout texte affiché en HTML (question, réponse, titre, extrait) est échappé.
- Les statistiques d'usage de Streamlit et la télémétrie de ChromaDB sont désactivées.

## Structure du code

```
application.py            Point d'entrée Streamlit (navigation à 5 pages)
config.py                 Réglages : nom, thèmes et couleurs, types de source, modèle, seuil, TOP_K...
indicateurs.py            Chiffres saisis à la main, avec source et page (Accueil, Espace données, CSV)
index_corpus.py           Indexation en ligne de commande
evaluate_rag.py           Évaluation par thème (même code que l'application)
llm/client.py             Client du modèle de langage (OpenAI, Gemini, Ollama)
rag/pdf.py                Extraction du texte, page par page
rag/chunking.py           Découpage en morceaux (jamais à cheval sur deux pages)
rag/sources.py            Registre des sources (documents/sources.json)
rag/store.py              Embeddings locaux, ChromaDB (une collection par thème), indexation
rag/retrieval.py          Recherche, filtre par type de source, seuil de pertinence
rag/generation.py         Prompt unique et citations
rag/pipeline.py           answer_question() : utilisée par l'application ET l'évaluation
ui/style.css              Feuille de style unique
ui/layout.py              En-tête et navigation
ui/formatting.py          Blocs HTML (texte échappé)
ui/pages/                 Une fonction render() par page
.streamlit/config.toml    Thème (couleurs, police Plus Jakarta Sans)
documents/<thème>/        PDF de chaque thème
documents/sources.json    Registre des sources
tests/                    Tests automatisés (pytest), sans clé d'API
```

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

## Configuration

Le choix du modèle et la clé d'API se règlent dans un fichier `.env`, jamais envoyé sur GitHub :

```
copy .env.example .env           (Windows)
cp .env.example .env             (Linux, macOS)
```

| Variable | Rôle |
|---|---|
| `LLM_PROVIDER` | `openai` (par défaut), `gemini` ou `ollama` |
| `LLM_MODEL` | Nom du modèle (facultatif pour OpenAI : `gpt-4o-mini`) |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | Clé d'API (si elle manque, un champ apparaît dans l'Assistant) |
| `LLM_BASE_URL` | Adresse du service, si elle diffère de l'adresse par défaut |

Ne mettez jamais de clé dans le code.

## Utilisation

```
streamlit run application.py
```

Au premier démarrage, l'application indexe les documents (une à deux minutes).

### Ajouter un document

1. Décrivez-le dans `documents/sources.json` :
   - `fichier`, `titre`, `producteur`, `annee`, `licence` (telle qu'elle est écrite dans le document) ;
   - `themes` : la **liste** de ses thèmes, par exemple `["paludisme", "cancer_sein", "cancer_col"]`. Le document est indexé dans la collection de **chacun** de ces thèmes ;
   - `type_source` : `enquete_nationale`, `oms`, `circ`, `pnlp` ou `ministere` ;
   - `url` : la page officielle du document ;
   - `publiable` : `true` si la licence permet de le publier sur GitHub (licence ouverte, par exemple CC BY-NC-SA de l'OMS), sinon `false`.
2. Placez le PDF dans le dossier de son **premier** thème : `documents/<themes[0]>/<fichier>`.
3. Si `publiable` vaut `false`, ajoutez son nom au `.gitignore` (une ligne `documents/**/<fichier>`). Un test vérifie cette règle.
4. Lancez `python index_corpus.py`, puis redémarrez l'application.

Un PDF absent du registre est indexé avec un avertissement, sans titre ni type de source : il n'apparaît pas quand on filtre par type. Un PDF déclaré mais introuvable, ou rangé dans un autre dossier que celui de son premier thème, est signalé et n'est pas indexé.

Les documents candidats, pas encore vérifiés, attendent dans `documents/_a_verifier/` (ignoré par Git et par l'indexation), avec leur fiche dans `documents/_a_verifier/CANDIDATS.md`.

### Documents non publiables : les télécharger soi-même

Seuls les documents à licence ouverte sont publiés dans ce dépôt. Les autres (licence non précisée ou « tous droits réservés ») sont listés dans `documents/sources.json` avec `"publiable": false` : ils sont ignorés par Git et restent sur la machine de chacun. Après un `git clone`, pour les obtenir :

1. Ouvrez `documents/sources.json` et repérez les entrées `"publiable": false`.
2. Téléchargez chaque document depuis son lien officiel (champ `url`).
3. Enregistrez-le **sous le nom exact** indiqué dans `fichier`, dans le dossier de son premier thème (`documents/<themes[0]>/`).
4. Lancez `python index_corpus.py`.

Si un document manque, l'application fonctionne quand même : l'indexation affiche un avertissement et l'assistant répond à partir des documents présents.

### Évaluer le pipeline

```
python evaluate_rag.py --theme tabac
python evaluate_rag.py --theme tabac --pause 13     (offre gratuite Gemini : 5 requêtes par minute)
```

Résumé à l'écran et détail dans `eval_results_tabac.csv`.

### Lancer les tests

```
pip install -r requirements-dev.txt
python -m pytest
```

## Base de connaissances (thème tabac)

- **Enquête mondiale sur le tabagisme chez les adultes (GATS) – Sénégal 2023, rapport principal**, ANSD et ministère de la Santé et de l'Action sociale (publié en octobre 2025) : `documents/tabac/GATS_2023.pdf`. Le document ne précise pas de licence ; il est maintenu dans le dépôt par décision du responsable du projet (rapport public de l'ANSD), en exception à la règle des licences ouvertes.
- **WHO global report on trends in prevalence of tobacco use 2000–2024 and projections 2025–2030** (en anglais), © Organisation mondiale de la Santé 2025, licence [CC BY-NC-SA 3.0 IGO](https://creativecommons.org/licenses/by-nc-sa/3.0/igo) : `documents/tabac/Rapport OMS_Tabagisme 2000_2024  projections 2025_2030.pdf`.

## Limites connues

- **La recherche retrouve mal certaines pages** : à l'évaluation, la page attendue du GATS n'est retrouvée que pour 4 questions sur 16. L'extraction du texte des PDF perd la structure des tableaux, et le rapport OMS remonte souvent avant le GATS.
- Le seuil de pertinence (0,78) a été réglé à la main ; les questions pièges proches du sujet (autre pays, autre tranche d'âge) le franchissent, et seul le modèle peut alors refuser.
- L'évaluation avec un vrai modèle de langage reste à mener (quotas OpenAI et Gemini épuisés lors des essais).
- Les thèmes cancer du sein, cancer du col et paludisme n'ont pas encore de documents ni de jeu d'évaluation.

## Technologies

Python · Streamlit · pypdf · sentence-transformers · ChromaDB · OpenAI SDK (OpenAI, Gemini, Ollama) · pandas · pytest

## Auteur

Birama TOGOLA · [github.com/Birama-cloud](https://github.com/Birama-cloud)

## Licence

Code sous licence MIT, voir `LICENSE`. Les rapports du dossier `documents/` restent la propriété de leurs auteurs (ANSD, OMS) et sont redistribués à des fins non commerciales, avec leurs licences d'origine.
