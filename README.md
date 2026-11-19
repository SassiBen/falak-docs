# Falak Docs

**Posez une question, l'assistant cherche la réponse dans les documents de l'entreprise et vous dit où il l'a trouvée.**

Dans toute entreprise, les réponses existent déjà : dans les conditions
générales, les procédures, la charte informatique, le guide d'accueil. Le
problème, c'est que personne ne les relit. Falak Docs les lit pour vous et
répond en citant le passage exact, avec la page.

Ce type d'assistant s'appelle un **RAG** (*retrieval-augmented generation*) :
d'abord on cherche les bons passages, ensuite seulement on demande à une IA de
rédiger la réponse à partir d'eux, et uniquement d'eux.

![Architecture](docs/architecture.svg)

## Un exemple

```
$ falak-docs ask "Sous combien de jours faut-il confirmer des réserves ?"

Les réserves doivent être confirmées par lettre recommandée dans les 3 jours
ouvrés suivant la livraison [1].
  [1] Conditions générales de transport — Réclamations > Avaries et manquants
```

Et quand la réponse n'est pas dans les documents, il le dit au lieu d'inventer :

```
$ falak-docs ask "Quel est le cours de l'action Delorme ?"
Je ne trouve pas cette information dans les documents disponibles.
```

## En local ou dans le cloud, au choix

La recherche se fait toujours sur la machine : les documents ne partent nulle
part. Pour la rédaction, on choisit :

- **En local, avec Ollama.** Rien ne sort de l'entreprise. Idéal pour les
  documents sensibles.
- **Dans le cloud, avec l'API Claude.** Seuls les cinq passages retenus pour
  la question sont envoyés, jamais le reste des documents. On gagne la qualité
  de rédaction d'un grand modèle, et c'est l'API elle-même qui relie chaque
  phrase de la réponse à sa source.

On passe de l'un à l'autre avec une option, sans rien réindexer.

## Comment il trouve les bons passages

1. **Il lit** les documents en Markdown, HTML et PDF, en gardant les titres et
   les numéros de page.
2. **Il découpe** chaque document en passages qui suivent ses sections, pour
   qu'un passage ne mélange jamais deux sujets.
3. **Il cherche de deux façons** à la fois : par le sens (les embeddings, pour
   retrouver une question formulée autrement) et par les mots exacts (BM25,
   pour retrouver « L133-3 » ou « B100 »). Les deux résultats sont ensuite
   fusionnés.
4. **Il ne demande rien à l'IA** si aucun passage n'est assez proche : c'est la
   meilleure protection contre une réponse inventée.

## Ce que les mesures m'ont appris

J'ai mesuré séparément la recherche et la rédaction, sur 14 questions dont
deux sans réponse dans les documents.

- **La recherche était parfaite** dès le départ : pour chaque question, le bon
  document arrive en premier.
- **Les erreurs venaient de la rédaction.** Avec un petit modèle, deux réponses
  sur la charte informatique (un PDF) étaient fausses. Le modèle répondait « je
  ne trouve pas » dès que la question n'utilisait pas les mots du document.
  Un prompt mieux formulé a fait passer le taux de bonnes réponses de 86 % à
  93 %.

Sans mesurer les deux étapes séparément, j'aurais cherché le problème au mauvais
endroit.

Le dossier complet est dans [docs/falak-docs.pdf](docs/falak-docs.pdf).

## L'essayer

```bash
pip install -e .
ollama pull bge-m3
ollama pull qwen2.5:7b-instruct-q4_K_S

falak-docs ingest corpus/
falak-docs ask "Puis-je brancher ma clé USB personnelle sur mon poste ?"
falak-docs eval --answers
```

Avec l'API Claude pour la rédaction :

```bash
export ANTHROPIC_API_KEY=...
falak-docs ask --provider claude "Quelle part du carburant est du B100 ?"
```

Le modèle utilisé par défaut est `claude-opus-5-5`, modifiable avec
`FALAKDOCS_CLAUDE_MODEL`. Si un filtre de sécurité refuse une requête, l'API la
rejoue sur un autre modèle au lieu de renvoyer une réponse vide.

Pour une API web : `falak-docs serve`, puis `POST /v1/ask`.

## Les documents d'exemple

Le dossier `corpus/` contient cinq documents d'une entreprise imaginaire,
Transports Delorme, dans trois formats : conditions générales, procédure de
sécurité, guide d'accueil (Markdown), charte informatique (PDF) et rapport RSE
(page web). Remplacez-les par les vôtres.

## Sous le capot

- **bge-m3** (via Ollama) pour transformer les textes en vecteurs, en local.
- **ChromaDB** pour stocker et chercher ces vecteurs.
- **BM25** pour la recherche par mots exacts, et une fusion par rang (RRF).
- **Ollama** ou le **SDK officiel Anthropic** pour la rédaction.
- **FastAPI** pour l'API.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Les 21 tests n'appellent aucun modèle : les API d'Ollama et d'Anthropic sont
simulées, ce qui vérifie aussi la forme exacte des requêtes envoyées.

---

Transports Delorme est une entreprise inventée pour l'exemple.
Projet sous licence MIT, par Sassi Benalouache ([Falakion](https://falakion.com)).
