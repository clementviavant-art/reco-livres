
import os
import re
import streamlit as st
import pandas as pd
from groq import Groq

# -----------------------------
# CONFIGURATION
# -----------------------------

MODEL = "qwen/qwen3.8-27b"
CSV_FILE = "livres.csv"
MAX_RECOMMENDATIONS = 3


# -----------------------------
# PAGE
# -----------------------------

st.set_page_config(
    page_title="Conseiller de lecture",
    page_icon="📚",
    layout="centered"
)

st.title("📚 On lit quoi ce soir ??")
st.write("Décris tes envies de lecture : l'IA choisira des livres dans ton catalogue.")


# -----------------------------
# LOAD THE CSV
# -----------------------------

@st.cache_data
def load_books():
    # Le fichier fourni utilise un point-virgule et un encodage UTF-8 avec BOM.
    df = pd.read_csv(
        CSV_FILE,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False
    )

    # Nettoyage des noms de colonnes
    df.columns = [col.strip() for col in df.columns]

    # Vérification des colonnes attendues
    required_columns = ["Auteur", "Titre du livre"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing)
            + ". Colonnes trouvées : "
            + ", ".join(df.columns)
        )

    # Nettoyage des données
    df["Auteur"] = df["Auteur"].str.strip()
    df["Titre du livre"] = df["Titre du livre"].str.strip()

    # Retirer les lignes sans titre
    df = df[df["Titre du livre"] != ""].copy()

    # Identifiant simple pour retrouver chaque livre
    df = df.reset_index(drop=True)
    df["_id"] = df.index.astype(str)

    return df


try:
    books = load_books()
except FileNotFoundError:
    st.error(
        f"Impossible de trouver {CSV_FILE}. "
        "Place livres.csv dans le même dossier que app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
    st.stop()

if books.empty:
    st.error("Le fichier CSV ne contient aucun livre avec un titre.")
    st.stop()

st.caption(f"📖 {len(books)} livres disponibles dans le catalogue.")


# -----------------------------
# API KEY
# -----------------------------

try:
    api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key = ""

# Possibilité d'utiliser une variable d'environnement
if not api_key:
    api_key = os.getenv("GROQ_API_KEY", "")

if not api_key:
    st.error(
        "La clé API Groq est introuvable. "
        "Configure GROQ_API_KEY dans le fichier .streamlit/secrets.toml."
    )
    st.stop()


# -----------------------------
# USER PREFERENCES
# -----------------------------

preferences = st.text_area(
    "Qu'as-tu envie de lire ?",
    placeholder=(
        "Exemple : un roman captivant, avec du suspense, "
        "des personnages complexes, sans science-fiction..."
    ),
    height=120
)

avoid = st.text_input(
    "Ce que tu veux éviter (facultatif)",
    placeholder="Ex. : romance, violence, livres trop longs..."
)

submitted = st.button(
    "Trouver des livres",
    type="primary",
    use_container_width=True
)


# -----------------------------
# BUILD A COMPACT CATALOGUE
# -----------------------------

def build_catalogue(df):
    lines = []

    for _, book in df.iterrows():
        author = book["Auteur"].replace("\n", " ").replace("|", "/")
        title = book["Titre du livre"].replace("\n", " ").replace("|", "/")

        # Format compact : identifiant | auteur | titre
        lines.append(f'{book["_id"]}|{author}|{title}')

    return "\n".join(lines)


# -----------------------------
# ASK GROQ
# -----------------------------

def recommend_books(preferences, avoid, df):
    client = Groq(api_key=api_key)

    catalogue = build_catalogue(df)

    prompt = f"""
Tu es un conseiller littéraire.
Choisis au maximum {MAX_RECOMMENDATIONS} livres dans le catalogue fourni.

ENVIES DU LECTEUR :
{preferences}

À ÉVITER :
{avoid if avoid.strip() else "Rien de précisé"}

CATALOGUE COMPLET :
ID|Auteur|Titre
{catalogue}

Ta mission :
1. Sélectionne 3 livres du catalogue qui correspondent aux envies du lecteur.
2. Propose obligatoirement une 4e recommandation extérieure au catalogue.
3. Cette 4e recommandation doit être un livre réel, publié et pertinent pour le lecteur.
4. Ne choisis pas pour la 4e recommandation un livre déjà présent dans le catalogue.
5. N'invente aucun titre ni auteur. Si tu as un doute sur l'existence d'un livre, choisis-en un autre.
6. Pour chaque livre, donne une justification courte et personnalisée.

Réponds uniquement avec 4 lignes, dans ce format exact :

ID: identifiant | raison : justification
ID: identifiant | raison : justification
ID: identifiant | raison : justification
EXTERIEUR: titre | auteur : nom de l'auteur | raison : justification

Pour les trois premières lignes, utilise exclusivement les identifiants du catalogue.
Pour la dernière ligne, indique le titre et l'auteur d'un livre absent du catalogue.
N'ajoute aucun texte avant ou après ces lignes.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un conseiller littéraire précis. "
                    "Tu dois sélectionner les livres uniquement dans "
                    "le catalogue fourni et respecter le format demandé."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.6,
        max_completion_tokens=1000
    )

    return response.choices[0].message.content or ""


# -----------------------------
# VALIDATE THE RECOMMENDATIONS
# -----------------------------

def parse_recommendations(answer, df):
    # On ne fait confiance qu'aux identifiants réellement présents dans le CSV.
    books_by_id = df.set_index("_id", drop=False)

    # Accepte par exemple :
    # ID: 12 | raison : Ce roman est captivant.
    pattern = re.compile(
        r"^\s*ID\s*:\s*(\d+)\s*\|\s*raison\s*:\s*(.+?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    matches = pattern.findall(answer)
    recommendations = []
    seen_ids = set()

    for book_id, reason in matches:
        if book_id not in books_by_id.index:
            continue

        if book_id in seen_ids:
            continue

        seen_ids.add(book_id)
        book = books_by_id.loc[book_id]

        recommendations.append({
            "titre": book["Titre du livre"],
            "auteur": book["Auteur"],
            "raison": reason.strip()
        })

        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

    return recommendations


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

if submitted:
    if not preferences.strip():
        st.warning("Décris un peu tes envies de lecture avant de lancer la recherche.")
        st.stop()

    with st.spinner("Je parcours le catalogue et cherche les livres qui te correspondent..."):
        try:
            answer = recommend_books(preferences, avoid, books)
            recommendations = parse_recommendations(answer, books)

            if not recommendations:
                st.warning(
                    "L'IA n'a pas renvoyé de recommandations dans le format attendu. "
                    "Réessaie. Si le problème persiste, vérifie le modèle Groq utilisé."
                )

                with st.expander("Voir la réponse brute de l'IA"):
                    st.text(answer)

            else:
                st.subheader("📚 Voici ma sélection")

                for i, book in enumerate(recommendations, start=1):
                    st.markdown(f"### {i}. {book['titre']}")
                    st.write(f"**Auteur :** {book['auteur']}")
                    st.write(book["raison"])
                    st.divider()

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'appel à Groq : {e}")
