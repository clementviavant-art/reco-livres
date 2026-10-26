
import os
import re
import unicodedata
import streamlit as st
import pandas as pd
from groq import Groq


# ==================================================
# CONFIGURATION
# ==================================================

MODEL = "qwen/qwen3.8-27b"
CSV_FILE = "livres.csv"
MAX_CATALOGUE_BOOKS = 3


# ==================================================
# PAGE STREAMLIT
# ==================================================

st.set_page_config(
    page_title="Mon conseiller de lecture",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Mon conseiller de lecture")
st.write(
    "Trouve tes prochaines lectures parmi tes livres, "
    "avec en bonus une découverte extérieure à ta bibliothèque."
)


# ==================================================
# CHARGEMENT DU CATALOGUE
# ==================================================

@st.cache_data
def load_books():
    df = pd.read_csv(
        CSV_FILE,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False
    )

    df.columns = [col.strip() for col in df.columns]

    required = ["Auteur", "Titre du livre"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}. "
            f"Colonnes trouvées : {list(df.columns)}"
        )

    df["Auteur"] = df["Auteur"].str.strip()
    df["Titre du livre"] = df["Titre du livre"].str.strip()

    df = df[df["Titre du livre"] != ""].copy()
    df = df.reset_index(drop=True)

    # Identifiant interne unique pour chaque livre
    df["_id"] = df.index.astype(str)

    return df


try:
    books = load_books()
except FileNotFoundError:
    st.error(
        "Fichier livres.csv introuvable. "
        "Place-le dans le même dossier que app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Erreur de lecture du catalogue : {e}")
    st.stop()

if books.empty:
    st.error("Aucun livre trouvé dans le catalogue.")
    st.stop()

st.caption(f"📖 {len(books)} livres dans ton catalogue.")


# ==================================================
# CLÉ API GROQ
# ==================================================

try:
    api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key = ""

if not api_key:
    api_key = os.getenv("GROQ_API_KEY", "")

if not api_key:
    st.error(
        "Clé API Groq introuvable. "
        "Configure GROQ_API_KEY dans "
        ".streamlit/secrets.toml."
    )
    st.stop()

client = Groq(api_key=api_key)


# ==================================================
# FORMULAIRE UTILISATEUR
# ==================================================

with st.form("preferences_form"):
    preferences = st.text_area(
        "Qu'as-tu envie de lire ?",
        placeholder=(
            "Ex. : un thriller captivant, avec du suspense, "
            "des personnages complexes et un rythme soutenu..."
        ),
        height=120
    )

    avoid = st.text_input(
        "Ce que tu veux éviter (facultatif)",
        placeholder="Ex. : romance, violence, science-fiction..."
    )

    submitted = st.form_submit_button(
        "Trouver mes prochaines lectures",
        type="primary",
        use_container_width=True
    )


# ==================================================
# OUTILS
# ==================================================

def normalize_text(value):
    """Normalise le texte pour comparer les titres."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", value).strip().casefold()


def build_catalogue(df):
    """Construit une liste compacte contenant tout le catalogue."""
    lines = []

    for _, book in df.iterrows():
        author = book["Auteur"].replace("\n", " ").replace("|", "/")
        title = book["Titre du livre"].replace("\n", " ").replace("|", "/")

        lines.append(f'{book["_id"]}|{author}|{title}')

    return "\n".join(lines)


def call_groq(prompt, max_tokens=1200):
    """Envoie une demande à Groq sans imposer le format JSON."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un conseiller littéraire rigoureux. "
                    "Tu respectes le format de réponse demandé, "
                    "tu n'inventes pas de livres du catalogue "
                    "et tu distingues clairement les suggestions "
                    "issues du catalogue des suggestions extérieures."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.6,
        max_completion_tokens=max_tokens
    )

    return response.choices[0].message.content or ""


# ==================================================
# PARSE DES RECOMMANDATIONS DU CATALOGUE
# ==================================================

def parse_catalogue_recommendations(answer, df):
    """Ne conserve que les identifiants réellement présents dans le CSV."""
    books_by_id = df.set_index("_id", drop=False)
    recommendations = []
    seen_ids = set()

    pattern = re.compile(
        r"^\s*ID\s*:\s*(\d+)\s*\|\s*raison\s*:\s*(.+?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    for book_id, reason in pattern.findall(answer):
        if book_id not in books_by_id.index:
            continue

        if book_id in seen_ids:
            continue

        seen_ids.add(book_id)
        book = books_by_id.loc[book_id]

        recommendations.append({
            "titre": book["Titre du livre"],
            "auteur": book["Auteur"],
            "raison": reason.strip(),
            "exterieur": False
        })

        if len(recommendations) >= MAX_CATALOGUE_BOOKS:
            break

    return recommendations


# ==================================================
# PARSE DE LA RECOMMANDATION EXTÉRIEURE
# ==================================================

def parse_external_recommendation(answer, df):
    """Récupère une recommandation extérieure et vérifie son titre."""

    # Format attendu :
    # EXTERIEUR: Titre | auteur : Nom | raison : Justification
    pattern = re.compile(
        r"^\s*EXTERIEUR\s*:\s*(.+?)\s*\|\s*"
        r"auteur\s*:\s*(.+?)\s*\|\s*"
        r"raison\s*:\s*(.+?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    match = pattern.search(answer)

    if not match:
        return None

    title, author, reason = [
        value.strip() for value in match.groups()
    ]

    if not title or not author or not reason:
        return None

    # Refuse un livre déjà présent dans le catalogue
    catalogue_titles = {
        normalize_text(value)
        for value in df["Titre du livre"]
    }

    if normalize_text(title) in catalogue_titles:
        return None

    return {
        "titre": title,
        "auteur": author,
        "raison": reason,
        "exterieur": True
    }


# ==================================================
# RECOMMANDATION DE SECOURS
# ==================================================

def get_external_fallback(preferences, avoid, df, existing_titles):
    """
    Si la première réponse oublie le livre extérieur,
    effectue une seconde demande dédiée à cette recommandation.
    """

    catalogue = build_catalogue(df)

    existing = "\n".join(
        f"- {title}" for title in existing_titles
    ) or "- Aucun"

    prompt = f"""
Tu es un conseiller littéraire.

Tu dois proposer UN SEUL livre réel, publié et pertinent.

Envies du lecteur :
{preferences}

Livres à éviter :
{avoid if avoid.strip() else "Rien de précisé"}

Livres déjà recommandés :
{existing}

Voici le catalogue COMPLET. Le livre choisi doit être absent
de ce catalogue.

ID|Auteur|Titre
{catalogue}

CONSIGNES :
- Choisis un livre réel dont tu connais le titre et l'auteur.
- Vérifie qu'il n'est pas dans le catalogue ni dans les livres
  déjà recommandés.
- Ne reprends aucun des livres déjà proposés.
- Donne une raison courte et personnalisée.
- Réponds sur une seule ligne avec exactement ce format :
EXTERIEUR: titre | auteur : nom de l'auteur | raison : justification
- Aucun texte avant ou après cette ligne.
"""

    answer = call_groq(prompt, max_tokens=500)

    return answer


# ==================================================
# DEMANDE PRINCIPALE
# ==================================================

def recommend_books(preferences, avoid, df):
    catalogue = build_catalogue(df)

    prompt = f"""
Tu es un conseiller littéraire.

ENVIES DU LECTEUR :
{preferences}

LIVRES À ÉVITER :
{avoid if avoid.strip() else "Rien de précisé"}

CATALOGUE COMPLET :
ID|Auteur|Titre
{catalogue}

MISSION OBLIGATOIRE :
Tu dois fournir EXACTEMENT 4 recommandations :
1. Un premier livre présent dans le catalogue.
2. Un deuxième livre présent dans le catalogue.
3. Un troisième livre présent dans le catalogue.
4. Un livre réel, absent du catalogue, pour faire découvrir
   une nouvelle lecture au lecteur.

Pour les trois premières recommandations :
- Utilise uniquement les identifiants exacts du catalogue.
- Ne recommande pas deux fois le même livre.
- Explique brièvement pourquoi chaque livre correspond aux envies.

Pour la quatrième recommandation :
- Choisis un livre réel, publié et pertinent.
- Ne choisis aucun titre présent dans le catalogue.
- Ne reprends aucun des trois livres déjà recommandés.
- Indique le titre, l'auteur et une justification courte.
- Cette quatrième recommandation est obligatoire.

FORMAT DE RÉPONSE STRICT :
ID: identifiant | raison : justification
ID: identifiant | raison : justification
ID: identifiant | raison : justification
EXTERIEUR: titre | auteur : nom de l'auteur | raison : justification

Remplace « identifiant » par un identifiant réellement présent
dans le catalogue.
Ne fournis ni introduction, ni conclusion, ni autre ligne.
"""

    # Premier appel : 3 livres du catalogue + 1 livre extérieur
    answer = call_groq(prompt)

    catalogue_recommendations = parse_catalogue_recommendations(
        answer, df
    )

    external_recommendation = parse_external_recommendation(
        answer, df
    )

    # Secours : si l'IA a oublié la recommandation extérieure,
    # on lui demande une recommandation séparée.
    if external_recommendation is None:
        existing_titles = [
            book["titre"] for book in catalogue_recommendations
        ]

        fallback_answer = get_external_fallback(
            preferences,
            avoid,
            df,
            existing_titles
        )

        external_recommendation = parse_external_recommendation(
            fallback_answer, df
        )

        # Conserve les réponses brutes pour aider au diagnostic.
        diagnostic = (
            "Réponse principale de l'IA :\n"
            + answer
            + "\n\nRéponse à la demande de secours :\n"
            + fallback_answer
        )
    else:
        diagnostic = answer

    recommendations = catalogue_recommendations.copy()

    if external_recommendation:
        recommendations.append(external_recommendation)

    return recommendations, diagnostic


# ==================================================
# AFFICHAGE
# ==================================================

if submitted:
    if not preferences.strip():
        st.warning("Décris tes envies de lecture avant de lancer la recherche.")
        st.stop()

    with st.spinner(
        "Analyse du catalogue et recherche de tes prochaines lectures..."
    ):
        try:
            recommendations, diagnostic = recommend_books(
                preferences,
                avoid,
                books
            )

            catalogue_count = sum(
                not book["exterieur"] for book in recommendations
            )
            external_count = sum(
                book["exterieur"] for book in recommendations
            )

            if len(recommendations) == 4:
                st.success("Voici tes 4 recommandations !")
            else:
                st.warning(
                    f"J'ai obtenu {catalogue_count} recommandation(s) "
                    f"du catalogue et {external_count} recommandation(s) "
                    "extérieure(s). La réponse de l'IA est incomplète."
                )

            if recommendations:
                st.subheader("📚 Voici ma sélection")

                for index, book in enumerate(recommendations, start=1):
                    if book["exterieur"]:
                        st.markdown(
                            f"### {index}. 🌟 {book['titre']}"
                        )
                        st.caption("Découverte hors catalogue")
                    else:
                        st.markdown(
                            f"### {index}. {book['titre']}"
                        )
                        st.caption("Issu de ton catalogue")

                    st.write(f"**Auteur :** {book['auteur']}")
                    st.write(book["raison"])
                    st.divider()

            # Affiche les réponses brutes uniquement en cas de problème.
            if len(recommendations) != 4:
                with st.expander(
                    "Diagnostic : voir la réponse brute de l'IA"
                ):
                    st.code(diagnostic)

        except Exception as e:
            st.error(f"Une erreur est survenue avec Groq : {e}")
