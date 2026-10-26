import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

# ---------------- CONFIGURATION ----------------

MODEL = "qwen/qwen3.8-27b"
MAX_CANDIDATS = 40
MAX_TOKENS_REPONSE = 220

st.set_page_config(
    page_title="Mon recommandeur de livres",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Mon recommandeur de livres")

# ---------------- API ----------------

try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except Exception:
    st.error(
        "Clé GROQ_API_KEY absente ou inaccessible. "
        "Configure-la dans les secrets Streamlit."
    )
    st.stop()

# ---------------- CHARGEMENT ----------------

@st.cache_data
def load_data():
    path = Path("livres.csv")
    if not path.exists():
        return None

    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = df.columns.str.strip()

    return df


df = load_data()

if df is None:
    st.error("Le fichier livres.csv est introuvable.")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("Le fichier livres.csv est vide.")
    st.stop()

# ---------------- DETECTION DES COLONNES ----------------

def find_column(names):
    lookup = {c.casefold(): c for c in df.columns}
    return next(
        (lookup[n.casefold()] for n in names if n.casefold() in lookup),
        None,
    )

col_titre = find_column(["Titre", "Nom", "Title", "Name"])
col_auteur = find_column(["Auteur", "Author"])

if col_titre is None:
    col_titre = df.columns[0]

# Écarter les lignes sans titre
df = df[df[col_titre].str.strip() != ""].copy()

if df.empty:
    st.error("Aucun livre avec un titre valide.")
    st.stop()

# ---------------- RECHERCHE LOCALE ----------------

STOPWORDS = {
    "avec", "dans", "pour", "mais", "donc", "quel", "quelle",
    "quels", "quelles", "livre", "livres", "envie", "avoir",
    "etre", "être", "suis", "mon", "ma", "mes", "les", "des",
    "une", "un", "du", "de", "la", "le", "et", "ou", "qui",
    "que", "quoi", "sur", "par", "pas", "plus", "tres", "très",
    "plutot", "plutôt", "comme", "moi", "je", "me", "ce", "ces",
}

def normalize(text):
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text

def keywords(text):
    return {
        word
        for word in re.findall(r"[a-z0-9]{2,}", normalize(text))
        if word not in STOPWORDS
    }

# Préparer l'index local une seule fois par version du CSV
@st.cache_data
def prepare_index(data):
    index = data.copy()

    # Recherche dans toutes les colonnes : genre, résumé,
    # catégories, titre, auteur, etc.
    index["_search"] = index.astype(str).agg(" ".join, axis=1).map(normalize)

    return index

index = prepare_index(df)

def find_candidates(query, limit=MAX_CANDIDATS):
    terms = keywords(query)

    if not terms:
        return pd.DataFrame()

    # Score simple, calculé localement sans appel à l'IA
    scores = index["_search"].map(
        lambda text: sum(
            1 for term in terms
            if re.search(r"\b" + re.escape(term) + r"\b", text)
        )
    )

    matching = scores[scores > 0].sort_values(ascending=False)

    # En cas d'égalité, conserver un ordre stable
    matching = matching.head(limit)

    return df.loc[matching.index].copy()

# ---------------- INTERFACE ----------------

st.caption(f"{len(df)} livres dans la bibliothèque.")

with st.expander("Voir les livres disponibles"):
    st.dataframe(df, use_container_width=True, hide_index=True)

query = st.text_input(
    "Que souhaites-tu lire ?",
    placeholder="Ex. : un polar sombre, un roman drôle, de la fantasy...",
    max_chars=300,
)

if st.button("Obtenir 3 recommandations", type="primary"):
    if not query.strip():
        st.warning("Décris le livre que tu recherches.")
        st.stop()

    candidats = find_candidates(query)

    if candidats.empty:
        st.warning(
            "Aucun livre correspondant trouvé. "
            "Essaie d'autres mots ou vérifie les colonnes de ton CSV."
        )
        st.stop()

    # Envoyer uniquement les titres et auteurs des candidats
    livres = []
    for i, (_, row) in enumerate(candidats.iterrows(), start=1):
        titre = str(row[col_titre]).strip()

        auteur = (
            str(row[col_auteur]).strip()
            if col_auteur and str(row[col_auteur]).strip()
            else "auteur inconnu"
        )

        livres.append(f"{i}|{titre}|{auteur}")

    catalogue = "\n".join(livres)

    prompt = (
        "Choisis 3 livres distincts parmi la liste, selon la demande. "
        "N'invente aucun livre. Réponds en français, brièvement, "
        "avec le titre, l'auteur et une justification courte. "
        "Si moins de 3 conviennent, recommande-en moins.\n"
        f"Demande: {query.strip()}\n"
        f"Livres (ID|titre|auteur):\n{catalogue}"
    )

    with st.spinner("Recherche des meilleures recommandations..."):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=MAX_TOKENS_REPONSE,
                reasoning_effort="none",
            )

            answer = response.choices[0].message.content

            if answer:
                st.markdown(answer)
            else:
                st.warning("L'IA n'a pas retourné de recommandations.")

            # Mesurer les tokens consommés
            usage = response.usage
            if usage:
                st.caption(
                    f"Tokens : {usage.prompt_tokens} en entrée · "
                    f"{usage.completion_tokens} en sortie · "
                    f"{usage.total_tokens} au total"
                )

        except Exception as e:
            st.error(f"Erreur Groq : {e}")
```
