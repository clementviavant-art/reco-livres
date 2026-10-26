import streamlit as st
import pandas as pd
from groq import Groq

# ---------------- CONFIGURATION ----------------

MODEL = "qwen/qwen3.8-27b"
MAX_COMPLETION_TOKENS = 180

st.set_page_config(
    page_title="Mon recommandeur de livres",
    page_icon="📚",
)

st.title("📚 Mon recommandeur de livres")

# ---------------- INITIALISATION GROQ ----------------

try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error(
        "Clé API Groq absente ou inaccessible. "
        "Configure GROQ_API_KEY dans les secrets Streamlit."
    )
    st.stop()

# ---------------- CHARGEMENT CSV ----------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("livres.csv", dtype=str).fillna("")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Le fichier livres.csv est introuvable sur GitHub.")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("Le fichier livres.csv est vide.")
    st.stop()

# ---------------- DETECTION DES COLONNES ----------------

def find_column(candidates):
    columns = {col.casefold(): col for col in df.columns}
    return next(
        (columns[name.casefold()] for name in candidates
         if name.casefold() in columns),
        None,
    )

col_titre = find_column(["Titre", "Nom", "Title", "Name"])
col_auteur = find_column(["Auteur", "Author"])

# Première colonne par défaut si aucun titre reconnu
if col_titre is None:
    col_titre = df.columns[0]

# Retirer les lignes sans titre
df = df[df[col_titre].str.strip().ne("")].copy()

if df.empty:
    st.error("Aucun livre avec un titre valide.")
    st.stop()

# ---------------- AFFICHAGE BIBLIOTHEQUE ----------------

st.caption(f"{len(df)} livres disponibles.")

if st.checkbox("Voir la liste des livres disponibles"):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------- DEMANDE UTILISATEUR ----------------

user_query = st.text_input(
    "Quel genre de livre recherches-tu ou quelle est ton humeur ?",
    placeholder="Ex. : un polar, un roman drôle, de la fantasy...",
    max_chars=300,
)

if st.button("Obtenir des recommandations", type="primary"):

    if not user_query.strip():
        st.warning("Veuillez entrer une demande.")
        st.stop()

    with st.spinner("L'IA analyse toute la bibliothèque..."):

        # ---------------- CATALOGUE COMPACT ----------------

        # Conserver uniquement titre et auteur
        colonnes = [col_titre]
        if col_auteur:
            colonnes.append(col_auteur)

        livres_df = df[colonnes].copy()

        for col in colonnes:
            livres_df[col] = livres_df[col].str.strip()

        # Supprimer les doublons exacts
        livres_df = livres_df.drop_duplicates()

        # Format compact : titre|auteur
        if col_auteur:
            lignes = [
                f"{titre}|{auteur or '?'}"
                for titre, auteur in livres_df.itertuples(
                    index=False, name=None
                )
                if titre
            ]
        else:
            lignes = [
                titre for titre in livres_df[col_titre]
                if titre
            ]

        catalogue = "\n".join(lignes)

        # ---------------- PROMPT MINIMAL ----------------

        prompt = (
            f"Demande:{user_query.strip()}\n"
            "Choisis jusqu'à 3 livres de cette liste, "
            "sans inventer de titres. Réponds en français, "
            "brièvement, avec titre, auteur si disponible "
            "et une raison courte.\n"
            f"{catalogue}"
        )

        # ---------------- APPEL GROQ ----------------

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="medium",
            )

            result = completion.choices[0].message.content

            if result:
                st.markdown(result)
            else:
                st.warning("Aucune recommandation générée.")

            # ---------------- SUIVI DES TOKENS ----------------

            usage = completion.usage

            if usage:
                st.caption(
                    f"Bibliothèque : {len(livres_df)} livres · "
                    f"Entrée : {usage.prompt_tokens:,} tokens · "
                    f"Sortie : {usage.completion_tokens:,} tokens · "
                    f"Total : {usage.total_tokens:,} tokens"
                )

        except Exception as e:
            st.error(f"Erreur lors de l'appel à Groq : {e}")
