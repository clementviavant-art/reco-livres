
import json
import streamlit as st
import pandas as pd
from groq import Groq

# ---------------- CONFIGURATION ----------------

MODEL = "qwen/qwen3.8-27b"
MAX_COMPLETION_TOKENS = 1200
MAX_DESCRIPTION_CHARS = 180

st.set_page_config(
    page_title="Mon recommandeur de livres",
    page_icon="📚",
)

st.title("📚 Mon recommandeur de livres")

# ---------------- CONNEXION GROQ ----------------

try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except Exception:
    st.error(
        "Clé API Groq absente ou inaccessible. "
        "Configure GROQ_API_KEY dans les secrets Streamlit."
    )
    st.stop()

# ---------------- CHARGEMENT CSV ----------------

@st.cache_data
def load_data():
    import csv
    from pathlib import Path

    path = Path("livres.csv")

    if not path.exists():
        return None

    erreurs = []

    # Essayer les encodages courants
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                sample = f.read(16384)

            if not sample.strip():
                raise ValueError("Le fichier est vide.")

            # Détecter le séparateur
            try:
                dialect = csv.Sniffer().sniff(
                    sample, delimiters=";,\t|"
                )
                separator = dialect.delimiter
            except csv.Error:
                # Le point-virgule est courant dans les CSV français
                separator = ";"

            data = pd.read_csv(
                path,
                sep=separator,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                engine="python",
                on_bad_lines="error",
            )

            data.columns = data.columns.str.strip()
            return data

        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as e:
            erreurs.append(f"{encoding}: {e}")

    st.error(
        "Impossible de lire livres.csv. Vérifie le séparateur, "
        "les guillemets et les lignes mal formées."
    )

    with st.expander("Détails techniques"):
        st.code("\n".join(erreurs))

    return None

# ---------------- DETECTION DES COLONNES ----------------

def find_column(candidates):
    lookup = {col.casefold(): col for col in df.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None

col_titre = find_column(["Titre", "Nom", "Title", "Name"])
col_auteur = find_column(["Auteur", "Author"])

if col_titre is None:
    col_titre = df.columns[0]

# Métadonnées utiles si elles existent dans le CSV
metadata_candidates = {
    "genre": ["Genre", "Genres", "Catégorie", "Categorie", "Category"],
    "resume": ["Résumé", "Resume", "Description", "Synopsis", "Summary"],
    "ambiance": ["Ambiance", "Mood", "Atmosphère", "Atmosphere"],
}

metadata_columns = {}
for key, candidates in metadata_candidates.items():
    column = find_column(candidates)
    if column and column not in [col_titre, col_auteur]:
        metadata_columns[key] = column

# Retirer uniquement les lignes sans titre
df = df[df[col_titre].str.strip().ne("")].copy().reset_index(drop=True)

if df.empty:
    st.error("Aucun livre avec un titre valide.")
    st.stop()

# Identifiant stable pour chaque ligne pendant cette session de données
df["_book_id"] = [str(i) for i in range(1, len(df) + 1)]

# ---------------- AFFICHAGE ----------------

st.caption(f"{len(df)} livres dans la bibliothèque.")

if st.checkbox("Voir la liste des livres disponibles"):
    st.dataframe(
        df.drop(columns=["_book_id"]),
        use_container_width=True,
        hide_index=True,
    )

# ---------------- CONSTRUCTION DU CATALOGUE ----------------

def clean_text(value, max_chars=None):
    text = " ".join(str(value).split())
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text

def build_catalogue(data):
    lines = []

    for _, row in data.iterrows():
        book_id = row["_book_id"]
        title = clean_text(row[col_titre])
        author = clean_text(row[col_auteur]) if col_auteur else ""

        # Format compact : ID|titre|auteur
        fields = [book_id, title, author or "?"]

        # Ajouter uniquement les métadonnées disponibles
        for key, column in metadata_columns.items():
            value = clean_text(
                row[column],
                MAX_DESCRIPTION_CHARS if key == "resume" else 80,
            )
            if value:
                fields.append(f"{key}:{value}")

        lines.append("|".join(fields))

    return "\n".join(lines)

catalogue = build_catalogue(df)

# ---------------- DEMANDE UTILISATEUR ----------------

user_query = st.text_input(
    "Quel genre de livre recherches-tu ou quelle est ton humeur ?",
    placeholder="Ex. : un polar psychologique sombre avec une fin surprenante",
    max_chars=500,
)

if st.button("Obtenir des recommandations", type="primary"):

    if not user_query.strip():
        st.warning("Décris le livre que tu recherches.")
        st.stop()

    # Prompt concis, avec critères explicites de sélection
    prompt = f"""Tu es un libraire expert.
Demande: {user_query.strip()}

Catalogue complet (ID|titre|auteur|métadonnées):
{catalogue}

Choisis les 3 livres les plus pertinents, dans l'ordre.
Compare le genre, l'ambiance, les thèmes et les attentes implicites.
Utilise les métadonnées disponibles. Ne déduis pas une intrigue inconnue
du seul titre. N'invente aucun livre ni aucune caractéristique.
Si aucun choix ne convient réellement, retourne moins de 3 résultats.
Retourne uniquement un objet JSON valide au format:
{{"recommendations":[{{"id":"ID_EXACT","reason":"Justification courte","confidence":"élevée|moyenne|faible"}}]}}
Chaque ID doit exister dans le catalogue. Aucun autre texte."""

    with st.spinner("Analyse de toute la bibliothèque..."):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu recommandes des livres avec discernement. "
                            "La pertinence prime sur le nombre de résultats. "
                            "Respecte strictement le catalogue."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort="medium",
                reasoning_format="hidden",
                temperature=1.0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content

            if not raw:
                st.warning("Le modèle n'a retourné aucune réponse.")
                st.stop()

            try:
                result = json.loads(raw)
                recommendations = result.get("recommendations", [])
            except (json.JSONDecodeError, AttributeError):
                st.error("La réponse du modèle n'est pas un JSON valide.")
                st.stop()

            # Associer les IDs aux livres réels du CSV
            books_by_id = df.set_index("_book_id", drop=False)
            shown_ids = set()
            valid_recommendations = []

            for item in recommendations:
                if not isinstance(item, dict):
                    continue

                book_id = str(item.get("id", "")).strip()

                if book_id not in books_by_id.index or book_id in shown_ids:
                    continue

                shown_ids.add(book_id)
                book = books_by_id.loc[book_id]

                # Le titre et l'auteur affichés viennent du CSV,
                # jamais du texte généré par le modèle.
                valid_recommendations.append(
                    {
                        "title": book[col_titre],
                        "author": (
                            book[col_auteur]
                            if col_auteur and book[col_auteur].strip()
                            else "Auteur non renseigné"
                        ),
                        "reason": str(item.get("reason", "")).strip(),
                        "confidence": str(
                            item.get("confidence", "non précisée")
                        ).strip(),
                    }
                )

            if not valid_recommendations:
                st.warning(
                    "Aucune recommandation valide. "
                    "Essaie de reformuler ta demande."
                )
            else:
                st.subheader("Tes recommandations")

                for i, book in enumerate(valid_recommendations, start=1):
                    st.markdown(
                        f"### {i}. {book['title']}\n"
                        f"**Auteur :** {book['author']}\n\n"
                        f"{book['reason']}\n\n"
                        f"Confiance estimée : {book['confidence']}"
                    )

            # Mesure des tokens consommés
            usage = response.usage
            if usage:
                st.caption(
                    f"{len(df)} livres envoyés · "
                    f"Entrée : {usage.prompt_tokens:,} tokens · "
                    f"Sortie : {usage.completion_tokens:,} tokens · "
                    f"Total : {usage.total_tokens:,} tokens"
                )

        except Exception as e:
            st.error(f"Erreur lors de l'appel à Groq : {e}")
