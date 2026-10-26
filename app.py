import json
import streamlit as st
import pandas as pd
from groq import Groq

# ---------------- CONFIGURATION ----------------

MODEL = "qwen/qwen3.8-27b"
MAX_COMPLETION_TOKENS = 1200

st.set_page_config(
    page_title="Mon recommandeur de livres",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Mon recommandeur de livres")

# ---------------- CONNEXION GROQ ----------------

try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error(
        "Clé API Groq absente ou inaccessible. "
        "Vérifie les secrets de ton application."
    )
    st.stop()

# ---------------- CHARGEMENT DU CSV ----------------

@st.cache_data
def load_data():
    try:
        data = pd.read_csv(
            "livres.csv",
            sep=";",
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )

        data.columns = data.columns.str.strip()

        # Vérification des colonnes du fichier fourni
        required = ["Auteur", "Titre du livre"]
        missing = [col for col in required if col not in data.columns]

        if missing:
            raise ValueError(
                "Colonnes manquantes : " + ", ".join(missing)
            )

        # Nettoyage des données
        data["Auteur"] = data["Auteur"].str.strip()
        data["Titre du livre"] = data["Titre du livre"].str.strip()

        data = data[
            data["Titre du livre"].ne("")
        ].copy()

        # Identifiant propre à chaque ligne
        data = data.reset_index(drop=True)
        data["_id"] = data.index.map(str)

        return data

    except FileNotFoundError:
        st.error("Le fichier livres.csv est introuvable.")
        return None
    except Exception as e:
        st.error(f"Erreur de lecture du CSV : {e}")
        return None


df = load_data()

if df is None:
    st.stop()

if df.empty:
    st.error("Aucun livre trouvé dans le fichier CSV.")
    st.stop()

# ---------------- BIBLIOTHEQUE ----------------

st.caption(f"📚 {len(df)} livres disponibles")

if st.checkbox("Voir la liste des livres"):
    st.dataframe(
        df[["Auteur", "Titre du livre"]],
        use_container_width=True,
        hide_index=True,
    )

# ---------------- CONSTRUCTION DU CATALOGUE ----------------

def build_catalogue(data):
    # Une ligne compacte par livre : ID|auteur|titre
    return "\n".join(
        f"{row['_id']}|{row['Auteur'] or '?'}|{row['Titre du livre']}"
        for _, row in data.iterrows()
    )


catalogue = build_catalogue(df)

# ---------------- DEMANDE UTILISATEUR ----------------

user_query = st.text_input(
    "Quel genre de livre recherches-tu ou quelle est ton humeur ?",
    placeholder="Ex. : un polar sombre avec une intrigue psychologique",
    max_chars=500,
)

if st.button("Obtenir mes recommandations", type="primary"):

    if not user_query.strip():
        st.warning("Décris le livre que tu recherches.")
        st.stop()

    prompt = f"""Tu es un libraire expert et exigeant.

DEMANDE DU LECTEUR :
{user_query.strip()}

BIBLIOTHÈQUE COMPLÈTE (ID|auteur|titre) :
{catalogue}

MISSION :
Sélectionne jusqu'à 3 livres réellement présents dans cette liste.
Classe-les du plus pertinent au moins pertinent.
Prends en compte le genre, l'ambiance, les thèmes, le rythme
et les attentes implicites du lecteur.
Compare les livres entre eux avant de choisir.
Ne choisis pas un livre uniquement parce que son titre contient
un mot similaire à la demande.
N'invente ni titre, ni auteur, ni intrigue.
Le catalogue ne contient que les auteurs et titres :
si tu ne connais pas suffisamment un livre, indique cette limite.
Si aucun livre ne correspond raisonnablement, retourne une liste vide.

Retourne uniquement un JSON valide au format :
{{"recommendations":[
  {{"id":"ID exact","reason":"Justification concise",
    "confidence":"élevée|moyenne|faible"}}
]}}
Les identifiants doivent exister dans le catalogue."""

    with st.spinner("Analyse de toute la bibliothèque..."):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu recommandes des livres avec discernement. "
                            "Privilégie la pertinence, l'honnêteté et "
                            "les correspondances littéraires réelles."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="medium",
                reasoning_format="hidden",
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                st.warning("Le modèle n'a retourné aucune réponse.")
                st.stop()

            result = json.loads(content)
            recommendations = result.get("recommendations", [])

            # Vérification des identifiants et récupération des
            # titres/auteurs directement depuis le CSV
            books_by_id = df.set_index("_id")
            seen = set()
            valid = []

            for item in recommendations:
                if not isinstance(item, dict):
                    continue

                book_id = str(item.get("id", "")).strip()

                if book_id not in books_by_id.index or book_id in seen:
                    continue

                seen.add(book_id)
                book = books_by_id.loc[book_id]

                valid.append({
                    "auteur": book["Auteur"],
                    "titre": book["Titre du livre"],
                    "raison": str(item.get("reason", "")).strip(),
                    "confiance": str(
                        item.get("confidence", "non précisée")
                    ).strip(),
                })

                if len(valid) == 3:
                    break

            if not valid:
                st.warning(
                    "Aucune recommandation suffisamment pertinente. "
                    "Essaie de préciser le genre ou l'ambiance recherchée."
                )
            else:
                st.subheader("Tes recommandations")

                for i, book in enumerate(valid, start=1):
                    st.markdown(f"### {i}. {book['titre']}")
                    st.write(f"**Auteur :** {book['auteur'] or 'Non renseigné'}")

                    if book["raison"]:
                        st.write(book["raison"])

                    st.caption(
                        f"Confiance estimée : {book['confiance']}"
                    )

            # ---------------- SUIVI DES TOKENS ----------------

            usage = response.usage

            if usage:
                st.divider()
                st.caption(
                    f"Livres transmis : {len(df)} · "
                    f"Tokens en entrée : {usage.prompt_tokens:,} · "
                    f"Tokens en sortie : {usage.completion_tokens:,} · "
                    f"Total : {usage.total_tokens:,}"
                )

        except json.JSONDecodeError:
            st.error("La réponse de l'IA n'est pas un JSON valide.")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à Groq : {e}")
