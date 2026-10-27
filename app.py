
import os
import random
import re
import time
import unicodedata

import groq
import pandas as pd
import streamlit as st
from groq import Groq


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Mon conseiller de lecture",
    page_icon="📚",
    layout="centered",
)

APP_VERSION = "8"
CSV_FILE = "livres.csv"

MAX_CATALOGUE_BOOKS = 3
TOTAL_BOOKS = 4

# Budgets de sortie réduits
MAIN_TOKENS = 300
DISCOVERY_TOKENS = 160

# Plafond configurable dans les secrets Streamlit
try:
    OUTPUT_TOKEN_CAP = int(
        os.getenv("GROQ_OUTPUT_TOKEN_CAP", "500")
    )
except ValueError:
    OUTPUT_TOKEN_CAP = 500

REQUEST_TIMEOUT = 40
MAX_RATE_LIMIT_WAIT = 10
DISCOVERY_COUNT = 2
DISCOVERY_SAMPLE_SIZE = 30

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
REASONING_EFFORT = os.getenv(
    "GROQ_REASONING_EFFORT", "none"
).strip().lower()

try:
    MODEL = st.secrets.get("GROQ_MODEL", MODEL)
    REASONING_EFFORT = st.secrets.get(
        "GROQ_REASONING_EFFORT", REASONING_EFFORT
    ).strip().lower()
    OUTPUT_TOKEN_CAP = int(
        st.secrets.get(
            "GROQ_OUTPUT_TOKEN_CAP", OUTPUT_TOKEN_CAP
        )
    )
except Exception:
    pass


# ==================================================
# TEXTE ET NORMALISATION
# ==================================================

def normalize_text(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", value).strip().casefold()


def title_key(title):
    return normalize_text(str(title).replace("|", "/"))


def clean_field(series):
    return (
        series.str.replace("\n", " ", regex=False)
        .str.replace("|", "/", regex=False)
        .str.strip()
    )


def strip_reasoning(text):
    text = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ==================================================
# CHARGEMENT DU CATALOGUE
# ==================================================

@st.cache_data
def load_catalogue():
    df = pd.read_csv(
        CSV_FILE,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    df.columns = [c.strip() for c in df.columns]

    required = ["Auteur", "Titre du livre"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}. "
            f"Colonnes trouvées : {list(df.columns)}"
        )

    df["Auteur"] = df["Auteur"].str.strip()
    df["Titre du livre"] = df["Titre du livre"].str.strip()
    df = df[df["Titre du livre"] != ""].reset_index(drop=True)

    df["_id"] = df.index.astype(str)

    lines = (
        df["_id"]
        + "|"
        + clean_field(df["Auteur"])
        + "|"
        + clean_field(df["Titre du livre"])
    )

    books = {}

    for book_id, author, title, line in zip(
        df["_id"], df["Auteur"], df["Titre du livre"], lines
    ):
        books[book_id] = {
            "id": book_id,
            "auteur": author,
            "titre": title,
            "ligne": line,
        }

    return {
        "count": len(df),
        "text": "\n".join(lines),
        "by_id": books,
        "id_by_title": {
            title_key(book["titre"]): book_id
            for book_id, book in reversed(list(books.items()))
        },
        "titles": {
            title_key(title) for title in df["Titre du livre"]
        },
    }


try:
    catalogue = load_catalogue()
except FileNotFoundError:
    st.error(
        "Fichier livres.csv introuvable. "
        "Place-le dans le même dossier que app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Erreur de lecture du catalogue : {e}")
    st.stop()

if catalogue["count"] == 0:
    st.error("Aucun livre trouvé dans le catalogue.")
    st.stop()


# ==================================================
# CONFIGURATION GROQ
# ==================================================

def get_setting(name, default=""):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, default)


api_key = get_setting("GROQ_API_KEY")

st.title("Conseiller spécial pour la famille Viavant")
st.write(
    "Cloé, Gilberte, trouvez vos prochaines lectures parmi "
    "les livres de la bibliothèque, avec une découverte extérieure."
)

st.caption(
    f"📖 {catalogue['count']} livres · version {APP_VERSION} "
    f"· modèle {MODEL}"
)

if not api_key:
    st.error(
        "Clé API Groq introuvable. Configure GROQ_API_KEY "
        "dans .streamlit/secrets.toml."
    )
    st.stop()


@st.cache_resource
def get_client(key):
    return Groq(
        api_key=key,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


client = get_client(api_key)


SYSTEM_PROMPT = (
    "Tu es un conseiller littéraire rigoureux. "
    "Respecte exactement le format demandé. "
    "N'invente aucun identifiant du catalogue. "
    "Justifications courtes, personnalisées et pertinentes."
)


def reasoning_options():
    if REASONING_EFFORT in ("", "off"):
        return {}

    options = {"reasoning_effort": REASONING_EFFORT}

    if REASONING_EFFORT != "none":
        options["reasoning_format"] = "hidden"

    return options


def retry_after_seconds(error):
    try:
        return float(error.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def call_groq(prompt, max_tokens):
    """Un appel, avec une unique nouvelle tentative si le délai 429 est court."""
    max_tokens = min(max_tokens, OUTPUT_TOKEN_CAP)

    kwargs = {}
    options = reasoning_options()

    if options:
        kwargs["extra_body"] = options

    for attempt in (1, 2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
            break

        except groq.RateLimitError as error:
            wait = retry_after_seconds(error)

            if (
                attempt == 2
                or wait is None
                or wait > MAX_RATE_LIMIT_WAIT
            ):
                raise

            time.sleep(wait + 0.5)

    choice = response.choices[0]

    return (
        strip_reasoning(choice.message.content or ""),
        choice.finish_reason,
    )


# ==================================================
# PROMPTS COMPACTS
# ==================================================

def build_main_prompt(preferences, avoid):
    return f"""
Envies : {preferences.strip()[:1000]}
À éviter : {avoid.strip()[:300] or "non précisé"}

CATALOGUE COMPLET (ID|Auteur|Titre) :
{catalogue["text"]}

Choisis exactement 3 livres distincts du catalogue et
1 livre réel extérieur au catalogue.

Respecte les envies et les exclusions. Le livre extérieur
ne doit pas être présent dans le catalogue.

Pour les livres du catalogue, utilise les ID exacts.
Ne répète aucun livre. Justification personnalisée de
15 mots maximum par livre.

FORMAT STRICT : exactement 4 lignes, sans introduction.

ID|identifiant|justification
ID|identifiant|justification
ID|identifiant|justification
EXT|titre|auteur|justification

Pour EXT, donne le titre, l'auteur et la justification.
""".strip()


def build_discovery_prompt(sample):
    lines = "\n".join(book["ligne"] for book in sample)

    return f"""
Choisis 2 livres différents de cette liste, si possible
d'auteurs différents, avec des ambiances contrastées.

{lines}

Réponds exactement sur 2 lignes :
ID|identifiant|justification de 12 mots maximum

Aucun autre texte.
""".strip()


# ==================================================
# PARSING DES RÉPONSES
# ==================================================

CATALOGUE_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:ID\s*\|\s*)?(\d+)\s*\|\s*(.+?)\s*$",
    re.IGNORECASE,
)

EXTERNAL_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?EXT(?:ERIEUR)?\s*\|\s*"
    r"(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$",
    re.IGNORECASE,
)


def clean_author(value):
    value = re.sub(
        r"\s*:\s*nom de l['’]auteur\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(' *[]"\u00ab\u00bb')


def parse_catalogue_recommendations(
    answer, exclude_ids=(), limit=MAX_CATALOGUE_BOOKS,
    allowed_ids=None,
):
    seen = set(exclude_ids)
    recommendations = []

    for line in answer.splitlines():
        match = CATALOGUE_LINE.match(line.strip())

        if not match:
            continue

        book_id, reason = match.groups()

        if book_id not in catalogue["by_id"]:
            continue

        if allowed_ids is not None and book_id not in allowed_ids:
            continue

        if book_id in seen:
            continue

        seen.add(book_id)
        book = catalogue["by_id"][book_id]

        recommendations.append({
            "id": book_id,
            "titre": book["titre"],
            "auteur": book["auteur"],
            "raison": reason.strip(),
            "exterieur": False,
        })

        if len(recommendations) >= limit:
            break

    return recommendations


def parse_external_candidate(answer):
    for line in answer.splitlines():
        match = EXTERNAL_LINE.match(line.strip())

        if not match:
            continue

        title, author, reason = [
            value.strip() for value in match.groups()
        ]

        title = title.strip(' *[]"\u00ab\u00bb')
        author = clean_author(author)

        if not title or not author or not reason:
            continue

        if title_key(title) in catalogue["titles"]:
            continue

        return {
            "id": None,
            "titre": title,
            "auteur": author,
            "raison": reason,
            "exterieur": True,
        }

    return None


# ==================================================
# RECOMMANDATIONS : UN SEUL APPEL PRINCIPAL
# ==================================================

def recommend_books(preferences, avoid):
    prompt = build_main_prompt(preferences, avoid)

    answer, finish_reason = call_groq(prompt, MAIN_TOKENS)

    # Vérification stricte des identifiants côté Python.
    catalogue_recs = parse_catalogue_recommendations(answer)

    external_rec = parse_external_candidate(answer)

    # Évite qu'un livre extérieur reprenne un titre sélectionné.
    if external_rec and any(
        title_key(book["titre"]) == title_key(external_rec["titre"])
        for book in catalogue_recs
    ):
        external_rec = None

    recommendations = catalogue_recs[:]

    if external_rec:
        recommendations.append(external_rec)

    diagnostic = (
        f"Fin de réponse : {finish_reason}\n"
        f"Livres du catalogue valides : {len(catalogue_recs)}/3\n"
        f"Livre extérieur valide : {'oui' if external_rec else 'non'}\n\n"
        f"Réponse brute :\n{answer}"
    )

    return recommendations, diagnostic


# ==================================================
# SUGGESTIONS D'ACCUEIL PAR IA
# ==================================================

def sample_books(size):
    books = list(catalogue["by_id"].values())
    random.shuffle(books)

    picked = []
    authors = set()

    for book in books:
        author = normalize_text(book["auteur"])

        if author in authors:
            continue

        authors.add(author)
        picked.append(book)

        if len(picked) >= min(size, len(books)):
            break

    if len(picked) < DISCOVERY_COUNT:
        picked = books[:DISCOVERY_COUNT]

    return picked


def get_discoveries():
    sample = sample_books(DISCOVERY_SAMPLE_SIZE)
    allowed_ids = {book["id"] for book in sample}

    try:
        answer, _ = call_groq(
            build_discovery_prompt(sample),
            DISCOVERY_TOKENS,
        )

        picks = parse_catalogue_recommendations(
            answer,
            limit=DISCOVERY_COUNT,
            allowed_ids=allowed_ids,
        )

    except groq.APIError:
        picks = []

    chosen = []
    used_authors = set()

    for pick in picks:
        author = normalize_text(pick["auteur"])

        if author in used_authors:
            continue

        used_authors.add(author)
        chosen.append(pick)

    # Complète aléatoirement en cas de réponse incomplète.
    for book in sample:
        if len(chosen) >= DISCOVERY_COUNT:
            break

        if book["id"] in {item["id"] for item in chosen}:
            continue

        author = normalize_text(book["auteur"])

        if author in used_authors:
            continue

        used_authors.add(author)

        chosen.append({
            "id": book["id"],
            "titre": book["titre"],
            "auteur": book["auteur"],
            "raison": "",
            "exterieur": False,
        })

    return chosen


# ==================================================
# AFFICHAGE
# ==================================================

def bordered_container():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def show_result(recommendations, diagnostic):
    catalogue_count = sum(
        not book["exterieur"] for book in recommendations
    )
    external_count = sum(
        book["exterieur"] for book in recommendations
    )

    if len(recommendations) == TOTAL_BOOKS:
        st.success("Voici tes 4 recommandations !")
    else:
        st.warning(
            f"{catalogue_count} livre(s) du catalogue et "
            f"{external_count} découverte(s) extérieure(s). "
            "La réponse de l'IA est incomplète."
        )

    if recommendations:
        st.subheader("📚 Voici ma sélection")

        for index, book in enumerate(recommendations, start=1):
            if book["exterieur"]:
                st.markdown(f"### {index}. 🌟 {book['titre']}")
                st.caption("Découverte hors catalogue")
            else:
                st.markdown(f"### {index}. {book['titre']}")
                st.caption("Issu de ton catalogue")

            st.write(f"**Auteur :** {book['auteur']}")
            st.write(book["raison"])
            st.divider()

    if len(recommendations) != TOTAL_BOOKS:
        with st.expander("Diagnostic : réponse brute de l'IA"):
            st.code(diagnostic)


# Suggestions d'accueil
with st.container():
    st.subheader("✨ Pour commencer : deux idées de la bibliothèque")

    if st.button("🔄 Autres idées"):
        st.session_state.pop("discoveries", None)

    if "discoveries" not in st.session_state:
        with st.spinner("Je feuillette les étagères..."):
            st.session_state["discoveries"] = get_discoveries()

    discoveries = st.session_state["discoveries"]

    columns = st.columns(DISCOVERY_COUNT)

    for column, book in zip(columns, discoveries):
        with column:
            with bordered_container():
                st.markdown(f"**{book['titre']}**")
                st.caption(book["auteur"])

                if book["raison"]:
                    st.write(book["raison"])

st.divider()


# Formulaire
with st.form("preferences_form"):
    preferences = st.text_area(
        "Qu'as-tu envie de lire ?",
        placeholder=(
            "Ex. : un thriller captivant, avec du suspense, "
            "des personnages complexes et un rythme soutenu..."
        ),
        height=120,
    )

    avoid = st.text_input(
        "Ce que tu veux éviter (facultatif)",
        placeholder="Ex. : romance, violence, science-fiction...",
    )

    submitted = st.form_submit_button(
        "Trouver mes prochaines lectures",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not preferences.strip():
        st.warning(
            "Décris tes envies de lecture avant de lancer la recherche."
        )
    else:
        st.session_state.pop("result", None)

        with st.status(
            "Recherche de tes prochaines lectures...",
            expanded=True,
        ) as status:
            try:
                st.session_state["result"] = recommend_books(
                    preferences, avoid
                )

                status.update(
                    label="Recherche terminée",
                    state="complete",
                    expanded=False,
                )

            except groq.RateLimitError as e:
                status.update(
                    label="Limite d'utilisation atteinte",
                    state="error",
                )
                st.error(
                    "Limite Groq atteinte. Attends un peu puis réessaie."
                )

            except groq.APIError as e:
                status.update(
                    label="Erreur Groq",
                    state="error",
                )
                st.error(f"Erreur de l'API Groq : {e}")

            except Exception as e:
                status.update(
                    label="Erreur inattendue",
                    state="error",
                )
                st.error(f"Erreur : {e}")


if "result" in st.session_state:
    show_result(*st.session_state["result"])
