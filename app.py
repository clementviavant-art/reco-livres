import os
import random
import re
import time
import unicodedata

import groq
import pandas as pd
import streamlit as st
from groq import Groq

# Erreur de délai dépassé (repli sur APIError si le SDK ne l'expose pas)
TIMEOUT_ERROR = getattr(groq, "APITimeoutError", groq.APIError)


# ==================================================
# PAGE STREAMLIT (doit être la première commande st.*)
# ==================================================

st.set_page_config(
    page_title="Mon conseiller de lecture",
    page_icon="📚",
    layout="centered"
)


# ==================================================
# CONFIGURATION
# ==================================================

def get_setting(name, default=""):
    """Lit un réglage dans st.secrets, puis dans l'environnement."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, default)


# Le modèle peut être changé sans toucher au code (GROQ_MODEL dans les secrets).
MODEL = get_setting("GROQ_MODEL", "qwen/qwen3.8-27b")

# Mode de réflexion de Qwen 3.8 sur Groq :
#   "none"                          -> mode « instruct » : rapide (conseillé ici)
#   "low" / "medium" / "high" / "default" -> mode « thinking » : lent
#   "off"                           -> n'envoie rien (autre modèle sans ce réglage)
REASONING_EFFORT = get_setting("GROQ_REASONING_EFFORT", "none").strip().lower()

CSV_FILE = "livres.csv"
MAX_CATALOGUE_BOOKS = 3
TOTAL_BOOKS = MAX_CATALOGUE_BOOKS + 1  # + 1 découverte hors catalogue
# Réponse courte en mode instruct ; marge pour la réflexion en mode thinking.
MAX_TOKENS = 1500 if REASONING_EFFORT in ("none", "off") else 4000
REQUEST_TIMEOUT = 40  # secondes, par appel à Groq
TIME_BUDGET = 80  # secondes, pour l'ensemble d'une recherche
MAX_RATE_LIMIT_WAIT = 10  # on n'attend une limite de débit que si elle est courte
MAX_EXTERNAL_ATTEMPTS = 2
DISCOVERY_COUNT = 2  # suggestions affichées à l'ouverture de la page
DISCOVERY_SAMPLE_SIZE = 30  # livres tirés au hasard et soumis au modèle


# ==================================================
# OUTILS TEXTE
# ==================================================

def normalize_text(value):
    """Normalise le texte pour comparer les titres."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", value).strip().casefold()


def title_key(title):
    """Clé de comparaison d'un titre tel que le modèle le voit dans le prompt."""
    return normalize_text(str(title).replace("|", "/"))


def titles_match(a, b):
    """Vrai si deux titres sont identiques, à la casse, aux accents
    ou à un sous-titre près."""
    a, b = title_key(a), title_key(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def clean_field(series):
    """Neutralise les retours à la ligne et les '|' d'une colonne."""
    return (
        series
        .str.replace("\n", " ", regex=False)
        .str.replace("|", "/", regex=False)
    )


def strip_reasoning(text):
    """Retire un éventuel bloc <think>, même s'il est tronqué."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ==================================================
# CHARGEMENT DU CATALOGUE (calculé une seule fois)
# ==================================================

@st.cache_data
def load_catalogue():
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

    df = df[df["Titre du livre"] != ""].reset_index(drop=True)

    # Identifiant interne unique pour chaque livre
    df["_id"] = df.index.astype(str)

    # Lignes du catalogue envoyées au modèle : construites une seule fois
    lines = (
        df["_id"]
        + "|" + clean_field(df["Auteur"])
        + "|" + clean_field(df["Titre du livre"])
    )
    text = "\n".join(lines)

    id_by_title = {}
    for book_id, title in zip(df["_id"], df["Titre du livre"]):
        id_by_title.setdefault(title_key(title), book_id)

    return {
        "count": len(df),
        "text": text,
        "by_id": {
            book_id: {
                "id": book_id,
                "titre": title,
                "auteur": author,
                "ligne": line,
            }
            for book_id, title, author, line in zip(
                df["_id"], df["Titre du livre"], df["Auteur"], lines
            )
        },
        "id_by_title": id_by_title,
        "titles": {normalize_text(t) for t in df["Titre du livre"]},
    }


@st.cache_resource
def get_client(api_key):
    """Client Groq créé une seule fois, avec un timeout court."""
    # max_retries=0 : les nouvelles tentatives sont gérées par l'application
    # (sinon le SDK peut attendre longtemps en silence).
    return Groq(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)


# ==================================================
# INTERFACE : EN-TÊTE ET CHARGEMENT
# ==================================================

st.title("Conseiller spécial pour la famille Viavant")
st.write(
    "Cloé, Gilberte, trouvez vos prochaines lectures parmi les livres, "
    "avec en bonus une découverte extérieure à notre bibliothèque."
)

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

st.caption(f"📖 {catalogue['count']} livres dans ton catalogue.")

api_key = get_setting("GROQ_API_KEY")

if not api_key:
    st.error(
        "Clé API Groq introuvable. "
        "Configure GROQ_API_KEY dans "
        ".streamlit/secrets.toml."
    )
    st.stop()

client = get_client(api_key)


# ==================================================
# ZONE DES SUGGESTIONS D'ACCUEIL (remplie plus bas)
# ==================================================

suggestions_box = st.container()


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
# APPEL À GROQ
# ==================================================

SYSTEM_PROMPT = (
    "Tu es un conseiller littéraire rigoureux. "
    "Tu respectes le format de réponse demandé, "
    "tu n'inventes pas de livres du catalogue "
    "et tu distingues clairement les suggestions "
    "issues du catalogue des suggestions extérieures."
)


def reasoning_options():
    """Paramètres de réflexion envoyés à Groq (voir REASONING_EFFORT)."""
    if REASONING_EFFORT in ("", "off"):
        return {}

    options = {"reasoning_effort": REASONING_EFFORT}
    if REASONING_EFFORT != "none":
        options["reasoning_format"] = "hidden"  # ne renvoyer que la réponse
    return options


def retry_after_seconds(error):
    """Délai conseillé par Groq après une erreur 429, s'il est indiqué."""
    try:
        return float(error.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def call_groq(prompt, max_tokens=MAX_TOKENS):
    """Envoie une demande à Groq. Renvoie (texte, finish_reason)."""
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
                temperature=0.6,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            break
        except groq.RateLimitError as error:
            wait = retry_after_seconds(error)
            # Une seule nouvelle tentative, et seulement si l'attente est courte.
            if attempt == 2 or wait is None or wait > MAX_RATE_LIMIT_WAIT:
                raise
            time.sleep(wait + 0.5)

    choice = response.choices[0]
    return strip_reasoning(choice.message.content or ""), choice.finish_reason


# ==================================================
# PROMPTS
# ==================================================

def format_lines(n):
    """Lignes d'exemple du format demandé pour les livres du catalogue."""
    line = "ID: identifiant | titre : titre exact | raison : justification"
    return "\n".join([line] * n)


def build_main_prompt(preferences, avoid):
    n = MAX_CATALOGUE_BOOKS
    avoid_text = avoid.strip() or "Rien de précisé"
    id_lines = format_lines(n)

    return f"""
Tu es un conseiller littéraire.

ENVIES DU LECTEUR :
{preferences}

LIVRES À ÉVITER :
{avoid_text}

CATALOGUE COMPLET :
ID|Auteur|Titre
{catalogue["text"]}

MISSION OBLIGATOIRE :
Tu dois fournir EXACTEMENT {n + 1} recommandations :
- {n} livres présents dans le catalogue ;
- 1 livre réel, absent du catalogue, pour faire découvrir
  une nouvelle lecture au lecteur.

Pour les {n} recommandations du catalogue :
- Utilise uniquement les identifiants exacts du catalogue.
- Ne recommande pas deux fois le même livre.
- Explique brièvement pourquoi chaque livre correspond aux envies.

Pour la dernière recommandation (hors catalogue) :
- Choisis un livre réel, publié et pertinent.
- Ne choisis aucun titre présent dans le catalogue.
- Ne reprends aucun des {n} livres déjà recommandés.
- Indique le titre, l'auteur et une justification courte.
- Cette recommandation est obligatoire.

FORMAT DE RÉPONSE STRICT :
{id_lines}
EXTERIEUR: titre | auteur : Prénom Nom | raison : justification

Pour la ligne EXTERIEUR, l'ordre est strict : le titre, puis « auteur : »
suivi du nom de l'auteur, puis « raison : ».
Remplace « identifiant » par un identifiant réellement présent
dans le catalogue et recopie le titre exactement comme dans le
catalogue. Commence chaque ligne par « ID: ».
Ne fournis ni introduction, ni conclusion, ni autre ligne.
"""


def build_completion_prompt(preferences, avoid, kept_ids, missing):
    """Demande les livres du catalogue qui manquent."""
    avoid_text = avoid.strip() or "Rien de précisé"
    id_lines = format_lines(missing)
    kept = ", ".join(kept_ids) or "aucun"

    return f"""
Tu es un conseiller littéraire.

ENVIES DU LECTEUR :
{preferences}

LIVRES À ÉVITER :
{avoid_text}

CATALOGUE COMPLET :
ID|Auteur|Titre
{catalogue["text"]}

Identifiants déjà retenus (à ne pas reprendre) : {kept}

Choisis EXACTEMENT {missing} autre(s) livre(s) du catalogue,
avec des identifiants exacts, différents de ceux déjà retenus.

FORMAT DE RÉPONSE STRICT :
{id_lines}

Ne fournis ni introduction, ni conclusion, ni autre ligne.
"""


def build_external_prompt(preferences, avoid, excluded_titles):
    """Demande uniquement le livre extérieur (le catalogue n'est pas
    renvoyé : la vérification est faite en Python)."""
    avoid_text = avoid.strip() or "Rien de précisé"
    excluded = "\n".join(f"- {t}" for t in excluded_titles) or "- Aucun"

    return f"""
Tu es un conseiller littéraire.

Tu dois proposer UN SEUL livre réel, publié et pertinent.

ENVIES DU LECTEUR :
{preferences}

LIVRES À ÉVITER :
{avoid_text}

LIVRES INTERDITS (déjà recommandés, déjà dans la bibliothèque
de la famille, ou déjà refusés) :
{excluded}

CONSIGNES :
- Choisis un livre réel dont tu connais le titre et l'auteur.
- Ne reprends aucun des livres interdits.
- Donne une raison courte et personnalisée.
- Réponds sur une seule ligne avec exactement ce format :
EXTERIEUR: titre | auteur : Prénom Nom | raison : justification
- L'ordre est strict : le titre, puis « auteur : » suivi du nom de
  l'auteur, puis « raison : ».
- Aucun texte avant ou après cette ligne.
"""


# ==================================================
# PARSING DES RÉPONSES
# ==================================================

# Tolère : puces/gras éventuels, préfixe « ID: » absent, titre facultatif.
CATALOGUE_LINE = re.compile(
    r"^[\s\-*•>]*(?:ID\s*:?\s*)?(\d+)\s*\*{0,2}\s*\|\s*"
    r"(?:titre\s*:\s*(.+?)\s*\|\s*)?"
    r"raison\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE
)

# Tolère : puces/gras, « EXTÉRIEUR » accentué, libellé « auteur : » absent.
EXTERNAL_LINE = re.compile(
    r"^[\s\-*•>]*EXT[ÉE]RIEUR\s*:\s*(.+?)\s*\|\s*"
    r"(?:auteur\s*:\s*)?(.+?)\s*\|\s*"
    r"raison\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE
)


def clean_author(value):
    """Retire les restes du format demandé que le modèle recopie parfois
    (ex. « Gaël Faye : nom de l'auteur »)."""
    value = re.sub(r"\s*:\s*nom de l['’]auteur\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:auteur|nom de l['’]auteur)\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip(" *[]\"«»")


def parse_catalogue_recommendations(answer, exclude_ids=(), limit=None):
    """Ne conserve que les identifiants réellement présents dans le CSV."""
    limit = limit or MAX_CATALOGUE_BOOKS
    seen_ids = set(exclude_ids)
    recommendations = []

    for book_id, title, reason in CATALOGUE_LINE.findall(answer):
        # Si le modèle a donné un titre, il doit correspondre à l'ID.
        # Sinon on se fie au titre (plus fiable pour un LLM qu'un numéro).
        if title:
            expected = catalogue["by_id"].get(book_id)
            if expected is None or not titles_match(expected["titre"], title):
                book_id = catalogue["id_by_title"].get(title_key(title))
                if book_id is None:
                    continue

        if book_id in seen_ids or book_id not in catalogue["by_id"]:
            continue

        seen_ids.add(book_id)
        book = catalogue["by_id"][book_id]

        recommendations.append({
            "id": book_id,
            "titre": book["titre"],
            "auteur": book["auteur"],
            "raison": reason.strip(),
            "exterieur": False
        })

        if len(recommendations) >= limit:
            break

    return recommendations


def parse_external_candidate(answer):
    """Extrait la ligne EXTERIEUR (sans encore valider le titre)."""
    match = EXTERNAL_LINE.search(answer)

    if not match:
        return None

    title, author, reason = [v.strip() for v in match.groups()]
    title = title.strip(" *[]\"«»")
    author = clean_author(author)

    if not (title and author and reason):
        return None

    return {
        "titre": title,
        "auteur": author,
        "raison": reason,
        "exterieur": True
    }


def is_new_title(title, refused_titles=()):
    """Vrai si le titre n'est ni dans le catalogue ni déjà refusé."""
    normalized = normalize_text(title)
    return (
        normalized not in catalogue["titles"]
        and normalized not in {normalize_text(t) for t in refused_titles}
    )


# ==================================================
# LOGIQUE DE RECOMMANDATION
# ==================================================

def recommend_books(preferences, avoid, progress=None):
    progress = progress or (lambda message: None)
    started = time.monotonic()
    logs = []
    failed = False  # une erreur Groq a déjà eu lieu : on arrête les appels

    def time_left():
        return TIME_BUDGET - (time.monotonic() - started)

    def ask(label, prompt):
        answer, finish_reason = call_groq(prompt)
        note = (
            " [réponse tronquée : max_tokens atteint]"
            if finish_reason == "length" else ""
        )
        logs.append(f"{label}{note} :\n{answer or '(vide)'}")
        return answer

    def ask_optional(label, prompt):
        """Appel facultatif : en cas d'erreur ou de délai dépassé,
        on garde ce qu'on a déjà au lieu de tout perdre."""
        nonlocal failed

        if failed or time_left() <= 0:
            logs.append(f"{label} : ignoré (erreur ou délai de {TIME_BUDGET} s dépassé)")
            return ""

        try:
            return ask(label, prompt)
        except groq.APIError as error:
            failed = True
            logs.append(f"{label} : erreur Groq ({type(error).__name__}: {error})")
            return ""

    # 1) Appel principal : livres du catalogue + livre extérieur
    progress("1/3 · Analyse de la bibliothèque…")
    answer = ask("Réponse principale", build_main_prompt(preferences, avoid))

    catalogue_recs = parse_catalogue_recommendations(answer)
    external_rec = None
    refused_titles = []

    candidate = parse_external_candidate(answer)
    if candidate:
        if is_new_title(candidate["titre"]):
            external_rec = candidate
        else:
            refused_titles.append(candidate["titre"])

    # 2) Complète si des livres du catalogue manquent (ID inventé, doublon...)
    missing = MAX_CATALOGUE_BOOKS - len(catalogue_recs)
    if missing > 0:
        progress(f"2/3 · Il manque {missing} livre(s) du catalogue, je complète…")
        extra_answer = ask_optional(
            "Complément catalogue",
            build_completion_prompt(
                preferences,
                avoid,
                [rec["id"] for rec in catalogue_recs],
                missing
            )
        )
        catalogue_recs += parse_catalogue_recommendations(
            extra_answer,
            exclude_ids=[rec["id"] for rec in catalogue_recs],
            limit=missing
        )

    # 3) Secours : demande dédiée pour le livre extérieur (2 essais max)
    attempts = 0
    while external_rec is None and attempts < MAX_EXTERNAL_ATTEMPTS:
        attempts += 1
        progress(f"3/3 · Recherche d'une découverte hors catalogue (essai {attempts})…")

        excluded = [rec["titre"] for rec in catalogue_recs] + refused_titles
        fallback_answer = ask_optional(
            f"Secours extérieur (essai {attempts})",
            build_external_prompt(preferences, avoid, excluded)
        )

        candidate = parse_external_candidate(fallback_answer)
        if not candidate:
            if failed or time_left() <= 0:
                break
            continue

        if is_new_title(candidate["titre"], refused_titles):
            external_rec = candidate
        else:
            refused_titles.append(candidate["titre"])

    recommendations = catalogue_recs
    if external_rec:
        recommendations.append(external_rec)

    return recommendations, "\n\n".join(logs)


# ==================================================
# SUGGESTIONS D'ACCUEIL (2 livres différents)
# ==================================================

def sample_books(size):
    """Tire des livres au hasard, un seul par auteur pour varier."""
    books = list(catalogue["by_id"].values())
    random.shuffle(books)

    picked, authors = [], set()
    for book in books:
        author = normalize_text(book["auteur"])
        if author in authors:
            continue
        authors.add(author)
        picked.append(book)
        if len(picked) >= size:
            break

    # Catalogue avec très peu d'auteurs : on complète sans contrainte.
    if len(picked) < DISCOVERY_COUNT:
        picked = books[:size]

    return picked


def build_discovery_prompt(sample):
    lines = "\n".join(book["ligne"] for book in sample)
    id_lines = format_lines(DISCOVERY_COUNT)

    return f"""
Tu es un conseiller littéraire.

Voici une sélection aléatoire de livres de la bibliothèque familiale :
ID|Auteur|Titre
{lines}

Choisis EXACTEMENT {DISCOVERY_COUNT} livres de cette liste, relativement
différents l'un de l'autre (genre, ambiance, époque ou style), pour
donner deux envies de lecture contrastées.

- Utilise uniquement les identifiants exacts de la liste.
- Explique en une phrase pourquoi chaque livre donne envie d'être lu.

FORMAT DE RÉPONSE STRICT :
{id_lines}

Ne fournis ni introduction, ni conclusion, ni autre ligne.
"""


def get_discoveries():
    """Renvoie 2 livres du catalogue, d'auteurs différents.
    Si Groq échoue, on complète au hasard : la page reste utilisable."""
    sample = sample_books(DISCOVERY_SAMPLE_SIZE)

    try:
        answer, _ = call_groq(build_discovery_prompt(sample))
        picks = parse_catalogue_recommendations(answer, limit=DISCOVERY_COUNT)
    except groq.APIError:
        picks = []

    chosen = []
    for pick in picks:
        author = normalize_text(pick["auteur"])
        if all(author != normalize_text(c["auteur"]) for c in chosen):
            chosen.append(pick)

    # Complément aléatoire (sans justification) si besoin
    for book in sample:
        if len(chosen) >= DISCOVERY_COUNT:
            break
        author = normalize_text(book["auteur"])
        if book["id"] in {c["id"] for c in chosen}:
            continue
        if any(author == normalize_text(c["auteur"]) for c in chosen):
            continue
        chosen.append({
            "id": book["id"],
            "titre": book["titre"],
            "auteur": book["auteur"],
            "raison": "",
            "exterieur": False
        })

    return chosen


# ==================================================
# AFFICHAGE
# ==================================================

def show_result(recommendations, diagnostic):
    catalogue_count = sum(not b["exterieur"] for b in recommendations)
    external_count = sum(b["exterieur"] for b in recommendations)

    if len(recommendations) == TOTAL_BOOKS:
        st.success(f"Voici tes {TOTAL_BOOKS} recommandations !")
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
                st.markdown(f"### {index}. 🌟 {book['titre']}")
                st.caption("Découverte hors catalogue")
            else:
                st.markdown(f"### {index}. {book['titre']}")
                st.caption("Issu de ton catalogue")

            st.write(f"**Auteur :** {book['auteur']}")
            st.write(book["raison"])
            st.divider()

    # Réponses brutes affichées uniquement en cas de problème.
    if len(recommendations) != TOTAL_BOOKS:
        with st.expander("Diagnostic : voir la réponse brute de l'IA"):
            st.code(diagnostic)


with suggestions_box:
    st.subheader("✨ Pour commencer : deux idées de la bibliothèque")

    reroll = st.button("🔄 Autres idées")

    if reroll or "discoveries" not in st.session_state:
        with st.spinner("Je feuillette les étagères..."):
            st.session_state["discoveries"] = get_discoveries()

    for column, book in zip(
        st.columns(DISCOVERY_COUNT), st.session_state["discoveries"]
    ):
        with column, st.container(border=True):
            st.markdown(f"**{book['titre']}**")
            st.caption(book["auteur"])
            if book["raison"]:
                st.write(book["raison"])

    st.divider()


if submitted:
    if not preferences.strip():
        st.warning("Décris tes envies de lecture avant de lancer la recherche.")
    else:
        st.session_state.pop("result", None)

        with st.status(
            "Recherche de tes prochaines lectures...", expanded=True
        ) as status:
            try:
                st.session_state["result"] = recommend_books(
                    preferences, avoid, progress=status.write
                )
                status.update(
                    label="Recherche terminée", state="complete", expanded=False
                )
            except groq.RateLimitError:
                status.update(label="Limite d'utilisation atteinte", state="error")
                st.error(
                    "Limite d'utilisation Groq atteinte. "
                    "Réessaie dans une minute."
                )
            except TIMEOUT_ERROR:
                status.update(label="Groq ne répond pas", state="error")
                st.error(
                    "Groq met trop de temps à répondre. Réessaie dans un instant."
                )
            except groq.APIError as e:
                status.update(label="Erreur Groq", state="error")
                st.error(f"Une erreur est survenue avec Groq : {e}")

# Le résultat reste affiché même après un autre clic sur la page.
if "result" in st.session_state:
    show_result(*st.session_state["result"])
