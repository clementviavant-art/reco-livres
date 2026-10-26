import streamlit as st
import pandas as pd
from groq import Groq

# Initialisation du client Groq avec les secrets Streamlit
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Erreur de configuration de la clé API Groq : {e}")

st.title("📚 Mon Recommandeur de Livres")

# Chargement de la base Notion
@st.cache_data
def load_data():
    try:
        return pd.read_csv("livres.csv")
    except FileNotFoundError:
        return None

df_livres = load_data()

if df_livres is not None:
    if st.checkbox("Voir la liste des livres disponibles"):
        st.dataframe(df_livres)

    user_query = st.text_input("Quel genre de livre recherchez-vous ou quelle est votre humeur actuelle ?")
    
    if st.button("Obtenir des recommandations"):
        if user_query:
            with st.spinner("L'IA analyse la bibliothèque..."):
                contexte_livres = df_livres.to_string(index=False)
                
                prompt = f"""
                Tu es un bibliothécaire expert. Voici la liste des livres disponibles dans ma bibliothèque :
                {contexte_livres}
                
                En te basant uniquement ou prioritairement sur cette liste, recommande 3 livres qui correspondent à la demande suivante : "{user_query}".
                Pour chaque livre, donne le titre, l'auteur et explique brièvement pourquoi il correspond à la demande.
                """
                
                try:
                    # Appel de l'IA via Groq (Llama 3 70B, extrêmement rapide et performant)
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                    )
                    st.markdown(completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'IA : {e}")
        else:
            st.warning("Veuillez entrer une demande.")
else:
    st.error("Le fichier `livres.csv` est introuvable sur votre dépôt GitHub.")
