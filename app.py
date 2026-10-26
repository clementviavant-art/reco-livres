import streamlit as st
import pandas as pd
from groq import Groq

# Initialisation sécurisée
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
                
                # 1. Détection intelligente des colonnes Titre et Auteur
                col_titre = next((col for col in ["Titre", "Nom", "Title", "Name"] if col in df_livres.columns), df_livres.columns[0])
                col_auteur = next((col for col in ["Auteur", "Author"] if col in df_livres.columns), None)
                
                # 2. Formatage ultra-léger ligne par ligne (ex: "- Le Petit Prince par Antoine de Saint-Exupéry")
                if col_auteur:
                    liste_livres = [f"- {row[col_titre]} (par {row[col_auteur]})" for _, row in df_livres.iterrows()]
                else:
                    liste_livres = [f"- {row[col_titre]}" for _, row in df_livres.iloc[:, 0].items()]
                
                # On joint l'ensemble avec des sauts de ligne
                contexte_livres = "\n".join(liste_livres)
                
                prompt = f"""
                Tu es un bibliothécaire expert. Voici la liste des livres disponibles dans ma bibliothèque :
                {contexte_livres}
                
                En te basant sur cette liste, recommande 3 livres qui correspondent à la demande suivante : "{user_query}".
                Pour chaque livre, donne le titre, l'auteur et explique brièvement pourquoi il correspond à la demande.
                """
                
                try:
                    completion = client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
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
