import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Chargement des artefacts
@st.cache(allow_output_mutation=True)
def load_artifacts():
    with open('encoder.pkl', 'rb') as f:
        label_encoders = pickle.load(f)  # Charger le dictionnaire de LabelEncoders
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return label_encoders, scaler, model

label_encoders, scaler, model = load_artifacts()

# Interface utilisateur
st.title('📊 Expresso Churn Prediction')

# Récupération des catégories depuis les LabelEncoders
try:
    region_categories = label_encoders['REGION'].classes_  # Classes pour REGION
    top_pack_categories = label_encoders['TOP_PACK'].classes_  # Classes pour TOP_PACK
except KeyError:
    st.error("Erreur de chargement de l'encodeur. Vérifiez les colonnes dans encoder.pkl")
    st.stop()

# Section de saisie avec formulaire
with st.form("prediction_form"):
    region = st.selectbox('Region', options=region_categories)
    top_pack = st.selectbox('Top Pack', options=top_pack_categories)
    tenure = st.number_input('Tenure (months)', min_value=0, step=1)
    revenue = st.number_input('Monthly Revenue (USD)', min_value=0.0, step=10.0)
    submitted = st.form_submit_button('Predict Churn')

if submitted:
    try:
        # Création du DataFrame d'entrée
        input_data = pd.DataFrame([[region, top_pack, tenure, revenue]], 
                                columns=['REGION', 'TOP_PACK', 'TENURE', 'REVENUE'])
        
        # Encodage des variables catégoriques avec LabelEncoder
        input_data['REGION'] = label_encoders['REGION'].transform(input_data['REGION'])
        input_data['TOP_PACK'] = label_encoders['TOP_PACK'].transform(input_data['TOP_PACK'])
        
        # Normalisation des variables numériques
        numerical_data = scaler.transform(input_data[['TENURE', 'REVENUE']])
        
        # Combinaison des features
        final_input = np.hstack(input_data[['REGION', 'TOP_PACK']], numerical_data)
        
        # Prédiction
        churn_prob = model.predict_proba(final_input)[0][1]
        
        # Affichage stylisé
        st.success(f"## 🔮 Churn Probability: {churn_prob:.1%}")
        
        # Interprétation
        if churn_prob > 0.7:
            st.warning("❗ Client à haut risque - Action recommandée immédiate")
        elif churn_prob > 0.4:
            st.info("⚠️ Client à risque modéré - Surveillance recommandée")
        else:
            st.success("✅ Client fidèle - Aucune action nécessaire")

    except Exception as e:
        st.error(f"🚨 Erreur lors de la prédiction: {str(e)}")
        st.write("Vérifiez que :")
        st.markdown("- Tous les fichiers artefacts sont présents")
        st.markdown("- Le format des données d'entrée correspond au modèle")