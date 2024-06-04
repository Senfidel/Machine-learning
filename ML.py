import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px
import os






# Spécifiez les chemins absolus pour chaque modèle
model_paths = {
'KNeighborsRegressor': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\KNeighborsRegressor_pipeline.pkl",
    'Polynomial Regression': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\Polynomial_Regression_pipeline.pkl",
    'ElasticNet': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\ElasticNet_pipeline.pkl",
    'BaggingRegressor': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\BaggingRegressor_pipeline.pkl",
    'RandomForestRegressor': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\RandomForestRegressor_pipeline.pkl",
    'GradientBoostingRegressor': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\GradientBoostingRegressor_pipeline.pkl",
    'AdaBoostRegressor': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\AdaBoostRegressor_pipeline.pkl",
    'SVR': r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Notbooks\SVR_pipeline.pkl"
}

# Chargement des modèles
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Chargement des données
df = pd.read_csv(
    r"C:\Users\etien\OneDrive\Documents\Master 2\D2SN\Machine learning\Machine learning2\Exam Machine learning\Data\salaries_cleaned.csv")

y = np.log1p(df['salary_in_usd'])
X = df.drop(["salary_in_usd", 'salary'], axis=1)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuration de l'application Streamlit
st.set_page_config(page_title='Vitrine des Modèles', layout='wide', initial_sidebar_state='expanded')

# Barre latérale pour la navigation entre les pages
st.sidebar.title("Navigation")
st.image( "C:/Users/etien/OneDrive/Documents/Master 2/D2SN/Machine learning/Machine learning2/Exam Machine learning/Data/th.jpeg")
pages = ["Contexte du Projet", "Vitrine des Modèles", "Prédictions avec SVR"]
page = st.sidebar.selectbox("Sélectionnez une Page:", pages)
if page == "Contexte du Projet":
    st.title("Contexte du Projet")
    st.markdown("""
    #### Objectif du projet
    Le but de ce projet est de produire une application à partir d'un modèle de machine learning qui prédit le salaire des data scientists en fonction de leurs caractéristiques. Il s'agit bien d'un problème de régression.

    #### Les étapes du projet
    1. **Collecte et récupération des données :**
        Selon [CHALONS, G., 2022](https://kaizen-solutions.net/kaizen-insights/articles-et-conseils-de-nos-experts/cycle-de-vie-projet-machine-learning-8-etapes/), "Les projets de Machine Learning sont des processus longs, du fait qu'ils nécessitent notamment de collecter une grande quantité de données afin d'apporter une réponse robuste, fiable et stable." Mais nous n'avons pas eu à faire ce travail puisque nous utilisons des données propres qui sont disponibles sur Kaggle. Vous pouvez accéder au dataset et à sa description à partir de ce [lien](https://www.kaggle.com/datasets/abhinavshaw09/data-science-job-salaries-2024).

    2. **Exploration et Visualisation des données :**
        Cette étape ainsi que la suivante nous ont permis de mieux comprendre l'environnement étudié, d'éviter un certain nombre de biais et de fournir des données de qualité aux algorithmes de Machine Learning choisis. Cette étape m'a permis de bien visualiser la distribution du salaire, et d'identifier les doublons et l'asymétrie de la distribution du salaire. Elle m'a aussi permis de voir qu'il n'y avait pas de valeurs manquantes dans le jeu de données.

    3. **Préparation des données :**
        Comme tout projet informatique, la qualité des données entrantes impacte fortement les données sortantes : "Garbage In, Garbage Out". Nous avons supprimé les doublons et corrigé les asymétries identifiées (voir le notebook) durant l'étape précédente. Nous avons aussi enrichi les données en convertissant les codes de pays en noms complets grâce à la librairie pycountry et recodé bien d'autres variables pour améliorer leur lisibilité par l'utilisateur.

    4. **Choix et implémentation des modèles**
        Nous avons testé plusieurs algorithmes de régression : k-NN Regression, Polynomial Regression, ElasticNet, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor et le SVR dont nous avons comparé les capacités de généralisation. Notre choix de ces modèles est en partie motivé par notre volonté de mise en pratique. D'ailleurs, c'est le défaut d'application des algorithmes de régression qui nous a poussés à travailler sur cette problématique de régression.

    5. **Entraînement des modèles :**
        Afin d'éviter d'introduire des biais statistiques, nous avons divisé les jeux de données en 2 parties. La première consiste à entraîner les modèles ("training set" 80% des données), la deuxième à les tester ("test set" 20% des données). Nous avons effectué cette séparation de façon aléatoire et reproductible. Pour optimiser les hyperparamètres, nous avons choisi des grilles de recherche plus ou moins simples en fonction du temps d'apprentissage des algorithmes utilisés puisqu'il fallait optimiser le temps d'apprentissage aussi. En effet, les algorithmes à entraîner sont relativement nombreux alors que cette phase du projet est très gourmande en ressources de calcul (CPU, GPU) et en temps.

    6. **Évaluation et Validation des modèles :**
        Après l'implémentation et l'entraînement des modèles sur le "training set", nous avons sélectionné les combinaisons des hyperparamètres qui se trompaient le moins dans la prédiction des données test. C'est-à-dire les hyperparamètres dont la combinaison minimisait le plus le MSE de chaque algorithme.

    7. **Comparaison des modèles et choix du modèle le plus robuste pour notre application: **
        Après l'optimisation des hyperparamètres, nous avons comparé la capacité des algorithmes à prédire les données test et sélectionné celui qui avait le MSE le plus faible. C'est ce modèle que nous avons déployé en production.

    8. **Déploiement du modèle: **
        Nous avons créé une application Streamlit dans laquelle nous avons encapsulé les différents modèles. Dans la première page de l'application, nous présentons le contexte du projet (description sommaire et quelques visualisations). Dans la seconde page, nous présentons des prédictions des algorithmes sur un échantillon des données test.
    """)

    st.write("### Exploration des données")
    st.dataframe(df.head())
    if st.checkbox("Dimensions du dataframe : "):
        st.write(df.shape)

    df['log_salary'] = np.log1p(df['salary_in_usd'])
    fig1 = px.box(df, y='salary_in_usd', template="seaborn", title="Salary Distribution")
    st.plotly_chart(fig1)

    fig2 = px.scatter(df, x='job_title', y='log_salary', template="seaborn", title="Job Title vs. Log Salary")
    st.plotly_chart(fig2)

    z3 = df['job_title'].value_counts().head(10)
    fig3 = px.bar(z3, x=z3.index, y=z3.values, color=z3.index, text=z3.values,
                  labels={'index': 'Job Title', 'y': 'Count', 'text': 'Count'}, template='seaborn',
                  title='<b>Top 10 Popular Roles in Data Science</b>')
    st.plotly_chart(fig3)

    top_paid_roles = df.groupby('job_title', as_index=False)['log_salary'].max().sort_values(by='log_salary',
                                                                                             ascending=False).head(10)
    fig4= px.bar(top_paid_roles, x='job_title', y='log_salary', color='job_title',
                   labels={'job_title': 'Job Title', 'log_salary': 'Log Salary'}, template='ggplot2',
                   text='log_salary', title='<b>Top 10 Highest Paid Roles in Data Science</b>')
    st.plotly_chart(fig4)

    avg_paid_roles = df.groupby('job_title', as_index=False)['log_salary'].mean().sort_values(by='log_salary',
                                                                                              ascending=False).head(10)
    avg_paid_roles['log_salary'] = round(avg_paid_roles['log_salary'], 2)
    fig5 = px.bar(avg_paid_roles, x='job_title', y='log_salary', color='job_title',
                   labels={'job_title': 'Job Title', 'log_salary': 'Avg Salary in USD'}, text='log_salary',
                   template='seaborn', title='<b>Top 10 Roles in Data Science based on Average Pay</b>')
    fig5.update_traces(textfont_size=8)
    st.plotly_chart(fig5)

    fig6 = px.histogram(df, x='log_salary', marginal='rug', template='seaborn',
                         labels={'log_salary': 'Salary in USD'}, title='<b>Salary Distribution</b>')
    st.plotly_chart(fig6)

    fig7 = px.violin(df, x='work_year', y='log_salary', color='work_year',
                      labels={'work_year': 'Year', 'log_salary': 'Log Salary'}, template='seaborn',
                      title='<b>Data Science Salaries by Year</b>')
    st.plotly_chart(fig7)

    fig8 = px.box(df, x='experience_level', y='log_salary', color='experience_level', template='ggplot2',
                   labels={'experience_level': 'Experience Level', 'log_salary': 'Log Salary'},
                   title='<b>Data Science Salaries by Experience</b>')
    st.plotly_chart(fig8)

    fig9 = px.box(df, x='company_size', y='log_salary', color='company_size', template='ggplot2',
                   labels={'company_size': 'Company Size', 'log_salary': 'Log Salary'},
                   title='<b>Data Science Salaries by Company Size</b>')
    st.plotly_chart(fig9)



if page == "Vitrine des Modèles":
    st.title("🏆 Vitrine des Modèles")

    # Sidebar pour la sélection du modèle
    st.sidebar.title("Sélectionner un Modèle")
    selected_model = st.sidebar.selectbox("Choisissez un modèle à explorer", list(models.keys()))

    # Affichage des détails du modèle
    st.header(f"Modèle : {selected_model}")

    # Affichage des détails du modèle
    model = models[selected_model]
    y_pred = model.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)

    st.write(f"### Score de Test (MSE) : {test_score}")

    #st.write("### Paramètres du Modèle :")
    #st.json(model.get_params())

    # Affichage des prédictions pour un échantillon de données de test
    st.write("### Prédictions sur un Échantillon des Données de Test :")
    sample_data = X_test.head(10)
    sample_predictions = model.predict(sample_data)

    st.write(pd.DataFrame({
        'Salaire Réel (log)': y_test.head(10).values,
        'Salaire Prédit (log)': sample_predictions
    }))

    # Comparaison des modèles
    st.sidebar.title("Comparer les Modèles")
    if st.sidebar.checkbox("Comparer tous les modèles"):
        st.subheader("Comparaison des Modèles")
        comparison_data = []

        for name, mdl in models.items():
            y_pred = mdl.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            comparison_data.append((name, score))

        comparison_df = pd.DataFrame(comparison_data, columns=['Modèle', 'MSE']).sort_values(by='MSE')
        st.write(comparison_df)

    # Stylisation de l'application
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stSidebar .stSelectbox>div>div {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    # Ajout d'un footer
    st.markdown("""
        <style>
        footer {
            visibility: hidden;
        }
        footer:after {
            content:'Développé par Votre Nom';
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px.
        }
        </style>
        """, unsafe_allow_html=True)

elif page == "Prédictions avec SVR":
    st.title("🔍 Prédictions de Salaire avec SVR")
    # Section pour la saisie des données utilisateur
    st.markdown("<h3 style='font-size:20px;'>Saisir les Caractéristiques de l'Usager :</h3>", unsafe_allow_html=True)
    with st.form("prediction_form"):
        work_year = st.number_input("**Année de Travail (2020-2024)**", min_value=2020, max_value=2024, step=1)
        salary_currency = st.selectbox("**Devise du Salaire (USD, EUR, GBP, etc.)**", df['salary_currency'].unique())
        experience_level = st.selectbox("**Niveau d'Expérience (Débutant, Intermédiaire, Cadre, Senior)**",
                                        ['Débutant', 'Intermédiaire', 'Cadre', 'Senior'])
        job_title = st.selectbox("**Titre du Poste**", df['job_title'].unique())
        employee_residence = st.selectbox("**Résidence de l'Usager**", df['employee_residence'].unique())
        remote_ratio = st.number_input("**Ratio de Télétravail (0-100)**", min_value=0, max_value=100, step=1)
        company_location = st.selectbox("**Localisation de l'Entreprise**", df['company_location'].unique())
        company_size = st.selectbox("**Taille de l'Entreprise (S, M, L)**", df['company_size'].unique())
        employment_type = st.selectbox("**Type d'emploi**", df['employment_type'].unique())

        submitted = st.form_submit_button("**Prédire**")

    if submitted:
            input_data = pd.DataFrame({
                'work_year': [work_year],
                'salary_currency': [salary_currency],
                'experience_level': [experience_level],
                'job_title': [job_title],
                'employee_residence': [employee_residence],
                'remote_ratio': [remote_ratio],
                'company_location': [company_location],
                'company_size': [company_size],
                'employment_type': [employment_type]
            })
            st.write("### Données Saisies :")
            st.write(input_data)

            # Make prediction using the SVR model
            prediction = models['SVR'].predict(input_data)
            prediction_usd = np.expm1(prediction)  # Convert back from log scale
            prediction_eur = prediction_usd * 0.85  # Convert to EUR using exchange rate 1 USD = 0.85 EUR

            st.write(f"### Salaire Prédit en Dollars : $ {prediction_usd[0]:,.2f}")
            st.write(f"### Salaire Prédit en Euros : € {prediction_eur[0]:,.2f}")

    # Apply some styling to the app
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stSidebar .stSelectbox>div>div {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    # Adding a footer
    st.markdown("""
        <style>
        footer {
            visibility: hidden;
        }
        footer:after {
            content:'Développé par Votre Nom';
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px.
        }
        </style>
        """, unsafe_allow_html=True)
