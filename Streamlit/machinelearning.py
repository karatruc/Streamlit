import streamlit as st
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt


from DataML import DataML

from ResultatML import ResulatatML


def afficher(dataML:DataML, df_accidents: DataFrame):

    # Modélisation
    st.markdown("## Modélisation")
    st.text("Nous allons expliqué la démarche entreprise pour la modélisation de notre modèle")

    # Variable Cible
    st.markdown("### 1 - Variable cible")

    # Classes de la variable cible
    st.markdown("### 1.1 - Classes de la variable cible")

    st.markdown("""
    - **0** : Indemne.
    - **1** : Blessé léger.
    - **2** : Blessé hospitalisé.
    - **3** : Tué.
    """)

    # Distribution de la variable cible
    st.markdown("### 1.2 - Distribution de la variable cible")
    plot_distribution_classe(df_accidents, st)

    # Choix de la métrique:
    st.markdown("### 2 - Choix de la métrique")
    st.markdown("""
        Nous faisons face à un déséquilibre des classes avec la gravité des accidents.
        Nous avons choisi le **recall** comme étant la métrique la plus adaptée à notre étude.
        Le **recall** nous permet ainsi d'éviter tout risque de faux négatifs lors d’une prédiction, 
        ce qui donne la priorité à la détection correcte des cas positifs.
    """)

    st.markdown("### 3 - Modélisation sur quatre classes")
    modelisation_4_classes(dataML)

    st.markdown("### 4 - Modélisation sur deux classes")
    modelisation_2_classes(dataML)

    st.markdown("### 5 - Conclusion")

    st.markdown(""" Nous avons choisit le modèle **CatBoost**, car il ressort avec de bons résultats sur les 
                    classes à la fois pour quatre et deux classes par rapport autres modèles.
                """)

def modelisation_2_classes(dataML):

    st.markdown("#### 4.1 - Distribution de la variable cible sur deux classes")
    plt.figure(figsize=(20, 3))
    sns.countplot(x=dataML.y_test_2_classes, stat='percent', hue=dataML.y_test_2_classes)

    st.pyplot(plt)

    st.markdown("#### 4.2 - Modélisation sur deux classes par defaut")
    ml_2_classes = ResulatatML("Model 2 classes",
                               dataML.X_test,
                               dataML.y_test,
                               dataML.resultat_2_classes,
                               nb_classes=2)
    ml_2_classes.afficher_resultat()

    st.markdown("#### 4.3 - Modélisation sur deux classes avec undersampling")
    ml_2_classes_under = ResulatatML("Model deux classes undersampling",
                                    dataML.X_test,
                                    dataML.y_test,
                                    dataML.resultat_2_classes_under,
                                     nb_classes=2)
    ml_2_classes_under.afficher_resultat()

    st.markdown("#### 4.4 - Recherche des hyperparamètres sur deux classes")
    st.markdown("**Paramètres de recherche :**")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("XGBoost"):
            st.code("""
            param_grid = { 'max_depth': [3, 4, 5,7,8],
                           'learning_rate': [0.01, 0.1, 0.2],
                           'n_estimators': [100, 200, 300]
                }
            """)
    with col2:
        with st.expander("CatBoost"):
            st.code("""
            param_grid = { 'iterations': [500, 1000, 1500],         
                           'learning_rate': [0.03,0.1,0.3],     
                           'depth': [6, 8, 10] 
                }
            """)
    st.markdown("**Résultats de recherche :**")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("XGBoost"):
            st.code("""
            Best Param : { 'learning_rate': 0.1,
                           'max_depth': 8, 
                           'n_estimators': 300}
            """)

            ResulatatML.rapport_classification(dataML.xgboost_2_classes,
                                               "",
                                               dataML.X_test,
                                               dataML.y_test_2_classes)
    with col2:
        with st.expander("CatBoost"):
            st.code("""
            Best Param : { 'depth': 10,
                           'iterations': 1500, 
                           'learning_rate': 0.03}
            """)
            ResulatatML.rapport_classification(dataML.catboost_2_classes,
                                               "",
                                               dataML.X_test,
                                               dataML.y_test_2_classes)


def modelisation_4_classes(dataML):
    st.markdown("#### 3.1 - Modélisation sur quatre classes par defaut")
    ml_4_classes = ResulatatML("Model 4 classes",
                               dataML.X_test,
                               dataML.y_test,
                               dataML.resultat_4_classes)
    ml_4_classes.afficher_resultat()

    st.markdown("#### 3.2 - Modélisation sur quatre classes avec oversampling")
    ml_4_classes_over = ResulatatML("Model 4 classes Oversampling",
                                    dataML.X_test,
                                    dataML.y_test,
                                    dataML.resultat_4_classes_over)
    ml_4_classes_over.afficher_resultat()

    st.markdown("#### 3.2 - Modélisation sur quatre classes avec undersampling")
    ml_4_classes_under = ResulatatML("Model 4 classes Undersampling",
                                     dataML.X_test,
                                     dataML.y_test,
                                     dataML.resultat_4_classes_under)
    ml_4_classes_under.afficher_resultat()

    st.markdown("#### 3.3 - Recherche des hyperparamètres sur 4 classes")
    st.markdown("**Paramètres de recherche :**")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("HistGradientBoostingClassifier"):
            st.code("""
            param_grid = { 'learning_rate': [0.001,0.01, 0.1, 0.3],      # Taux d'apprentissage
                            'max_depth': [None,3, 5, 7],                 # Profondeur maximale des arbres
                            }
            """)
    with col2:
        with st.expander("CatBoost"):
            st.code("""
            param_grid = {  'iterations': [500, 1000, 1500],          # Nombre d'itérations (nombre d'arbres)
                            'learning_rate': [0.03,0.1,0.3],      # Taux d'apprentissage
                            'depth': [6, 8, 10]                     # Profondeur des arbres
                            }
            """)
    st.markdown("**Résultat de recherche :**")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("HistGradientBoostingClassifier"):
            st.code("""
            Best Param : {'learning_rate': 0.1,
                          'max_depth': None}
            """)

            ResulatatML.rapport_classification(dataML.histoboost_4_classes,
                                               "",
                                               dataML.X_test,
                                               dataML.y_test)
    with col2:
        with st.expander("CatBoost"):
            st.code("""
            Best Param : {'depth': 6,'iterations': 1000,
                          'learning_rate': 0.1}
            """)
            ResulatatML.rapport_classification(dataML.catboost_4_classes,
                                               "",
                                               dataML.X_test,
                                               dataML.y_test)


# @st.cache_data()
# def load_data_ml():
#     dataML = DataML()
#     return dataML



def plot_distribution_classe(df, st):
    #plt.figure(figsize=(3, 3))
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.countplot(x=df['grav'], stat='percent', hue=df['grav'], ax=ax)

    #plt.title("Répartition de la variable cible")
    plt.show()
    st.pyplot(fig)
