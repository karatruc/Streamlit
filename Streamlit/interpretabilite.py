import streamlit as st
import shap
import pandas as pd
from DataML import DataML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

thispath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(thispath)

st.cache_data()
def afficher_inter_4_classes(dataML: DataML, df):
    with st.expander("**Interpretabilite sur quatre classes**"):
        st.markdown("#### Interpétabilité global")
        st.markdown("**Importance des variables explicatives pour la prédiction des tués**")

        # shap_df = pd.read_csv("{}/..//Models/final_shapes_values/1_final_catboost_values_data_classe_3_geo.zip".format(thispath),
        #                       compression=dict(method='zip', archive_name='1_final_values_data_classe_3_geo.csv'))

        shap_df = df

        shap_values_csv = shap_df.drop(columns=['x_test_instance', 'base_values']).values

        df_shap = pd.DataFrame(np.abs(shap_values_csv), columns=dataML.X_test.columns)

        plt.figure(figsize=(9, 3))

        #shap.summary_plot(shap_exp)
        #st.pyplot(plt)
        column_groups = regroupement_variable(dataML.X_test)
        df_avg = calcul_moyenne(column_groups, df_shap)
        afficher_global_value(df_avg)

        st.markdown("#### Interprétabilité d'une variable")
        selected_option = st.selectbox("Choisissez une variable", sorted(df_avg.columns), key="interpre_quatre")

        if selected_option:
            feature_summary_plot(selected_option, shap_df, shap_values_csv, dataML.X_test)
            show_categoriel_bar_plot(selected_option, df_shap, column_groups, plot_size=(9, 1))

def get_final_shap_values_3(path) :
    shap_df = pd.read_csv("{}/..//Models/final_shapes_values/1_final_catboost_values_data_classe_3_geo.zip".format(path),
                              compression=dict(method='zip', archive_name='1_final_values_data_classe_3_geo.csv'))
    return shap_df


def get_final_shap_values(path) :
    df_part1 = pd.read_csv("{}/..//Models/final_shapes_values/1_final_catboost_values_data_geo_part1.zip".format(path)
                               , compression=dict(method='zip'))
    df_part2 = pd.read_csv("{}/..//Models/final_shapes_values/1_final_catboost_values_data_geo_part2.zip".format(path)
                            , compression=dict(method='zip'))

    shap_df = pd.concat([df_part1, df_part2], ignore_index=True)
    
    return shap_df

st.cache_data()
def afficher_inter_2_classes(dataML: DataML, df):
    print('debut affichage')
    with st.expander("**Interpretabilite sur deux classes**"):
        st.markdown("#### Interprétabilité globale")
        st.markdown("**Importance des variables explicatives pour la prédiction des tués**")
        
        #shap_df = pd.read_csv("{}/..//Models/final_shapes_values/1_final_catboost_values_data_geo.zip".format(thispath),
        #                      compression=dict(method='zip', archive_name='1_final_catboost_values_data_geo.csv'))
        
        # df_part1 = pd.read_csv("{}/..//Models/final_shapes_values/1_final_catboost_values_data_geo_part1.zip".format(thispath)
        #                        , compression=dict(method='zip'))
        # df_part2 = pd.read_csv("{}/..//Models/final_shapes_values/1_final_catboost_values_data_geo_part2.zip".format(thispath)
        #                        , compression=dict(method='zip'))

        # shap_df = pd.concat([df_part1, df_part2], ignore_index=True)
        
        shap_df = df

        shap_values_csv = shap_df.drop(columns=['x_test_instance', 'base_values']).values

        df_shap = pd.DataFrame(np.abs(shap_values_csv), columns=dataML.X_test.columns)

        

        plt.figure(figsize=(9, 3))

        column_groups = regroupement_variable(dataML.X_test)
        df_avg = calcul_moyenne(column_groups, df_shap)
        afficher_global_value(df_avg)

        st.markdown("#### Interpétabilité d'une variable")
        selected_option = st.selectbox("Choisissez une variable", sorted(df_avg.columns), key="interpre_deux")

        if selected_option:
            print('1')
            feature_summary_plot(selected_option, shap_df, shap_values_csv, dataML.X_test)
            print('2')
            show_categoriel_bar_plot(selected_option, df_shap, column_groups, plot_size=(9, 1))
        
        print('fin affichage')


def calcul_moyenne(column_groups, df_shap):
    df_avg = pd.DataFrame()
    # Calculer les moyennes pour chaque groupe de variable
    for prefix, columns in column_groups.items():
        df_avg[f'{prefix}'] = df_shap[columns].mean(axis=1)
    return df_avg


def regroupement_variable(X_test):
    column_groups = {}

    # Regroupement des colonnes
    for col in X_test.columns:
        prefix = col.split('_')[0]
        if prefix not in column_groups:
            column_groups[prefix] = []
        column_groups[prefix].append(col)

    return column_groups

st.cache_data()
def afficher_global_value(df_avg):
    df_final_var_mean = pd.DataFrame(df_avg.mean().sort_values(ascending=False)).T

    plt.figure(figsize=(19, 4))

    # Bar plot de l'importance des variables explicatives
    sns.barplot(df_final_var_mean)

    # La moyenne total des features
    cat_moy = df_avg.mean().mean()
    plt.axhline(y=cat_moy, color=plt.cm.tab10(3), linestyle='--', linewidth=2,
                label=f"Moyenne de l'ensemble des features : {cat_moy:.6f}")
    plt.text(-0.6, cat_moy, "Moyenne", color='r', va='center', ha='right')

    plt.legend()
    plt.title("Importance des variables explicatives de la classe des Tués")

    st.pyplot(plt)

st.cache_data()
def feature_summary_plot(var_name: str, shap_df, shap_values_csv, X_test):
    mois_columns_with_index = [(index, col) for index, col in enumerate(shap_df.columns) if col.startswith(var_name)]

    index_min = min(mois_columns_with_index)[0]
    index_max = max(mois_columns_with_index)[0]

    plt.figure(figsize=(10, 4))
    shap.summary_plot(shap_values_csv[:, index_min:index_max + 1], X_test.iloc[:, index_min:index_max + 1])

    st.pyplot(plt)

st.cache_data()
def show_categoriel_bar_plot(col_name: str, df_shap, column_groups, plot_size=(5, 5)):
    #print(f"***** Annalyse des shap values de la variables {col_name} *****")
    df_shap_var_cat = pd.DataFrame(df_shap.mean().sort_values(ascending=False)).T
    for prefix, columns in column_groups.items():
        if prefix == col_name:
            var_plot = list()
            for c in df_shap_var_cat.columns:
                if c.startswith(prefix):
                    var_plot.append(c)

            #Plot
            plt.figure(figsize=plot_size)
            ax = sns.barplot(df_shap_var_cat[var_plot])

            #Ligne de la moyenne
            cat_moy = df_shap_var_cat[var_plot].mean().mean()
            #plt.axhline(y=cat_moy, color=plt.cm.tab10(3), linestyle='--', linewidth=2, label='Moyenne')
            plt.axhline(y=cat_moy, color=plt.cm.tab10(3), linestyle='--', linewidth=2, label=f"Moyenne : {cat_moy:.6f}")
            plt.text(-0.6, cat_moy, "Moyenne", color='r', va='center', ha='right')

            #Titre
            plt.title(f"Bar plot shap value de la variable {col_name}")

            plt.legend()
            st.pyplot(plt)
