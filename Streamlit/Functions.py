import streamlit as st
from joblib import load
import tensorflow as tf
from scipy.stats import chi2_contingency
from tensorflow.keras.models import Sequential
import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#chargements des variables
@st.cache_data
def get_variables(path) :
    """ récupères le dictionnaire contenant variables, libellés
    """
    variables = load('{}/../Data/libelles_variables.joblib'.format(path))
    variables['hh']['valeurs'] = { x : str(x) for x in range(0,24) }
    return variables

#chargements des images pour places 
@st.cache_data(show_spinner=False)
def get_html_places(path, url) :
    """ insere les images des places dans les fichiers html
    """
    cat_vehicules = ['moto','car','tc']
    cat_vehicule_html = {}
    for cat in cat_vehicules :
        with open('{}/images/{}.html'.format(path,cat),'r') as html:
            cat_vehicule_html[cat] = html.read().replace('{path}',url).replace('.jpg','.jpg?raw=true')
    return cat_vehicule_html

#chargements des modèles
@st.cache_data(show_spinner=False)
def get_models(path) :
    """ Récupère les modèles sauvegardés
    """
    models = {}
    models['4'] = {}
    models['4']['catboost'] = load('{}/..//Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/catboost_4_classes.gz'.format(path))
    models['4']['histgradientboost'] = load('{}/..//Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/HistGradientBoosting_4_classes.gz'.format(path))
    models['4']['Réseau de neurones'] = tf.keras.models.load_model('{}/..//neural_network/results/4class_neural_network_model.h5'.format(path))

    models['2'] = {}
    models['2']['catboost'] = load('{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/catboost_2_classes.gz'.format(path))
    models['2']['xgboost'] = load('{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/xgboost_2_classes.gz'.format(path))
    models['2']['Réseau de neurones'] = tf.keras.models.load_model('{}/..//neural_network/results/2class_neural_network_model.h5'.format(path))

    return models

def remove_NR( dico ) :
    """ Supprime les valeurs -1 du dictionn,aire des variables et valeurs
    """
    return {i:v for i, v in dico.items() if i not in  ['-1',-1]}

@st.cache_data(show_spinner=False)
def get_data(path) :
    """ charge les données issues du preprocessing
    """
    df = pd.read_csv('{}/../Data/accidents.zip'.format(path))
    return df

@st.cache_data(show_spinner=False)
def get_geoloc_map(path) :
    """ Chargement du clustering de geolocalisation
    """
    kmeans = load('{}/../Models/clustering_geoloc.joblib'.format(path))
    centers = kmeans.cluster_centers_
    labels = range(0,80)

    map = folium.Map(location=[0,0], zoom_start=1)

    for label , center in zip(labels, centers) :
        folium.Marker(center, 
                    popup = folium.map.Popup(label, parse_html=True),
                    tooltip= folium.map.Tooltip(permanent=True, text='<b>{}</b><br/>({:.2f}/{:.2f})'.format(label, center[0], center[1]), sticky=False),
                    
                    ).add_to(map)
    
    return map

def plot_cat(df, variable, normalize, dico_vars) :
    """ renvoi un barplot de la gravité en foicntion de la variable fournie
    """
    palette = ['blue','green', 'orange','red']
    labels = ['Indemne','Blessé léger','Blessé Grave', 'Tué']
    
    fig, ax = plt.subplots(figsize=(12, 5) )
    
    if normalize :
        df2plot = (df.groupby([variable,'grav'],observed=True).size()*100 / df.groupby(variable, observed=False).size()).reset_index(name='percent')
        sns.barplot(data = df2plot, x=variable, y='percent', hue='grav', palette = palette)
        
        plt.ylabel('Pourcentages d\'usagers par gravité')
    else :
        df2plot=df[[variable, 'grav']]
        sns.countplot(data=df2plot, x=variable, hue='grav', palette = palette)   
        plt.ylabel('Nombres d\'usagers par gravité')     
    
    hands, labs = ax.get_legend_handles_labels()

    plt.legend(title='Gravité', handles = hands, labels = labels, bbox_to_anchor=(1.35, 1))
    plt.xlabel(dico_vars[variable]['variable'])
    plt.title('Répartition des gravités selon {}'.format(dico_vars[variable]['variable']));
    
    value_keys = list(df[variable].unique())
    value_keys.sort()
    
    values = [dico_vars[variable]['valeurs'][v] for v in value_keys]

    plt.xticks(ticks = range(len(value_keys)),  labels = [dico_vars[variable]['valeurs'][v] for v in value_keys], rotation = 80);
    st.pyplot(fig)
    khi, cramer = test_chi2(df,variable,'grav')
    st.write(khi)
    st.write(cramer)

def plot_num(df, variable, normalize, dico_vars) :
    palette = ['blue','green', 'orange','red']

    df2plot = (df.groupby([variable,'grav'],observed=True).count())
    st.write(df2plot)

    fig = plt.figure(figsize=(10, 4))

    sns.lineplot(data=df2plot,x=variable,y='Count',palette=palette,hue='grav')
    st.pyplot(fig)

def test_chi2(data_frame , var:str,var_cible:str):
    
    ct= pd.crosstab(data_frame[var], data_frame[var_cible])

    chi2_stat, p_value, dof, expected_freq =chi2_contingency(ct)
    
    #print('Test chi2 (',var,',',var_cible,')','p-value :',p_value)

    result = 'Test Chi² : p =  {}'.format(p_value)

    if p_value<0.05:
        # Cramer
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        cramer_v = np.sqrt(chi2_stat / (n * min_dim))
        
        if cramer_v >= 0.5 :
            pot = 'forte'
        elif cramer_v >=0.3 :
            pot = 'modérée'
        elif cramer_v >=0.1 :
            pot = 'faible'
        else :
            pot = 'nulle'
        
        cramer = 'Score de Cramer-V : {} (Potentielle Correlation {})'.format(cramer_v, pot)
        
    return result, cramer
        # Afficher le coefficient de Cramér-V
        #print("Potentiel Correlation entre :",var,"et",var_cible,"Coef Cramér-V :", cramer_v)
        # V = 0 : Aucune association entre les variables.
        # V proche de 0.1 : Faible association.
        # V proche de 0.3 : Association modérée.
        # V proche de 0.5 et plus : Association forte

def highlight_max(x, color):
    return np.where(x == np.max(x.to_numpy()), f"color: {color};", None)

def model_predict(model, dict, pipeline, variables) :
    x = pipeline.transform(pd.DataFrame.from_dict({k:[v] for k, v in dict.items()}))
    st.write(x)

    if model.__class__.__name__ == 'Sequential' :
        pred = model.predict(x)
        
    else :
        pred = model.predict_proba(x)

    df_pred = pd.DataFrame(columns=variables['grav']['valeurs'].values(), data = pred).style.highlight_max(color = 'red', axis = 1).format('{:,.2%}'.format)
    
    return df_pred
