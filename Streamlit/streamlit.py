#import des bibliothèques
import streamlit as st
#paramètres de la page streamlit
st.set_page_config(layout="wide")
st.title('Accidents de la route en France 2019-2023')
st.markdown(
    """
    <style>
    [data-testid="stTabs"] > div:first-child{
        width: 1024px;
    }
    """,
    unsafe_allow_html=True,
)

with st.spinner ('Chargement des bibliothèques') :

    from streamlit_folium import st_folium
    import folium
    from joblib import load
    import os
    from st_click_detector import click_detector
    import numpy as np
    import datetime
    from catboost import CatBoostClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    from sklearn.cluster import KMeans
    #from tensorflow.keras.models import Sequential
    import tensorflow as tf
    from scipy.stats import chi2_contingency

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)

from Pipeline import *

#listes
url_images = 'https://github.com/karatruc/Streamlit/blob/Streamlit/Streamlit/images'

old_catv = ['04','05','06','08','09','11','12','18','19']

catv_moto = ['01','02','30','31','32','33','34','35','36','41','42','43','80','50','60']
catv_tc = ['37','38','39','40']

var_usag = ['place','catu', 'sexe','age','trajet','secu1','secu2','secu3']
var_cara = ['mois','jour','lum','agg','int','atm','col']
var_lieu = ['catr','circ','vosp','prof','plan','surf','infra','situ']
var_vehi = ['catv','obs', 'obsm','choc','manv','motor']


#chargements des variables
@st.cache_data
def get_variables() :
    """ récupères le dictionnaire contenant variables, libellés
    """
    variables = load('{}/../Data/libelles_variables.joblib'.format(path))
    return variables

#chargements des images pour places 
@st.cache_data(show_spinner=False)
def get_html_places() :
    """ insere les images des places dans les fichiers html
    """
    cat_vehicules = ['moto','car','tc']
    cat_vehicule_html = {}
    for cat in cat_vehicules :
        with open('{}/images/{}.html'.format(path,cat),'r') as html:
            cat_vehicule_html[cat] = html.read().replace('{path}',url_images).replace('.jpg','.jpg?raw=true')
    return cat_vehicule_html

#chargements des modèles
@st.cache_data(show_spinner=False)
def get_models() :
    """ Récupère les modèles sauvegardés
    """
    models = {}
    models['4'] = {}
    models['4']['catboost'] = load('Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/catboost_4_classes.gz'.format(path))
    models['4']['histgradientboost'] = load('{}/..//Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/HistGradientBoosting_4_classes.gz'.format(path))
    #models['4']['Réseau de neurones'] = load('{}/../neural_network/results/2class_neural_network_model.joblib'.format(path))

    models['2'] = {}
    models['2']['catboost'] = load('{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/catboost_2_classes.gz'.format(path))
    models['2']['xgboost'] = load('{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/xgboost_2_classes.gz'.format(path))
    #models['2']['Réseau de neurones'] = load('{}/../neural_network/results/4class_neural_network_model.joblib'.format(path))

    return models

def remove_NR( dico ) :
    """ Supprime les valeurs -1 du dictionn,aire des variables et valeurs
    """
    return {i:v for i, v in dico.items() if i != '-1'}

@st.cache_data(show_spinner=False)
def get_data() :
    """ charge les données issues du preprocessing
    """
    df = pd.read_csv('{}/../Data/accidents.zip'.format(path))
    return df

@st.cache_data(show_spinner=False)
def get_geoloc_map() :
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
    #fig = plt.figure(figsize=(10, 4))
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

tabExploration, tabGeolocalisation, tabPrevision = st.tabs(["Exploration","Géolocalisation", "Prévisions"])

with st.spinner('Chargements des données...') :
    html_places = get_html_places()
    vars = get_variables()
    models = get_models()
    df_accidents = get_data()
    map = get_geoloc_map()

with tabExploration :
    variables = list(df_accidents.columns)
    variables = [v for v in variables if v not in ['lat','long','nbv','vma','grav']]
    
    var2plot = st.selectbox(label = 'Variable', options = variables, format_func = lambda x : vars[x]['variable'] )
    norm = st.toggle('Normalisation')
    
    with st.spinner('Tracage du graphique') :
        plot_cat(df_accidents, var2plot, norm, vars) 

with tabGeolocalisation :
    f_map = st_folium(map, use_container_width=True)# width=725)

with tabPrevision :

    # génère le formulaire : une liste de choix par variable
    options = {}

    #caractéristiques

    st.subheader("Caractéristiques de l'accident", divider="gray")
    for k in [x for x in var_cara if x not in ['mois','jour']]:
        vals = remove_NR(vars[k]['valeurs'])
        
        options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )

    ddmmyyyy = st.date_input('Date de l\'accident')
    hhmm = st.time_input('Heure de l\'accident')

    options['jour'] = ddmmyyyy.strftime('%d')
    options['mois'] = ddmmyyyy.strftime('%m')
    options['an'] = ddmmyyyy.strftime('%Y')

    options['hrmn'] = hhmm.strftime('%H%M') 


    # map
    map_cont = st.container(height=735, border=True)
    DEFAULT_LATITUDE = 46.3
    DEFAULT_LONGITUDE = 2.85

    with map_cont:
        m = folium.Map(location=[DEFAULT_LATITUDE, DEFAULT_LONGITUDE], zoom_start=6)

        # The code below will be responsible for displaying 
        # the popup with the latitude and longitude shown
        m.add_child(folium.LatLngPopup())
        f_map = st_folium(m, use_container_width=True)# width=725)
        options['lat'] = DEFAULT_LATITUDE
        options['long'] = DEFAULT_LONGITUDE

    if f_map.get("last_clicked"):
        options['lat'] = f_map["last_clicked"]["lat"]
        options['long'] = f_map["last_clicked"]["lng"]

        st.write("latitude : {}".format(options['lat']))
        st.write("longitude : {}".format(options['long']))

    #lieux
    st.subheader("Lieu de l'accident", divider="gray")
    for k in var_lieu :
        vals = remove_NR(vars[k]['valeurs'])
        options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )

    #options['nbv'] = st.number_input('Nombre de voies de circulation')
    options['nbv'] = st.slider('Nombre de voies de circulation', min_value = 1, max_value = 12, value = 1, step = 1)

    options['vma'] = st.slider('Vitesse maximale autorisée', min_value = 10, max_value = 130, value = 50, step = 5)
    #vehicules
    st.subheader("Véhicule transportant la victime", divider="gray")
    for k in var_vehi :
        vals = remove_NR(vars[k]['valeurs'])
        if k == 'catv' :
            vals = {x:vals[x] for x in vals.keys() if x not in old_catv}
        options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )

    # usagers
    st.subheader("Victime", divider="gray")
    for k in [x for x in var_usag if x not in ['place','age']] :
        vals = remove_NR(vars[k]['valeurs'])
        options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )

    options['an_nais'] = st.selectbox('Année de naissance', range(1900,2025), index = 100)

    if options['catv'] in catv_tc :
        content = html_places['tc']
    elif options['catv'] in catv_moto :
        content = html_places['moto']
    else  :
        content = html_places['car']

    options['place'] = click_detector(content)
    st.write("place : {}".format(options['place']))


    st.subheader("Modèle de prédiction")



    nb_classes = st.selectbox('Nb de classes prédites', list(models.keys()))

    model = st.selectbox('Modèle de prédiction', list(models[nb_classes].keys()))


    if st.button('Effectuer la prédiction') :
        m = models[nb_classes][model]
        x = pipe.transform(pd.DataFrame.from_dict({k:[v] for k, v in options.items()}))

        pred = m.predict_proba(x)

        st.write(pred)







