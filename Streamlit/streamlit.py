import streamlit as st
from streamlit_folium import st_folium

import folium
from joblib import load
import os
from st_click_detector import click_detector
#from streamlit_drawable_canvas import st_canvas 
#from PIL import Image
import numpy as np
import datetime
from catboost import CatBoostClassifier

import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)

from Pipeline import *


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
    variables = load('{}/../Data/libelles_variables.joblib'.format(path))
    return variables

#chargements des images pour places 
@st.cache_data
def get_html_places() :
    cat_vehicules = ['moto','car','tc']
    cat_vehicule_html = {}
    for cat in cat_vehicules :
        with open('{}/images/{}.html'.format(path,cat),'r') as html:
            cat_vehicule_html[cat] = html.read().replace('{path}',url_images).replace('.jpg','.jpg?raw=true')
    return cat_vehicule_html

#chargements des modèles
@st.cache_data
def get_models() :
    models = {}
    models['4'] = {}
    models['4']['catboost'] = load('Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/catboost_4_classes.gz'.format(path))
    models['4']['histgradientboost'] = load('{}/..//Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/HistGradientBoosting_4_classes.gz'.format(path))

    models['2'] = {}
    models['2']['catboost'] = load('{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/catboost_2_classes.gz'.format(path))
    models['2']['xgboost'] = load('{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/xgboost_2_classes.gz'.format(path))

    return models

def remove_NR( dico ) :
    return {i:v for i, v in dico.items() if i != '-1'}

@st.cache_data()
def get_data() :
    df = pd.read_csv('{}/../Data/accidents.zip'.format(path))
    return df


tabExploration, tabPrevision = st.tabs(["Exploration", "Prévisions"])

html_places = get_html_places()
vars = get_variables()
models = get_models()

df_accidents = get_data()

with tabExploration :
    variables = list(df_accidents.columns)
    variables = [v for v in variables if v not in ['lat','long']]
    
    #new_vars = {'mois': {'variable':'Mois'}, 'nbv' : {'variable': 'Nombre de véhicules'}}

    var2plot = st.selectbox(label = 'variable', options = variables, format_func = lambda x : vars[x]['variable'])



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







