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
    from tensorflow.keras.models import Sequential

    thispath = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(thispath)

    from Pipeline import *
    from Functions import *


#listes
url_images = 'https://github.com/karatruc/Streamlit/blob/Streamlit/Streamlit/images'


old_catv = [int(i) for i in ['04','05','06','08','09','11','12','18','19']]

catv_moto = [int(i) for i in ['01','02','30','31','32','33','34','35','36','41','42','43','80','50','60']]
catv_tc = [int(i) for i in ['37','38','39','40']]

var_usag = ['place','catu', 'sexe','age','trajet','secu1','secu2','secu3']
var_cara = ['mois','jour','lum','agg','int','atm','col']
var_lieu = ['catr','circ','vosp','prof','plan','surf','infra','situ']
var_vehi = ['catv','obs', 'obsm','choc','manv','motor']

with st.spinner('Chargements des données...') :
    html_places = get_html_places(thispath, url_images)
    vars = get_variables(thispath)
    models = get_models(thispath)
    df_accidents = get_data(thispath)
    map = get_geoloc_map(thispath)


tabPreprocessing, tabExploration, tabGeolocalisation, tabML, tabNN, tabInterpretabilite,  tabPrevision = \
    st.tabs(["Preprocessing","Exploration","Géolocalisation", "Machine Learning","Réseaux de Neurones","Interprétabilité", "Prévisions"])



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
        
        df_pred = model_predict(m, options, pipe, vars)

        st.dataframe(df_pred)

        # x = pipe.transform(pd.DataFrame.from_dict({k:[v] for k, v in options.items()}))

        # if model == 'Réseau de neurones' :
        #     pred = m.predict(x)
        #     st.write(pred)

        # else :
        #     pred = m.predict_proba(x)

        # df_pred = pd.DataFrame(columns=vars['grav']['valeurs'].values(), data = pred)
        
        # st.dataframe(df_pred.style.highlight_max(color = 'red', axis = 1).format('{:,.2%}'.format))

    






