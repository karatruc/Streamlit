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
    import Functions

    from DataML import DataML
    import machinelearning as ml
    import interpretabilite as inter

@st.cache_data(show_spinner = False)
def plot_cat(df, variable, normalize, dico_vars, stacked) :
    return Functions.plot_cat(df, variable, normalize, dico_vars, stacked)

@st.cache_data(show_spinner = False)
def plot_cat_old_data(df, variable, normalize, dico_vars) :
    return Functions.plot_cat_old_data(df, variable, normalize, dico_vars)

# st.cache_data()
# def afficher_inter_4_classes(data) :
#     return inter.afficher_inter_4_classes(data)

# st.cache_data()
# def afficher_inter_2_classes(data) :
#     return inter.afficher_inter_2_classes(data)

st.cache_data()
def afficher(data,df) :
    return ml.afficher(data,df)




# @st.cache_data(show_spinner = False)
# def load_data_ml():
#     dataML = DataML()
#     return dataML


#listes
url_images = 'https://github.com/karatruc/Streamlit/blob/main/Streamlit/images'


old_catv = [int(i) for i in ['04','05','06','08','09','11','12','18','19']]

catv_moto = [int(i) for i in ['01','02','30','31','32','33','34','35','36','41','42','43','80','50','60']]
catv_tc = [int(i) for i in ['37','38','39','40']]

var_usag = ['place','catu', 'sexe','age','trajet','secu1','secu2','secu3']
var_cara = ['mois','jour','lum','agg','int','atm','col']
var_lieu = ['catr','circ','vosp','prof','plan','surf','infra','situ']
var_vehi = ['catv','obs', 'obsm','choc','manv','motor']

with st.spinner('Chargements des données...') :

    if 'html_places' not in st.session_state :
        st.session_state['html_places'] = Functions.get_html_places(thispath, url_images)
    
    html_places = st.session_state['html_places']

    if 'vars' not in st.session_state :
        st.session_state['vars'] = Functions.get_variables(thispath)

    vars = st.session_state['vars']

    if 'models' not in st.session_state :
        st.session_state['models'] = Functions.get_models(thispath)
    
    models = st.session_state['models']

    if 'df_accidents' not in st.session_state :
        st.session_state['df_accidents'] = Functions.get_data(thispath)
    
    df_accidents = st.session_state['df_accidents']

    if 'df_brutes' not in st.session_state :
        st.session_state['df_brutes'] = Functions.get_data_brutes(thispath)
    
    df_brutes = st.session_state['df_brutes']
    
    if 'map' not in st.session_state :
        st.session_state['map'] = Functions.get_geoloc_map(thispath)
    
    map = st.session_state['map']


    if 'dataML' not in st.session_state :    
        st.session_state['dataML'] = DataML()

    dataML = st.session_state['dataML']

    if 'final_shap_values' not in st.session_state :
        st.session_state['final_shap_values'] =  inter.get_final_shap_values(thispath)
    
    if 'final_shap_values_3' not in st.session_state :
        st.session_state['final_shap_values_3'] =  inter.get_final_shap_values_3(thispath)


    
    
    #mapLatLong = Functions.localisationLatLong(df_brutes, thispath)


tabPreprocessing, tabExploration, tabExplorationPreprocessing, tabGeolocalisation, tabML, tabNN, tabInterpretabilite,  tabPrevision = \
    st.tabs(["Preprocessing","Exploration ( données brutes )","Exploration ( données traitées )", "Géolocalisation", "Machine Learning","Réseaux de Neurones","Interprétabilité", "Prévisions"])

with tabPreprocessing :
    expPreprocessing_1 = st.expander("Fusion des fichiers..." )
    expPreprocessing_1.image('{}/images/jointure.png'.format(thispath)   
    )
    
    expPreprocessing_2 = st.expander("Nettoyage...")
    expPreprocessing_2.write(
        '''- Suppression des variables pour lesquelles plus de 25 % des valeurs n'étaient pas renseignées.\n
- Suppression des lignes sans variable d'intérêt (la gravité des blessures).\n
- Suppression des variables difficlement interprétables ( l'adresse,le numéro ou le type de voie, ou encore le département, le sens de circulation, etc.\n
- Suppression de 14 % de lignes pour lesquelles au moins une valeur n'était pas renseignée.\n
- Le jeu de données contient après ce nettoyage environ 443 000 enregistrements.\n

'''
    )
    
    expPreprocessing_3 = st.expander("Recodage...")
    expPreprocessing_3.write(
        '''- Recodage des latitude et longitude (séparateur décimal).\n
- Recodage des variables relatives aux équipements de sécurité ont été recodées 
(Une variable binaire par type d'équipement)\n
- De l'horodatage, nous avons conservé uniquement l'heure, le mois et la notion de weekend oui semaine.\n
- À partir de l'année de naissance des usagers nous avons calculé l'âge au moment dans l'année de l'accident.\n
- Après ces traitements l'ensemble des variables catégorielles restantes a été recodées en entier.\n
- La variable d'intérêt a été recodée de 0 à 3 de façon progressive avec la gravité.\n
- Après les premières modélisation, il nous a semblé judicieux de transformer la variable âge en classe d'âge.
'''
    )

    expPreprocessing_4 = st.expander("Géolocalisation...")
    expPreprocessing_4.write(
        """
- Les coordonnées Latitude et longitude ont été transformées en cluster.\n
Nous avons pour cela utiliser un algorithme de K-Moyennes, et avons opté pour 80 clusters.\n
La localisation des clusters sur une carte nous a permis de detecter des erreurs de saisies des coordonnées.
""")

    expPreprocessing_6 = st.expander("Binarisation et standardisation...")
    expPreprocessing_6.write(
        """
- L'ensemble des variables catégorielles a été recodé en variable binaire (OneHotEncoder)
- Les variables numériques ont été standardisées (StandardScaler)
""")

    expPreprocessing_7 = st.expander("Séparation du jeu de données")
    expPreprocessing_7.image('{}/images/split.png'.format(thispath)   )
                              
    expPreprocessing_5 = st.expander("Pipeline...")
    expPreprocessing_5.write(
        '''
Les principales étapes de ce preprocessing ont été intégrées dans un pipeline qui sera utilisé pour traiter les données en vue de prédiction.

'''
    )

with tabExploration :
    variablesBrutes = ['an_nais','dep']

    colExp21, colExp22, colExp23 = st.columns(3)

    varBrutes2plot = colExp21.selectbox(label = 'Variable', options = variablesBrutes, format_func = lambda x : vars[x]['variable'] )
    normBrutes = colExp22.toggle('Normalisation', key='norm1')
    stackedBrutes = colExp23.toggle('Données Empilées', key='stacked1')
    with st.spinner('Tracage du graphique') :
        plot_cat(df_brutes, varBrutes2plot, normBrutes, vars, stackedBrutes) 


with tabExplorationPreprocessing :
    
    variables = list(df_accidents.columns)
    variables = [v for v in variables if v not in ['lat','long','nbv','vma','grav','geoloc']]
    
    colExp11, colExp12, colExp13 = st.columns(3)

    var2plot = colExp11.selectbox(label = 'Variable', options = variables, format_func = lambda x : vars[x]['variable'], label_visibility = 'hidden' )
    norm = colExp12.toggle('Normalisation', key = 'norm2')
    stacked = colExp13.toggle('Données Empilées', key = 'stacked2')
    with st.spinner('Tracage du graphique') :
        plot_cat(df_accidents, var2plot, norm, vars, stacked) 

with tabGeolocalisation :
    
    with st.expander('Localisations tués et blessés graves en France') :
        st.image('{}/images/map_monde.png'.format(thispath))

    with st.expander('Localisations tués et blessés graves en métropole', expanded = False) :
        st.image('{}/images/france.png'.format(thispath))

    with st.expander('Localisations des clusters', expanded = False) :
    #f_map = st_folium(map, use_container_width=True)# width=725)
        f_map = st_folium(map, use_container_width=True)# width=725)

    #f_map_pie = st_folium(map, use_container_width=True)# width=725)

    #f_remap = st.folium(mapLatLong, use_container_width=True)

with tabPrevision :
    
    # génère le formulaire : une liste de choix par variable
    options = {}

    cas = st.radio("Préremplissage", ["aucun",'pas grave', 'grave'])
    default_options = {"lum":1,"agg":1,"int":1,"atm":1,"col":1,"jour":"11","mois":"01","an":"2025","hrmn":"1637","lat":47.57652571374621,"long":2.4169921875000004,"catr":1,"circ":1,"vosp":0,"prof":1,"plan":1,"surf":1,"infra":0,"situ":1,"nbv":1,"vma":50,"catv":0,"obs":0,"obsm":0,"choc":0,"manv":0,"motor":0,"catu":1,"sexe":1,"trajet":0,"secu1":0,"secu2":0,"secu3":0,"an_nais":2000,"place":"1"}
    if cas == 'grave' :
        default_options = {"lum":3,"agg":1,"int":2,"atm":1,"col":1,"jour":"11","mois":"01","an":"2025","hrmn":"1623","lat":46.66451741754238,"long":1.9775390625000002,"catr":3,"circ":2,"vosp":0,"prof":1,"plan":4,"surf":2,"infra":0,"situ":1,"nbv":2,"vma":130,"catv":33,"obs":0,"obsm":2,"choc":3,"manv":0,"motor":0,"catu":1,"sexe":1,"trajet":0,"secu1":0,"secu2":0,"secu3":0,"an_nais":2000,"place":"1"}
    if cas == 'pas grave' :
        default_options = {"lum":1,"agg":2,"int":9,"atm":1,"col":7,"jour":"11","mois":"01","an":"2025","hrmn":"1637","lat":47.57652571374621,"long":2.4169921875000004,"catr":7,"circ":1,"vosp":0,"prof":1,"plan":1,"surf":1,"infra":0,"situ":1,"nbv":1,"vma":30,"catv":7,"obs":1,"obsm":0,"choc":2,"manv":0,"motor":1,"catu":1,"sexe":2,"trajet":0,"secu1":5,"secu2":1,"secu3":0,"an_nais":2000,"place":"8"}
    #caractéristiques
    st.subheader("Caractéristiques de l'accident", divider="gray")

    col = 0
    cols_cara = st.columns(2)

    for k in [x for x in var_cara if x not in ['mois','jour']]:
        vals = Functions.remove_NR(vars[k]['valeurs'])
        
        options[k] = cols_cara[col].selectbox(key = k, 
                                              label = vars[k]['variable'], 
                                              options = list(vals.keys()), 
                                              format_func = lambda x : vals[x], 
                                              index = list(vals.keys()).index(default_options[k]) )
        col = (col == 0) * 1

    ddmmyyyy = cols_cara[col].date_input('Date de l\'accident')
    col = (col == 0) * 1
    hhmm = cols_cara[col].time_input('Heure de l\'accident')
    col = (col == 0) * 1

    options['jour'] = ddmmyyyy.strftime('%d')
    options['mois'] = ddmmyyyy.strftime('%m')
    options['an'] = ddmmyyyy.strftime('%Y')

    options['hrmn'] = hhmm.strftime('%H%M') 


    # map
    map_cont = st.container(height=600, border=True)
    DEFAULT_LATITUDE = default_options['lat']#46.3
    DEFAULT_LONGITUDE = default_options['long']#2.85

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
    cols_lieu = st.columns(2)
    col = 0
    for k in var_lieu :
        vals = Functions.remove_NR(vars[k]['valeurs'])
        options[k] = cols_lieu[col].selectbox(key = k, 
                                              label = vars[k]['variable'], 
                                              options = list(vals.keys()), 
                                              format_func = lambda x : vals[x], 
                                              index = list(vals.keys()).index(default_options[k]) )
        col = (col == 0) * 1

    #options['nbv'] = st.number_input('Nombre de voies de circulation')
    options['nbv'] = cols_lieu[col].slider('Nombre de voies de circulation', min_value = 1, max_value = 12,  step = 1, value = default_options['nbv'] )
    col = (col == 0) * 1

    options['vma'] = cols_lieu[col].slider('Vitesse maximale autorisée', min_value = 10, max_value = 130,  step = 5, value = default_options['vma'])
    col = (col == 0) * 1
    #vehicules

    

    st.subheader("Véhicule transportant la victime", divider="gray")
    cols_vehi = st.columns(2)
    col = 0

    for k in var_vehi :
        vals = Functions.remove_NR(vars[k]['valeurs'])
        if k == 'catv' :
            vals = {x:vals[x] for x in vals.keys() if x not in old_catv}
        options[k] = cols_vehi[col].selectbox(key = k, 
                                              label = vars[k]['variable'], 
                                              options = list(vals.keys()), 
                                              format_func = lambda x : vals[x] , 
                                              index = list(vals.keys()).index(default_options[k]) )
        col = (col == 0) * 1

    # usagers
    st.subheader("Victime", divider="gray")

    cols_usager = st.columns(2)
    col = 0
    for k in [x for x in var_usag if x not in ['place','age']] :
        vals = Functions.remove_NR(vars[k]['valeurs'])
        options[k] = cols_usager[col].selectbox(key = k, 
                                                label = vars[k]['variable'], 
                                                options = list(vals.keys()), 
                                                format_func = lambda x : vals[x], 
                                                index = list(vals.keys()).index(default_options[k]) )
        col = (col == 0) * 1
    options['an_nais'] = cols_usager[col].selectbox('Année de naissance', range(1900,2025), index = 100)
    col = (col == 0) * 1

    if options['catv'] in catv_tc :
        content = html_places['tc']
    elif options['catv'] in catv_moto :
        content = html_places['moto']
    else  :
        content = html_places['car']

    #options['place'] = default_options['place']
    options['place'] = click_detector(content)

    if options['place'] == "" :
        options['place'] = default_options['place']

    st.write("place : {}".format(options['place']))

    st.subheader("Modèle de prédiction")

    cols_pred = st.columns(2)

    nb_classes = cols_pred[0].selectbox('Nb de classes prédites', list(models.keys()))

    model = cols_pred[1].selectbox('Modèle de prédiction', list(models[nb_classes].keys()))

    if st.button('Effectuer la prédiction') :
        
        m = models[nb_classes][model]

        df_pred = Functions.model_predict(m, options, pipe, vars)

        st.dataframe(df_pred)


with tabML:
    
    #import machinelearning as ml
    #ml.afficher(dataML,df_accidents)
    print('debut ML')
    afficher(dataML,df_accidents)
    print('fin ML')

with tabInterpretabilite:
    
    #import interpretabilite as inter
    inter.afficher_inter_4_classes(dataML, st.session_state['final_shap_values_3'])
    inter.afficher_inter_2_classes(dataML, st.session_state['final_shap_values'])
    

    
with tabNN:
    
    image1_path = os.path.join(thispath, "./images/img1.jpg")
    image2_path = os.path.join(thispath, "./images/img2.jpg")
    
    #st.subheader("Réseaux de Neurones", divider="gray")

    st.subheader("Visualisation des Réseaux de Neurones")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Réseaux de Neurones 4 classes")
        st.image(image1_path, caption="Première Image", use_container_width=True)  

    with col2:
        st.markdown("### Réseaux de Neurones 2 classes")
        st.image(image2_path, caption="Seconde Image", use_container_width=True)





