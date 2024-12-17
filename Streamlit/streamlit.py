import streamlit as st
from streamlit_folium import st_folium
from st_clickable_images import clickable_images
import folium
from joblib import load
import os
from st_click_detector import click_detector
from streamlit_drawable_canvas import st_canvas 
from PIL import Image
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
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
    variables = load('d:/GIT/Data/libelles_variables.joblib')
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

html_places = get_html_places()


vars = get_variables()



# génère le formulaire : une liste de choix par variable
options = {}

#caractéristiques
for k in [x for x in var_cara if x not in ['mois','jour']]:
    vals = vars[k]['valeurs']
    options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )

#lieux
for k in var_lieu :
    vals = vars[k]['valeurs']
    options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )

#vehicules
for k in var_vehi :
    vals = vars[k]['valeurs']
    if k == 'catv' :
        vals = {x:vals[x] for x in vals.keys() if x not in old_catv}
    options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )

# usagers
for k in [x for x in var_usag if x not in ['place','age']] :
    vals = vars[k]['valeurs']
    options[k] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )


st.write(options['catv'])


if options['catv'] in catv_tc :
    content = html_places['tc']
elif options['catv'] in catv_moto :
    content = html_places['moto']
else  :
    content = html_places['car']

st.write(content)


options['catv'] = click_detector(content)

# for k, v in vars.items() :
#     if k not in ['place'] :
#         vals = vars[k]['valeurs']
        
#         if k == 'catv' :
#             vals = vals - old_catv

#         options['k'] = st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )






    


    



#clicked = click_detector(content)

#st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")
# #st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

# st.write(content)












# DEFAULT_LATITUDE = 46.3
# DEFAULT_LONGITUDE = 2.85


# m = folium.Map(location=[DEFAULT_LATITUDE, DEFAULT_LONGITUDE], zoom_start=6)

# # The code below will be responsible for displaying 
# # the popup with the latitude and longitude shown
# m.add_child(folium.LatLngPopup())

# f_map = st_folium(m, width=725)

# selected_latitude = DEFAULT_LATITUDE
# selected_longitude = DEFAULT_LONGITUDE

# if f_map.get("last_clicked"):
#     selected_latitude = f_map["last_clicked"]["lat"]
#     selected_longitude = f_map["last_clicked"]["lng"]


# form = st.form("Position entry form")

# submit = form.form_submit_button()

# if submit:
#     if selected_latitude == DEFAULT_LATITUDE and selected_longitude == DEFAULT_LONGITUDE:
#         st.warning("Selected position has default values!")
#     st.success(f"Stored position: {selected_latitude}, {selected_longitude}")
