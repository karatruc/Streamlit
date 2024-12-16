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

#chargements des variables
@st.cache_data
def variables() :
    variables = load('d:/GIT/Data/libelles_variables.joblib')
    return variables

@st.cache_data
def get_images() :
    images = {}
    images['tc'] = []
    images['moto'] = []
    images['car'] = []

    

    with os.scandir(path + '\images') as files :
        for file in files :
            img=file.path
            
            if file.name.startswith('tc') :
                images['tc'].append(img)
            if file.name.startswith('car') :
                images['car'].append(img)
            if file.name.startswith('moto') :
                images['moto'].append(img)
         
    return images

images = get_images()



#@st_cache_data
#def get_html_images() :


#images = get_images()

with open('d:/git/Streamlit/images/tc.html','r') as html:
    content = html.read().replace('{path}',path)

st.write(content)
    



clicked = click_detector(content)

st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")
# #st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

# st.write(content)


# vars = variables()

# def keyValues(key) :
#     return vars[key]


# # génère le formulaire : une liste de choix par variable
# for k, v in vars.items() :
#     vals = vars[k]['valeurs']
    
#     st.selectbox(key = k, label = vars[k]['variable'], options = list(vals.keys()), format_func = lambda x : vals[x] )









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
