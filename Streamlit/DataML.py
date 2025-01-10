import os
from pathlib import Path
import streamlit as st

import pandas as pd
import joblib as joblib
from pandas.core.frame import DataFrame
import sys

thispath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(thispath)


class DataML:
    def __init__(self):
        self.X_test = pd.read_csv("{}/..//Data/X_test.zip".format(thispath))
        self.y_test = pd.read_csv("{}/..//Data/y_test.zip".format(thispath))['grav']
        y_ = self.y_test.replace({1: 0})
        self.y_test_2_classes = y_.replace({2: 1, 3: 1})

        #Modèles 4 classe par defaut
        #self.modele_4_names = self.__model_names("{}/..//Models/model_4_classes".format(thispath))
        #self.model_4_classes = self.__load_all_model(self.modele_4_names, "model_4_classes")
        self.resultat_4_classes: DataFrame = pd.read_csv("{}/..//Models/model_4_classes/final_result_4_classes.csv".format(thispath),
                                                         index_col=0)

        # Modèles 4 classe par Over
        #self.modele_4_names_over = self.__model_names("{}/..//Models/model_4_classes_over".format(thispath))
        #self.model_4_classes_over = self.__load_all_model(self.modele_4_names_over, "model_4_classes_over")
        self.resultat_4_classes_over: DataFrame = pd.read_csv(
            "{}/..//Models/model_4_classes_over/final_result_4_classes.csv".format(thispath),
            index_col=0)
        
        #st.dataframe(self.resultat_4_classes_over.index)

        # Modèles 4 classe par under
        #self.modele_4_names_under = self.__model_names("{}/..//Models/model_4_classes_under".format(thispath))
        #self.model_4_classes_under = self.__load_all_model(self.modele_4_names_under, "model_4_classes_under")
        self.resultat_4_classes_under: DataFrame = pd.read_csv(
            "{}/..//Models/model_4_classes_under/final_result_4_classes.csv".format(thispath),
            index_col=0)

        self.catboost_4_classes = joblib.load(
            "{}/..//Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/catboost_4_classes.gz".format(thispath))
        self.histoboost_4_classes = joblib.load(
            "{}/..//Notebooks/MachineLearning/1_ML_4_classes/2_grid_search_best_model/best_models/HistGradientBoosting_4_classes.gz".format(thispath))

        # Modèles 2 classe par defaut
        #self.modele_2_names = self.__model_names("{}/..//Models/model_2_classes".format(thispath))
        # self.model_4_classes = self.__load_all_model(self.modele_4_names, "model_4_classes")
        self.resultat_2_classes: DataFrame = pd.read_csv("{}/..//Models/model_2_classes/final_result_2_classes.csv".format(thispath),
                                                         index_col=0)

        # Modèles 2 classe par under
        #self.modele_2_names_under = self.__model_names("{}/..//Models/model_2_classes_under".format(thispath))
        # self.model_4_classes_under = self.__load_all_model(self.modele_4_names_under, "model_4_classes_under")
        self.resultat_2_classes_under: DataFrame = pd.read_csv(
            "{}/..//Models/model_2_classes_under/final_result_2_classes.csv".format(thispath),
            index_col=0)

        self.catboost_2_classes = joblib.load(
            "{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/catboost_2_classes.gz".format(thispath))
        self.xgboost_2_classes = joblib.load(
            "{}/..//Notebooks/MachineLearning/2_ML_2_classes/2_grid_search_best_model/best_models/xgboost_2_classes.gz".format(thispath))

    def __load_all_model(self, noms_fichiers: [], rep: str):
        models = {}
        for name in noms_fichiers:
            model = joblib.load("{}/..//Models/{}/{}_4_classes.gz".format(thispath,rep, name))
            models[name] = model

        return models

    def __model_names(self, directory: str) -> []:
        repertoire = Path(directory)
        fichiers_gz = [str(fichier) for fichier in repertoire.rglob("*.gz")]

        noms_fichiers = [os.path.basename(f).split('_')[0] for f in fichiers_gz]
        noms_fichiers.sort()

        return noms_fichiers
