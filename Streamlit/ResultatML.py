import streamlit as st
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ResulatatML:
    #def __init__(self, name: str, modele_names, modeles, X_test, y_test, df_plot):
    def __init__(self, name: str, X_test, y_test, df_plot, nb_classes=4):
        #self.modele_names = modele_names
        #self.modeles = modeles
        self.X_test = X_test
        self.y_test = y_test
        self.df_plot = df_plot
        self.name = name
        self.nb_classes = nb_classes

    def afficher_resultat(self):
        options = ["Résultat par model", "Résultat par classe"]

        selected_option = st.selectbox("Choisissez le format du résultat", options, key="afficher_resultat" + self.name)

        if selected_option == options[0]:
            self.__resultat_par_model()
        else:
            self.__resulatat_par_classe()

    def __resultat_par_model(self):
        model_selectionne = st.selectbox("Sélectionnez un modèle", self.df_plot.index,
                                         key="resultat_par_model" + self.name)
        if model_selectionne:
            #model = self.modeles[model_selectionne]
            df_m = self.df_plot.loc[model_selectionne]
            df_m = df_m.drop(index="mean")
            # Affichage du tableau
            st.dataframe(df_m)

            is_bar_plot = False
            if is_bar_plot:
                plt.figure(figsize=(9, 2))
                var_ = df_m.sort_values(ascending=False)
                sns.barplot(x=var_.index, y=var_.values, hue=var_.index)
                st.pyplot(plt)
            #self.__rapport_classification(model, model_selectionne, self.X_test, self.y_test)

    def __resulatat_par_classe(self):
        classes_4 = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3']
        classes_2 = ['Classe 0', 'Classe 1']
        classe_liste = classes_4 if self.nb_classes == 4 else classes_2
        var_classe = st.selectbox("Sélectionnez une classe", classe_liste, key="resulatat_par_classe" + self.name)
        index_classe = classe_liste.index(var_classe)
        # Définition de la taille de la figure
        plt.figure(figsize=(9, 3))

        var_ = self.df_plot[str(index_classe)].sort_values(ascending=False)

        ax = sns.barplot(x=var_.values, y=var_.index, hue=var_.index)
        plt.title(f"Recall variable cible {index_classe}")

        for p in ax.patches:
            ax.annotate('{:.2f}'.format(p.get_width()), (p.get_width(), p.get_y() + p.get_height()),
                        ha='center', va='baseline', fontsize=12, color='white', xytext=(-20, 0),
                        textcoords='offset points')

        # Affichage du graphique avec Streamlit
        #plt.tight_layout()
        st.pyplot(plt)

    def rapport_classification(model, nom: str, X_test, y_test):
        # Faire des prédictions
        y_pred = model.predict(X_test)

        # Calculer le rapport de classification imbalancé
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.text("Rapport de classification du modèle {}".format(nom))
        st.dataframe(report_df.style.background_gradient(cmap='Blues'))
