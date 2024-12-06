from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.metrics import classification_report_imbalanced
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import GridSearchCV

import time

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#df_best_model_grid_search = None

def fit(name:str,estmator,X_train_p,y_train_p,X_test_p,y_test_p):

    start_time = time.time()

    # FIT
    estmator.fit(X_train_p, y_train_p)
    
    end_time = time.time()

    # Temps 
    training_time_seconds = end_time - start_time

    if training_time_seconds < 60:
        print(f"Temps d'entraînement : {training_time_seconds:.2f} seconds")
    else:
        # Convertir en minutes
        training_time = training_time_seconds / 60
        print(f"Temps d'entraînement : {training_time:.2f} minutes")
    
    #PREDICT
    y_pred = estmator.predict(X_test_p)

    
    # classification report
    print('Classification_Report:',name)
    print()
    print(classification_report(y_test_p, y_pred))
    report = classification_report_imbalanced(y_test_p, y_pred,output_dict=True)
    
    df_recall = pd.DataFrame.from_dict(report).transpose()
    
    # Suppression des lignes accuracy,avg_ et total_support
    df_recall = df_recall[~df_recall.index.str.contains('accuracy|avg|total_support')]
    
    # Ne garder que la colonne 'recall'
    df_recall = df_recall[['rec']]
    
    df_recall= df_recall.rename(columns={'rec':name})

    

    ## PLOT
    plt.figure(figsize = (3, 3))
    ax = sns.barplot(df_recall,x=df_recall.index.values,y=name,hue=df_recall.index.values)
    plt.title(name + ' with default parameters : Recall')

    for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()/2),
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 0),
                    textcoords='offset points')
    
    plt.show()

    df_recall = df_recall.transpose()
    df_recall['Training_Time'] = training_time_seconds
    df_recall['Best_Params'] = "Default"

    #display(df_recall)
    return ( estmator ,df_recall)


def fit_grid_search(name:str,estmator,param_grid,cv,X_train_p,y_train_p,X_test_p,y_test_p):
    
    #global df_best_model_grid_search

    # Grid
    grid_search = GridSearchCV(estimator=estmator, param_grid=param_grid, scoring='recall_macro',cv=cv,n_jobs=-1)

    start_time = time.time()

    # FIT
    grid_search.fit(X_train_p, y_train_p)
    
    end_time = time.time()

    # Temps 
    training_time_seconds = end_time - start_time

    if training_time_seconds < 60:
        print(f"Temps d'entraînement : {training_time_seconds:.2f} seconds")
    else:
        # Convertir en minutes
        training_time = training_time_seconds / 60
        print(f"Temps d'entraînement : {training_time:.2f} minutes")
    
    #PREDICT
    y_pred = grid_search.best_estimator_.predict(X_test_p)

    
    # classification report
    print('Classification_Report:',name)
    print()
    print(classification_report(y_test_p, y_pred))
    report =classification_report_imbalanced(y_test_p, y_pred,output_dict=True)
    
    df_recall = pd.DataFrame.from_dict(report).transpose()
    
    # Suppression des lignes accuracy,avg_ et total_support
    df_recall = df_recall[~df_recall.index.str.contains('accuracy|avg|total_support')]
    
    # Ne garder que la colonne 'recall'
    df_recall = df_recall[['rec']]
    
    df_recall= df_recall.rename(columns={'rec':name})

    

    ## PLOT
    plt.figure(figsize = (3, 3))
    ax = sns.barplot(df_recall,x=df_recall.index.values,y=name,hue=df_recall.index.values)
    plt.title(name + ' with default parameters : Recall')

    for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()/2),
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 0),
                    textcoords='offset points')
    
    plt.show()

    df_recall = df_recall.transpose()
    df_recall['Training_Time'] = training_time_seconds
    df_recall['Best_Params'] = str(grid_search.best_params_)
    
    print("*** Best Param :",grid_search.best_params_,"***")

    display(df_recall)
    return ( grid_search.best_estimator_ ,df_recall)

def uac_roc(model,x_test,y_test):
    
    # Prédictions des probabilités sur le jeu de test
    y_pred_proba = model.predict_proba(x_test)
    
    # Binarisation des labels (nécessaire pour calculer la ROC courbe)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])  # Binarisation des labels pour les 4 classes
    
    # Nombre de classes
    n_classes = y_test_bin.shape[1]
    
    # Tracer la courbe ROC pour chaque classe
    plt.figure(figsize=(10, 8))
    
    # Couleurs pour chaque classe
    colors = ['green', 'yellow', 'orange', 'red']
    
    # Calculer la courbe ROC pour chaque classe
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Classe {i} (AUC = {roc_auc:.2f})')
    
    # Tracer la ligne diagonale (AUC = 0.5, modèle aléatoire)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    
    # Ajouter des labels et un titre
    plt.xlabel('Taux de faux positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de vrais positifs (TPR)', fontsize=12)
    plt.title('Courbes ROC pour une classification multi-classes', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    
    # Afficher la courbe ROC
    plt.grid(True)
    plt.show()
