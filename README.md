# Projet gravité d'accident

Projet de modélisation de la gravité des accidents de la route

## Auteurs du projet
* Eric FAVRE
* Vitor BIOULAC
* Thierno SIDIBE

## Installer les dépendances

```shell
# Création d'un environnement virtuel
python3 -m venv .venv
# Activation de l'environement virtuel
source .venv/bin/activate
# Installation des dépendances
pip install -r requirements.txt
```

## Lancer Jupyter
```shell
jupyter notebook
```

## Désactiver l'environement virtuel
```shell
deactivate
```

## Arborescence du projet

    ├── Data                     			<-- Données CSV
    ├── Models                  			<-- Modèles
    ├── Notebooks                			<-- Ensemble de notebooks
    │   ├── Data_Exploration     			<-- Notebooks d'exploration et de visualisation des données
    │   ├── MachineLearning      			<-- Notebooks dédiés au machine learning
    │   │   ├── 1_ML_4_classes   			<-- Modélisation du machine learning sur quatre classes
    │   │   │   ├── 1_train_basics_models    	<-- Entraînement des modèles de base sur quatre classes
    │   │   │   ├── 2_grid_search_best_model 	<-- Recherche d'hyperparamètres des modèles sur quatre classes
    │   │   │   └── 3_best_model_interpretation   <-- Interprétation des meilleurs modèles sur quatre classes
    │   │   ├── 2_ML_2_classes   			<-- Modélisation du machine learning sur deux classes
    │   │   │   ├── 1_train_basics_models    	<-- Entraînement des modèles de base sur deux classes
    │   │   │   ├── 2_grid_search_best_model 	<-- Recherche d'hyperparamètres des modèles sur deux classes
    │   │   │   └── 3_best_model_interpretation 	<-- Interprétation des meilleurs modèles sur deux classes
    │   │   └── python_scripts    			<-- Scripts Python
    │   └── Preprocessing        			<-- Notebooks de prétraitement
    ├── README.md
    └── requirements.txt         			<-- Librairies et dépendances liées au projet

