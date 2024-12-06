import joblib
import os


def save(model,file_name:str):
    
    # Supprimer le fichier s'il existe
    if os.path.exists(file_name):
        os.remove(file_name)
    
    with open(file_name , 'wb') as file:
        joblib.dump(model,file, compress=("bz2", 3))


def load(file_name:str):
    
    with open(file_name , 'rb') as file:
        model = joblib.load(file)

    return model

