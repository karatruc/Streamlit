
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
import os

path = os.path.dirname(os.path.realpath(__file__))
# repertoire (relatif) des ficheirs de données
data_path = '../../Data'
models_path = '{}/../Models'.format(path)

# récupération du modèle de clustering pour la géolocalisation
clusteringModel = load("{}/clustering_geoloc.joblib".format(models_path))
scaler = load("{}/scaler.joblib".format(models_path))
# récupération du transformer pour la nibarisation
transformer = load("{}/OneHotEncoderTransformer.joblib".format(models_path))

def replaceByNan(df) :
    df =  df.replace({'-1':np.nan,
                            -1:np.nan,
                            ' -1':np.nan,
                            '#ERREUR':np.nan
                        })
    return df

class RecodeMissingValues(BaseEstimator, TransformerMixin) :
    def __init__(self) :
        return None

    def fit(self, X, y = None) :
        return self

    def transform(self, X) :
        x = X.copy()
        return replaceByNan(x)

def addSecu(ds) :
    for i in range(1,10) :
        ds['secu_'+str(i)] = ((ds['secu1']==str(i)) | (ds['secu2']==str(i)) | (ds['secu3']==str(i)))*1
    ds = ds.drop(['secu1','secu2','secu3'], axis = 1)
    return ds

class RecodeSecu(BaseEstimator, TransformerMixin) :
    def __init__(self):
        return None
    
    def fit(self, X, y = None) :
        return self
    
    def transform(self, X) :
        #secu  : un champ binaire par équipement
        x = X.copy()
        return addSecu(x)

class DropColumns(BaseEstimator, TransformerMixin) :
    def __init__(self, columns_list):
        self.columns = columns_list
        return None
    def fit(self, X, y = None) :
        return self
    def transform(self, X) :
        x = X.copy()
        x = x.drop(self.columns, axis = 1)
        return x

class DropNa(BaseEstimator, TransformerMixin) :
    def __init__(self) :
        return None
    
    def fit(self, X, y = None) :
        return self
    
    def transform(self, X) :
        x= X.copy()
        x = x.dropna(how = 'any', axis=0)
        return x
    
class ConvertCoords(BaseEstimator, TransformerMixin) :
    def __init__(self) :
        return None
    
    def fit(self, X, y = None) :
        return self
    
    def transform(self, X) :
        x = X.copy()
        x['lat']= x['lat'].str.replace(',','.').astype('float')
        x['long']= x['long'].str.replace(',','.').astype('float')

        return x

def recodeDateTime(ds) :
    ds['hh'] = ds['hrmn'].str[:2].astype('int')
        
    #conversin mois
    ds['mois'] = ds['mois'].astype(int)

    #ajout d'une variable nbinaire weekend
    ds['weekend']=(pd.to_datetime(dict(day=ds['jour'], month=ds['mois'].astype(int), year=ds['an'])).dt.weekday>=5)*1

    #calcul de l'age de la victime dans l'année de l'accident
    ds['age'] = ds['an'].astype(int) - ds['an_nais'].astype(int)

    #suppression des variables inutiles
    ds = ds.drop(['jour','an', 'an_nais','hrmn'], axis = 1)

    return ds

class RecodeDatetime(BaseEstimator, TransformerMixin) :
    def __init__(self) :
        return None
    
    def fit(self, X, y = None) :
        return self
    
    def transform(self, X) :
        x = X.copy()
        
        x = recodeDateTime(x)

        return x

def addGeoloc(df, model) :
    df['geoloc'] = model.predict(df[['lat','long']])
    df = df.drop(['lat','long'], axis = 1) 
    return df                              


class Geolocalization(BaseEstimator, TransformerMixin) :
    def __init__(self, clustering_model) :
        self.clusteringModel = clustering_model
        return None
    
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        x= X.copy()
        x = addGeoloc(x, self.clusteringModel)
        return x


def classesAges(ds) :
    bins = [0,9,17,24,44,64,999]
    labels=[0,1,2,3,4,5]
    ds['classe_age'] = pd.cut(ds['age'], bins=bins, labels=labels, include_lowest=True)
    ds['classe_age'] = ds['classe_age'].astype(int)
    ds = ds.drop('age', axis = 1)
    return ds

class RecodeAges(BaseEstimator, TransformerMixin) :
    def __init__(self) :
        return None
    
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        x = X.copy()
        x = classesAges(x)
        return x

def binarisation(ds, cols, transformer) :

    ds[cols] = ds[cols].astype(int)
    
    #encoder = OneHotEncoder(sparse_output=False)
    #transformer = make_column_transformer((encoder, cols), remainder='passthrough', verbose_feature_names_out=False) 
    #transformed=transformer.fit_transform(ds)
    ds = pd.DataFrame(transformer.transform(ds), columns=transformer.get_feature_names_out(), index=ds.index)
    
    return ds


class Binarisation(BaseEstimator, TransformerMixin) :
    def __init__(self, cat_cols, transformer) :
        self.catCols = cat_cols
        self.transformer = transformer
        return None
    
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        x = X.copy()
        
        x = binarisation(x, self.catCols, transformer)

        return x

def scaling(ds, cols, scaler) :
    ds[cols] = scaler.transform(ds[cols])
    
    return ds

class Scaling(BaseEstimator, TransformerMixin) :
    def __init__(self, num_cols, scaler) :
        self.scaler = scaler
        self.numCols = num_cols
        return None
    
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        x = X.copy()
        x = scaling(x, self.numCols, self.scaler)
        return x

cat_cols=['mois', 'lum', 'agg', 'int', 'atm', 'col', 'catr', 'circ', 'vosp', 'prof', 'plan', 
          'surf', 'infra', 'situ', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor', 
          'place', 'catu', 'trajet','sexe','hh','weekend','geoloc','classe_age']
num_cols = ['nbv','vma']

pipe = Pipeline(
    steps = [
        ('Recodage Valeurs Manquantes', RecodeMissingValues()),
        ('Recodage des équipements de sécurité',RecodeSecu()),
        #('Suppression Colonnes',DropColumns(['adr', 'voie','v1', 'v2', 'lartpc', 'larrout', 'occutc', 'locp', 'actp', 'etatp', 'pr','pr1', 'senc', 'dep','com'])),
        ('Suppression des lignes avec données manquantes',DropNa()),
        #('Conversion des coordonnées géographiques', ConvertCoords()),
        ('Recodage des variables temporelles', RecodeDatetime()),
        ('Clusterisation des géolocalisation',Geolocalization(clusteringModel)),
        ('Tranches d\'ages',RecodeAges()),
        ('Binarisation des variables catégorielle',Binarisation(cat_cols, transformer)),
        ('Standardisation des variables numériques', Scaling(num_cols, scaler))
    ]
    
)
