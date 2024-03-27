import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme() # permet d'obtenir le fonc gris avec les lignes blanches

import geopandas as gpd 

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as confusion_matrix

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

sns.axes_style(style=None, rc=None);

url1 = "https://github.com/leaajo/TP_ISD/blob/master/all_df/fr-en-adresse-et-geolocalisation-etablissements-premier-et-second-degre.csv"
url2 = "https://github.com/leaajo/TP_ISD/all_df/fr-en-lycee_gt-effectifs-niveau-sexe-lv.csv"

tous_les_etablissements = pd.read_csv(r"https://github.com/leaajo/TP_ISD/blob/master/all_df/fr-en-adresse-et-geolocalisation-etablissements-premier-et-second-degre.csv",sep=";")
lycees = pd.read_csv(r"https://github.com/leaajo/TP_ISD/all_df/fr-en-lycee_gt-effectifs-niveau-sexe-lv.csv",sep=";")

lycees_2 = lycees.copy()
lycees_2 = lycees_2.merge(tous_les_etablissements, left_on="numero_lycee", right_on="numero_uai")

# Liste des colonnes à conserver
colonnes_a_garder = ['rentree_scolaire', 'denomination_principale_x', 'secteur', 'nombre_d_eleves', 'latitude', 'longitude', 'appariement', 'nature_uai_libe']

# Sélectionner les colonnes spécifiées dans lycees_2
lycees_2 = lycees_2.loc[:, colonnes_a_garder]

# On va passer la latitude et la longitude en geometry (pour les cartes)

# Conversion
geometry = gpd.points_from_xy(lycees_2["longitude"], lycees_2["latitude"])

# Create a DataFrame with a geometry containing the Points
geo_lycee = gpd.GeoDataFrame(
    lycees_2, crs="EPSG:4326", geometry=geometry
)

geo_lycee_metropole = geo_lycee[(geo_lycee['latitude'] > 40) &
                                (geo_lycee['longitude']> -10)]

url3 = "https://github.com/leaajo/TP_ISD/all_df/metropole.geojson"
donnees_france = gpd.read_file(url3)

# Créer une figure et des axes
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor('white')

# Tracer la carte de la France
donnees_france.plot(ax=ax, color='lightgrey', edgecolor = 'grey')

# Tracer les points des lycées sur la carte de la France
geo_lycee_metropole.plot(ax=ax, color='blue', markersize=8)

# Afficher la carte
plt.show()