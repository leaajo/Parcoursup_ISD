import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme() # permet d'obtenir le fonc gris avec les lignes blanches

import geopandas as gpd 

sns.axes_style(style=None, rc=None);

url1 = "https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-en-adresse-et-geolocalisation-etablissements-premier-et-second-degre.csv"
url2 = "https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-en-lycee_gt-effectifs-niveau-sexe-lv.csv"

tous_les_etablissements = pd.read_csv(url1,sep=";")
lycees = pd.read_csv(url2,sep=";")

# On va fusionner les deux dataframes pour avoir les informations sur les lycées
lycees_2 = lycees.copy()
lycees_2 = lycees_2.merge(tous_les_etablissements, left_on="numero_lycee", right_on="numero_uai")
# Liste des colonnes à conserver
colonnes_a_garder = ['rentree_scolaire', 'denomination_principale_x', 'secteur', 'nombre_d_eleves', 'latitude', 'longitude', 'appariement', 'nature_uai_libe', 'numero_uai', 'code_departement']
lycees_2 = lycees_2.loc[:, colonnes_a_garder]


# On va passer la latitude et la longitude en geometry (pour les cartes)
# Conversion
geometry = gpd.points_from_xy(lycees_2["longitude"], lycees_2["latitude"])

# Création du GeoDataFrame
geo_lycee = gpd.GeoDataFrame(
    lycees_2, crs="EPSG:4326", geometry=geometry
)

# On garde que les données en France métropolitaine
geo_lycee_metropole = geo_lycee[(geo_lycee['latitude'] > 40) &
                                (geo_lycee['longitude']> -10)]

# On va charger les données géographiques de la France
url3 = "https://raw.githubusercontent.com/leaajo/TP_ISD/master/donnees_geo/metropole.geojson"
donnees_france = gpd.read_file(url3)

# Créer une figure et des axes
fig, ax = plt.subplots(figsize=(10, 10))
# Tracer la carte de la France
donnees_france.plot(ax=ax, color='lightgrey', edgecolor = 'grey')
# Tracer les points des lycées sur la carte de la France
geo_lycee_metropole.plot(ax=ax, color='blue', markersize=8)

# Ajouter le titre
ax.set_title("Répartition des lycées en France métropolitaine")

plt.show()


#
#
# Sauvegarder le DataFrame lycee_2 dans un fichier CSV
lycees_2.to_csv('lycees_2.csv', index=False)
#
#
#