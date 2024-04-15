import pandas as pd 
import matplotlib.pyplot as plt
import geoviews as gv
import geoviews.feature as gf
gv.extension('matplotlib', 'bokeh')
import geopandas as gpd

import holoviews as hv
from holoviews import dim, opts
hv.extension('bokeh')

from cartopy import crs

import traitement as trt

# Importation du fichier des départements (GeoDataFrame)
france_dep = gpd.read_file("./donnees_geo/departements-version-simplifiee.geojson")

# On ajoute un zéro devant le code du département pour pouvoir le joindre avec les données des lycées
france_dep["code_du_departement"] = "0" + france_dep["code"]
france_dep = france_dep.drop(columns=['code'])

# Importation des données de tous les lycées de France
tous_les_lycees = pd.read_csv('./all_df/fr-en-ips-lycees-ap2022.csv', sep=';')

# Merge du GeoDataFrame et du DataFrame
taux_res_dep0 = france_dep.merge(tous_les_lycees, on='code_du_departement', suffixes=('','_y'))

# Importation des données des résultats des lycées supérieurs
lycees_resultats = pd.read_csv('./all_df/fr-en-indicateurs-de-resultat-des-lycees-denseignement-general-et-technologique.csv', sep=';')
# On merge les données des lycées avec les données des départements
taux_res_dep = taux_res_dep0.merge(lycees_resultats, left_on='uai', right_on='code_etablissement', suffixes=('','_y'))
taux_res_dep_2023 = taux_res_dep[taux_res_dep['annee'] == 2023]

# On garde les colonnes qui nous intéressent
colonnes_a_garder = ['nom', 'geometry', 'code_du_departement', 'taux_reu_brut_gnle', 'taux_men_brut_gnle']
taux_res_dep_2023 = taux_res_dep_2023.loc[:, colonnes_a_garder]

# On calcule le taux de réussite brut moyen par département 
carte_res2023 = taux_res_dep_2023.groupby(['nom', 'geometry', 'code_du_departement']).mean()
carte_res2023 = carte_res2023.reset_index()

carte_res2023 = france_dep.merge(carte_res2023, on='code_du_departement', suffixes=('','_y'))
carte_res2023 = carte_res2023.drop(columns = ['nom_y', 'geometry_y'])

# TAUX REUSSITE BRUT BAC SESSION 2023 PAR DEPARTEMENT :
regions = gv.Polygons(carte_res2023, vdims=['nom', 'code_du_departement', 'taux_reu_brut_gnle', 'taux_men_brut_gnle'])
regions.opts(width=600, height=600, toolbar='above', color=dim('taux_reu_brut_gnle'), colorbar=True, tools=['hover'], aspect='equal')
regions.opts(title='Taux de réussite brut au bac session 2023 par département')

from bokeh.plotting import show
# Convertir GeoViews en plot Bokeh
#plot = gv.render(regions, backend='bokeh')

#show(plot)

# Nombre d'élèves venant de la meme academie (2023):
lycees_resultats2 = pd.read_csv('./all_df/fr-esr-parcoursup_2023.csv', sep=';')
nb_mm_acc0 = france_dep.merge(tous_les_lycees, on='code_du_departement', suffixes=('','_y'))

nb_mm_acc = nb_mm_acc0.merge(lycees_resultats2, left_on='uai', right_on='cod_uai', suffixes=('','_y'))


nb_mm_acc_2023 = taux_res_dep[taux_res_dep['session'] == 2023]

cg = ['nom', 'geometry', 'code_du_departement', 'departement', 'academie', 'acc_aca_orig', 'acc_term']
nb_mm_acc_2023 = nb_mm_acc_2023.loc[:, cg]

carte_acc2023 = nb_mm_acc_2023.groupby(['nom', 'geometry', 'code_du_departement', 'departement', 'academie']).mean()
carte_acc2023 = carte_acc2023.reset_index()

carte_acc2023 = france_dep.merge(carte_acc2023, on='code_du_departement', suffixes=('','_y'))
carte_acc2023 = carte_acc2023.drop(columns = ['nom_y', 'geometry_y'])


regions_2 = gv.Polygons(carte_acc2023, vdims=['nom', 'code_du_departement', 'departement', 'academie', 'acc_aca_orig', 'acc_term'])
regions_2.opts(width=600, height=600, toolbar='above', color=dim('acc_aca_orig'), colorbar=True, tools=['hover'], aspect='equal')

plot = gv.render(regions_2, backend='bokeh')    
show(plot)