import pandas as pd 
spe_premiere = pd.read_csv(r"1 - Sélection des données\Jeux de données\fr-en-effectifs-specialites-triplettes-1ere-generale.csv", sep=";")
spe_terminale = pd.read_csv(r"1 - Sélection des données\Jeux de données\fr-en-effectifs-specialites-doublettes-terminale-generale.csv", sep=";")
spe_premiere["numero_etablissement"] = spe_premiere["numero_etablissement"].astype("string")
#Garder les informations pour 2021-2022
spe_premiere = spe_premiere[spe_premiere["rentree_scolaire"] == 2021]
#Retirer les variables déjà étudiées (avec le jeu de données Parcoursup) ou inutilisables
liste = ["rentree_scolaire", "region_academique", "academie", "departement", "commune", "denomination", "patronyme","secteur"]
for col in liste :
    spe_premiere.pop(col)

#Mêmes manipulations
spe_terminale["numero_etablissement"] = spe_terminale["numero_etablissement"].astype("string")
spe_terminale = spe_terminale[spe_terminale["rentree_scolaire"] == 2021]
liste = ["rentree_scolaire", "region_academique", "academie", "departement", "commune", "denomination", "patronyme","secteur"]
for col in liste :
    spe_terminale.pop(col)

#Fusion des spécialités pour première et terminale
spe = spe_terminale.merge(spe_premiere, on="numero_etablissement")
spe.info()
spe = spe.to_csv("1 - Sélection des données\Jeux créés\specialites", sep=";")