import pandas as pd 
spe_premiere = pd.read_csv('./all_df/fr-en-effectifs-specialites-triplettes-1ere-generale.csv', sep=";")
spe_terminale = pd.read_csv('./all_df/fr-en-effectifs-specialites-doublettes-terminale-generale.csv', sep=";")
spe_premiere["numero_etablissement"] = spe_premiere["numero_etablissement"].astype("string")
spe_premiere = spe_premiere[spe_premiere["rentree_scolaire"] == 2021]
liste = ["rentree_scolaire", "region_academique", "academie", "departement", "commune", "denomination", "patronyme","secteur"]
for col in liste :
    spe_premiere.pop(col)

spe_terminale["numero_etablissement"] = spe_terminale["numero_etablissement"].astype("string")
spe_terminale = spe_terminale[spe_terminale["rentree_scolaire"] == 2021]
liste = ["rentree_scolaire", "region_academique", "academie", "departement", "commune", "denomination", "patronyme","secteur"]
for col in liste :
    spe_terminale.pop(col)


spe = spe_terminale.merge(spe_premiere, on="numero_etablissement")
spe.info()
spe = spe.to_csv("./Sélection des variables/df_inter/specialites", sep=";")