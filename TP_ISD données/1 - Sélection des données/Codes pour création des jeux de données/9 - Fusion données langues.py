import pandas as pd 
import numpy as np 

lycee_df = pd.read_csv(r"1 - Sélection des données\Jeux créés\lycee_df", sep=";") #Créé avec 1 - Données Parcoursup 2021-2022
#Du dataframe Parcoursup, on garde seulement les variables à prédire et les données géographiques
lycee_df = lycee_df.filter(["code_uai", "fili", "acc_term", "acc_term_f", "acc_aca_orig", "latitude", "longitude"], axis=1)
lycee_df = pd.get_dummies(lycee_df, columns = ["fili"], drop_first=True)
langues_gt = pd.read_csv(r"1 - Sélection des données\Jeux créés\langues_gt", sep=";") #Créé avec 7 - Langues de chaque lycée gt
langues_pro = pd.read_csv(r"1 - Sélection des données\Jeux créés\langues_pro", sep=";") #Créé avec 8 - Langues de chaque lycée pro

#Rassemblement des données
lycee_l = lycee_df.merge(langues_gt, how="left", left_on="code_uai", right_on="numero_lycee") #MODIF !!!!!!!!!!!!!!!!!!!!
lycee_l = lycee_l.merge(langues_pro, how="left", left_on="code_uai", right_on="numero_lycee")
lycee_l = lycee_l.drop(["numero_lycee_x", "numero_lycee_y", "Unnamed: 0_x", "Unnamed: 0_y"], axis=1)

#On retire les lycées pour lesquels toutes les valeurs des langues sont nulles : on n'a pas d'informations pour ces lycées
lycee_test_null = lycee_l.drop(lycee_df.columns, axis=1)
index = lycee_test_null.index[lycee_test_null.isnull().all(1)]
lycee_l = lycee_l.drop(index, axis=0)

#Pour le reste, on remplace par des 0
for i in lycee_l.columns :
    lycee_l[i] = lycee_l[i].replace('nan', 0)
    lycee_l[i] = lycee_l[i].replace(np.NaN, 0)
lycee_l.info()
lycee_l.to_csv(r"1 - Sélection des données\Jeux créés\lycee_l", sep=";")