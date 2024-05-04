import pandas as pd 
import numpy as np 

lycee_df = pd.read_csv(r"1 - Sélection des données\Jeux créés\lycee_df", sep=";") #Créé avec 1 - Données Parcoursup 2021-2022
lycee_df = lycee_df.filter(["code_uai", "fili", "acc_term", "acc_term_f", "acc_aca_orig", "latitude", "longitude", "contrat_etab_Public"], axis=1)
lycee_df = pd.get_dummies(lycee_df, columns=["fili"], drop_first=True)
spe = pd.read_csv(r"1 - Sélection des données\Jeux créés\specialites", sep=";") #Créé avec 10 - Fusion données spécialités

lycee_s = lycee_df.merge(spe, left_on="code_uai", right_on="numero_etablissement")
lycee_s.pop("code_uai")
lycee_s.pop("numero_etablissement")
lycee_s.info()
lycee_s = lycee_s.to_csv(r"1 - Sélection des données\Jeux créés\lycee_s", sep=";")