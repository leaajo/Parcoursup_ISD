import pandas as pd
import numpy as np

lycee_df2 = pd.read_csv(r'1 - Sélection des données\Jeux créés\lycee_df', sep=";") #Créé avec 1 - Données Parcoursup 2021-2022
val_aj_df2 = pd.read_csv(r"1 - Sélection des données\Jeux créés\val_aj", sep=";") #Créé avec 4 - Valeur ajoutée ensemble
ips_df1 = pd.read_csv(r'1 - Sélection des données\Jeux créés\ips_df', sep=";") #Créé avec 5 - Indice de position sociale
lycee_df3 = lycee_df2.copy()
lycee_df3 = lycee_df3.merge(val_aj_df2, left_on="code_uai", right_on="code_etablissement")

lycee_df4 = lycee_df3.copy()
lycee_df4 = lycee_df4.merge(ips_df1, left_on="code_uai", right_on="UAI")

#Classification des filières
lycee_df4 = pd.get_dummies(lycee_df4, columns = ["fili"], drop_first=True)
lycee_df4.info()
#Création des indices indiquant Etablissement et formation
indices = []
Etablissement = lycee_df4["g_ea_lib_vx"]
Filiere = lycee_df4["lib_comp_voe_ins"]
for i in range(0, len(Etablissement)) :
  indice = Etablissement[i] + ' ' + Filiere[i]
  indices.append(indice)
lycee_df4["Indices"] = indices
lycee_df4 = lycee_df4.set_index("Indices")

#On retire les colonnes créées dans la fusion et les colonnes concernant les titres de la formation
lycee_df4["code_uai"] = lycee_df4["code_uai"].astype("string")
lycee_df4 = lycee_df4.select_dtypes(exclude=[object]) #MODIFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
lycee_df4 = lycee_df4.drop(["Unnamed: 0_x", "Unnamed: 0_y", "Unnamed: 0", "session"], axis=1)
lycee_df4.info()
lycee_df4.to_csv(r"1 - Sélection des données\Jeux créés\lycee_df2", sep=";")

