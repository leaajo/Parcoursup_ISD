import pandas as pd
import numpy as np

lycee_df2 = pd.read_csv('./Sélection des variables/df_inter/lycee_df2', sep=";")
val_aj_df2 = pd.read_csv('./all_df/fr-en-indicateurs-de-resultat-des-lycees-denseignement-general-et-technologique.csv', sep=";")
ips_df1 = pd.read_csv('.all_df/fr-en-ips_lycees.csv', sep=";")
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
Filiere = lycee_df4["formation"]
for i in range(0, len(Etablissement)) :
  indice = Etablissement[i] + ' ' + Filiere[i]
  indices.append(indice)
lycee_df4["Indices"] = indices
lycee_df4 = lycee_df4.set_index("Indices")

#On garde seulement les valeurs numériques
lycee_df5 = lycee_df4.copy()
lycee_df5 = lycee_df5.select_dtypes(exclude=[object])
lycee_df5 = lycee_df5.drop(["Unnamed: 0_x", "Unnamed: 0_y", "Unnamed: 0"], axis=1)
lycee_df5.info()
lycee_df5 = lycee_df5.to_csv('./Sélection des données/df_inter/lycee_df5', sep=";")

