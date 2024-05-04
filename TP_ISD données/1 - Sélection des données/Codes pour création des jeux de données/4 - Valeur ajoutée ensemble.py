import pandas as pd 
import numpy as np 
val_aj_gnle_tech = pd.read_csv(r"1 - Sélection des données\Jeux créés\val_aj_gnle", sep=";")
val_aj_pro = pd.read_csv(r"1 - Sélection des données\Jeux créés\val_aj_pro", sep=";")
val_aj = val_aj_gnle_tech.merge(val_aj_pro, how="outer", left_on="code_etablissement", right_on="code_etablissement")
val_aj = val_aj.drop(["Unnamed: 0_x", "Unnamed: 0_y"], axis=1)
for i in val_aj.columns :
    val_aj[i] = val_aj[i].replace('nan', 0)
    val_aj[i] = val_aj[i].replace(np.NaN, 0)
val_aj["code_etablissement"] = val_aj["code_etablissement"].astype("string")
val_aj.info()
val_aj.to_csv(r"1 - Sélection des données\Jeux créés\val_aj", sep=";")
