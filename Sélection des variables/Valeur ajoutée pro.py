import pandas as pd 
import numpy as np
val_aj_pro = pd.read_csv(r"Sélection des données\Jeux de données\fr-en-indicateurs-de-resultat-des-lycees-denseignement-professionnels.csv", sep=";")
val_aj_pro = val_aj_pro[val_aj_pro["annee"]==2021]
var = ["code_etablissement", "taux_brut_de_reussite_total_secteurs","taux_acces_brut_seconde_bac", 
       "taux_acces_brut_premiere_bac", "taux_acces_brut_terminale_bac", "va_reu_total", "va_acc_seconde",  "va_acc_premiere", 
       "va_acc_terminale", "va_men_total", "effectifs_presents_total_secteurs", "taux_brut_de_mentions_total_secteurs"]

val_aj_pro = val_aj_pro.filter(var, axis=1)
val_aj_pro["code_etablissement"] = val_aj_pro["code_etablissement"].astype("string")

val_aj_pro["va_acc_premiere"] = val_aj_pro["va_acc_premiere"].replace('.', '0.0')
val_aj_pro["va_acc_terminale"] = val_aj_pro["va_acc_terminale"].replace('.', '0.0')

val_aj_pro["va_acc_premiere"] = val_aj_pro["va_acc_premiere"].astype("float")
val_aj_pro["va_acc_terminale"] = val_aj_pro["va_acc_terminale"].astype("float")
val_aj_pro["lycee_pro"] = [1]*val_aj_pro.shape[0]
val_aj_pro = val_aj_pro.rename({"taux_brut_de_reussite_total_secteurs" : "taux_brut_de_reussite_total_secteurs_p",
                                "taux_acces_brut_seconde_bac" : "taux_acces_brut_seconde_bac_p", 
       "taux_acces_brut_premiere_bac" : "taux_acces_brut_premiere_bac_p", 
       "taux_acces_brut_terminale_bac" : "taux_acces_brut_terminale_bac_p", 
       "va_reu_total" : "va_reu_total_p", "va_acc_seconde" : "va_acc_seconde_p",  
       "va_acc_premiere" : "va_acc_premiere_p", 
       "va_acc_terminale" : "va_acc_terminale_p", "va_men_total" : "va_men_total_p", 
       "effectifs_presents_total_secteurs" : "effectifs_presents_total_secteurs_p", 
       "taux_brut_de_mentions_total_secteurs" : "taux_brut_de_mentions_total_secteurs_p"}, axis='columns')
val_aj_pro = val_aj_pro.dropna(axis=0)
val_aj_pro.info()
val_aj_pro.to_csv(r"Sélection des données\val_aj_pro", sep=";")