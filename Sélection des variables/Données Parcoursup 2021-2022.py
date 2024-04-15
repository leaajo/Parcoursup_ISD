import pandas as pd 
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
lycee_df = pd.read_csv('./all_df/fr-esr-parcoursup_2022.csv', sep=";")

liste = ["Session", "Statut de l’établissement de la filière de formation (public, privé…)", "Code UAI de l'établissement", "Établissement", "Département de l’établissement", "Région de l’établissement",
                         "Académie de l’établissement", "Commune de l’établissement", "Filière de formation", "Sélectivité",
                         "Filière de formation très agrégée", "Filière de formation", "Filière de formation détaillée bis", "Filière de formation très détaillée", "tri",
                         "Coordonnées GPS de la formation", "Capacité de l’établissement par formation", "Effectif total des candidats pour une formation", "Effectif total des candidats en phase principale",
                         "Effectif des candidats néo bacheliers généraux en phase principale", "Effectif des candidats néo bacheliers technologiques en phase principale", "Effectif des candidats néo bacheliers professionnels en phase principale",
                         "Effectif total des candidats en phase complémentaire", "Effectif des candidats néo bacheliers généraux en phase complémentaire", "Effectif des candidats néo bacheliers technologique en phase complémentaire",
                         "Effectif total des candidats ayant reçu une proposition d’admission de la part de l’établissement",
                         "Effectif total des candidats ayant accepté la proposition de l’établissement (admis)", "Dont effectif des candidates admises", "Effectif des admis en phase principale", "Effectif des admis en phase complémentaire", "Dont effectif des admis en internat", "Dont effectif des admis boursiers néo bacheliers",
                         "Effectif des admis néo bacheliers généraux", "Effectif des admis néo bacheliers technologiques", "Effectif des admis néo bacheliers professionnels", "Dont effectif des admis néo bacheliers sans mention au bac", "Dont effectif des admis néo bacheliers avec mention Assez Bien au bac", "Dont effectif des admis néo bacheliers avec mention Bien au bac",
                         "Dont effectif des admis néo bacheliers avec mention Très Bien au bac", "Dont effectif des admis néo bacheliers avec mention Très Bien avec félicitations au bac", "Effectif des admis néo bacheliers généraux ayant eu une mention au bac", "Effectif des admis néo bacheliers technologiques ayant eu une mention au bac", "Effectif des admis néo bacheliers professionnels ayant eu une mention au bac",
                         "Dont effectif des admis issus du même établissement (BTS/CPGE)", "Dont effectif des admises issues du même établissement (BTS/CPGE)", "Dont effectif des admis issus de la même académie", "Dont effectif des admis issus de la même académie (Paris/Créteil/Versailles réunies)"]
lycee_df = lycee_df.filter(liste, axis=1)

for col in ["Statut de l’établissement de la filière de formation (public, privé…)", "Code UAI de l'établissement", "Établissement", "Département de l’établissement", "Région de l’établissement",
                         "Académie de l’établissement", "Commune de l’établissement", "Filière de formation", "Sélectivité",
                         "Filière de formation très agrégée", "Filière de formation détaillée bis", "Filière de formation très détaillée", "tri",
                         "Coordonnées GPS de la formation"]:
    lycee_df[col] = lycee_df[col].astype('string')

#Renommer les variables
changer = {"Session" : "session", "Statut de l’établissement de la filière de formation (public, privé…)" : "contrat_etab", "Code UAI de l'établissement" : "code_uai", "Établissement" : "g_ea_lib_vx","Département de l’établissement" : "dep_lib", "Région de l’établissement" : "region_etab_aff", 
           "Académie de l’établissement" : "acad_mies", "Commune de l’établissement" : "ville_etab", "Filière de formation" : "lib_for_voe_ins", "Sélectivité" : "select_form", 
           "Filière de formation très agrégée" : "fili", "Filière de formation" : "lib_for_voe_ins", "Filière de formation détaillée bis" : "lib_comp_voe_ins", "Filière de formation très détaillée" : "detail_forma", "tri" : "tri", 
           "Coordonnées GPS de la formation" : "g_olocalisation_des_formations", "Capacité de l’établissement par formation" : "capa_fin", "Effectif total des candidats pour une formation" : "voe_tot", "Effectif total des candidats en phase principale" : "nb_voe_pp", 
           "Effectif des candidats néo bacheliers généraux en phase principale" : "nb_voe_pp_bg", "Effectif des candidats néo bacheliers technologiques en phase principale" : "nb_voe_pp_bt", "Effectif des candidats néo bacheliers professionnels en phase principale" : "nb_voe_pp_bp",
           "Effectif total des candidats en phase complémentaire" : "nb_voe_pc", "Effectif des candidats néo bacheliers généraux en phase complémentaire" : "nb_voe_pc_bg", "Effectif des candidats néo bacheliers technologique en phase complémentaire" : "nb_voe_pc_bt", "Effectif total des candidats ayant reçu une proposition d’admission de la part de l’établissement" : "prop_tot", 
           "Effectif total des candidats ayant accepté la proposition de l’établissement (admis)" : "acc_tot", "Dont effectif des candidates admises" : "acc_tot_f", "Effectif des admis en phase principale" : "acc_pp", "Effectif des admis en phase complémentaire" : "acc_pc", "Dont effectif des admis en internat" : "acc_internat", "Dont effectif des admis boursiers néo bacheliers" : "acc_brs", 
           "Effectif des admis néo bacheliers généraux" : "acc_bg", "Effectif des admis néo bacheliers technologiques" : "acc_bt", "Effectif des admis néo bacheliers professionnels" : "acc_bp",  "Dont effectif des admis néo bacheliers sans mention au bac" : "acc_sansmention", "Dont effectif des admis néo bacheliers avec mention Assez Bien au bac" : "acc_ab", "Dont effectif des admis néo bacheliers avec mention Bien au bac" : "acc_b", 
           "Dont effectif des admis néo bacheliers avec mention Très Bien au bac" : "acc_tb", "Dont effectif des admis néo bacheliers avec mention Très Bien avec félicitations au bac" : "acc_tbf", "Effectif des admis néo bacheliers généraux ayant eu une mention au bac" : "acc_bg_mention", "Effectif des admis néo bacheliers technologiques ayant eu une mention au bac" : "acc_bt_mention", "Effectif des admis néo bacheliers professionnels ayant eu une mention au bac" : "acc_bp_mention", 
           "Dont effectif des admis issus du même établissement (BTS/CPGE)" : "acc_term", "Dont effectif des admises issues du même établissement (BTS/CPGE)" : "acc_term_f", "Dont effectif des admis issus de la même académie" : "acc_aca_orig", "Dont effectif des admis issus de la même académie (Paris/Créteil/Versailles réunies)" : "acc_aca_orig_idf"}
lycee_df = lycee_df.rename(columns = changer)

#Sélection des lycées
lycee_df1 = lycee_df.copy()
i = 0
retirer = []
for s in lycee_df["g_ea_lib_vx"] :
  if "Lycée" not in s :
    retirer.append(i)
  i+=1
lycee_df1 = lycee_df1.drop(retirer, axis=0)
lycee_df1=lycee_df1.reset_index(drop=True)

#On garde seulement CPGE, BTS, Ecole d'ingénieur, EFTS
autre_form = []
i=0
for s in lycee_df1["fili"] :
  if s == "Autre formation" :
    autre_form.append(i)
  i+=1
lycee_df1 = lycee_df1.drop(autre_form, axis=0)


#Garder juste la formation
lycee_df1["formation"] = lycee_df1.iloc[:,8][2]
lycee_df1["formation"] = lycee_df1["formation"].astype('string')
lycee_df1.pop("lib_for_voe_ins")


#Création des variables géographiques
lycee_df2 = lycee_df1.copy()
latitude = []
longitude = []
for val in lycee_df2["g_olocalisation_des_formations"] :
  virgule = val.find(',')
  latitude.append(val[0:virgule])
  longitude.append(val[virgule+1:])
lycee_df2["latitude"] = latitude
lycee_df2["longitude"] = longitude
lycee_df2["latitude"] = lycee_df2["latitude"].astype("float")
lycee_df2["longitude"] = lycee_df2["longitude"].astype("float")

#Garder seulement l'académie, pas la région
lycee_df2.pop("g_olocalisation_des_formations")
lycee_df2.pop("ville_etab")
lycee_df2.pop("region_etab_aff")
lycee_df2.pop("dep_lib")
lycee_df2 = pd.get_dummies(lycee_df2, columns=["contrat_etab", "acad_mies"], drop_first=True)

lycee_df2 = lycee_df2.to_csv('./Sélection des variables/df_inter/lycee_df2', sep=";")