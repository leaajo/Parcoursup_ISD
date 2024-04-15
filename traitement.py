# Importation des modules
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
    
def traitement_des_donnees(annee=int):

    # Importation des données
    url = f'./all_df/fr-esr-parcoursup_{annee}.csv'
    parcoursup = pd.read_csv(url, sep=';')
    
    # On passe certaines variables en string pour mieux les manipuler
    for col in ["cod_uai",                          # code de l'établissement
                "g_ea_lib_vx",                      # Nom de l'établissement
                "dep",                              # Département numéro
                "dep_lib",                          # Nom du département
                "region_etab_aff",                  # Région de l'établissement
                "acad_mies",                        # Académie
                "ville_etab",                       # Ville de l'établissement  
                "lib_for_voe_ins",                  # Filière de formation (Domaine + type :BTS, DUT, CPGE, ...+ Bac demandé)
                "select_form",                      # Sélectivité ou non
                "fili",                             # Filière de formation très agrégée (Ecole d'ingénieur, BUT, DUT...)
                "lib_comp_voe_ins",                 # Filière de formation détaillée bis (Exemple : Concours Puissance Alpha - Formation d'ingénieur Bac + 5 - Bacs généraux - 2 Sciences)
                "form_lib_voe_acc",                 # Domaine de formation (Exemple : BTS - Agricole, Formation des écoles d'ingénieurs,...)
                "fil_lib_voe_acc",                  # Filière de formation très détaillée (Exemple : Métiers de l'audio-visuel opt : métiers de l'image)
                "detail_forma",                     # Tri : Bac demandé
                "g_olocalisation_des_formations"    # Localisation : latitude et longitude séparés par une virgule
                ]: 
        parcoursup[col] = parcoursup[col].astype('string')  # On passe les variables en string
    #
    #
    #    
    # On sélectionne que les lycées maitenant :
    #
    #
    #
    lycees_avec_sup = parcoursup.copy()
    i = 0
    retirer = []
    for s in parcoursup["g_ea_lib_vx"] :
        if "Lycée" not in s :
             retirer.append(i)
        i+=1
    lycees_avec_sup = lycees_avec_sup.drop(retirer, axis=0)
    lycees_avec_sup = lycees_avec_sup.reset_index(drop=True)
    #
    #
    # On crée des données géométriques à partir de la variable "g_olocalisation_des_formations"
    #
    #
    #
    lycees_avec_sup_2 = lycees_avec_sup.copy()
    latitude = []
    longitude = []

    for val in lycees_avec_sup["g_olocalisation_des_formations"]:
       if pd.notna(val):  # Vérifier si la valeur n'est pas manquante
            try:
                lat, lon = val.split(',')
                latitude.append(float(lat.strip()))  # Convertir en float et extraire la latitude
                longitude.append(float(lon.strip()))  # Convertir en float et extraire la longitude
            except ValueError:
                latitude.append(None)  # Ajouter None si la valeur ne peut pas être divisée
                longitude.append(None)  # Ajouter None si la valeur ne peut pas être divisée
       else:
            latitude.append(None)  # Ajouter None si la valeur est manquante
            longitude.append(None)  # Ajouter None si la valeur est manquante
            
    lycees_avec_sup_2["latitude"] = latitude
    lycees_avec_sup_2["longitude"] = longitude

    lycees_avec_sup_2 = lycees_avec_sup_2.drop(columns = ['g_olocalisation_des_formations', 'dep', 'dep_lib', 'region_etab_aff', 'ville_etab'])
    #
    #
    #
    # On traite l'IPS avant de l'ajouter au dataframe
    #
    #
    #
    val_aj_df=pd.read_csv('./all_df/fr-en-indicateurs-de-resultat-des-lycees-denseignement-general-et-technologique.csv', sep=";", low_memory=False)
    val_aj_df = val_aj_df[val_aj_df["annee"] == annee] # On prend que les données de l'année choisie

    liste_val = ['code_etablissement', 'taux_brut_de_reussite_total_series', 'taux_reussite_attendu_acad_total_series', 'taux_reussite_attendu_france_total_series',
         'taux_mention_attendu_toutes_series', 'pourcentage_bacheliers_sortants_2de_1re_term_etab', 'pourcentage_bacheliers_sortants_terminales_etab',
         'pourcentage_bacheliers_sortants_2de_1re_term_acad', 'pourcentage_bacheliers_sortants_terminales_acad', 'pourcentage_bacheliers_sortants_2de_1re_term_france',
         'pourcentage_bacheliers_sortants_terminales_france', 'taux_acces_brut_seconde_bac', 'taux_acces_attendu_acad_seconde_bac', 'taux_acces_attendu_france_seconde_bac',
         'taux_acces_brut_premiere_bac', 'taux_acces_attendu_acad_premiere_bac', 'taux_acces_attendu_france_premiere_bac', 'taux_acces_brut_terminale_bac',
         'taux_acces_attendu_france_terminale_bac', 'va_reu_total', 'va_acc_seconde', 'va_men_total', 'presents_gnle', 'taux_reu_brut_gnle', 'va_reu_gnle',
         'taux_men_brut_gnle', 'va_men_gnle', 'nombre_de_mentions_tb_avec_felicitations_g', 'nombre_de_mentions_tb_sans_felicitations_g', 'nombre_de_mentions_b_g',
         'nombre_de_mentions_ab_g', 'nombre_de_mentions_tb_avec_felicitations_t', 'nombre_de_mentions_tb_sans_felicitations_t', 'nombre_de_mentions_b_t',
         'nombre_de_mentions_ab_t']
    val_aj_df = val_aj_df.filter(liste_val, axis=1)
    
    # On renomme les colonnes pour les fusionner puis on les supprime
    val_aj_df['nombre_de_mentions_tb_g'] = val_aj_df['nombre_de_mentions_tb_avec_felicitations_g'] + val_aj_df['nombre_de_mentions_tb_sans_felicitations_g']
    val_aj_df['nombre_de_mentions_tb_t'] = val_aj_df['nombre_de_mentions_tb_avec_felicitations_t'] + val_aj_df['nombre_de_mentions_tb_sans_felicitations_t']
    val_aj_df.pop('nombre_de_mentions_tb_avec_felicitations_g')
    val_aj_df.pop('nombre_de_mentions_tb_sans_felicitations_g')
    val_aj_df.pop('nombre_de_mentions_tb_avec_felicitations_t')
    val_aj_df.pop('nombre_de_mentions_tb_sans_felicitations_t')
    

    # On supprime les variables vides (donc inutiles)
    null = val_aj_df.isnull()
    val_aj_df1 = val_aj_df.copy()
    for col in val_aj_df.columns :  
        if list(set(list(null[col]))) == [True] :
             val_aj_df1.pop(col)

    # On change le type du code établissement car on va fusionner les dataframes
    val_aj_df1['code_etablissement'] = val_aj_df1['code_etablissement'].astype('string')
    val_aj_df1 = val_aj_df1.dropna(axis=0, subset="presents_gnle")

    # Pour éviter les bugs, on remplace les valeurs manquantes par des 0.0 ou des NaN et on passe en float
    liste = ["taux_acces_brut_seconde_bac", "va_reu_total", "va_acc_seconde", "va_men_total", "presents_gnle", "va_reu_gnle", "va_men_gnle"]
    val_aj_df2 = val_aj_df1.copy()
    for var in liste :
         val_aj_df2[var] = val_aj_df2[var].replace('.', '0.0')
         val_aj_df2[var] = val_aj_df2[var].replace('ND', np.NaN)
    val_aj_df2 = val_aj_df2.dropna(axis=0, subset = liste)

    for var in liste :
        val_aj_df2[var] = val_aj_df2[var].astype('float')

    # On fusionne les dataframes
    lycees_avec_sup_3 = lycees_avec_sup_2.merge(val_aj_df2, left_on='cod_uai', right_on='code_etablissement', how='left')
    #
    #
    #
    # On traite l'IPS avant de l'ajouter au dataframe
    #
    #
    #
    if annee == 2021 :
        ips = pd.read_csv('./all_df/fr-en-ips_lycees.csv', sep=";")
        ips_df = ips[ips["Rentrée scolaire"] == '2020-2021'] # On prend que les données de l'année choisie
        liste_ips = ['UAI', 'IPS voie GT', 'IPS voie PRO', 'IPS Ensemble GT-PRO', 'Ecart-type de l\'IPS voie GT', 'Ecart-type de l\'IPS voie PRO']
        nouveau_noms = {'UAI': 'uai', 'IPS voie GT': 'ips_voie_gt', 'IPS voie PRO': 'ips_voie_pro', 'IPS Ensemble GT-PRO': 'ips_ensemble_gt_pro', "Ecart-type de l'IPS voie GT": 'ecart_type_ips_voie_gt', "Ecart-type de l'IPS voie PRO": 'ecart_type_ips_voie_pro'}
        ips_df1 = ips_df.filter(liste_ips, axis=1)
        ips_df1["UAI"] = ips_df1["UAI"].astype("string")
        # On fusionne les dataframes pour avoir les IPS :
        lycees_avec_sup_4 = lycees_avec_sup_3.copy()
        lycees_avec_sup_4 = lycees_avec_sup_4.merge(ips_df1, left_on="cod_uai", right_on="UAI")
        lycees_avec_sup_4.rename(columns=nouveau_noms, inplace=True)
    
    if annee == 2023 :
        ips = pd.read_csv('./all_df/fr-en-ips-lycees-ap2022.csv', sep=";")
        ips_df = ips[ips["rentree_scolaire"] == '2022-2023']
        liste_ips = ['uai', 'ips_voie_gt', 'ips_voie_pro', 'ips_ensemble_gt_pro', 'ecart_type_ips_voie_gt', 'ecart_type_ips_voie_pro']
        ips_df1 = ips_df.filter(liste_ips, axis=1)
        ips_df1["uai"] = ips_df1["uai"].astype("string")
        lycees_avec_sup_4 = lycees_avec_sup_3.copy()
        lycees_avec_sup_4 = lycees_avec_sup_4.merge(ips_df1, left_on="cod_uai", right_on="uai")


    return lycees_avec_sup_4

print(traitement_des_donnees(2021))