# Importation des modules
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
    
def traitement_des_donnees(annee=int):
    
    # Importation des modules
    from math import pi
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px

    # Importation des données
    url = f'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_{annee}.csv'
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
    
    return lycees_avec_sup_2

print(traitement_des_donnees(2021))