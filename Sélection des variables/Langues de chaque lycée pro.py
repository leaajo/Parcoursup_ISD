import pandas as pd 
langues = pd.read_csv('./all_df/fr-en-lycee_pro-effectifs-niveau-sexe-lv.csv', sep=";")
langues = langues[langues["rentree_scolaire"] == 2021]
langues['numero_lycee'] = langues['numero_lycee'].astype('string')
langues_pro = langues.drop(["region_academique", "academie", "commune", "denomination_principale", "patronyme", "secteur",
                        "departement", "rentree_scolaire"], axis=1)
langues_pro.to_csv('./Sélection des variables/df_inter/langues_pro', sep=";")