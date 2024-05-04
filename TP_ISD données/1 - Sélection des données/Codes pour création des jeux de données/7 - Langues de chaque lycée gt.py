import pandas as pd 
langues = pd.read_csv(r"1 - Sélection des données\Jeux de données\fr-en-lycee_gt-effectifs-niveau-sexe-lv.csv", sep=";")
langues = langues[langues["rentree_scolaire"] == 2021]
langues['numero_lycee'] = langues['numero_lycee'].astype('string')
langues_gt = langues.drop(["region_academique", "academie", "commune", "denomination_principale", "patronyme", "secteur",
                        "departement", "rentree_scolaire"], axis=1)
langues_gt.to_csv(r"1 - Sélection des données\Jeux créés\langues_gt", sep=";")