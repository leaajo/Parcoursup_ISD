import pandas as pd

# Avec la réforme du baccalauréat, les données de Parcoursup ont été modifiées depuis 2021. Nous allons donc concaténer les données de 2021,2022 et 2023 pour avoir un DataFrame complet.
annee_2021 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2021.csv', sep=';')
annee_2023 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup.csv', sep=';')
annee_2022 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2022.csv', sep=';')
annee_2023_aveclesnomsdesvariablesdeannee_2022 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2023_avec_les_noms.csv.csv', sep=';')


deux_ans = pd.concat([annee_2021, annee_2023], axis=0)

# Supposons que `annee_2023` et `annee_2023_aveclesnomsdesvariablesdeannee_2022` sont vos DataFrames

# Trier `annee_2023_aveclesnomsdesvariablesdeannee_2022` en utilisant l'ordre de `annee_2023`
colonne_commune = 'session'  
annee_2022_ordered = annee_2023_aveclesnomsdesvariablesdeannee_2022.set_index(colonne_commune).loc[annee_2023[colonne_commune]].reset_index()


# Concaténer les deux DataFrames alignés
parcoursup_total = pd.concat([annee_2022_ordered, deux_ans], ignore_index=True)
parcoursup_total.to_csv('parcoursup_total.csv', index=False)    
