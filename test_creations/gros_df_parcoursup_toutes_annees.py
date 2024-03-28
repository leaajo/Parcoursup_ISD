import pandas as pd

# Avec la réforme du baccalauréat, les données de Parcoursup ont été modifiées depuis 2021. Nous allons donc concaténer les données de 2021,2022 et 2023 pour avoir un DataFrame complet.
annee_2021 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2021.csv', sep=';')
annee_2023 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup.csv', sep=';')
annee_2022 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2022.csv', sep=';')

annee_2022 = annee_2022.rename(columns=dict(zip(annee_2022.columns, annee_2021.columns.to_list)))

# Concaténer tous les DataFrames en un seul DataFrame
parcoursup_total = pd.concat([annee_2021, annee_2023], ignore_index=True)

parcoursup_total.to_csv('parcoursup_total.csv', index=False) # Sauvegarde du DataFrame
