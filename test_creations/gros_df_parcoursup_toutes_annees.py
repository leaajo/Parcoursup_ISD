import pandas as pd

# Avec la réforme du baccalauréat, les données de Parcoursup ont été modifiées depuis 2021. Nous allons donc concaténer les données de 2021,2022 et 2023 pour avoir un DataFrame complet.
annee_2021 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2021.csv', sep=';')
annee_2023 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup.csv', sep=';')


colonnes_communes = annee_2021.columns.intersection(annee_2023.columns)

# Sélectionner les colonnes des deux DataFrames
annee_2021 = annee_2021[colonnes_communes]
annee_2023 = annee_2023[colonnes_communes]

# Concaténer en ajoutant les lignes (axis=0)
concatenated_df = pd.concat([annee_2021, annee_2023], axis=0)
concatenated_df.to_csv('parcoursup_2021_2023.csv', sep=';', index=False)
