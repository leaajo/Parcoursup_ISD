import pandas as pd

# Avec la réforme du baccalauréat, les données de Parcoursup ont été modifiées depuis 2021. Nous allons donc concaténer les données de 2021,2022 et 2023 pour avoir un DataFrame complet.
annee_2021 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2021.csv', sep=';')
annee_2023 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup.csv', sep=';')
annee_2022 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2022.csv', sep=';')


# Concatenate annee_2021 and annee_2023 along their columns
parcoursup_2021_2023 = pd.concat([annee_2021, annee_2023], axis=1)
parcoursup_2021_2023.to_csv('parcoursup_2021_2023.csv', sep=';', index=False)
