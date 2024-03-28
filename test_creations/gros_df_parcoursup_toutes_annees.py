import pandas as pd

annee_2018 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup-2018.csv', sep=';')
annee_2019 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup-2019.csv', sep=';')
annee_2020 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2020.csv', sep=';')
annee_2021 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2021.csv', sep=';')
annee_2023 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup.csv', sep=';')

parcoursup_total = pd.concat([annee_2018, annee_2019, annee_2020, annee_2021, annee_2023])  # On concatène les données
parcoursup_total.to_csv('parcoursup_total.csv', index=False) # Sauvegarde du DataFrame
