import pandas as pd

annee_2023 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup.csv', sep=';')
annee_2022 = pd.read_csv(r'https://raw.githubusercontent.com/leaajo/TP_ISD/master/all_df/fr-esr-parcoursup_2022.csv', sep=';')

X_2022 = annee_2022.columns.to_list()
X_2023 = annee_2023.columns.to_list()

print("Colonnes pour l'année 2022 :")
for x_2022, x_2023 in zip(X_2022, X_2023):
    print(f"{x_2022}\t{x_2023}")