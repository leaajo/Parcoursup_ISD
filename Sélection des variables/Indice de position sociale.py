import pandas as pd 
ips = pd.read_csv('https://raw.githubusercontent.com/Bleunvenn/TP_ISD/master/all_df/fr-en-ips_lycees.csv', sep=";")
ips_df = ips[ips["Rentrée scolaire"] == '2021-2022']

#Sélection des variables
liste_ips = ['UAI', 'IPS voie GT', 'IPS voie PRO', 'IPS Ensemble GT-PRO', 'Ecart-type de l\'IPS voie GT', 'Ecart-type de l\'IPS voie PRO']
ips_df1 = ips_df.filter(liste_ips, axis=1)

#Conversion en string
ips_df1["UAI"] = ips_df1["UAI"].astype("string")

ips_df1.info()

ips_df1 = ips_df1.to_csv("ips_df1", sep=";")