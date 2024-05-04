import pandas as pd
import numpy as np

val_aj_df=pd.read_csv(r'1 - Sélection des données\Jeux de données\fr-en-indicateurs-de-resultat-des-lycees-denseignement-general-et-technologique.csv', sep=";")
X = val_aj_df.columns.to_list()
for x in X :
  print(x)
val_aj_df = val_aj_df[val_aj_df["annee"] == 2021]

#Tri des variables global
liste_val = ['code_etablissement', 'taux_brut_de_reussite_total_series', 'taux_reussite_attendu_acad_total_series', 'taux_reussite_attendu_france_total_series',
         'taux_mention_attendu_toutes_series', 'pourcentage_bacheliers_sortants_2de_1re_term_etab', 'pourcentage_bacheliers_sortants_terminales_etab',
         'pourcentage_bacheliers_sortants_2de_1re_term_acad', 'pourcentage_bacheliers_sortants_terminales_acad', 'pourcentage_bacheliers_sortants_2de_1re_term_france',
         'pourcentage_bacheliers_sortants_terminales_france', 'taux_acces_brut_seconde_bac', 'taux_acces_attendu_acad_seconde_bac', 'taux_acces_attendu_france_seconde_bac',
         'taux_acces_brut_premiere_bac', 'taux_acces_attendu_acad_premiere_bac', 'taux_acces_attendu_france_premiere_bac', 'taux_acces_brut_terminale_bac',
         'taux_acces_attendu_france_terminale_bac', 'va_reu_total', 'va_acc_seconde', 'va_men_total', 'presents_gnle', 'taux_reu_brut_gnle', 'va_reu_gnle',
         'taux_men_brut_gnle', 'va_men_gnle', 'nombre_de_mentions_tb_avec_felicitations_g', 'nombre_de_mentions_tb_sans_felicitations_g', 'nombre_de_mentions_b_g',
         'nombre_de_mentions_ab_g', 'nombre_de_mentions_tb_avec_felicitations_t', 'nombre_de_mentions_tb_sans_felicitations_t', 'nombre_de_mentions_b_t',
         'nombre_de_mentions_ab_t']
val_aj_df = val_aj_df.filter(liste_val, axis=1)

#Réécriture des mentions
val_aj_df['nombre_de_mentions_tb_g'] = val_aj_df['nombre_de_mentions_tb_avec_felicitations_g'] + val_aj_df['nombre_de_mentions_tb_sans_felicitations_g']
val_aj_df['nombre_de_mentions_tb_t'] = val_aj_df['nombre_de_mentions_tb_avec_felicitations_t'] + val_aj_df['nombre_de_mentions_tb_sans_felicitations_t']
val_aj_df.pop('nombre_de_mentions_tb_avec_felicitations_g')
val_aj_df.pop('nombre_de_mentions_tb_sans_felicitations_g')
val_aj_df.pop('nombre_de_mentions_tb_avec_felicitations_t')
val_aj_df.pop('nombre_de_mentions_tb_sans_felicitations_t')

#Retirer les valeurs nulles
null = val_aj_df.isnull()
val_aj_df1 = val_aj_df.copy()
for col in val_aj_df.columns :
  if list(set(list(null[col]))) == [True] :
    val_aj_df1.pop(col)

#Conversion en string
val_aj_df1['code_etablissement'] = val_aj_df1['code_etablissement'].astype('string')

val_aj_df1 = val_aj_df1.dropna(axis=0, subset="presents_gnle")

#Remplacer les vides
liste = ["taux_acces_brut_seconde_bac", "va_reu_total", "va_acc_seconde", "va_men_total", "presents_gnle", "va_reu_gnle", "va_men_gnle"]
val_aj_df2 = val_aj_df1.copy()
for var in liste :
  val_aj_df2[var] = val_aj_df2[var].replace('.', '0.0')
  val_aj_df2[var] = val_aj_df2[var].replace('ND', np.NaN)
val_aj_df2 = val_aj_df2.dropna(axis=0, subset = liste)

#Convertir en float
for var in liste :
  val_aj_df2[var] = val_aj_df2[var].astype('float')

# Ajout d'une variable pour indiquer que c'est un lycée général et technologique
val_aj_df2["lycee_gnle_et_tech"] = [1]*val_aj_df2.shape[0]

val_aj_df2.info()

val_aj_df2 = val_aj_df2.to_csv(r'1 - Sélection des données\Jeux créés\val_aj_gnle', sep=";")