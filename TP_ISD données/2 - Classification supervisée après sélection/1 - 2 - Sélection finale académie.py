from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Sélection des meilleures variables et réalisation du RandomForest utilisant ces variables pour la prédiction du nombre d'élèves venant de la même académie que le lycée de la formation

def Modele(X, y, test_size, param_grid) :
    X_no_test, X_test, y_no_test, y_test = train_test_split(X, y, test_size=test_size, random_state=3)
    rf_model = RandomForestRegressor()
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="r2").fit(X_no_test, y_no_test)
    best_params_forest = grid_search.best_params_
    best_score_forest = grid_search.best_score_
    print("Meilleur paramètre : " + str(best_params_forest))
    print("Meilleur score évaluation : " + str(best_score_forest))

    #Appliquer le modèle aux données test
    rf_model = RandomForestRegressor(**best_params_forest)
    rf_model.fit(X_no_test, y_no_test)
    best_score_forest = rf_model.score(X_test, y_test)
    print("Meilleur score : " + str(best_score_forest))
    return(X_no_test, y_no_test, X_test, y_test, best_params_forest)

def Permutation(X_no_test, y_no_test, rf_model, index) :
    #Vérifier l'importance de chaque variable
    result = permutation_importance(
    rf_model, X_no_test, y_no_test, n_repeats=10, random_state=2, scoring='r2')
    importance_var_lycee_complet = pd.Series(result.importances_mean, index=index)
    fig, ax = plt.subplots()
    importance_var_lycee_complet.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Importance des variables déterminée par permutation")
    ax.set_ylabel("Diminution moyenne du score R2")
    fig.tight_layout()
    plt.xticks(rotation = 'vertical')
    plt.show()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(importance_var_lycee_complet)
        

#Préparation du tableau final
lycee_df = pd.read_csv(r"1 - Sélection des données\Jeux créés\lycee_RF", sep=";") #Créé avec 2 - 1 - Second Random Forest conservant le maximum d'individus
lycee_l = pd.read_csv(r"1 - Sélection des données\Jeux créés\lycee_l", sep=";") #Créé avec 9 - Fusion données langues
specialites = pd.read_csv("1 - Sélection des données\Jeux créés\specialites", sep=";") #Créé avec 10 - Spécialités de chaque lycée

#On inscrit ici les variables qui, pour chaque RandomForest, se sont révélées être les plus significatives par test de permutation
var_lycee = ["Indices","code_uai", "capa_fin", "prop_tot", "acc_tot", 
             "acc_tot_f", "acc_pp", "acc_bg", "acc_bt", "acc_bp", "acc_sansmention", "acc_ab", "acc_b", 
             "acc_bg_mention", "acad_mies_Paris", "taux_men_brut_gnle", 
             "va_men_gnle", "IPS Ensemble GT-PRO", "acc_term", "acc_term_f", "acc_aca_orig"]
var_langues = ["code_uai", "latitude", "longitude", "nombre_d_eleves_x", "2ndes_gt_filles", "2ndes_gt_lv2_allemand", 
               "1eres_g_lv2_allemand", "1eres_g_lv2_espagnol",
               "1eres_stmg", "1eres_stmg_garcons", "terminales_g_garcons", "terminales_g_lv1_anglais", "terminales_g_lv2_allemand", 
               "terminales_g_lv2_espagnol", "terminales_stmg_garcons"]
var_spes = ["numero_etablissement",
            "0001_mathematiques_physique_chimie_filles", "0001_mathematiques_physique_chimie_garcons",
            "0002_hist_geo_geopolitique_sc_politiques_sciences_economiques_et_sociales_garcons",
            "0105_humanites_litterature_et_philosophie_filles_y", 
            "0300_langues_litterature_et_cultures_etrangeres_et_regionales_filles_y", "0439_hist_geo_geopolitique_sc_politiques_garcons_y",
            "0629_sciences_de_la_vie_et_de_la_terre_filles_y", 
            "05_mathematiques_numerique_et_sciences_informatiques_physique_chimie_garcons"]


#Rassemblement de toutes les variables en un DataFrame
lycee = lycee_df.filter(var_lycee, axis=1)
langues = lycee_l.filter(var_langues, axis=1)
langues = langues.drop_duplicates(subset=["code_uai"]) #Il y a des doublons car on a plusieurs fois la même chose pour des formations du même lycée : comme on va refusionner après, on retire ici les doublons
specialites = specialites.filter(var_spes, axis=1)
specialites = specialites.drop_duplicates(subset=["numero_etablissement"])

formations = lycee.merge(langues, how="left", left_on="code_uai", right_on="code_uai")
formations = formations.merge(specialites, how="left", left_on="code_uai", right_on="numero_etablissement")
for spe in ["0001_mathematiques_physique_chimie_filles", "0001_mathematiques_physique_chimie_garcons",
            "0002_hist_geo_geopolitique_sc_politiques_sciences_economiques_et_sociales_garcons",
            "0105_humanites_litterature_et_philosophie_filles_y", 
            "0300_langues_litterature_et_cultures_etrangeres_et_regionales_filles_y", "0439_hist_geo_geopolitique_sc_politiques_garcons_y",
            "0629_sciences_de_la_vie_et_de_la_terre_filles_y", 
            "05_mathematiques_numerique_et_sciences_informatiques_physique_chimie_garcons"] :
    formations[spe] = formations[spe].replace(np.NaN, 0) #Pour les lycées non généraux, le nombre d'élèves en spécialités de formations générale est fixé à 0
formations.pop("code_uai")
formations.pop("numero_etablissement")

formations = formations.dropna(axis=0, subset=formations.columns)
formations.to_csv(r'2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_acad', sep=";")
formations.pop("Indices")
y_admis_lycee = formations["acc_term"]
y_admis_lycee_f = formations["acc_term_f"]
y_admis_acad = formations["acc_aca_orig"]
formations = formations.drop(["acc_term", "acc_term_f", "acc_aca_orig"], axis=1)

scaler = StandardScaler()
X_formations = scaler.fit_transform(formations)


param_grid = {
    'n_estimators': [150, 100],
    'max_depth': [None, 20],
}
index = formations.columns.to_list()

X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele(X_formations, y_admis_acad, 0.15, param_grid)
rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
Permutation(X_no_test, y_no_test, rf_model, index)

#Résultats
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.8858535164297198
#Meilleur score : 0.8137881776739502