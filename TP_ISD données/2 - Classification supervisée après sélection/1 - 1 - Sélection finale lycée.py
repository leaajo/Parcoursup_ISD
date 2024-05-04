from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Sélection des meilleures variables et réalisation du RandomForest utilisant ces variables pour la prédiction du nombre d'élèves venant du même lycée que la formation
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
    plt.show()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(importance_var_lycee_complet)
        

#Préparation du tableau final
lycee_df = pd.read_csv(r"1 - Sélection des données\Jeux créés\lycee_RF", sep=";") #créé avec 2 - 1 - Second Random Forest conservant le maximum d'individus
langues = pd.read_csv("1 - Sélection des données\Jeux créés\lycee_l", sep=";") #Créé avec 9 - Fusion données langues
specialites = pd.read_csv("1 - Sélection des données\Jeux créés\specialites", sep=";") #Créé avec 10 - Spécialités de chaque lycée

#On inscrit ici les variables qui, pour chaque RandomForest, se sont révélées être les plus significatives par test de permutation
var_lycee = ["Indices", "code_uai", "capa_fin", "voe_tot", "nb_voe_pp", "nb_voe_pp_bg", "nb_voe_pp_bt", "nb_voe_pp_bp", 
             "acc_brs", "acc_bg", "prop_tot", "acc_tot", 
             "acc_tot_f", "acc_pp", "acc_bt", "acc_bp", "acc_sansmention", "acc_ab", "acc_b", "acc_tbf", "acc_bt_mention", 
             "presents_gnle", "taux_men_brut_gnle", "va_men_gnle", 
             "taux_brut_de_reussite_total_secteurs_p", "taux_acces_brut_seconde_p", 
             "taux_brut_de_mentions_total_secteurs_p", "IPS Ensemble GT-PRO",
             "acc_term", "acc_term_f", "acc_aca_orig"]
var_langues = ["code_uai", "latitude", "longitude",  "nombre_d_eleves", "2ndes_gt_lv2_allemand", 
               "2ndes_gt_lv2_espagnol", "1eres_g_lv2_allemand", "1eres_g_lv2_espagnol",
               "1eres_g_lv2_italien", "1eres_sti2d_garcons", "terminales_g_lv2_allemand", 
               "terminales_g_lv2_espagnol", "terminales_stmg_filles", "terminales_stmg_garcons",
               "2ndes_pro_garcons", "1eres_pro_garcons", "1eres_pro_lv1_anglais", "terminales_pro", "terminales_pro_garcons",
               "terminales_pro_lv2_espagnol"]
var_spes = ["numero_etablissement", 
            "0001_mathematiques_physique_chimie_filles",
            "0010_humanites_litterature_et_philosophie_langues_litt_et_cultures_etra_et_r_filles",
            "0300_langues_litterature_et_cultures_etrangeres_et_regionales_garcons_y",
            "0629_sciences_de_la_vie_et_de_la_terre_filles_y", "0673_sciences_de_l_ingenieur_garcons", 
            "14_hist_geo_geopolitique_sc_politiques_mathematiques_physique_chimie_filles"]

#Rassemblement de toutes les variables en un DataFrame
lycee = lycee_df.filter(var_lycee, axis=1)
langues = langues.filter(var_langues, axis=1)
langues = langues.drop_duplicates(subset=["code_uai"]) #Il y a des doublons car on a plusieurs fois la même chose pour des formations du même lycée : comme on va refusionner après, on retire ici les doublons
specialites = specialites.filter(var_spes, axis=1)
specialites = specialites.drop_duplicates(subset=["numero_etablissement"])

formations = lycee.merge(langues, left_on="code_uai", right_on="code_uai")
formations = formations.merge(specialites, left_on="code_uai", right_on="numero_etablissement")
for spe in ["0001_mathematiques_physique_chimie_filles",
            "0010_humanites_litterature_et_philosophie_langues_litt_et_cultures_etra_et_r_filles",
            "0300_langues_litterature_et_cultures_etrangeres_et_regionales_garcons_y",
            "0629_sciences_de_la_vie_et_de_la_terre_filles_y", "0673_sciences_de_l_ingenieur_garcons", 
            "14_hist_geo_geopolitique_sc_politiques_mathematiques_physique_chimie_filles"] :
    formations[spe] = formations[spe].replace(np.NaN, 0) #Pour les lycées non généraux, le nombre d'élèves en spécialités de formations générale est fixé à 0
formations.pop("code_uai")
formations.pop("numero_etablissement")

formations = formations.dropna(axis=0, subset=formations.columns) #Retirer les lignes avec des valeurs manquantes
formations.to_csv(r'2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_lycee', sep=";") #DataFrame comportant les meilleures variables pour la prédiction

#Séparation des features et valeurs à prédire
y_admis_lycee = formations["acc_term"]
y_admis_lycee_f = formations["acc_term_f"]
y_admis_acad = formations["acc_aca_orig"]
formations = formations.drop(["Indices","acc_term", "acc_term_f", "acc_aca_orig"], axis=1) 

scaler = StandardScaler()
X_formations = scaler.fit_transform(formations)


param_grid = {
    'n_estimators': [150, 100],
    'max_depth': [None, 20],
}
index = formations.columns.to_list()

#RandomForest
X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele(X_formations, y_admis_lycee, 0.15, param_grid)
rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
Permutation(X_no_test, y_no_test, rf_model, index)

X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele(X_formations, y_admis_lycee_f, 0.15, param_grid) #Prédiction pour le nombre d'admises issues du même lycée
rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
Permutation(X_no_test, y_no_test, rf_model, index)

#Résultats
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.Meilleur score évaluation : 0.46419748612144185
#Meilleur score : 0.3753053156250864