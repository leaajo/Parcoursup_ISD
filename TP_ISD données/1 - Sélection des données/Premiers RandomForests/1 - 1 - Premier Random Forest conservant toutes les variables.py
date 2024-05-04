import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Premier modèle de prédiction du nombre d'élèves de la formation venant du même lycée
#Pour ce RandomForest, on garde toutes les variables dont certaines qui diminuent grandement l'échantillon de formations sans valeurs manquantes (l'écart-type IPS)


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

#On retire les individus avec des infos manquantes pour ce premier Random Forest
lycee_df = pd.read_csv(r"1 - Sélection des données\Jeux créés\lycee_df2", sep=";") #Créé avec Fusion des données Parcoursup, Val ajoutée et IPS
lycee_df.pop("code_uai")
lycee_df = lycee_df.dropna(axis=0, subset=["Ecart-type de l'IPS voie PRO"]) #Variable qui a le plus d'éléments manquants

#On remplace les nan qui perturbent le code
lycee_df["acc_term"] = lycee_df["acc_term"].replace('nan', np.NaN)
lycee_df = lycee_df.dropna(axis=0, subset=lycee_df.columns)

#Création des features et variables à prédire
y_admis_lycee = lycee_df["acc_term"]
y_admis_lycee_f = lycee_df["acc_term_f"]
y_admis_acad = lycee_df["acc_aca_orig"]
X = lycee_df.copy()

for col in ["acc_term", "acc_term_f",
            "acc_aca_orig", "acc_aca_orig_idf"] :
            X.pop(col)
X.pop("Indices")
#Normalisation
scaler = StandardScaler()
X_n = scaler.fit_transform(X)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 20],
}
index = X.columns.to_list()

X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele(X_n, y_admis_lycee, 0.15, param_grid)
rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
Permutation(X_no_test, y_no_test, rf_model, index)

#Résultats : 
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : 0.2569980614182703
#Meilleur score (calculé sur l'ensemble test) : 0.11977971784536445
#Le modèle est proche d'un modèle complètement aléatoire, donc assez peu efficace.
#Les variables avec beaucoup de valeurs manquantes (Ecart-type d'IPS) sont peu significatives, on pourra les retirer pour conserver plus d'individus dans la suite