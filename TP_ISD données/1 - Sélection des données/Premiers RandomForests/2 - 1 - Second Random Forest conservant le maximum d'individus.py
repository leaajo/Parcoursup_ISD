from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Second modèle de prédiction du nombre d'élèves de la formation venant du même lycée
#Pour ce RandomForest, on retire les variables avec trop de données manquantes (les écarts-types) pour conserver un maximum de formations
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


#On garde tous les individus avec des valeurs non vides
lycee_df = pd.read_csv("1 - Sélection des données\Jeux créés\lycee_df2", sep=";") #Créé avec Fusion des données Parcoursup, Val ajoutée et IPS
var_y = ["acc_term", "acc_term_f",
         "acc_aca_orig"]
lycee_df = lycee_df.dropna(axis=0, subset=var_y)
lycee_df = lycee_df.dropna(axis=1) #On retire toutes les variables avec des valeurs manquantes, on sait qu'il y en a peu

#Séparer les y et normaliser les données
y_lycee = lycee_df["acc_term"]
y_lycee_f = lycee_df["acc_term_f"]
y_lycee_acad = lycee_df["acc_aca_orig"]
X_complet = lycee_df.copy()
X_complet.info()
lycee_RF = X_complet.to_csv(r"1 - Sélection des données\Jeux créés\lycee_RF", sep=";") #Avoir le fichier avant sa normalisation

X_complet = X_complet.drop(["acc_term", "acc_term_f", "acc_aca_orig", "acc_aca_orig_idf", "Indices", "code_uai"], axis=1) 
scaler = StandardScaler()
X_complet_n = scaler.fit_transform(X_complet)


param_grid = {
    'n_estimators': [150, 100],
    'max_depth': [None, 20],
}
index = X_complet.columns.to_list()

#Détermination du modèle et test par permutation
X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele(X_complet_n, y_lycee, 0.15, param_grid)
rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
Permutation(X_no_test, y_no_test, rf_model, index)

#Exemple d'affichage : valeurs prédites et réelles selon le nombre d'admis avec un bac technologique
y_predict = rf_model.predict(X_test)
plt.plot(X_test[:,17], y_test, 'o', color="blue")
plt.plot(X_test[:,17], y_predict, '.', color="red")
plt.title("Données prédites et réelles selon les effectifs des admis bacheliers technologiques")
plt.ylabel('Nombre d élèves restant dans le lycée')
plt.xlabel('Réussite au bac technologique')
plt.legend()
plt.show()

#Résultats : 
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.515592979315785
#Meilleur score (calculé sur l'ensemble test) : 0.46463287539245934
#Le modèle est déjà plus performant, même si le score r2 est encore assez faible. 
