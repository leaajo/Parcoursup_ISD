from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

formations = pd.read_csv(r'2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_acad', sep=";")
formations.pop("Indices")

y_admis_acad = formations["acc_aca_orig"]
formations = formations.drop(["acc_term", "acc_term_f", "acc_aca_orig"], axis=1)

scaler = StandardScaler()
X_formations = scaler.fit_transform(formations)

#Exemple de graphique donnant les données réelles et prédites
X_no_test, X_test, y_no_test, y_test = train_test_split(X_formations, y_admis_acad, test_size=0.15, random_state=3)
best_params_forest = {'max_depth': None, 'n_estimators': 150}
rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
y_predict = rf_model.predict(X_test)
plt.plot(X_test[:,0], y_test, 'o', color="blue") #capa_fin
plt.plot(X_test[:,0], y_predict, '.', color="red")
plt.title("Données prédites et réelles selon la capacité de la formation")
plt.ylabel("Taux d'élèves venant de la même académie")
plt.xlabel('Nombre de places de la formation')
plt.legend()
plt.show()