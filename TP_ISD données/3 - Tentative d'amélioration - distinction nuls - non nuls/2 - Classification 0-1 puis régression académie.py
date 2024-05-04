from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
#Première tentative d'amélioration du RandomForest : on prédit d'abord si le nombre d'élèves de la même académie est nul ou pas, puis on applique un RandomForest pour les valeurs prédites comme étant non nulles

def Modele_class(X, y, test_size, param_grid, critere) :
    X_no_test, X_test, y_no_test, y_test = train_test_split(X, y, test_size=test_size, random_state=3)
    rf_model = RandomForestClassifier()
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring=critere).fit(X_no_test, y_no_test)
    best_params_forest = grid_search.best_params_
    best_score_forest = grid_search.best_score_
    print("Meilleur paramètre : " + str(best_params_forest))
    print("Meilleur score évaluation : " + str(best_score_forest))

    #Appliquer le modèle aux données test
    rf_model = RandomForestClassifier(**best_params_forest)
    rf_model.fit(X_no_test, y_no_test)
    best_score_forest = rf_model.score(X_test, y_test)
    print("Meilleur score : " + str(best_score_forest))
    return(X_no_test, y_no_test, X_test, y_test, best_params_forest)

def Matrice(modele, X, y) :
  y_pred = modele.predict(X)
  matrice = confusion_matrix(y, y_pred)
  tn, fp, fn, tp = matrice.flatten()
  names = ["True neg", "False pos", "False neg", "True pos"]
  counts = [values for values in matrice.flatten()]
  percentages = ['{0:.2%}'.format(value) for value in matrice.flatten()/np.sum(matrice)]
  labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
  labels = np.asarray(labels).reshape(2,2)
  plt.figure(figsize=(8,6))
  sns.heatmap(matrice, annot=labels, fmt='', cmap="Blues",
            xticklabels=["Pas d'etudiants", "Au moins un étudiant"],
            yticklabels=["Pas d'étudiants", "Au moins un étudiant"])
  plt.xlabel("Valeur prédite")
  plt.ylabel("Véritable valeur")
  plt.title("Matrice de confusion")
  plt.show()
  tn,fp,fn,tp = matrice.flatten()
  # ACCURACY
  print('ACCURACY : ','{0:.2%}'.format((tp + tn)/(tp + fp + tn + fn)))
  # PRECISION
  print('PRECISION : ','{0:.2%}'.format(tp/(tp + fp)))
  # RECALL
  print('RECALL : ','{0:.2%}'.format(tp/(tp + fn)))

def Modele_reg(X, y, test_size, param_grid) :
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


formations = pd.read_csv(r"2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_acad", sep=";")
y_admis_acad = formations["acc_aca_orig"]
y_admis_acad_c = []
for i in y_admis_acad :
    if i == 0 :
        y_admis_acad_c.append(0)
    else :
        y_admis_acad_c.append(1)
formations["y_admis_acad_c"] = y_admis_acad_c
X_formations = formations.drop(["Indices","acc_term", "acc_term_f", "acc_aca_orig", "y_admis_acad_c"], axis=1)
scaler = StandardScaler()
X_formations = scaler.fit_transform(X_formations)

param_grid = {
    'n_estimators': [150, 100],
    'max_depth': [None, 20],
}
index = formations.columns.to_list()

#Prédiction de la valeur nulle ou pas du nombre d'élèves de la même académie
X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele_class(X_formations, y_admis_acad_c, 0.15, param_grid, "roc_auc")
rf_model = RandomForestClassifier(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)
Matrice(rf_model, X_formations, y_admis_acad_c)

formations["pred_0_1"] = rf_model.predict(X_formations)
formations = formations.sort_values("pred_0_1")

#RandomForest dépendant de la classe précédente
predictions_r = [0]*list(formations["pred_0_1"].values).count(0) #On insère déjà tous les 0 prédits

#Régression pour les valeurs prédites comme étant non nulles
X_formations = formations[formations["pred_0_1"]!=0]
y_admis_acad_non_nul = X_formations["acc_aca_orig"]
X_formations = X_formations.drop(["Indices", "acc_term", "acc_term_f", "acc_aca_orig", "y_admis_acad_c", "pred_0_1"], axis=1)
scaler = StandardScaler()
X_formations = scaler.fit_transform(X_formations)
X_no_test, y_no_test, X_test, y_test, best_params_forest = Modele_reg(X_formations, y_admis_acad_non_nul, 0.15, param_grid)

rf_model = RandomForestRegressor(**best_params_forest)
rf_model.fit(X_no_test, y_no_test)

predictions_r.extend(rf_model.predict(X_formations)) #Liste contenant l'ensemble des valeurs prédites


#Résultats : 

#Classification 0-1 : 
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.9623288211589761
#Meilleur score : 0.9946380697050938
#Matrice de confusion calculée sur l'ensemble total des lycées (pas seulement l'ensemble test) ;
#ACCURACY :  99.92%
#PRECISION :  99.94%
#RECALL :  99.98%

#Régression pour les valeurs prédites non nulles : 
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.8603917496951702
#Meilleur score : 0.8950694073827014

#Conclusion : le modèle permettant de prédire si le nombre d'élèves venant du même lycée est nul ou pas est assez bon.
#De plus, même si le nombre de formations pour lesquelles ce nombre est nul est assez faible, on remarque tout de même qu'en isolant les formations pour lesquelles on a prédit une valeur non nulle, le score r2 du modèle est amélioré.