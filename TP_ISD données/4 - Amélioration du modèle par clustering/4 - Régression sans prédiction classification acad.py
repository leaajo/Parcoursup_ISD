from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, make_scorer, f1_score, r2_score
import seaborn as sns
import plotly.express as px

#On prédit le nombre d'élèves venant de la même académie de la même manière que dans le 4 - Méthode classification - régression acad.
#La différence est qu'ici, on n'établit pas de classification pour tenter de prédire la classe à laquelle appartient chaque formation.
#On part donc du principe que l'on connaît cette classe et, à partir de cela, on configure, pour chaque classe, une régression pour prédire le nombre d'élèves de la même académie.

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

#Préparation du tableau final
formations_sc = pd.read_csv(r"4 - Amélioration du modèle par clustering\Jeux créés\lycee_sc_acad", sep=";")
#Récupérer les valeurs non standardisées pour les valeurs que l'on cherche à prédire
val_non_standardisees = pd.read_csv(r"2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_acad", sep=";")
for var in ["acc_term", "acc_term_f", "acc_aca_orig"] :
    formations_sc[var] = val_non_standardisees[var]

formations_sc = formations_sc.sort_values("Classe KMeans")

formations_sc = formations_sc.dropna(axis=0, subset=formations_sc.columns)
y_admis_lycee = formations_sc["acc_term"]
y_admis_lycee_f = formations_sc["acc_term_f"]
y_admis_acad = formations_sc["acc_aca_orig"]
KMeans = formations_sc["Classe KMeans"]
formations = formations_sc.drop(["Indices", "acc_term", "acc_term_f", "acc_aca_orig", "Classe CAH", "Classe KMeans", "Classe DBSCAN"], axis=1)

scaler = StandardScaler()
X_formations = scaler.fit_transform(formations)


param_grid = {
    'n_estimators': [150, 100],
    'max_depth': [None, 20],
}
index = formations.columns.to_list()

#Modèle de régression dépendant de la classe de la formation
vraies_valeurs = [] #Ensemble des vraies valeurs pour les sous-ensembles de tests
predictions_r = [] #Valeurs prédites pour les sous-ensembles de tests
predictions_total = [] #Valeurs prédites pour le tableau entier
scores = [] #Scores selon la classe
best_params = [] #Meilleurs paramètres pour chaque classe
for classe in range(0,10) : 
    X = formations_sc[formations_sc["Classe KMeans"]==classe] #On isole les formations d'une certaine classe
    y = X["acc_aca_orig"]
    X = X.drop(["Indices", "acc_term", "acc_term_f", "acc_aca_orig", "Classe CAH", "Classe KMeans", "Classe DBSCAN"], axis=1)
    if len(y) != 0 : 
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_no_test_r, y_no_test_r, X_test_r, y_test_r, best_params_forest = Modele_reg(X, y, 0.15, param_grid)
        best_params.append(best_params_forest)
        rf_model = RandomForestRegressor(**best_params_forest)
        rf_model.fit(X_no_test_r, y_no_test_r)
        y_predict_r = rf_model.predict(X_test_r)
        y_predict_total = rf_model.predict(X)
        vraies_valeurs.extend(y_test_r)
        predictions_r.extend(y_predict_r)
        predictions_total.extend(y_predict_total)
        scores.append(r2_score(y_test_r, y_predict_r))

print("Score r2 total" + str(r2_score(vraies_valeurs, predictions_r)))

#Représentation graphique de la performance du modèle
formations_sc["predictions_reg"] = predictions_total

fig = px.scatter(formations_sc,
                 x = "acc_aca_orig",
                 y = "predictions_reg",
                 color = "Classe KMeans",
                 hover_name = "Indices",
                labels={
                     "Valeur réelle",
                     "Valeur prédite"
                 },
                 color_continuous_scale="ylgnbu"
)
fig.update_traces(textposition='top center')
fig.update_traces(
                  marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.update_layout(
    height=600,width=1000, font=dict(size=8),
    title_text="Nombre d'élèves venant de la même académie prédit selon le nombre réel et la classe KMean réelle"
)

fig.show()

plt.bar(range(0,10), scores, color="b", align="center")
plt.title("Performance du modèle par classe K-Means")
plt.ylabel("Score r2")
plt.xlabel("Classe")
plt.show()

#Affichage des véritables valeurs et valeurs prédites selon une variable choisie
plt.plot(formations_sc["acad_mies_Paris"], formations_sc["acc_aca_orig"], 'o', color="blue", label="Vraie classe")
plt.plot(formations_sc["acad_mies_Paris"], formations_sc["predictions_reg"], '.', color="red", label="Classe prédite")
plt.title("Véritables classes KMeans et classes prédites")
plt.ylabel("Classes")
plt.legend()
plt.show()

#Résultats : 
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.715401488417936
#Meilleur score : 0.6316193728102338
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.7382441776543993
#Meilleur score : 0.800701937686327
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.5083484868618269
#Meilleur score : -0.5373324458098656
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.4240750571240576
#Meilleur score : 0.3963277613207461
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : 0.7560486284284854
#Meilleur score : 0.8062886235308167
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 100}
#Meilleur score évaluation : 0.548919967501223
#Meilleur score : 0.6133929726831581
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.7207374330442379
#Meilleur score : 0.7566440253098248
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.766894728741948
#Meilleur score : 0.7426906827177077
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.7322830410148314
#Meilleur score : 0.7057944501493563
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.7763401807728398
#Meilleur score : 0.786948031055976

#Score r2 total : 0.8770342912351943
#On peut faire les mêmes remarques que pour la méthode effectuant au préalable une prédiction de la classe.
#Le score final est assez bon, et on peut s'intéresser plus au détail aux performances pour chaque classe.