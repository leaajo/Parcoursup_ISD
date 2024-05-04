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

#Pour ce dernier modèle, on va utiliser le clustering K-Means réalisé précédemment pour tenter d'affiner les RandomForests :
#On va effectuer une forêt aléatoire pour chaque classe, de manière itérative.
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
formations_sc = pd.read_csv(r"4 - Amélioration du modèle par clustering\Jeux créés\lycee_sc", sep=";")
#Récupérer les valeurs non standardisées pour les valeurs que l'on cherche à prédire
val_non_standardisees = pd.read_csv(r"2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_lycee", sep=";")
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
f1 = make_scorer(f1_score , average='weighted') #Score pour classification avec plus de deux options
X_no_test_c, y_no_test_c, X_test_c, y_test_c, best_params_forest = Modele_class(X_formations, KMeans, 0.15, param_grid, f1)
rf_model = RandomForestClassifier(**best_params_forest)
rf_model.fit(X_no_test_c, y_no_test_c)
predictions = rf_model.predict(X_formations)
formations_sc["predictions"] = predictions

#Affichage des véritables classes KMeans et des classes prédites
plt.plot(range(0, len(formations_sc["Classe KMeans"])), formations_sc["Classe KMeans"], 'o', color="blue", label="Vraie classe")
plt.plot(range(0, len(formations_sc["predictions"])), formations_sc["predictions"], '.', color="red", label="Classe prédite")
plt.title("Véritables classes KMeans et classes prédites")
plt.ylabel("Classes")
plt.legend()
plt.show()

formations_sc = formations_sc.sort_values("predictions") #Ranger dans l'ordre des classes prédites
vraies_valeurs = []
predictions_r = []
predictions_total = []
best_params = []
for classe in range(0,10) : 
    X = formations_sc[formations_sc["predictions"]==classe]
    y = X["acc_term"]
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

print("Score r2 total" + str(r2_score(vraies_valeurs, predictions_r)))

#Représentation graphique de la performance du modèle
formations_sc["predictions_reg"] = predictions_total

fig = px.scatter(formations_sc,
                 x = "acc_term",
                 y = "predictions_reg",
                 color = "predictions",
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
    title_text="Nombre d'élèves venant du même lycée prédit selon le nombre réel et la classe KMean prédite"
)

fig.show()

#Résultats
#Prédicion de la classe K-Means de la formation :
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.91881238564253
#Meilleur score : 0.8989280245022971

#Prédiction du nombre d'élèves du même lycée admis dans la formation :
#Classe 0
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : 0.3060757063061089
#Meilleur score : 0.3239947312219139
#Classe 1
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : 0.34851848504203664
#Meilleur score : 0.3292031581717695
#Classe 2
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.2681640793775889
#Meilleur score : 0.28908603181852754
#Classe 3
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.2858321282313566
#Meilleur score : 0.6463293507362784
#Classe 4
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.24808099377171447
#Meilleur score : 0.23834615027270223
#Classe 5
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.3248375685669755
#Meilleur score : 0.3595084646818447
#Classe 6
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : -0.056787370594831924
#Meilleur score : 0.5324880269248883
#Classe 7
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : 0.42739671430607495
#Meilleur score : 0.32052377294067436
#Classe 8
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 100}
#Meilleur score évaluation : 0.018294680106787142
#Meilleur score : 0.2873672908527549
#Classe 9
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : 0.3165079176720488
#Meilleur score : 0.4621809542245915

#Score r2 total (déterminé en prenant l'ensemble des valeurs des échantillons utilisés pour les tests) : 0.4669888389577247

#Ce modèle a un meilleur score que celui avec les variables sélectionnées des différents DataFrames, mais les performances ne sont toujours pas très élevées.
#Cependant, en passant par le clustering, on observe une certaine variabilité dans les scores de prédiction : par exemple, le modèle de prédiction est un peu plus satisfaisant pour les classes 3 et 9.
#Il devient alors possible de réfléchir sur la pertinence des features pour chaque classe, et de faire le lien avec le type de formation, puisque l'on a précédemment noté une certaine correspondance entre le sujet de la formation et la classe K-Means.
#Pour prolonger cette étude, il serait intéressant d'effectuer un test par permutation pour chaque classe et d'étudier, dans tous les cas, les variables qui s'avèrent intéressantes ou pas.