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

#On prédit la classe K-Means à laquelle appartient chaque formation, et on va ensuite faire la régression selon cette classe prédite
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

#On effectue la régression en fonction des classes prédites
formations_sc = formations_sc.sort_values("predictions") #Ranger dans l'ordre des classes prédites
vraies_valeurs = [] #Véritable nombre d'élèves venant de la même académie pour les ensembles tests
predictions_r = [] #Valeurs prédites pour les ensembles tests
predictions_total = [] #Ensemble des valeurs prédites
scores = [] #Scores pour chaque classe
best_params = [] #Meilleurs paramètres de prédiction pour chaque classe
for classe in range(0,10) : 
    X = formations_sc[formations_sc["predictions"]==classe] #On isole les formations selon les 10 classes K-Means
    y = X["acc_aca_orig"]
    X = X.drop(["Indices", "acc_term", "acc_term_f", "acc_aca_orig", "Classe CAH", "Classe KMeans", "Classe DBSCAN"], axis=1)
    index = X.columns
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
        Permutation(X_no_test_r, y_no_test_r, rf_model, index)

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
    title_text="Nombre d'élèves venant de la même académie prédit selon le nombre réel et la classe KMean prédite"
)

fig.show()


scores_2 = [] #Obtenir les scores selon la classe réelle
for i in range(0,10) :
    X = formations_sc[formations_sc["Classe KMeans"]==i]
    if len(X["Classe KMeans"]) != 0 :
        scores_2.append(r2_score(X["acc_aca_orig"], X["predictions_reg"]))
    else : 
        scores_2.append(0)
plt.bar(range(0,10), scores_2, color="b", align="center")
plt.title("Performance du modèle par classe K-Means")
plt.ylabel("Score r2")
plt.xlabel("Classe")
plt.show()

sns.boxplot(x="Classe KMeans",y="acc_aca_orig",
            data=formations_sc)
sns.despine(offset=10, trim=True)
plt.title("Nombre d'élèves de la même académie pour chaque classe")
plt.show()

sns.boxplot(x="Classe KMeans",y="predictions_reg",
            data=formations_sc)
sns.despine(offset=10, trim=True)
plt.title("Nombre prédit d'élèves de la même académie pour chaque classe")
plt.show()

#Affichage des véritables valeurs et valeurs prédites selon une variable choisie
var = "capa_fin"
plt.plot(formations_sc[var], formations_sc["acc_aca_orig"], 'o', color="blue", label="Vraie classe")
plt.plot(formations_sc[var], formations_sc["predictions_reg"], '.', color="red", label="Classe prédite")
plt.title("Véritables valeurs et valeurs prédites")
plt.ylabel("Classes")
plt.xlabel(var)
plt.legend()
plt.show()

#Résultats
#Prédicion de la classe K-Means de la formation :
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.9192568008502473
#Meilleur score : 0.9155495978552279

#Prédiction du nombre d'élèves du même lycée admis dans la formation :
#Classe 0
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.6827579635466081
#Meilleur score : 0.7152972249195073
#Classe 1
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.7432937120479142
#Meilleur score : 0.7432937120479142
#Classe 2
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.3019279685806936
#Meilleur score : 0.7341758671047828
#Classe 3
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.46473573691129977
#Meilleur score : 0.255562705568134
#Classe 4
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.8111138274371843
#Meilleur score : 0.7282444145959257
#Classe 5
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 100}
#Meilleur score évaluation : 0.5830697277437833
#Meilleur score : 0.47056310204403495
#Classe 6
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.6766404808764153
#Meilleur score : 0.7256031660917738
#Classe 7
#Meilleur paramètre : {'max_depth': None, 'n_estimators': 150}
#Meilleur score évaluation : 0.7342254654851845
#Meilleur score : 0.7799553946730577
#Classe 8
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 150}
#Meilleur score évaluation : 0.7106922859487533
#Meilleur score : 0.7582284464644613
#Classe 9
#Meilleur paramètre : {'max_depth': 20, 'n_estimators': 100}
#Meilleur score évaluation : 0.7988527294503094
#Meilleur score : 0.6955070770105177

#Score r2 total (déterminé en prenant l'ensemble des valeurs des échantillons utilisés pour les tests): 0.854677309040468

#Le score final est amélioré par rapport à la Classification supervisée après sélection sans clustering. 
#Le modèle est au final assez satisfaisant.
#Là aussi, la performance du modèle varie selon les classes, et une étude plus approfondie des variables intéressantes à considérer ou pas pour chaque classe serait intéressante.
