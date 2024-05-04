import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px

#On représente dans le plan factoriel les classes K-Means déterminées selon les variables sélectionnées pour la prédiction du nombre d'élèves du même lycée.
lycee_sc = pd.read_csv(r"4 - Amélioration du modèle par clustering\Jeux créés\lycee_sc", sep=";")
lycee_n = lycee_sc.copy()
lycee_n = lycee_n.drop(["Classe CAH", "Classe KMeans", "Classe DBSCAN", "Indices"], axis=1)
pca = PCA()
components = pca.fit_transform(lycee_n)

#Interprétation des quatre premiers axes factoriels
elements = lycee_n.columns
pcs = pca.fit(lycee_n)
n_axes = 4
loadings = pd.DataFrame(pcs.components_[0:n_axes, :],
                        columns=elements)
print(np.round(loadings,2))
maxPC = 1.01 * np.max(np.max(np.abs(loadings.loc[0:n_axes, :])))
f, axes = plt.subplots(n_axes, 1, figsize=(15, 10), sharex=True)
for i, ax in enumerate(axes):
    pc_loadings = loadings.loc[i, :]
    colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
    ax.axhline(color='#888888')
    pc_loadings.plot.bar(ax=ax, color=colors)
    ax.set_ylabel(f'PC{i+1}')
f.suptitle("Composition des " + str(n_axes) + " premières composantes principales")
plt.show()

#Intégrer les valeurs selon les quatre axes principaux dans le DataFrame
lycee_sc["PC 1"] = components[:,0]
lycee_sc["PC 2"] = components[:,1]
lycee_sc["PC 3"] = components[:,2]
lycee_sc["PC 4"] = components[:,3]

#AFFICHAGES : Représentation des clusterings selon les axes factoriels
#KMeans
fig = px.scatter(lycee_sc,
                 x = "PC 1",#+str(round(pca.explained_variance_ratio_[0]*100,2)),
                 y = "PC 2",
                 color = "Classe KMeans",
                 hover_name = "Indices",
                labels={
                     "PC 1": "PC 1 ("+str(round(pca.explained_variance_ratio_[0]*100,2))+"%)",
                     "PC 2": "PC 2 ("+str(round(pca.explained_variance_ratio_[1]*100,2))+"%)"
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
    title_text='ACP des formations (axes 1 et 2) selon les classes de K-means'
)

fig.show()

fig = px.scatter(lycee_sc,
                 x = "PC 3",#+str(round(pca.explained_variance_ratio_[0]*100,2)),
                 y = "PC 4",
                 color = "Classe KMeans",
                 hover_name = "Indices",
                labels={
                     "PC 3": "PC 3 ("+str(round(pca.explained_variance_ratio_[2]*100,2))+"%)",
                     "PC 4": "PC 4 ("+str(round(pca.explained_variance_ratio_[3]*100,2))+"%)"
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
    title_text='ACP des formations (axes 3 et 4) selon les classes de K-means'
)

fig.show()

#En étudiant les indices des différentes classes, on se rend compte que, malgré l'absence de variables à ce sujet, on retrouve une classification qui rassemble les formations en partie selon leurs thèmes.
#Si l'on prélève les thèmes récurrents pour chaque classe, on obtient les correspondances suivantes :
#Classe 0 : Management, comptabilité, industrialisation
#Classe 1 : Management commercial, gestion de la relation client
#Classe 2 : Industrialisation, maintenance, management, électronique
#Classe 3 : Prépa PCSI, Lettres, BCPST, MPSI
#Classe 4 : Comptabilité, management, électrotechnique 
#Classe 5 : Commerce, comptabilité, mathématiques appliquées
#Classe 6 : Mathématiques appliquées, services informatiques, management
#Classe 7 : Conception de produits et systèmes industriels, énergie
#Classe 8 : Mathématiques approfondies et appliquées, prépas Lettres, MPSI
#Classe 9 : Services informatiques, management, comptabilité
