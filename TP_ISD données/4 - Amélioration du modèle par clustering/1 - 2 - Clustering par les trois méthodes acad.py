import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from sklearn import metrics # for evaluations
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

#On effectue un clustering des formations selon les variables sélectionnées dans le second point.
#Ici on réalise les trois méthodes de CHA, K-Means et DBSCAN. Dans la suite on utilisera seulement les classes K-Means.
lycee_df = pd.read_csv(r"2 - Classification supervisée après sélection\Jeux créés\tableau_pred_eleves_acad", sep=";")
indices = lycee_df["Indices"] #Utilisés pour l'étiquetage des formations
lycee_df.index = indices
lycee_df.pop("Indices")
lycee_df.pop("Unnamed: 0")
#Centrer et réduire
scaler = StandardScaler()
lycee_n = scaler.fit_transform(lycee_df)

#Dataframe avec indices et données normalisées
lycee_sc = pd.DataFrame(lycee_n) #C'est aussi dans ce Dataframe qu'on rentrera les classes des individus
lycee_sc.columns = lycee_df.columns
lycee_sc.index = indices

# PREMIER CLUSTERING : Classification hiérarchique ascendante
linkage_matrix = linkage(lycee_n, method='ward')

#Dendrogramme
plt.figure(figsize=(10, 15))
dendrogram(linkage_matrix,
           orientation="left",
           leaf_font_size=8)
plt.title('Dendrogramme pour la classification hiérarchique ascendante')
plt.ylabel('Groupes')
plt.xlabel('Distance')
plt.show()

nbre_classes_CAH = 7 #Déterminé à partir du dendrogramme

#On insère les numéros des classes dans le Dataframe lycee_sc
clusters = cut_tree(linkage_matrix, n_clusters=nbre_classes_CAH)
L = []
for i in clusters:
    L.append(i[0])
L = np.array(L)
lycee_sc["Classe CAH"] = L


#DEUXIEME CLUSTERING : KMeans
range_n_clusters = np.arange(4,16) #Liste des nombres de classes testés
inertia = []
silhouette = []
for n_clusters in range_n_clusters:
    # k-means.
    kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
    y_pred = kmeans.fit_predict(lycee_n)
    inertia.append(kmeans.inertia_) #Variance
    #Coefficient de silhouette
    s_mean = metrics.silhouette_score(lycee_n, y_pred)
    silhouette.append(s_mean)

#Affichage de l'inertie et du coefficient de silhouette pour déterminer le nombre de classes
plt.plot(range_n_clusters,silhouette,".-",label="Silhouette")
plt.plot(range_n_clusters,(inertia-np.min(inertia))/(np.max(inertia)-np.min(inertia)),".-",label="Variance intraclasse normalisée")
plt.xticks(range_n_clusters)
plt.xlabel("Nombre de classes")
plt.title("Inertie et coefficient de silhouette selon le nombre de classe")
plt.legend()
plt.show()

#Ajout des classes KMeans
nbre_classes_KMeans = 10 #Déterminé avec la figure précédente
kmeans = KMeans(n_clusters=nbre_classes_KMeans, random_state=10) #Classification
y_pred = kmeans.fit_predict(lycee_n)
lycee_sc["Classe KMeans"] = y_pred


#TROISIEME CLUSTERING : DBSCAN
#Affichage des résultats pour divers epsilon
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(lycee_n)
distances, indices = neighbors_fit.kneighbors(lycee_n)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.ylabel("Epsilon")
plt.title("Voisinage epsilon considéré")
plt.show()

#Ajout des classes DBSCAN dans le Dataframe
epsilon = 4.1 #Déterminé par le graphique
y_pred = DBSCAN(eps=epsilon, min_samples=2).fit_predict(lycee_n)
lycee_sc["Classe DBSCAN"] = y_pred + 1

lycee_sc.to_csv(r"4 - Amélioration du modèle par clustering\Jeux créés\lycee_sc_acad", sep=";")