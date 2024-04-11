import matplotlib.pyplot as plt
import numpy as np
import traitement as trt

df = trt.traitement_des_donnees(2021) # On récupère les données de 2023

# On crée un pie chart pour voir la répartition des lycées par type de contrat POUR L'ACADEMIE DE RENNES
regions = df['fili'].value_counts()
x = regions.index
print(x)
y = regions.values
#myexplode = [0.3, 0.3 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # On fait ressortir la première région
plt.figure(figsize=(10, 10))
plt.pie(y, labels=x, autopct='%1.1f%%')
plt.show()