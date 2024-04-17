from math import pi

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from bokeh.palettes import Category20c
from bokeh.plotting import figure, show
from bokeh.transform import cumsum

lycees_2 = pd.read_csv('./all_df/lycees_2.csv', sep=',')
# Compter le nombre d'occurrences de chaque secteur
secteur_counts = lycees_2['secteur'].value_counts()

# Créer un graphique à secteurs en utilisant pyplot
plt.figure(figsize=(8, 8))  # Créez une figure en utilisant la fonction figure de pyplot
plt.pie(secteur_counts, labels=secteur_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
plt.title('Répartition des lycées par secteur')
plt.axis('equal')  # Assurez-vous que le graphique est un cercle
plt.show()

fig = px.sunburst(lycees_2, path=['rentree_scolaire', 'secteur'])

# Afficher le graphique interactif
fig.show()
