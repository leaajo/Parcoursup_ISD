from math import pi

import pandas as pd
import matplotlib as plt
from bokeh.palettes import Category20c
from bokeh.plotting import figure, show
from bokeh.transform import cumsum

lycees_2 = pd.read_csv('./TP_ISD/all_df/lycees_2.csv', sep=';')
# Compter le nombre d'occurrences de chaque secteur
secteur_counts = lycees_2['secteur'].value_counts()

# Créer un graphique à secteurs
plt.figure(figsize=(8, 8))
plt.pie(secteur_counts, labels=secteur_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
plt.title('Répartition des lycées par secteur')
plt.axis('equal')  # Assurez-vous que le graphique est un cercle
plt.show()

x = {
    'United States': 157,
    'United Kingdom': 93,
    'Japan': 89,
    'China': 63,
    'Germany': 44,
    'India': 42,
    'Italy': 40,
    'Australia': 35,
    'Brazil': 32,
    'France': 31,
    'Taiwan': 31,
    'Spain': 29,
}

data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = Category20c[len(x)]

p = figure(height=350, title="Pie Chart", toolbar_location=None,
           tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='country', source=data)

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None

show(p)