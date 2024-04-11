import geoviews as gv
import geoviews.feature as gf
gv.extension('bokeh')
import geopandas as gpd
import plotly.express as px
import matplotlib.cm as cm

France_dep = gpd.read_file("/content/drive/MyDrive/departements-version-simplifiee.geojson")

France_dep["code_du_departement"] = "0"+ France_dep["code"]
France_dep = France_dep.drop(columns = "code")
France_dep.head()

from cartopy import crs
# TAUX REUSSITE BRUT BAC SESSION 2023 PAR DEPARTEMENT :

import holoviews as hv
from holoviews import dim, opts
hv.extension('bokeh')

regions = gv.Polygons(carte_res2023, vdims=['nom', 'code_du_departement', 'departement', 'taux_reu_brut_gnle', 'taux_men_brut_gnle'])
regions.opts(width=600, height=600, toolbar='above', color=dim('taux_reu_brut_gnle'), colorbar=True, tools=['hover'], aspect='equal')
