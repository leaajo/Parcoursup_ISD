import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme() # permet d'obtenir le fonc gris avec les lignes blanches

import geopandas as gpd 

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score, 
    cross_val_predict, 
    train_test_split, 
    GridSearchCV, 
    KFold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import sklearn.metrics as confusion_matrix
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import plotly.express as px
import matplotlib.cm as cm

import geoviews as gv
import geoviews.feature as gf
gv.extension('bokeh')

from cartopy import crs