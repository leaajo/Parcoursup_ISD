# Cette fonction sert à transformer un dataframe pandas en un geodataframe geopandas. Cela nous permet de visualiser les données sur une carte. (plus tard)

def carte(annee=int) :
    import traitement as trt 
    import geopandas as gpd
    
    df = trt.traitement_des_donnees(annee)
    
    # Convert the longitude and latitude to a format recognized by geoPandas
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])

    # Create a DataFrame with a geometry containing the Points
    geo_lycee = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)

    geo_lycee_metropole = geo_lycee[geo_lycee['latitude'] > 40]
    
    return geo_lycee_metropole

print(carte(2023))