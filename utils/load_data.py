import pandas as pd
import yaml
import geopandas as gpd

with open("../config.yaml") as f:
    config = yaml.safe_load(f)

def load_stations(path:str=None)-> pd.DataFrame:
    if path is None:
        path = '../' + config['stations_file']
    try:
        stations= pd.read_csv(path)
    except FileNotFoundError:
        stations = pd.read_csv(config['stations_file'])
    stations[['lat', 'lon', '1']] = stations.Coordinates.str.split(",", expand = True)
    stations = stations.drop(columns=['1'])
    stations = stations.drop('H2 Conversion', axis=1)
    return stations


def load_rrn_vsmap(path:str=None)-> pd.DataFrame:
    if path is None:
        path = '../' + config['rrn_vsmap_file']
    try:
        rrn_vsmap = gpd.read_file(path)
    except FileNotFoundError:
        rrn_vsmap = gpd.read_file(path)
    rrn_vsmap[['dep', 'route', '1', '2', '3', '4']] = rrn_vsmap.route.str.split(" ", expand = True)
    rrn_vsmap = rrn_vsmap.drop(columns = ['1', '2', '3', '4'])
    return rrn_vsmap