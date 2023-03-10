from __future__ import annotations
import warnings
import yaml

import geopandas as gpd
import pandas as pd

from utils.helpers import Data

warnings.simplefilter('ignore', FutureWarning)

with open('/Users/charlesnicholas/Documents/Sustainability challenge/h2_station_distributor/config.yaml') as f:
    print('Opened')

with open(Data.find_file("config.yaml")) as f:
    print(f.name)
    config = yaml.safe_load(f)


def load_stations(path: str = None) -> pd.DataFrame:
    if path is None:
        path = config['stations_file']
    try:
        stations = pd.read_csv('../' + path)
    except FileNotFoundError:
        stations = pd.read_csv(config['stations_file'])
    stations[['lat', 'lon', '1']] = stations.Coordinates.str.split(",", expand=True)
    stations = stations.drop(columns=['1'])
    stations = stations.drop('H2 Conversion', axis=1)
    return stations


def load_rrn_vsmap(path: str = None) -> pd.DataFrame:
    if path is None:
        path = config['rrn_vsmap_file']
    try:
        rrn_vsmap = gpd.read_file('../' + path)
    except FileNotFoundError:
        rrn_vsmap = gpd.read_file(path)
    rrn_vsmap[['dep', 'route', '1', '2', '3', '4']] = rrn_vsmap.route.str.split(" ", expand=True)
    rrn_vsmap = rrn_vsmap.drop(columns=['1', '2', '3', '4'])
    return rrn_vsmap


def load_rrn_bornage(path: str = None) -> pd.DataFrame:
    if path is None:
        path = config['rrn_bornage_file']
    try:
        rrn_bornage = gpd.read_file('../' + path)
    except FileNotFoundError:
        rrn_bornage = gpd.read_file(path)
    return rrn_bornage


def load_regions(path: str = None) -> pd.DataFrame:
    if path is None:
        path = config['regions_file']
    try:
        regions = gpd.read_file('../' + path)
    except FileNotFoundError:
        regions = gpd.read_file(path)
    regions.drop(regions[regions['nomnewregi'] == 'Corse'].index, inplace=True)
    return regions


def load_traffic(path: str = None) -> pd.DataFrame:
    if path is None:
        path = config['traffic_file']
    try:
        traffic = gpd.read_file('../' + path)
    except FileNotFoundError:
        traffic = gpd.read_file(path)
    return traffic


def load_depreg(path: str = None) -> pd.DataFrame:
    if path is None:
        path = config['depreg_file']
    try:
        depreg = pd.read_csv('../' + path)
    except FileNotFoundError:
        depreg = pd.read_csv(path)
    depreg.drop(columns=['dep_name'], inplace=True)
    return depreg


def load_airesPL(path: str = None) -> pd.DataFrame:
    if path is None:
        path = config['airesPL']
    try:
        airesPL = gpd.read_file('../' + path)
    except FileNotFoundError:
        airesPL = gpd.read_file(path)
    return airesPL
