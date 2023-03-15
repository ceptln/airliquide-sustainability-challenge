from __future__ import annotations

from typing import Union, Any

import geopandas as gpd
import pandas as pd
import numpy as np
import math


from ..utils.helpers import Data


class ProductionPlants:
    def __init__(self):
        self.stations = pd.read_csv(Data.find_file(
            "stations_10000_2030.csv")).drop("Unnamed: 0", axis=1)
        self.dfae = gpd.read_file(Data.find_file(
            "Aires_logistiques_elargies.shp"))
        self.dfad = gpd.read_file(
            Data.find_file("Aires_logistiques_denses.shp"))
        self.dfad = gpd.read_file(
            Data.find_file("Aires_logistiques_denses.shp"))
        self.aires = None

    def preprocess(self):
        self.dfad["center"] = self.dfad["geometry"].apply(
            lambda x: tuple(x.centroid.coords)[0])
        self.dfae["center"] = self.dfae["geometry"].apply(
            lambda x: tuple(x.centroid.coords)[0])
        self.stations["type_n"] = self.stations["type"].apply(
            lambda x: 0.75 if x == "small" else 1.0)
        self.aires = pd.concat([self.dfae.drop("d", axis=1).assign(dense=0), self.dfad[~self.dfad.e1.isin(
            self.dfae.e1)].assign(dense=1)]).drop(["geometry", "dense"], axis=1)

    @classmethod
    def fitness_production(cls, X: Union[gpd.GeoDataFrame, pd.DataFrame],
                           stations: Union[gpd.GeoDataFrame, pd.DataFrame],
                           num_big: int,
                           capacity_big: Union[int, float],
                           capacity_small: Union[int, float]) -> list[Any]:
        df = X.copy()
        # x is a df of selected spots
        # we need columns: e1, coords

        lsdf = []
        cpdists = []
        for i in range(df.shape[0]):
            sdf = pd.DataFrame({'station': [], 'dist': [], 'tps': []})
            for j in range(stations.shape[0]):
                # assume type is in last column for stations and encoded from 1-3 (small-big)
                # assume coords are in second column for points and second for stations
                sdf.loc[j] = [stations.iloc[j, 0], math.dist(
                    df.iloc[i, 1], eval(stations.iloc[j, 1])), stations.iloc[j, -1]]
            sdf['cpdist'] = sdf.dist*sdf.tps
            lsdf.append(sdf.sort_values(by='dist').reset_index(drop=True))
            cpdists.append(np.mean(sdf.cpdist))

        # Now start from 'best' prod in our permutation
        rank = np.argsort(cpdists)
        bigp = rank[:num_big]
        smallp = rank[num_big:]

        # we will compute the fitness of our permutation at the same time to avoid too many loops
        # For now, we do not consider spatial distance between prods, as the way we have designed the alg ensures that
        antifitness = 0
        split = []

        for i in bigp:
            capb = 0
            counter = 0
            subsplit = []
            while capb <= capacity_big:
                itm = list(lsdf[i][~lsdf[i].station.isin(
                    [item for sublist in split for item in sublist])].reset_index(drop=True).iloc[counter, :])
                capb += int(itm[2])*2400
                subsplit.append(itm[0])
                antifitness += int(itm[2])*itm[1]
                counter += 1
            split.append(subsplit)

        for i in smallp:
            caps = 0
            counter = 0
            subsplit = []
            while caps <= capacity_small:
                itm = list(lsdf[i][~lsdf[i].station.isin(
                    [item for sublist in split for item in sublist])].reset_index(drop=True).iloc[counter, :])
                caps += int(itm[2])*2400
                subsplit.append(itm[0])
                antifitness += int(itm[2])*itm[1]
                counter += 1
            split.append(subsplit)

        return [antifitness, bigp, smallp]

    def get_best_prod_plants(self, num_prod, num_big, capacity_big, capacity_small, iterations):
        locations = self.aires
        stations = self.stations
        best = []
        for i in range(iterations):
            if i % 50 == 0:
                print(f'iteration{i}')
            temp = locations.sample(num_prod).reset_index(drop=True)
            fitr = self.fitness_production(
                temp, stations, num_big, capacity_big, capacity_small)
            if len(best) == 0 or fitr[0] > best[0]:
                temp['big'] = [1.0 if x in fitr[1]
                               else 0.0 for x in temp.index.values]
                best = [fitr[0], temp]
        return best
