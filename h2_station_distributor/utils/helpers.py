from __future__ import annotations
import io
import os
import requests
import time
from typing import Union
import warnings
import zipfile

import geopandas as gpd
import shapely
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


class LoadData:
    @staticmethod
    def load_stations(file_path: str = None, convert_to_gdf: bool = True,
                      only_h2_stations: bool = True) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        if file_path is None:
            file_path = Data.find_file('stations.csv')
        df = pd.read_csv(file_path)
        if convert_to_gdf:
            df = Data.convert_pd_stations_to_gpd(df_stations=df)
        if only_h2_stations:
            df = df.loc[df['H2 Conversion'] == 1, :]
        return df

    @staticmethod
    def load_regions(file_path: str = None, translate_regions_to_official_names: bool = True,
                     exclude_corsica: bool = True) -> gpd.GeoDataFrame:
        if file_path is None:
            file_path = Data.find_file('regions_2016.shp')
        df = gpd.read_file(file_path)
        if translate_regions_to_official_names:
            df = Data.translate_regions_regions_to_official_names(df_regions=df)
        if exclude_corsica:
            df = df.loc[df['nomnewregi'] != 'Corse', :]
            df = df.reset_index(drop=True)
        return df

    @staticmethod
    def load_traffic(file_path: str = None, clean_data: bool = True) -> gpd.GeoDataFrame:
        if file_path is None:
            file_path = Data.find_file('TMJA2018.shp')
        df = gpd.read_file(file_path)
        if clean_data:
            df = Data.clean_traffic_data(df=df)
        return df

    @staticmethod
    def load_depreg(file_path: str = None, drop_name: bool = False) -> pd.DataFrame:
        if file_path is None:
            file_path = Data.find_file('depreg.csv')
        df = pd.read_csv(file_path)
        if drop_name:
            df = df.drop(columns=['dep_name'])
        return df

    @staticmethod
    def load_logistic_hubs(file_path: str = None) -> gpd.GeoDataFrame:
        if file_path is None:
            file_path = Data.find_file('Aires_logistiques_denses.shp')
        return gpd.read_file(file_path)

    @staticmethod
    def load_water_bodies(file_path: str = None, simplify: bool = True) -> gpd.GeoDataFrame:
        if file_path is None:
            file_path = Data.find_file('water_bodies.shp')
        df_water_bodies = gpd.read_file(file_path)
        # The geometries are very precise which makes calculations with them very expensive, by simplifying them,
        # calculations are sped up considerably
        if simplify:
            # A tolerance of 15 meters remains accurate: the area change on the smallest lake is only 5.2 %
            # Increasing the tolerance is not recommended since it will create invalid geometries (which could be fixed
            # with adding .buffer(0) to create valid ones again)
            df_water_bodies = df_water_bodies.simplify(15)
        return df_water_bodies


class Data:
    @classmethod
    def find_file(cls, file_name: str) -> str:
        """This method searches for file_name and returns the full path to it."""
        for root, folders, files in os.walk(cls._find_root()):
            if file_name in files:
                return f'{root}/{file_name}'

    @classmethod
    def find_folder(cls, folder_name: str) -> str:
        """This method searches for folder_name and returns the full path to it."""
        for root, folders, files in os.walk(cls._find_root()):
            if folder_name in folders:
                return f'{root}/{folder_name}'

    @staticmethod
    def clean_traffic_data(df: gpd.GeoDataFrame, reset_index: bool = True) -> gpd.GeoDataFrame:
        """This method cleans the traffice data set. The provided DataFrame should be the output of the function
        load_traffic."""
        df = df.copy()
        # Removing 0 traffic records
        df = df.loc[df['tmja'].values > 0, :]
        # Since we are only interested in trucks, we remove the records that didn't see any heavy duty vehicles
        df = df.loc[df['pctPL'].values > 0, :]
        # They mentioned in the first coaching session that road section with more than 40 % of trucks on them are a
        # data error and should therefore be divided by 10
        mask = df['pctPL'] > 40
        df.loc[mask, 'pctPL'] = df.loc[mask, 'pctPL'] / 10

        if reset_index:
            df = df.reset_index(drop=True)
        return df

    @staticmethod
    def add_truck_traffic(df_traffic: Union[gpd.GeoDataFrame, pd.DataFrame]) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """This method adds a column with the truck traffic, since 'tmja' is the traffic of all vehicles.
        It is assumed that the df_traffic contains the following columns: 'tmja', 'pctPL'"""
        df_traffic = df_traffic.copy(deep=True)
        df_traffic['truck_tmja'] = df_traffic['tmja'] * df_traffic['pctPL'] / 100
        return df_traffic

    @staticmethod
    def add_weighted_traffic(df_traffic: gpd.GeoDataFrame,
                             traffic_column: str = 'truck_tmja') -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """This method adds a column with road section length / max(road section length) * traffic_column called
        'weighted_traffic' to df_traffic.
        It is assumed that the road geometry is contained in the column 'geometry'."""
        df_traffic = df_traffic.copy(deep=True)
        max_length = df_traffic['geometry'].length.max()
        df_traffic['weighted_traffic'] = df_traffic['geometry'].length / max_length * df_traffic[traffic_column]
        return df_traffic

    @staticmethod
    def add_regions_to_departments(df: Union[gpd.GeoDataFrame, pd.DataFrame], df_depreg: pd.DataFrame,
                                   df_department_col: str = 'depPrD') -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """This method can be used to add the region data to e.g. the traffic DataFrame. The matching is done based on
        the department number."""
        df = df.copy(deep=True)
        df = df.merge(df_depreg, how='inner', left_on=df_department_col, right_on='num_dep')
        useless_columns = ['num_dep', 'dep_name']
        for column in useless_columns:
            try:
                df = df.drop(columns=[column])
            except KeyError:
                pass
        return df

    @staticmethod
    def add_region_shapes(df: Union[gpd.GeoDataFrame, pd.DataFrame], df_regions: gpd.GeoDataFrame,
                          df_region_col: str = 'region_name') -> gpd.GeoDataFrame:
        """This method adds a column with the region shapes based on the regions defined in df_region_col. It is assumed
        that these names match up with the values in the column nomnewregi in the regions DataFrame."""
        df = df.copy(deep=True)
        df_regions = df_regions.copy(deep=True)

        # Get the type of the input DataFrame
        df_type = type(df)
        # Add the region shapes to df
        df = df.merge(df_regions.rename(columns={'geometry': 'region_geometry'}).set_geometry('region_geometry'),
                      left_on=df_region_col, right_on='nomnewregi', how='inner')
        if df_type == pd.DataFrame:
            df = gpd.GeoDataFrame(df, geometry='region_geometry', crs=df_regions.crs)

        # Since we are only interested in the shapes from the regions DataFrame we can get rid of the other column that
        # came with the regions DataFrame during the merge
        useless_columns = ['id_newregi', 'nomnewregi']
        for column in useless_columns:
            try:
                df = df.drop(columns=[column])
            except KeyError:
                pass
        return df

    @staticmethod
    def add_road_section_length_per_region(df: gpd.GeoDataFrame,
                                           df_regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method returns what length of the road section of a record is in what region. It is important to note
        that the provided lengths in the column 'longueur' deviate from the shape lengths.
        df is assumed to contain the following columns: 'geometry'
        df_regions is assumed to contain the following columns: 'nomnewregi', 'geometry'"""
        df = df.copy(deep=True)
        df_regions = df_regions.copy(deep=True)
        # We create a GeoSeries with the region name as index and the corresponding geometry as value
        r_with_shapes = df_regions.set_index('nomnewregi')['geometry']
        # We loop over all regions, take the intersection between a road section and a region and fill its length into
        # a new column with the region name and '_length' as name
        for index, value in r_with_shapes.items():
            df[index + '_length'] = df['geometry'].intersection(value).length
        return df

    @staticmethod
    def add_buffer_traffic(df_traffic: gpd.GeoDataFrame,
                           buffer_size: Union[int, float] = 10000,
                           traffic_column: str = 'weighted_traffic') -> gpd.GeoDataFrame:
        """This method adds a column containing the sum of the traffic within the buffer. It is recommended to use
        'weighted_traffic' as the traffic column (which can be created with add_weighted_traffic) to not give
        unreasonable weight to many small road sections.
        It is assumed that the geometry in df_traffic is contained in the column 'geometry'."""
        df_traffic = df_traffic.copy(deep=True)
        buffer_shapes = df_traffic.buffer(distance=buffer_size)
        for index, geometry in df_traffic['geometry'].items():
            mask = buffer_shapes.intersects(geometry)
            buffer_traffic = df_traffic.loc[mask, traffic_column].sum()
            df_traffic.loc[index, 'buffer_traffic'] = buffer_traffic
        return df_traffic

    @staticmethod
    def add_buffer_distance(df_traffic: gpd.GeoDataFrame,
                            buffer_size: Union[int, float] = 10000,
                            traffic_column: str = 'h2_truck_tmja') -> gpd.GeoDataFrame:
        """This method adds a column to df_traffic that contains how many km are covered by all trucks inside the
        buffer_distance around the centroid of a road section."""
        df_traffic = df_traffic.copy(deep=True)
        for index, area in df_traffic['geometry'].centroid.buffer(distance=buffer_size).items():
            df_traffic.loc[index, 'buffer_distance'] = (df_traffic.intersection(area).length / 1000
                                                        * df_traffic[traffic_column]).sum()
        return df_traffic

    @classmethod
    def add_h2_truck_traffic(cls, df_traffic: gpd.GeoDataFrame, n_trucks: int = 10000,
                             data_year: int = 2018, target_year: int = 2030) -> gpd.GeoDataFrame:
        """This method adds a column containing how many H2 trucks pass a road section per day. This is done by finding
        what portion of the truck traffic n_trucks corresponds to and then multiplying 'tmja' * 'pctPL' with that value
        to create 'h2_truck_tmja'."""
        if target_year < data_year:
            raise ValueError("target_year can't be smaller than data_year.")
        df_traffic = df_traffic.copy(deep=True)
        # Truck traffic is predicted to grow by 1.4 percent annually: (Source: https://www.ecologie.gouv.fr/sites/
        # default/files/Th%C3%A9ma%20-%20Projections%20de%20la%20demande%20de%20transport%20sur%20le%20long%20terme.pdf)
        truck_traffic_growth_rate = 0.014
        truck_traffic_multiplier = (1 + truck_traffic_growth_rate) ** (target_year - data_year)
        # We calculate what distance the n_trucks H2 trucks are assumed to cover, then we calculate how much distance
        # all trucks are assumed to cover in target_year. Since both distances are assumed to be from target_year, we
        # can see what percentage the H2 trucks are making up.
        h2_trucks_portion = (cls.calculate_h2_distance(n_trucks=n_trucks, range_in_km=400)
                             / (df_traffic['tmja']
                                * df_traffic['pctPL']
                                * truck_traffic_multiplier
                                * df_traffic['geometry'].length / 1000).sum())
        # For 'h2_truck_tmja' we are simply applying the same percentage to the current traffic values. Of course it
        # would make sense to project 'truck_tmja' as well instead of using 'truck_tmja' unchanged but hey ...
        df_traffic['h2_truck_tmja'] = df_traffic['tmja'] * df_traffic['pctPL'] * h2_trucks_portion
        return df_traffic

    @staticmethod
    def add_distance_to_nearest_logistic_hub(df_traffic: gpd.GeoDataFrame,
                                             df_logistic_hubs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method calculates the distance in meters between each road section and its nearest logistic hub. The
        distance is saved in new column called 'distance_to_next_hub'."""
        df_traffic = df_traffic.copy(deep=True)
        df_traffic['distance_to_next_hub'] = df_traffic.distance(df_logistic_hubs.unary_union)
        # To get the distance to the center of a logistc hub (as previously), simply uncomment the following line
        # df_traffic['distance_to_next_hub'] = df_traffic.distance(df_logistic_hubs.centroid.unary_union)
        return df_traffic

    @staticmethod
    def add_distance_to_point(df_traffic: gpd.GeoDataFrame, point: shapely.Point) -> gpd.GeoDataFrame:
        """This method calculates the distance between each road section and point. The distance is saved in a new
        column called 'distance_to_point'."""
        df_traffic = df_traffic.copy(deep=True)
        df_traffic['distance_to_point'] = df_traffic.distance(point)
        return df_traffic

    @staticmethod
    def add_distance_to_nearest_body_of_water(df_traffic: gpd.GeoDataFrame,
                                              df_water_body: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method adds a column called 'distance_to_water' to df_traffic, containing the distance of reach road
        section to next body of water. This is a very expensive function: takes between 2 and 3 min to run."""
        df_traffic = df_traffic.copy(deep=True)
        # Initialize the column with a first distance measurement
        df_traffic['distance_to_water'] = df_traffic.distance(df_water_body.unary_union)
        # To get the distance to the center of a body of water (much faster), simply uncomment the following line
        # df_traffic['distance_to_water'] = df_traffic.distance(df_water_body.centroid.unary_union)
        return df_traffic

    @staticmethod
    def calculate_h2_distance(n_trucks: int, range_in_km: Union[int, float] = 400) -> Union[int, float]:
        """This method calculates how much distance n_trucks are assumed to cover in a day. range_in_km determines how
        far a truck is assumed to travel.
        We make several assumptions here based on the truck driving regulation of France:
        Based on the regulation, one driver may, on average, not drive more than 45 hours a week and at most 10 hours in
        any given day, however, we assume that the trucks are owned by the companies -> they can change drivers once a
        driver arrived at his/her destination.
        The average driving speed for a truck is assumed to be 66.9 km/h which based on the information from
        the provided H2 truck manufacturers should not be a problem (source for the average truck speed:
        https://www.cnr.fr/download/file/publications/eld%202012.pdf, source for the max speed of H2 trucks:
        https://www.youtube.com/watch?v=ShgYjFb4Pp8).
        The radius of France is roughly 419 km -> most routes can be be completed within one shift
        (https://www.ecologie.gouv.fr/temps-travail-des-conducteurs-routiers-transport-marchandises).
        The average loading time is assumed ot be 3 hours, the average break time, which has to be taken every
        4.5 hours, is 45 min, if a tank does not last for the 4.5 hours, an additional 15 minutes are added for
        refueling which is otherwise assumed to be done during the break."""
        max_daily_driving_hours = 10
        avg_speed_in_km = 66.9
        avg_route_length_in_km = 419.
        avg_route_length_in_km = min(avg_route_length_in_km, max_daily_driving_hours * avg_speed_in_km)
        max_uninterrupted_driving_time_in_hours = 4.5
        # After max_uninterrupted_driving_time_in_hours a driver has to take a 45 min (or 0.75 hour) break
        avg_break_time_in_hours = (avg_route_length_in_km / avg_speed_in_km
                                   / max_uninterrupted_driving_time_in_hours * 0.75)
        # If the range is smaller than the distance covered within one interrupted driving session, extra refueling
        # breaks have to be added. Such a break is assumed to last 15 min.
        extra_breaks = np.ceil((max_uninterrupted_driving_time_in_hours * avg_speed_in_km / range_in_km) - 1)
        avg_break_time_in_hours += 15 * extra_breaks

        avg_loading_time_in_hours = 3
        avg_driving_time = avg_route_length_in_km / avg_speed_in_km

        shifts_per_day = 24 / (avg_driving_time + avg_break_time_in_hours + avg_loading_time_in_hours)
        h2_distance_in_km_per_day = n_trucks * avg_route_length_in_km * shifts_per_day
        return h2_distance_in_km_per_day

    @staticmethod
    def calculate_total_distance_covered(df_traffic: gpd.GeoDataFrame,
                                         traffic_column: str = 'truck_tmja') -> Union[int, float]:
        """This method calculates the total distance covered by all trucks in df_traffic (in km). It is assumed that
        the records contain mutually exclusive road sections."""
        return (df_traffic[traffic_column] * df_traffic['geometry'].length / 1000).sum()

    @staticmethod
    def calculate_distance_covered_by_area(df_traffic: gpd.GeoDataFrame,
                                           area: shapely.Polygon,
                                           traffic_column: str = 'truck_tmja') -> Union[int, float]:
        """This method calculates the total distance covered by all trucks inside area in km. It is assumed that
        the records contain mutually exclusive road sections."""
        return (df_traffic[traffic_column] * df_traffic.intersection(area).length / 1000).sum()

    @staticmethod
    def calculate_refuelings_per_day(h2_distance: Union[int, float], range_in_km: Union[int, float],
                                     min_fuel_level: float = 0.2) -> Union[int, float]:
        """This method calculated how many refuelings are necessary to cover h2_distance. range_in_km defines how far a
        truck can drive on a full tank and min_fuel_level determines at what tank level a truck will be refueled."""
        return h2_distance / ((1 - min_fuel_level) * range_in_km)

    @staticmethod
    def calculate_fuel_from_km(h2_distance_in_km: Union[int, float],
                               h2_in_kg_per_km: Union[int, float] = 0.08) -> Union[int, float]:
        """This method calculates how many kg of H2 are needed to cover h2_distance assuming an H2 consumption per km of
        h2_per_km."""
        return h2_distance_in_km * h2_in_kg_per_km

    @staticmethod
    def calculate_km_from_fuel(h2_in_kg: Union[int, float],
                               h2_in_kg_per_km: Union[int, float] = 0.08) -> Union[int, float]:
        """This method calculates how many km can be driven with h2_in_kg kilos of H2."""
        return h2_in_kg / h2_in_kg_per_km

    @staticmethod
    def translate_regions_regions_to_official_names(df_regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method translates the region names from the regions DataFrame to names as they are found online and in
        the depreg DataFrame."""
        name_mapping = dict()
        name_mapping['Alsace, Champagne-Ardenne et Lorraine'] = 'Grand Est'
        name_mapping['Aquitaine, Limousin et Poitou-Charentes'] = 'Nouvelle-Aquitaine'
        name_mapping['Auvergne et RhÃ´ne-Alpes'] = 'Auvergne-Rhône-Alpes'
        name_mapping['Basse-Normandie et Haute-Normandie'] = 'Normandie'
        name_mapping['Bourgogne et Franche-ComtÃ©'] = 'Bourgogne-Franche-Comté'
        name_mapping['Bretagne'] = 'Bretagne'
        name_mapping['Centre'] = 'Centre-Val de Loire'
        name_mapping['Corse'] = 'Corse'
        name_mapping['Ile-de-France'] = 'Île-de-France'
        name_mapping['Languedoc-Roussillon et Midi-PyrÃ©nÃ©es'] = 'Occitanie'
        name_mapping['Nord-Pas-de-Calais et Picardie'] = 'Hauts-de-France'
        name_mapping['Pays de la Loire'] = 'Pays de la Loire'
        name_mapping["Provence-Alpes-CÃ´te d'Azur"] = "Provence-Alpes-Côte d'Azur"
        df = df_regions.copy(deep=True)
        df.loc[:, 'nomnewregi'] = df.loc[:, 'nomnewregi'].apply(lambda x: name_mapping[x])
        return df

    @staticmethod
    def translate_depreg_regions_to_regions_regions(df_depreg: pd.DataFrame) -> pd.DataFrame:
        """This method maps the region names as they are found in the depreg DataFrame to the names as they are found in
        the regions DataFrame."""
        name_mapping = dict()
        name_mapping['Auvergne-Rhône-Alpes'] = 'Auvergne et RhÃ´ne-Alpes'
        name_mapping['Bourgogne-Franche-Comté'] = 'Bourgogne et Franche-ComtÃ©'
        name_mapping['Bretagne'] = 'Bretagne'
        name_mapping['Centre-Val de Loire'] = 'Centre'
        name_mapping['Corse'] = 'Corse'
        name_mapping['Grand Est'] = 'Alsace, Champagne-Ardenne et Lorraine'
        name_mapping['Guadeloupe'] = np.nan
        name_mapping['Guyane'] = np.nan
        name_mapping['Hauts-de-France'] = 'Nord-Pas-de-Calais et Picardie'
        name_mapping['La Réunion'] = np.nan
        name_mapping['Martinique'] = np.nan
        name_mapping['Mayotte'] = np.nan
        name_mapping['Normandie'] = 'Basse-Normandie et Haute-Normandie'
        name_mapping['Nouvelle-Aquitaine'] = 'Aquitaine, Limousin et Poitou-Charentes'
        name_mapping['Occitanie'] = 'Languedoc-Roussillon et Midi-PyrÃ©nÃ©es'
        name_mapping['Pays de la Loire'] = 'Pays de la Loire'
        name_mapping["Provence-Alpes-Côte d'Azur"] = "Provence-Alpes-CÃ´te d'Azur"
        name_mapping['Île-de-France'] = 'Ile-de-France'
        df = df_depreg.copy(deep=True)
        df.loc[:, 'region_name'] = df.loc[:, 'region_name'].apply(lambda x: name_mapping[x])
        return df

    @staticmethod
    def convert_pd_stations_to_gpd(df_stations: pd.DataFrame, target_crs: str = 'EPSG:2154') -> gpd.GeoDataFrame:
        """This method creates a GeoDataFrame from the stations DataFrame."""
        df_stations = df_stations.copy(deep=True)
        # Cleaning the coordinates column
        df_stations['Coordinates'] = df_stations['Coordinates'].str.replace(',,', ',')
        # Creating the column with shapely Points
        df_stations[['y', 'x']] = df_stations['Coordinates'].str.split(',', n=1, expand=True)
        df_stations['geometry'] = gpd.points_from_xy(x=df_stations['x'], y=df_stations['y'])
        # Converting the DataFrame into a GeoDataFrame
        gdf_stations = gpd.GeoDataFrame(df_stations, geometry='geometry', crs='EPSG:4326')
        # Converting the crs to the target_crs
        gdf_stations = gdf_stations.to_crs(crs=target_crs)
        # Remove the columns we just created
        gdf_stations = gdf_stations.drop(columns=['y', 'x'])
        return gdf_stations

    @staticmethod
    def limit_gdfs_to_main_land_france(df: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method filters out records from df that contain geometry that is not inside main land France."""
        df = df.copy(deep=True)
        df_regions = df_regions.copy(deep=True)

        main_land = df_regions[df_regions['nomnewregi'] != 'Corse']
        mask = df.intersects(main_land.unary_union)
        return df[mask]

    @staticmethod
    def get_nearest_point_on_road(df_traffic: gpd.GeoDataFrame, point: shapely.Point) -> shapely.Point:
        """This method returns the location on a road that is closest to point."""
        closest_road_index = df_traffic.distance(point).argmin()
        closest_road = df_traffic['geometry'].iloc[closest_road_index]
        point_on_road = closest_road.interpolate(closest_road.project(point))
        return point_on_road

    @classmethod
    def get_daily_production_of_h2_in_kg(cls, plant_type: str) -> float:
        """This method returns the daily output of an H2 production plant of type plant_type in kg.
        plant_type can only be 'small' or 'large'."""
        production_plants_hourly_energy_usage = {'small': 5000, 'large': 10 ** 5}
        production_plant_kw_per_kg_of_h2 = {'small': 55, 'large': 50}
        kg_per_hour = cls._convert_kw_to_kg_of_h2(energy=production_plants_hourly_energy_usage[plant_type],
                                                  kw_per_kg_of_h2=production_plant_kw_per_kg_of_h2[plant_type])
        return kg_per_hour * 24

    @staticmethod
    def change_traffic_to_target_year(df_traffic: gpd.GeoDataFrame, data_year: int = 2018,
                                      target_year: int = 2030) -> gpd.GeoDataFrame:
        """This method changes the column 'tmja' in df_traffic to values of 2030. Concretely, we are assuming an annual
        truck traffic growth rate of 1.4 %."""
        # The annual truck traffic growth rate is predicted to be 1.4% (Source: https://www.ecologie.gouv.fr/sites/
        # default/files/Th%C3%A9ma%20-%20Projections%20de%20la%20demande%20de%20transport%20sur%20le%20long%20terme.pdf)
        df_traffic = df_traffic.copy(deep=True)
        truck_traffic_growth_rate = 0.014
        truck_traffic_multiplier = (1 + truck_traffic_growth_rate) ** (target_year - data_year)
        df_traffic['tmja'] = df_traffic['tmja'] * truck_traffic_multiplier
        return df_traffic

    @staticmethod
    def _convert_kw_to_kg_of_h2(energy: int, kw_per_kg_of_h2: int) -> float:
        """This method converts KW of energy to the equivalent in kg of H2."""
        return energy / kw_per_kg_of_h2

    @staticmethod
    def _check_exclusivity_of_road_sections(df_traffic: gpd.GeoDataFrame, threshold: Union[int, float] = 1,
                                            geometry_column: str = 'geometry') -> bool:
        """This method checks if all the records in df_traffic have mutually exclusive road sections. threshold defines
        beyond how many meters of overlap a road section is not considered as mutually exclusive."""
        for index, value in df_traffic[geometry_column]:
            intersection = df_traffic[geometry_column].intersection(value).length.sum()
            if (intersection - value.length) > threshold:
                print(f'{index} has more than {threshold} meter(s) of overlap with other road sections.')
                return False
        return True

    @classmethod
    def _find_root(cls, starting_dir: str = '.') -> str:
        """This method finds the relative location of h2_station_distributor in the current working directory. This is
        need for the find_file and find_folder methods."""
        while 'h2_station_distributor' not in os.listdir(starting_dir):
            starting_dir += './.'
            cls._find_root(starting_dir=starting_dir)
        return starting_dir

    @classmethod
    def _create_cleaned_water_bodies_file_from_original_file(cls, df_water_bodies: gpd.GeoDataFrame,
                                                             df_regions: gpd.GeoDataFrame,
                                                             save: bool = True) -> gpd.GeoDataFrame:
        """This method can be used to create the """
        # Limiting the data to France and only the relevant columns
        relevant_columns = ['nameTxtInt', 'nameText', 'spZoneType', 'geometry']
        df_water_bodies = df_water_bodies.loc[df_water_bodies['country'] == 'FR', relevant_columns]
        # Converting it to the correct coordinate reference system (crs)
        df_water_bodies = df_water_bodies.to_crs(df_regions.crs)
        # Limiting the data to mainland France
        df_water_bodies = cls.limit_gdfs_to_main_land_france(df=df_water_bodies, df_regions=df_regions)
        df_water_bodies = df_water_bodies.reset_index(drop=True)
        if save:
            data_folder = cls.find_folder('raw')
            target_folder = f'{data_folder}/water_bodies'
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            df_water_bodies.to_file(f'{target_folder}/water_bodies.shp')
        return df_water_bodies


class DownloadData:
    raw_data_folder = 'data/raw'

    @classmethod
    def download_zip_file(cls, url: str, target_folder: str) -> None:
        """This method downloads a zip file from url and unpacks it to data/target_folder."""
        zip_file = requests.get(url=url).content
        zipfile.ZipFile(io.BytesIO(zip_file)).extractall(
            path=f'{Data.find_folder(cls.raw_data_folder)}/{target_folder}')

    @classmethod
    def download_csv_file(cls, url: str, target_file: str) -> None:
        """This method downloads a csv file from url and saves it to data/target_file."""
        csv_file = requests.get(url=url).content
        with open(f'{Data.find_folder(cls.raw_data_folder)}/{target_file}', 'wb') as f:
            f.write(csv_file)

    @classmethod
    def download_rrn_files(cls, shape_files: bool = True, csv_file: bool = False) -> None:
        """This method downloads the 'liaisons du réseau routier national' information from data.gouv.fr."""
        if shape_files:
            zip_url = 'https://www.data.gouv.fr/fr/datasets/r/92d86944-52e8-44c1-b4cc-b17ac82d70ed'
            cls.download_zip_file(url=zip_url, target_folder='RRN')

        if csv_file:
            csv_url = 'https://www.data.gouv.fr/fr/datasets/r/89f8bc85-f923-4540-bbc0-6e010b9b6339'
            cls.download_csv_file(url=csv_url, target_file='routes-2022.csv')

    @classmethod
    def download_traffic_files(cls, shape_files: bool = True, csv_file: bool = False) -> None:
        if shape_files:
            zip_url = 'https://www.data.gouv.fr/fr/datasets/r/bf95beb9-08e8-4bd1-a734-3d9a66b2caff'
            cls.download_zip_file(url=zip_url, target_folder='traffic')

        if csv_file:
            csv_url = 'https://www.data.gouv.fr/fr/datasets/r/d5d894b4-b58d-440c-821b-c574e9d6b175'
            cls.download_csv_file(url=csv_url, target_file='tmja-2019.csv')

    @classmethod
    def download_regions_files(cls) -> None:
        """The regions data comes from the following website: """
        zip_url = 'https://www.data.gouv.fr/fr/datasets/r/5e7b3100-80c3-48ae-9c9f-57b7977e7a69'
        cls.download_zip_file(url=zip_url, target_folder='regions')
        # The files are downloaded with an é in the name, but the original files are with an e -> we change the names
        regions_folder = Data.find_folder('regions')
        for file in os.listdir(regions_folder):
            os.rename(f'{regions_folder}/{file}', f'{regions_folder}/{file.replace("é", "e")}')

    @classmethod
    def download_aires_pl(cls) -> None:
        zip_url = 'https://www.statistiques.developpement-durable.gouv.fr/sites/default/files/2018-11/' \
                  'aire-logistiques-donnees-detaillees.zip'
        cls.download_zip_file(url=zip_url, target_folder='airesPL')

    @classmethod
    def download_stations(cls) -> None:
        """This method downloads the csv file with the existing fuel stations. Since we can't download the file directly
        from a link, we have to scrape with with selenium. It as therefore assumed that a geckodriver executable is
        installed in directory not higher than the home directory. If that's not the case download the fitting version
        from: https://github.com/mozilla/geckodriver/releases"""
        data_folder_path = Data.find_folder(cls.raw_data_folder)
        # Opening a browser
        options = Options()
        options.add_argument('--headless')
        options.set_preference('browser.download.dir', data_folder_path)
        options.set_preference('browser.download.folderList', 2)
        executable_path = Data.find_file('../geckodriver')
        driver = webdriver.Firefox(service=Service(executable_path=executable_path), options=options)
        # Loading the Google spreadsheet page
        google_doc_url = 'https://docs.google.com/spreadsheets/d/1TYjPlSC0M2VTDPkQHqLrshnkItETbSwl/edit#gid=855090481'
        driver.get(google_doc_url)
        # Clicking through the navigation to download the spreadsheet as a .csv file
        # First we click on file
        file_element = driver.find_element(By.ID, 'docs-file-menu')
        file_element.click()
        # Then we click on download
        download_element = driver.find_element(By.CSS_SELECTOR,
                                               '.docs-icon-editors-ia-download').find_element(By.XPATH, '..')
        download_element.click()
        # Finally we select the .csv as download file type
        for element in driver.find_elements(By.CLASS_NAME, 'goog-menuitem-label'):
            try:
                if 'csv' in element.get_attribute('aria-label').lower():
                    element.click()
                    # We have to wait a little, otherwise the download won't be triggered
                    time.sleep(1)
                    # Now we can close the driver again
                    driver.quit()
                    break
            except (AttributeError, StaleElementReferenceException):
                pass

        driver.quit()
        # Renaming the downloaded file to stations.csv
        downloaded_name = 'Données de stations TE_DV.xlsx - export_data_te.csv'
        if downloaded_name in os.listdir(data_folder_path):
            os.rename(f'{data_folder_path}/{downloaded_name}', f'{data_folder_path}/stations.csv')
        else:
            raise ValueError('The stations.csv file could not be downloaded')

    @classmethod
    def download_depreg(cls) -> None:
        csv_url = 'https://www.data.gouv.fr/fr/datasets/r/987227fb-dcb2-429e-96af-8979f97c9c84'
        cls.download_csv_file(url=csv_url, target_file='depreg.csv')

    @classmethod
    def download_water_bodies(cls) -> None:
        zip_url = 'https://www.eea.europa.eu/data-and-maps/data/wise-wfd-spatial/surface-water-body/shapefile-2016'
        cls.download_zip_file(url=zip_url, target_folder='airesPL')


class Strategies:
    @classmethod
    def split_into_n_areas(cls, df_traffic: gpd.GeoDataFrame, n_areas: int, rank_column: str = 'buffer_distance',
                           initial_buffer: Union[int, float] = 5000,
                           buffer_step: Union[int, float] = 5000) -> gpd.GeoDataFrame:
        """This method creates n_areas areas that each contain at least 1 / n_areas * total distance covered by trucks
        in all of France. The output DataFrame will have n_areas new columns, each telling which records belong to
        an area. One record can belong to multiple areas.
        The downside of this method is that some records (=road sections) do not belong to any area, especially the
        sections close to the border are affected by this.
        It is assumed that the geometry in df_traffic is stored in a column called 'geometry'."""
        df_traffic = df_traffic.copy(deep=True)
        # We will later need the DataFrame in sorted order over and over again
        df_traffic = df_traffic.sort_values(by=rank_column, ascending=False)
        # Calculate how much distance is to cover by all the stations (=areas)
        total_distance = Data.calculate_total_distance_covered(df_traffic, traffic_column='h2_truck_tmja')
        # Calculate how much distance is to cover by one station (=area)
        distance_per_area = total_distance / n_areas
        # Initialize variable to keep track of the number of created areas
        created_areas = 0
        # Initialize variable to keep track of the smallest buffer which will determine the distance from the covered
        # areas
        min_buffer = 10 ** 6
        # Define the starting point for the first area
        reference = df_traffic[rank_column].argmax()
        # Initialize shape to keep track of all the covered areas
        covered_area = df_traffic.loc[reference, 'geometry'].centroid.buffer(1)
        # Define a mask that will make sure that we exclude areas we have already covered
        mask = pd.Series(index=df_traffic.index)
        mask.loc[:] = False
        # Create a variable to which we will add all created areas
        df_areas = gpd.GeoDataFrame(columns=['priority', 'geometry'], geometry='geometry', crs='EPSG:2154')
        # Create areas until we have reached the desired number
        while created_areas < n_areas:
            # Size the area to cover distance_per_area
            buffer, used_records_mask, _, area = cls._size_buffer_for_distance(df_traffic=df_traffic,
                                                                               index=reference,
                                                                               target_distance=distance_per_area,
                                                                               initial_buffer=initial_buffer,
                                                                               buffer_step=buffer_step,
                                                                               step_direction='up')
            # Fill the DataFrame with the results
            df_areas.loc[created_areas, 'geometry'] = area
            df_areas.loc[created_areas, 'priority'] = created_areas + 1
            # Create a shape of all covered areas
            covered_area = covered_area.union(area)
            # Alter the geometry of the reference road section to only include parts that have not been covered yet. It
            # is important to note that the removed parts are also not considered in the sizing of a different point
            df_traffic.loc[reference, 'geometry'] = df_traffic.loc[reference, 'geometry'].difference(covered_area)
            # Create a mask to only use road parts that aren't in covered_area as starting point
            if buffer < min_buffer:
                min_buffer = buffer
            mask = df_traffic.difference(covered_area.buffer(min_buffer)).length > 0
            # Create a new starting point
            reference = df_traffic.loc[mask, rank_column].index[0]
            created_areas += 1
            # Inform on the progress
            if created_areas % 50 == 0:
                print(f'Created areas: {created_areas}, '
                      f'open records: {mask.sum()}, '
                      f'centroid: {area.centroid}, '
                      f'last buffer size: {buffer}, '
                      f'min buffer size: {min_buffer}')
        return df_areas

    @classmethod
    def split_into_areas_fuel_based(cls, df_traffic: gpd.GeoDataFrame, h2_in_kg: Union[int, float],
                                    rank_column: str = 'buffer_distance',
                                    buffer_step: Union[int, float] = 10000) -> gpd.GeoDataFrame:
        """This method creates areas based on the provided station size (in kg of H2 they can refuel) and the number of
        H2 trucks on the road. It is important to note that it is assumed that the provided df_traffic has the traffic
        number from 2030, since the distance_to_cover we are calculating is based on the number of trucks in 2030 as
        well as the range of trucks in 2030.
        station_size is in kg of H2 a station stores"""
        df_traffic = df_traffic.copy(deep=True)
        # Calculate the total distance to cover
        distance_to_cover = Data.calculate_total_distance_covered(df_traffic, traffic_column='h2_truck_tmja')
        # Calculate how much distance one station covers
        distance_covered_by_fuel = Data.calculate_km_from_fuel(h2_in_kg)
        # Initialize variables to keep track of the distance covered by all created areas (=stations)
        total_covered_distance = 0
        created_areas = 0
        # Define a starting point
        reference = df_traffic[rank_column].argmax()
        mask = pd.Series(index=df_traffic.index)
        mask.loc[:] = False
        mask = mask.astype(bool)
        # Create a DataFrame to store the created geometries in
        df_areas = gpd.GeoDataFrame(columns=['priority', 'geometry'], geometry='geometry', crs='EPSG:2154')
        # Create new stations until they all together cover distance_to_cover
        while total_covered_distance < distance_to_cover:
            # Get the buffer size that will cover a bit more than the distance a station can cover
            buffer_size, _, _, _ = cls._size_buffer_for_distance(df_traffic=df_traffic,
                                                                 index=reference,
                                                                 target_distance=distance_covered_by_fuel,
                                                                 buffer_step=buffer_step)
            # Reduce that buffer until a station can cover it
            _, used_records_mask, covered_distance, area = \
                cls._size_buffer_for_distance(df_traffic=df_traffic,
                                              index=reference,
                                              target_distance=distance_covered_by_fuel,
                                              initial_buffer=buffer_size,
                                              buffer_step=1000,
                                              step_direction='down')
            total_covered_distance += covered_distance
            # Fill the DataFrame with the results
            df_areas.loc[created_areas, 'geometry'] = area
            df_areas.loc[created_areas, 'priority'] = created_areas + 1
            # We update the mask such that a road section can have at most one station
            mask += used_records_mask
            # Get a new starting point
            reference = df_traffic.loc[~mask, rank_column].sort_values(ascending=False).index[0]
            created_areas += 1
            # Inform on the progress
            if created_areas % 50 == 0:
                print(f'Created areas: {created_areas}, '
                      f'open records: {(~mask).sum()}')
        return df_areas

    @staticmethod
    def find_closest_station_in_area(area: shapely.Polygon, df_stations: gpd.GeoDataFrame) -> tuple[str, shapely.Point]:
        """This method returns the location of the stations closest to the center of area."""
        centroid = area.centroid
        closest_station = df_stations['Station de service'].iloc[df_stations.distance(centroid).argmin()]
        closest_station_location = df_stations['geometry'].iloc[df_stations.distance(centroid).argmin()]
        return closest_station, closest_station_location

    @classmethod
    def create_h2_station_distribution_from_areas(cls, df_areas: gpd.GeoDataFrame,
                                                  df_with_hub_distance: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method creates a GeoDataFrame with the location of the suggested station for each area."""
        df = gpd.GeoDataFrame(columns=['priority', 'geometry'], geometry='geometry', crs='EPSG:2154')
        # We loop over all areas to find the best location for a station
        for index, (area_index, area) in enumerate(df_areas['geometry'].items()):
            # We only consider road sections that are within that area
            mask = df_with_hub_distance.intersects(area)
            df_temp = df_with_hub_distance[mask].copy(deep=True)
            df_temp['geometry'] = df_temp.intersection(area)
            # We calculate the score for each road section inside area
            scores = cls._add_station_score_for_point(df_temp, area.centroid)
            # Extract the best road section
            best_score_index = scores['score'].sort_values().index[0]
            # Get the closest point to the optimal location on that road section
            best_location = Data.get_nearest_point_on_road(df_traffic=df_temp.loc[[best_score_index], :],
                                                           point=area.centroid)
            df.loc[index, 'geometry'] = best_location
            df.loc[index, 'priority'] = df_areas.loc[area_index, 'priority']
        return df

    @classmethod
    def create_h2_plant_distribution_from_areas(cls, df_areas: gpd.GeoDataFrame,
                                                df_hubs_water: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method creates a GeoDataFrame with the location of the suggested plant for each area."""
        df = gpd.GeoDataFrame(columns=['priority', 'geometry'], geometry='geometry', crs='EPSG:2154')
        # We loop over all areas to find the best location for a station
        for index, (area_index, area) in enumerate(df_areas['geometry'].items()):
            # We only consider road sections that are within that area
            mask = df_hubs_water.intersects(area)
            df_temp = df_hubs_water[mask].copy(deep=True)
            df_temp['geometry'] = df_temp.intersection(area)
            # We calculate the score for each road section inside area
            scores = cls._add_plant_score_for_point(df_temp, area.centroid)
            # Extract the best road section
            best_score_index = scores['score'].sort_values().index[0]
            # Get the closest point to the optimal location on that road section
            best_location = Data.get_nearest_point_on_road(df_traffic=df_temp.loc[[best_score_index], :],
                                                           point=area.centroid)
            df.loc[index, 'geometry'] = best_location
            df.loc[index, 'priority'] = df_areas.loc[area_index, 'priority']
        return df

    @staticmethod
    def _add_station_score_for_point(df_next_hub: gpd.GeoDataFrame, point: shapely.Point) -> gpd.GeoDataFrame:
        """This method adds a column with the score of each road section, that will then define where a station is
        going to be built. The score is a weighted sum between the distance of a road section to the nearest hub, the
        distance of a road section to point and the negative of the H2 truck traffic on a road section.
        It is assumed that df_next_hub contains a column with the predicted traffic of H2 trucks called 'h2_truck_tmja'
        and a column with the distance to the nearest logistic hub called 'distance_to_next_hub' (which can be created
        with Data.add_distance_to_nearest_logistic_hub(df_traffic, df_logistic_hubs).
        A lower score is BETTER!"""
        # Creating the columns for the distance to the point
        df_next_hub = Data.add_distance_to_point(df_next_hub, point)
        # To give all features the same weight, we normalize the values
        score_columns = ['distance_to_next_hub', 'distance_to_point', 'h2_truck_tmja']
        df_next_hub[score_columns] = df_next_hub[score_columns] / df_next_hub[score_columns].sum()
        w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3
        # Finally we calculate the score as the weighted sum of the three features
        df_next_hub['score'] = (w1 * df_next_hub[score_columns[0]]
                                + w2 * df_next_hub[score_columns[1]]
                                - w3 * df_next_hub[score_columns[2]])
        return df_next_hub

    @staticmethod
    def _add_plant_score_for_point(df_hubs_water: gpd.GeoDataFrame, point: shapely.Point) -> gpd.GeoDataFrame:
        """This method adds a column with the score of each road section, that will then define where a plant is
        going to be built. The score is a weighted sum between the distance of a road section to the nearest body of
        water, the distance of a road section to point and the negative of the H2 truck traffic on a road section.
        It is assumed that df_hubs_water contains a column with the predicted traffic of H2 trucks called 'h2_truck_tmja'
        and a column with the distance to the nearest logistic hub called 'distance_to_water'.
        A lower score is BETTER"""
        # Creating the columns for the distance to the point
        df_hubs_water = Data.add_distance_to_point(df_hubs_water, point)
        # To give all features the same weight, we normalize the values
        score_columns = ['distance_to_water', 'distance_to_point', 'h2_truck_tmja']
        df_hubs_water[score_columns] = df_hubs_water[score_columns] / df_hubs_water[score_columns].sum()
        w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3
        # Finally we calculate the score as the weighted sum of the three features
        df_hubs_water['score'] = (w1 * df_hubs_water[score_columns[0]]
                                  + w2 * df_hubs_water[score_columns[1]]
                                  - w3 * df_hubs_water[score_columns[2]])
        return df_hubs_water

    @staticmethod
    def _size_buffer_for_distance(df_traffic: gpd.GeoDataFrame,
                                  index: int,
                                  target_distance: Union[int, float],
                                  initial_buffer: Union[int, float] = 10000,
                                  buffer_step: Union[int, float] = 10000,
                                  step_direction: str = 'up') -> tuple[Union[int, float],
                                                                       pd.Series,
                                                                       Union[int, float],
                                                                       shapely.Polygon]:
        """This method provides the size of the buffer that is needed around the record at index to cover
        target_distance when summing the traffic * road over all road sections inside the buffer.
        target_distance is what that sum is supposed to be
        initial_buffer is the buffer from which we start our calculation
        buffer_step is the step size by which we want to increase/decrease the buffer
        step_direction defines whether we want to increase the buffer from initial_buffer or decrease it"""
        if target_distance <= 0:
            raise ValueError('The target_distance has to be larger than 0.')
        if initial_buffer < 0:
            raise ValueError('The initial_buffer cannot be smaller than 0.')
        if buffer_step <= 0:
            raise ValueError('The buffer_step has to be larger than 0.')
        if step_direction not in ['down', 'up']:
            raise ValueError("The step direction can only be up or down.")
        if step_direction == 'up':
            comparison_operator = '<'
        else:
            comparison_operator = '>'

        buffer = initial_buffer
        area = df_traffic.loc[index, 'geometry'].centroid.buffer(buffer)
        mask = df_traffic.intersects(area)

        covered_distance = Data.calculate_distance_covered_by_area(df_traffic.loc[mask, :],
                                                                   area=area,
                                                                   traffic_column='h2_truck_tmja')
        while eval(f'{covered_distance}{comparison_operator}{target_distance}'):
            # If else conditions are faster than eval
            if step_direction == 'up':
                buffer += buffer_step
            else:
                buffer -= buffer_step
            area = df_traffic.loc[index, 'geometry'].centroid.buffer(buffer)
            mask = df_traffic.intersects(area)

            covered_distance = Data.calculate_distance_covered_by_area(df_traffic.loc[mask, :],
                                                                       area=area,
                                                                       traffic_column='h2_truck_tmja')

        return buffer, mask, covered_distance, area


class Accounting:
    @classmethod
    def calculate_cost_of_station_configuration(cls, stations: dict[str, dict[str, int]]) -> int:
        """This method calculates the total cost of the stations. stations is assumed to have 'small', 'medium' and/or
        'large' as first key and as value a dictionary with 'n_stations' and 'duration' as key and the number of
        stations and the years of operation as values respectively. E.g.:
        {'small': {'n_stations': 2, 'duration': 10},
        'medium': {'n_stations': 5, 'duration': 10}}
        The output will be in euros."""
        dict_for_inital_cost = {station_type: stations[station_type]['n_stations'] for station_type in stations}
        initial_cost = cls._calculate_initial_cost_of_station_configuration(stations=dict_for_inital_cost)
        ongoing_cost = cls._calculate_variable_cost_of_station_configuration(stations=stations)
        return initial_cost + ongoing_cost

    @classmethod
    def calculate_profit_of_station_fuel_based(cls, h2_in_kg: Union[int, float], station_type: str) -> float:
        """This method calculates the profit a station of type station_type makes in a year. h2_in_kg is the average
        amount of H2 the station served in a day, station_type can take the following values: 'small', 'medium' or
        'large'."""
        revenues = h2_in_kg * cls.get_avg_h2_margin_per_kg_over_all_station_types()
        cost = cls._get_annual_cost_for_station_type(station_type=station_type)
        profit = revenues - cost
        return profit

    @classmethod
    def calculate_profit_of_station_distance_based(cls, covered_distance_in_km: Union[int, float],
                                                   station_type: str) -> float:
        """This method calculates the profit a station of type station_type makes in a year. covered_distance_in_km is
        the average daily distance in km covered by all the H2 trucks that are assumed to charge at this station,
        station_type can take the following values: 'small', 'medium'
        or 'large'."""
        served_h2_in_kg = Data.calculate_fuel_from_km(covered_distance_in_km)
        return cls.calculate_profit_of_station_fuel_based(h2_in_kg=served_h2_in_kg, station_type=station_type)

    @classmethod
    def calculate_transportation_cost_to_stations(cls, df_stations: gpd.GeoDataFrame,
                                                  plant_location: shapely.Point, h2_in_kg: Union[int, float]) -> float:
        """This method returns the total tansportation cost to all stations in df_stations, each station gets the same
        amount of H2."""
        distance_in_km = df_stations.difference(plant_location) / 1000
        cost_per_kg_per_km = 0.008
        return (distance_in_km * h2_in_kg * cost_per_kg_per_km).sum()

    @classmethod
    def get_avg_h2_margin_per_kg_over_all_station_types(cls) -> float:
        """This method returns the margin of H2 per kg inferred from the information on the three different station
        types."""
        margin_from_small_station = cls._get_h2_margin_from_station_type('small')
        margin_from_medium_station = cls._get_h2_margin_from_station_type('medium')
        margin_from_large_station = cls._get_h2_margin_from_station_type('large')
        avg_margin = (margin_from_small_station + margin_from_medium_station + margin_from_large_station) / 3
        return avg_margin

    @classmethod
    def get_production_cost_per_kg_for_plant_type(cls, plant_type: str) -> float:
        """This method calculates the price to produce one kg of H2 in euros for a production plant of type plant_type.
        """
        annual_cost = cls._get_annual_cost_for_production_plant_type(plant_type=plant_type)
        daily_production_in_kg = Data.get_daily_production_of_h2_in_kg(plant_type=plant_type)
        cost_per_kg = annual_cost / (daily_production_in_kg * 365)
        return cost_per_kg

    @staticmethod
    def _calculate_h2_transportation_cost(h2_in_kg: Union[int, float], transport_distance_in_km: Union[int, float],
                                          cost_per_kg_per_km: float = 0.008) -> float:
        return h2_in_kg * transport_distance_in_km * cost_per_kg_per_km

    @staticmethod
    def _get_annual_cost_for_station_type(station_type: str) -> float:
        """Tihs method returns the annual cost of a station of type station_type. station_type can only take one of the
        following values: 'small', 'medium' or 'large'."""
        capex = {'small': 3 * 10 ** 6, 'medium': 5 * 10 ** 6, 'large': 8 * 10 ** 6}
        if station_type not in capex:
            raise ValueError("Only 'small', 'medium' and 'large' are allowed as values for station_type.")
        opex = {'small': 0.1 * capex['small'], 'medium': 8 / 100 * capex['medium'], 'large': 7 / 100 * capex['large']}
        depreciation_horizon_in_years = 15
        annual_cost = capex[station_type] / depreciation_horizon_in_years + opex[station_type]
        return annual_cost

    @staticmethod
    def _calculate_initial_cost_of_station_configuration(stations: dict[str, int]) -> int:
        """This method calculates the cost of the initial investment of the configuration provided in stations.
        stations is assumed to have 'small', 'medium' and/or 'large' as key(s) and the number of stations of each type
        as value(s). E.g.:
        {'small': 10, 'medium': 15, 'large': 3}"""
        cost_per_station = {'small': 3 * 10 ** 6, 'medium': 5 * 10 ** 6, 'large': 8 * 10 ** 6}
        if len([station for station in stations if station in cost_per_station]) != len(stations):
            raise ValueError("Only 'small', 'medium' and 'large' are allowed as keys to the stations dictionary.")
        return int(sum([cost_per_station[key] * stations[key] for key in stations]))

    @staticmethod
    def _calculate_variable_cost_of_station_configuration(stations: dict[str, dict[str, int]]):
        """This method calculates the ongoing cost of the stations. stations is assumed to have
        'small', 'medium' and/or 'large' as first key and as value a dictionary with 'n_stations' and 'duration' as key
        and the number of stations and the years of operation as values respectively. E.g.:
        {'small': {'n_stations': 2, 'duration': 10},
        'medium': {'n_stations': 5, 'duration': 10}}"""
        cost_per_year_per_station = {'small': 3 * 10 ** 5,
                                     'medium': 8 / 100 * 5 * 10 ** 6,
                                     'large': 7 / 100 * 8 * 10 ** 6}
        if len([station for station in stations if station in cost_per_year_per_station]) != len(stations):
            raise ValueError("Only 'small', 'medium' and 'large' are allowed as keys to the stations dictionary.")
        total_cost = 0
        for type_key in stations:
            var_cost = cost_per_year_per_station[type_key]
            for key in stations[type_key]:
                var_cost *= stations[type_key][key]
            total_cost += var_cost
        return int(total_cost)

    @classmethod
    def _get_h2_margin_from_station_type(cls, station_type: str) -> float:
        station_info = {'small': cls._infer_h2_margin_form_station_info(0.9, 2000, 3_000_000, 15, 0.1),
                        'medium': cls._infer_h2_margin_form_station_info(0.8, 3000, 5_000_000, 15, 0.08),
                        'large': cls._infer_h2_margin_form_station_info(0.6, 4000, 8_000_000, 15, 0.07)}
        if station_type not in station_info:
            raise ValueError("Only 'small', 'medium' and 'large' are allowed as values for station_type.")
        return station_info[station_type]

    @staticmethod
    def _infer_h2_margin_form_station_info(profitability_threshold: float, storage_capacity: int, capex: int,
                                           years_to_depreciate: int, opex_as_fraction_of_capex) -> float:
        """This method infers the margin of a kg of H2 based on the information they provided on the stations.
        The formula for the breakeven (= 0 profit) is:
            breakeven = profitability_threshold * storage_capacity * 365 * x - capex / years_to_depreciate
                        - opex_as_fraction_of_capex * capex
            x is the margin per kg of H2 we are looking for
            x = (capex / years_to_depreciate + opex_as_fraction_of_capex * capex)
                / (profitability_threshold * storage_capacity * 365)
        It is important to note, that we are making the assumption here, that the cost includes the margin a station has
        to pay for the H2."""
        return ((capex / years_to_depreciate + opex_as_fraction_of_capex * capex)
                / (profitability_threshold * storage_capacity * 365))

    @staticmethod
    def _get_annual_cost_for_production_plant_type(plant_type: str) -> float:
        """Tihs method returns the annual cost of a production plant of type plant_type. plant_type can only take one of
        the two values: 'small' or 'large'."""
        capex = {'small': 2 * 10 ** 7, 'large': 1.2 * 10 ** 8}
        if plant_type not in capex:
            raise ValueError("Only 'small' and 'large' are allowed as values for plant_type.")
        opex = {'small': 0.03 * capex['small'], 'large': 0.03 * capex['large']}
        depreciation_horizon_in_years = 15
        annual_cost = capex[plant_type] / depreciation_horizon_in_years + opex[plant_type]
        return annual_cost


class Plots:
    @staticmethod
    def plot_roads_over_regions(df_traffic: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame) -> plt.Figure:
        """This method plots the road network over a map of France."""
        fig, ax = plt.subplots()
        df_regions.plot(ax=ax)
        df_traffic.plot(ax=ax, color='red')
        ax.set_title('Road network of France')
        return fig

    @staticmethod
    def plot_traffic_per_region(df_traffic: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame, df_depreg) -> plt.Figure:
        """This method maps the total daily traffic per region."""
        t_with_r = Data.add_regions_to_departments(df_traffic, df_depreg)
        t_with_r_and_s = Data.add_region_shapes(t_with_r, df_regions).set_geometry('region_geometry')
        plot_data = t_with_r_and_s.groupby('region_name').agg({'tmja': sum, 'region_geometry': 'first'}).reset_index()
        plot_data = gpd.GeoDataFrame(plot_data, geometry='region_geometry', crs=df_regions.crs)

        fig, ax = plt.subplots()
        plot_data.plot(ax=ax, column='tmja', edgecolor='black', legend=True, legend_kwds={'label': 'Traffic per day'})
        ax.set_title('Daily traffic in France')
        return fig

    @classmethod
    def plot_calculated_areas(cls, df_areas: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame,
                              heat_map: bool = False) -> plt.Figure:
        """This method plots the roads on top of France, with a color indicating which area a road section belongs to.
        df_traffic_with_area_assignment is assumed to contain a geometry column as well as one or more boolean columns starting with 'area_'
        telling which record belong to which area (as output by Strategies.split_into_areas()), as well as (optionally)
        a column 'buffer_traffic' (as output by Data.add_buffer_traffic)."""
        warnings.warn('This plotting method has been deprecated since the output split_into_n_areas and'
                      'split_into_areas_fuel_based do not include the required DataFrame anymore.',
                      DeprecationWarning)
        df_areas = df_areas.copy(deep=True)
        # Creating geometries that represent a buffer of 10 km around the road sections
        df_areas['buffer_geometry'] = df_areas.buffer(10000)
        df_areas = df_areas.set_geometry('buffer_geometry')

        fig, ax = plt.subplots()
        # Drawing the outline of the French regions
        df_regions.plot(ax=ax)
        # The following line would have to be uncommented if one wanted to see a heat map of the traffic
        if heat_map:
            df_areas.plot(ax=ax, column='buffer_traffic', legend=True, cmap='hot')

        # Creating list of area columns
        areas = [col for col in df_areas if col.startswith('area_')] + ['no_area']
        # Creating a list of len(areas) columns
        colors = cls._get_colors(len(areas))
        # Drawing the roads with a buffer of 10 km, each with a different color
        for area, color in zip(areas, colors):
            df_areas[df_areas[area]].plot(ax=ax, color=color, alpha=0.5)
        # Adding the legend for the different areas
        custom_handles = cls._create_legend_handles(colors=colors, labels=areas)
        existing_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=[
                *existing_handles,
                *custom_handles],
            title="",
            loc=(1.04, 0.9),
            ncol=1,
            frameon=False,
            shadow=False)
        fig.tight_layout()
        return fig

    @classmethod
    def plot_chosen_stations(cls, df_station_distribution: gpd.GeoDataFrame,
                             df_regions: gpd.GeoDataFrame) -> plt.Figure:
        """This method displays the chosen locations of the H2 stations in France."""
        fig, ax = plt.subplots()
        df_regions.plot(ax=ax)
        df_station_distribution.plot(ax=ax, color='lightgreen', markersize=10)
        ax.set_title('H2 charging stations')
        # Adding the legend for the different areas
        custom_handles = cls._create_legend_handles(colors=['lightgreen'],
                                                    labels=['H2 station'])
        existing_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=[
                *existing_handles,
                *custom_handles],
            title="",
            loc=(1.04, 0.9),
            ncol=1,
            frameon=False,
            shadow=False)
        fig.tight_layout()
        return fig

    @classmethod
    def plot_chosen_stations_over_optimal_locations(cls, df_station_distribution: gpd.GeoDataFrame,
                                                    df_areas: gpd.GeoDataFrame,
                                                    df_regions: gpd.GeoDataFrame) -> plt.Figure:
        """This method plots the stations chosen over the locations picked to put a station (the centers of the areas).
        """
        fig, ax = plt.subplots()
        df_regions.plot(ax=ax)
        df_areas.centroid.plot(ax=ax, color='red', markersize=10)
        df_station_distribution.plot(ax=ax, color='lightgreen', markersize=10)
        ax.set_title('Optimal location vs actual station')
        # Adding the legend for the different areas
        custom_handles = cls._create_legend_handles(colors=['red', 'lightgreen'],
                                                    labels=['Optimal location', 'Nearest station'])
        existing_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=[
                *existing_handles,
                *custom_handles],
            title="",
            loc=(1.04, 0.9),
            ncol=1,
            frameon=False,
            shadow=False)
        fig.tight_layout()
        return fig

    @staticmethod
    def plotly_plot_chosen_stations(df_station_distribution: gpd.GeoDataFrame,
                                    df_regions: gpd.GeoDataFrame) -> go.Figure:
        """This method mirrors plot_chosen_stations, but the plotly library is used instead of matplotlib."""
        # Defining that don't want a background: no grid, no axis
        layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)', xaxis={'visible': False}, yaxis={'visible': False})
        # Setting up a graph object
        fig = go.Figure(layout=layout)
        # Plotting the regions
        for index, geometry in df_regions['geometry'].items():
            # Polygons and MultiPolygons have to be treated differently
            if isinstance(geometry, shapely.Polygon):
                x, y = geometry.exterior.coords.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y),
                                         fill='toself',
                                         line={'color': '#444445', 'width': 1},
                                         fillcolor='rgb(60, 117, 175)',
                                         mode='lines',
                                         text=f"Region: {df_regions.loc[index, 'nomnewregi']}",
                                         hoverinfo='text',
                                         showlegend=False))
            else:
                # MultiPolygons first have to be split up into individual Polygons
                for p in geometry.geoms:
                    x, y = p.exterior.coords.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y),
                                             fill='toself',
                                             line={'color': '#444445', 'width': 1},
                                             fillcolor='rgb(60, 117, 175)',
                                             mode='lines',
                                             text=f"Region: {df_regions.loc[index, 'nomnewregi']}",
                                             hoverinfo='text',
                                             showlegend=False))
        # Plotting the stations
        fig.add_trace(go.Scatter(x=df_station_distribution['geometry'].x, y=df_station_distribution['geometry'].y,
                                 mode='markers',
                                 line_color='rgb(166, 235, 154)',
                                 hovertemplate='Location: %{x}, %{y}<extra></extra>',
                                 # text=df_station_distribution['station'],
                                 showlegend=True,
                                 name='H2 station'))
        # Unsqueezing the plot
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        # Setting the plot title
        fig.update_layout(title={'text': 'H2 charging stations', 'x': 0.5})
        return fig

    @classmethod
    def plotly_plot_chosen_stations_over_optimal_locations(cls, df_station_distribution: gpd.GeoDataFrame,
                                                           df_areas: gpd.GeoDataFrame,
                                                           df_regions: gpd.GeoDataFrame) -> go.Figure:
        """This method mirrors plot_chosen_stations_over_optimal_locations, except that the plotly library is used."""
        # Loading the figure with the chosen stations
        fig = cls.plotly_plot_chosen_stations(df_station_distribution, df_regions)
        # We change the legend entry of the stations
        fig['data'][-1]['name'] = 'Nearest station'
        # Adding a trace with the optimal location (which is assumed to be the center of the areas)
        fig.add_trace(go.Scatter(x=df_areas.centroid.x, y=df_areas.centroid.y,
                                 mode='markers',
                                 line_color='red',
                                 showlegend=True,
                                 name='Optimal location',
                                 hoverinfo='skip'))
        fig.update_layout(title={'text': 'Optimal location vs actual station', 'x': 0.5})
        # Reordering the traces with the actual stations and the optimal locations to have the actual station in front
        # Direct assignment is not possible, we need to first turn it into a list from a tuple (which is unchangeable)
        traces = list(fig['data'])
        optimal_location_trace = traces[-1]
        traces[-1] = traces[-2]
        traces[-2] = optimal_location_trace
        fig['data'] = tuple(traces)
        return fig

    @staticmethod
    def _create_legend_handles(colors: list[Union[str, tuple[float, float, float, float]]],
                               labels: list[str]) -> list[Line2D]:
        assert len(colors) == len(labels), 'colors and labels have to have the same length.'
        legend_handles = list()
        for color, label in zip(colors, labels):
            legend_handles.append(Line2D(
                xdata=[],
                ydata=[],
                color=color,
                linewidth=4,
                label=label))
        return legend_handles

    @staticmethod
    def _get_colors(n_colors: int) -> list[tuple[float, float, float, float]]:
        cmap = plt.cm.get_cmap('plasma', n_colors + 1)
        colors = list()
        for i in range(n_colors):
            colors.append(cmap(i))
        return colors
