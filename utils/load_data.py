from __future__ import annotations

import geopandas as gpd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

import io
import os
import requests
import time
from typing import Union
import warnings
import yaml
import zipfile

warnings.simplefilter('ignore', FutureWarning)


with open("../config.yaml") as f:
    config = yaml.safe_load(f)


class Data:
    @staticmethod
    def find_file(file_name: str) -> str:
        """This method searches for file_name and returns the full path to it."""
        current_working_directory = os.getcwd()
        for root, folders, files in os.walk('.'):
            if file_name in files:
                return f'{current_working_directory}{root.lstrip(".")}/{file_name}'

    @staticmethod
    def find_folder(folder_name: str) -> str:
        """This method searches for folder_name and returns the full path to it."""
        current_working_directory = os.getcwd()
        for root, folders, files in os.walk('.'):
            if folder_name in folders:
                return f'{current_working_directory}{root.lstrip(".")}/{folder_name}'

    @staticmethod
    def clean_traffic_data(df: gpd.GeoDataFrame, reset_index: bool = True) -> gpd.GeoDataFrame:
        """This method cleans the traffice data set. The provided DataFrame should be the output of the function
        load_traffic."""
        df = df.copy()
        # Removing 0 traffic records
        df = df.loc[df['tmja'].values > 0, :]
        # Since we are only interested in trucks, we remove the records that didn't see any heavy duty vehicles
        df = df.loc[df['pctPL'].values > 0, :]
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
    def add_weighted_traffic(df_traffic: Union[gpd.GeoDataFrame, pd.DataFrame],
                             traffic_column: str = 'truck_tmja') -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """This method adds a column with 'longueur' / max('longueur') * traffic_column called 'weighted_traffic' to
        df_traffic.
        It is assumed that the road length is contained in the column 'longueur'."""
        df_traffic = df_traffic.copy(deep=True)
        max_length = df_traffic['longueur'].max()
        df_traffic['weighted_traffic'] = df_traffic['longueur'] / max_length * df_traffic[traffic_column]
        return df_traffic

    @staticmethod
    def _convert_pd_stations_to_gpd(df_stations: pd.DataFrame, target_crs: str = 'EPSG:2154') -> gpd.GeoDataFrame:
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

    @staticmethod
    def limit_gdfs_to_main_land_france(df: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """This method filters out records from df that contain geometry that is not inside main land France."""
        df = df.copy(deep=True)
        df_regions = df_regions.copy(deep=True)

        main_land = df_regions[df_regions['nomnewregi'] != 'Corse']
        mask = df.loc[:, 'geometry'].apply(lambda x: any(main_land.contains(x)))
        return df[mask]

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
        that the provided lengths in the column 'longueur' deviate from what we would have using the shape lengths
        however, we normalize the calculated shape lengths and then multiply with 'longueur' thus keeping with the
        lenghts provided in 'longueur'.
        df is assumed to contain the following columns: 'geometry', 'longueur'
        df_regions is assumed to contain the following columns: 'nomnewregi', 'geometry'"""
        df = df.copy(deep=True)
        df_regions = df_regions.copy(deep=True)
        r_with_shapes = df_regions.set_index('nomnewregi')['geometry']
        df['total_length'] = 0
        for index, value in r_with_shapes.items():
            df[index + '_geometry'] = value
            df = df.set_geometry(col=index + '_geometry')
            df[index + '_geometry'].crs = df['geometry'].crs
            df[index + '_intersection'] = df['geometry'].intersection(df[index + '_geometry'])
            df[index + '_length'] = df[index + '_intersection'].length
            df = df.drop(columns=[index + '_geometry', index + '_intersection'])
            df['total_length'] += df[index + '_length']

        for index, value in r_with_shapes.items():
            df[index + '_length'] = df[index + '_length'] / df['total_length'] * df['longueur']
        return df.drop(columns=['total_length'])

    @staticmethod
    def add_buffer_traffic(df_traffic: gpd.GeoDataFrame,
                           buffer_distance: Union[int, float] = 10000,
                           traffic_column: str = 'weighted_traffic') -> gpd.GeoDataFrame:
        """This method adds a column containing the sum of the traffic within the buffer. It is recommended to use
        'weighted_traffic' as the traffic column (which can be created with add_weighted_traffic) to not give
        unreasonable weight to many small road sections.
        It is assumed that the geometry in df_traffic is contained in the column 'geometry'."""
        df_traffic = df_traffic.copy(deep=True)
        buffer_shapes = df_traffic.buffer(distance=buffer_distance)
        for index, geometry in df_traffic['geometry'].items():
            mask = buffer_shapes.intersects(geometry)
            buffer_traffic = df_traffic.loc[mask, traffic_column].sum()
            df_traffic.loc[index, 'buffer_traffic'] = buffer_traffic
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
        avg_break_time_in_hours = avg_route_length_in_km / avg_speed_in_km \
                                  / max_uninterrupted_driving_time_in_hours * 0.75
        # If the range is smaller than the distance covered within one interrupted driving session, extra refueling
        # breaks have to be added. Such a break is assumed to last 15 min.
        extra_breaks = np.ceil((max_uninterrupted_driving_time_in_hours * avg_speed_in_km / range_in_km) - 1)
        avg_break_time_in_hours += 15 * extra_breaks

        avg_loading_time_in_hours = 3
        avg_driving_time = avg_route_length_in_km / avg_speed_in_km

        shifts_per_day = 24 / (avg_driving_time + avg_break_time_in_hours + avg_loading_time_in_hours)
        h2_distance_in_km_per_day = n_trucks * avg_route_length_in_km * shifts_per_day
        return h2_distance_in_km_per_day

    @classmethod
    def add_h2_truck_traffic(cls, df_traffic: gpd.GeoDataFrame, n_trucks: int = 10000,
                             data_year: int = 2018, target_year: int = 2030) -> gpd.GeoDataFrame:
        """This method adds a column containing how many H2 trucks pass a road section per day. This is done by finding
        what portion of the truck traffic n_trucks corresponds to and then multiplying 'truck_tmja' with that value to
        create 'h2_truck_tmja'."""
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
                             / (df_traffic['truck_tmja']
                                * truck_traffic_multiplier
                                * df_traffic['longueur'] / 1000).sum())
        # For 'h2_truck_tmja' we are simply applying the same percentage to the current traffic values. Of course it
        # would make sense to project 'truck_tmja' as well instead of using 'truck_tmja' unchanged but hey ...
        df_traffic['h2_truck_tmja'] = df_traffic['truck_tmja'] * h2_trucks_portion
        return df_traffic

    @staticmethod
    def calculate_total_distance_covered(df_traffic: Union[gpd.GeoDataFrame, pd.DataFrame],
                                         traffic_column: str = 'truck_tmja',
                                         road_length_column: str = 'longueur') -> Union[int, float]:
        """This method calculates the total distance covered by all trucks in df_traffic (in km). It is assumed that
        the records contain mutually exclusive road sections."""
        return (df_traffic[traffic_column] * df_traffic[road_length_column] / 1000).sum()

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


class Download:
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
        executable_path = Data.find_file('geckodriver')
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
    def download_depreg(cls):
        csv_url = 'https://www.data.gouv.fr/fr/datasets/r/987227fb-dcb2-429e-96af-8979f97c9c84'
        cls.download_csv_file(url=csv_url, target_file='depreg.csv')


class Strategies:
    @classmethod
    def split_into_n_areas(cls, df_traffic: gpd.GeoDataFrame, n_areas: int, rank_column: str = 'buffer_traffic',
                           buffer_step: Union[int, float] = 10000) -> gpd.GeoDataFrame:
        """This method creates n_areas areas that each contain at least 1 / n_areas * total distance covered by trucks
        in all of France. The output DataFrame will have n_areas new columns, each telling which records belong to
        an area. One record can belong to multiple areas.
        The downside of this method is that some records (=road sections) do not belong to any area, especially the
        sections close to the border are affected by this.
        It is assumed that the geometry in df_traffic is stored in a column called 'geometry'."""
        df_traffic = df_traffic.copy(deep=True)
        total_distance = Data.calculate_total_distance_covered(df_traffic)
        distance_per_area = total_distance / n_areas
        created_areas = 0
        reference = df_traffic[rank_column].argmax()
        mask = pd.Series(index=df_traffic.index)
        mask.loc[:] = False
        mask = mask.astype(bool)
        areas = dict()
        while created_areas < n_areas:
            buffer_size, used_records_mask = cls._size_buffer_for_distance(df_traffic=df_traffic, index=reference,
                                                                           target_distance=distance_per_area,
                                                                           buffer_step=buffer_step)
            areas[reference] = buffer_size
            mask += used_records_mask
            reference = df_traffic.loc[~mask, rank_column].sort_values(ascending=False).index[0]
            created_areas += 1
            df_traffic.loc[used_records_mask, f'area_{created_areas}'] = True
            df_traffic.loc[~used_records_mask, f'area_{created_areas}'] = False
        df_traffic['no_area'] = ~mask
        return df_traffic

    @staticmethod
    def _size_buffer_for_distance(df_traffic: gpd.GeoDataFrame, index: int, target_distance: Union[int, float],
                                  buffer_step: Union[int, float] = 10000) -> tuple[Union[int, float], pd.Series]:
        """This method provides the size of the buffer that is needed around the record at index to cover
        target_distance when summing 'longueur' * 'tmja' over all road sections inside the buffer."""
        if target_distance <= 0:
            raise ValueError('The target_distance has to be larger than 0.')
        covered_distance = 0
        buffer = buffer_step
        while covered_distance < target_distance:
            area = df_traffic.loc[index, 'geometry'].buffer(buffer)
            mask = df_traffic.intersects(area)
            covered_distance = Data.calculate_total_distance_covered(df_traffic.loc[mask, :])
            buffer += buffer_step
        return buffer - buffer_step, mask

    @classmethod
    def split_into_areas_station_based(cls, df_traffic: gpd.GeoDataFrame, station_size: Union[int, float],
                                       n_h2_trucks: int, rank_column: str = 'buffer_traffic',
                                       buffer_step: Union[int, float] = 10000) -> gpd.GeoDataFrame:
        """This method creates areas based on the provided station size (in kg of H2 they can refuel) and the number of
        H2 trucks on the road."""
        df_traffic = df_traffic.copy(deep=True)
        distance_to_cover = Data.calculate_h2_distance(n_trucks=n_h2_trucks)
        total_covered_distance = 0
        created_areas = 0

        reference = df_traffic[rank_column].argmax()
        mask = pd.Series(index=df_traffic.index)
        mask.loc[:] = False
        mask = mask.astype(bool)
        areas = dict()
        while total_covered_distance < distance_to_cover:
            buffer_size, used_records_mask, covered_distance = \
                cls._size_buffer_for_station(df_traffic=df_traffic, index=reference, target_h2_consumption=station_size,
                                             buffer_step=buffer_step)
            total_covered_distance += covered_distance
            areas[reference] = buffer_size
            mask += used_records_mask
            reference = df_traffic.loc[~mask, rank_column].sort_values(ascending=False).index[0]
            created_areas += 1
            df_traffic.loc[used_records_mask, f'area_{created_areas}'] = True
            df_traffic.loc[~used_records_mask, f'area_{created_areas}'] = False
        df_traffic['no_area'] = ~mask
        print(f'Created {created_areas + 1} areas.')
        return df_traffic

    @classmethod
    def _size_buffer_for_station(cls, df_traffic: gpd.GeoDataFrame, index: int,
                                 target_h2_consumption: Union[int, float],
                                 buffer_step: Union[int, float] = 10000) -> tuple[Union[int, float],
                                                                                  pd.Series,
                                                                                  Union[int, float]]:
        """This method calculates the maximum area around the record at index such that target_h2_consumption is still
        able to cover the H2 demand.
        target_h2_consumption is the H2 consumption in kg."""
        if target_h2_consumption <= 0:
            raise ValueError('The target_h2_consumption has to be larger than 0.')
        target_distance = cls._calculate_km_from_fuel(target_h2_consumption)
        covered_distance = 0
        buffer = buffer_step
        while covered_distance < target_distance:
            area = df_traffic.loc[index, 'geometry'].buffer(buffer)
            mask = df_traffic.intersects(area)
            covered_distance = Data.calculate_total_distance_covered(df_traffic.loc[mask, :],
                                                                     traffic_column='h2_truck_tmja')
            buffer += buffer_step
        return buffer - buffer_step, mask, covered_distance

    @staticmethod
    def _calculate_refuelings_per_day(h2_distance: Union[int, float], range_in_km: Union[int, float],
                                      min_fuel_level: float = 0.2) -> Union[int, float]:
        """This method calculated how many refuelings are necessary to cover h2_distance. range_in_km defines how far a
        truck can drive on a full tank and min_fuel_level determines at what tank level a truck will be refueled."""
        return h2_distance / ((1 - min_fuel_level) * range_in_km)

    @staticmethod
    def _calculate_fuel_need(h2_distance_in_km: Union[int, float],
                             h2_in_kg_per_km: Union[int, float] = 0.08) -> Union[int, float]:
        """This method calculates how many kg of H2 are needed to cover h2_distance assuming an H2 consumption per km of
        h2_per_km."""
        return h2_distance_in_km * h2_in_kg_per_km

    @staticmethod
    def _calculate_km_from_fuel(h2_fuel_in_kg: Union[int, float],
                                h2_in_kg_per_km: Union[int, float] = 0.08) -> Union[int, float]:
        return h2_fuel_in_kg / h2_in_kg_per_km


class Plots:
    @staticmethod
    def plot_roads_over_regions(df_traffic: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame) -> None:
        """This method plots the road network over a map of France."""
        fig, ax = plt.subplots()
        df_regions.plot(ax=ax)
        df_traffic.plot(ax=ax, color='red')
        ax.set_title('Road network of France')
        plt.show()

    @staticmethod
    def plot_traffic_per_region(df_traffic: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame, df_depreg) -> None:
        """This method maps the total daily traffic per region."""
        t_with_r = Data.add_regions_to_departments(df_traffic, df_depreg)
        t_with_r_and_s = Data.add_region_shapes(t_with_r, df_regions).set_geometry('region_geometry')
        plot_data = t_with_r_and_s.groupby('region_name').agg({'tmja': sum, 'region_geometry': 'first'}).reset_index()
        plot_data = gpd.GeoDataFrame(plot_data, geometry='region_geometry', crs=df_regions.crs)

        fig, ax = plt.subplots()
        plot_data.plot(ax=ax, column='tmja', edgecolor='black', legend=True, legend_kwds={'label': 'Traffic per day'})
        ax.set_title('Daily traffic in France')
        plt.show()

    @classmethod
    def plot_calculated_areas(cls, df_areas: gpd.GeoDataFrame, df_regions: gpd.GeoDataFrame) -> None:
        """This method plots the roads on top of France, with a color indicating which area a road section belongs to.
        df_areas is assumed to contain a geometry column as well as one or more boolean columns starting with 'area_'
        telling which record belong to which area (as output by Strategies.split_into_areas()), as well as a column
        'buffer_traffic' (as output by Data.add_buffer_traffic)."""
        df_areas = df_areas.copy(deep=True)
        # Creating geometries that represent a buffer of 10 km around the road sections
        df_areas['buffer_geometry'] = df_areas.buffer(10000)
        df_areas = df_areas.set_geometry('buffer_geometry')

        fig, ax = plt.subplots()
        # Drawing the outline of the French regions
        df_regions.plot(ax=ax)
        # The following line would have to be uncommented if one wanted to see a heat map of the traffic
        # df_areas.plot(ax=ax, column='buffer_traffic', legend=True, cmap='hot')

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
