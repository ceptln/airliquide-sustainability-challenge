from __future__ import annotations

import geopandas as gpd
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
import yaml
import zipfile


# with open("../config.yaml") as f:
#     config = yaml.safe_load(f)


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
