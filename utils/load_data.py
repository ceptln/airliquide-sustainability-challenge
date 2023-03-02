from __future__ import annotations

import geopandas as gpd
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
import yaml
import zipfile

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


class Download:
    raw_data_folder = 'data'

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
