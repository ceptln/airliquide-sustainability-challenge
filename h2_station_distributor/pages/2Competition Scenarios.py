from __future__ import annotations

import warnings

import pandas as pd
import geopandas as gpd
import streamlit as st
from models.stations import Stations
import pyproj
import folium
from streamlit_folium import st_folium

from utils.helpers import Data

warnings.filterwarnings("ignore")
st.set_page_config(page_title='H2 Stations Plan', page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Estimation of different competition scenarios')
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "Competition Scenarios";
            margin-left: 20px;
            margin-top: 20px;
            font-size: 30px;
            position: relative;
            top: 100px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

statA = pd.read_csv(Data.find_file('statA.csv'))
statB = pd.read_csv(Data.find_file('statB.csv'))

# statB.geometry = wkt.dumps(statB['geometry'])
statB[['dump', 'lat', 'lon']] = statB.geometry.str.split(" ", expand=True)
statB = statB.drop(columns={'dump', 'geometry'})
statB.lon = statB.lon.str[:-1]
statB.lat = statB.lat.str[1:]
statB = statB.sort_values('priority', ascending=True)
statA = statA.sort_values('priority', ascending=True)

lambert93 = pyproj.Proj("+init=EPSG:2154")
wgs84 = pyproj.Proj("+init=EPSG:4326")
# Convert the Lambert 93 coordinates to latitude and longitude
statB["lon"], statB["lat"] = pyproj.transform(
    lambert93, wgs84, statB["lat"], statB["lon"]
)

stations_2040 = pd.read_csv(Data.find_file('stations_10000.csv'))

x_small = stations_2040[['coords', 'type', 'fitness']]
x_small = x_small[x_small['type'] == 'small']
x_small = x_small.sort_values('fitness', ascending=False)
x_small[['lat', 'lon']] = x_small.coords.str.split(", ", expand=True)
x_small.lon = x_small.lon.str[:-1]
x_small.lat = x_small.lat.str[1:]

x_med = stations_2040[['coords', 'type', 'fitness']]
x_med = x_med[x_med['type'] == 'medium']
x_med = x_med.sort_values('fitness', ascending=False)
x_med[['lat', 'lon']] = x_med.coords.str.split(", ", expand=True)
x_med.lon = x_med.lon.str[:-1]
x_med.lat = x_med.lat.str[1:]

x_large = stations_2040[['coords', 'type', 'fitness']]
x_large = x_large[x_large['type'] == 'big']
x_large = x_large.sort_values('fitness', ascending=False)
x_large[['lat', 'lon']] = x_large.coords.str.split(", ", expand=True)
x_large.lon = x_large.lon.str[:-1]
x_large.lat = x_large.lat.str[1:]

Scenario = st.selectbox(label='Select the competition scenario', options=['Monopoly', 'Duopoly', 'Late Entrant'])

if Scenario == 'Monopoly':
    year = st.slider('Time step', min_value=2024, max_value=2040, value=2024, step=1)
    if year <= 2027:
        final = x_small.iloc[:19].append(x_med.iloc[:18].append(x_large.iloc[:4]))

        lambert93 = pyproj.Proj("+init=EPSG:2154")
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        # Convert the Lambert 93 coordinates to latitude and longitude
        final["lon"], final["lat"] = pyproj.transform(
            lambert93, wgs84, final["lat"], final["lon"]
        )

        mean_lat = final["lat"].mean()
        mean_lon = final["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in final.iterrows():
            x, y, type = row["lat"], row["lon"], row["type"]
            if type == 'big':
                color = "red"
            elif type == "medium":
                color = "blue"
            else:
                color = "green"
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color=color, icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2028 <= year < 2031:
        final = x_small.iloc[:59].append(x_med.iloc[:36].append(x_large.iloc[:7]))

        lambert93 = pyproj.Proj("+init=EPSG:2154")
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        # Convert the Lambert 93 coordinates to latitude and longitude
        final["lon"], final["lat"] = pyproj.transform(
            lambert93, wgs84, final["lat"], final["lon"]
        )

        mean_lat = final["lat"].mean()
        mean_lon = final["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in final.iterrows():
            x, y, type = row["lat"], row["lon"], row["type"]
            if type == 'big':
                color = "red"
            elif type == "medium":
                color = "blue"
            else:
                color = "green"
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color=color, icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2031 <= year < 2033:
        final = x_small.iloc[:105].append(x_med.iloc[:49].append(x_large.iloc[:21]))

        lambert93 = pyproj.Proj("+init=EPSG:2154")
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        # Convert the Lambert 93 coordinates to latitude and longitude
        final["lon"], final["lat"] = pyproj.transform(
            lambert93, wgs84, final["lat"], final["lon"]
        )

        mean_lat = final["lat"].mean()
        mean_lon = final["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in final.iterrows():
            x, y, type = row["lat"], row["lon"], row["type"]
            if type == 'big':
                color = "red"
            elif type == "medium":
                color = "blue"
            else:
                color = "green"
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color=color, icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2033 <= year < 2036:
        final = x_small.iloc[:140].append(x_med.iloc[:79].append(x_large.iloc[:40]))

        lambert93 = pyproj.Proj("+init=EPSG:2154")
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        # Convert the Lambert 93 coordinates to latitude and longitude
        final["lon"], final["lat"] = pyproj.transform(
            lambert93, wgs84, final["lat"], final["lon"]
        )

        mean_lat = final["lat"].mean()
        mean_lon = final["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in final.iterrows():
            x, y, type = row["lat"], row["lon"], row["type"]
            if type == 'big':
                color = "red"
            elif type == "medium":
                color = "blue"
            else:
                color = "green"
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color=color, icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2036 <= year:
        final = x_small.append(x_med.iloc.append(x_large))

        lambert93 = pyproj.Proj("+init=EPSG:2154")
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        # Convert the Lambert 93 coordinates to latitude and longitude
        final["lon"], final["lat"] = pyproj.transform(
            lambert93, wgs84, final["lat"], final["lon"]
        )

        mean_lat = final["lat"].mean()
        mean_lon = final["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in final.iterrows():
            x, y, type = row["lat"], row["lon"], row["type"]
            if type == 'big':
                color = "red"
            elif type == "medium":
                color = "blue"
            else:
                color = "green"
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color=color, icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)


elif Scenario == 'Duopoly':

    year = st.slider('Time step', min_value=2024, max_value=2040, value=2024, step=1)

    if year < 2026:
        mean_lat = statB["lat"].mean()
        mean_lon = statB["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        out = st_folium(m, width=1200, height=800)

    if 2026 <= year < 2028:

        mean_lat = statB["lat"].mean()
        mean_lon = statB["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in statB[:74].iterrows():
            x, y = row["lat"], row["lon"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='green', icon="glyphicon-flash")).add_to(m)

        for index, row in statA[:74].iterrows():
            x, y = row["gps_lat"], row["gps_lng"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='red', icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2028 <= year:

        mean_lat = statB["lat"].mean()
        mean_lon = statB["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in statB.iterrows():
            x, y = row["lat"], row["lon"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='green', icon="glyphicon-flash")).add_to(m)

        for index, row in statA.iterrows():
            x, y = row["gps_lat"], row["gps_lng"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='red', icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)


elif Scenario == 'Late Entrant':

    year = st.slider('Time step', min_value=2024, max_value=2040, value=2024, step=1)

    if year < 2026:

        mean_lat = statB["lat"].mean()
        mean_lon = statB["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)

        for index, row in statA[:20].iterrows():
            x, y = row["gps_lat"], row["gps_lng"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='red', icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2026 <= year < 2028:

        mean_lat = statB["lat"].mean()
        mean_lon = statB["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in statB[:15].iterrows():
            x, y = row["lat"], row["lon"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='green', icon="glyphicon-flash")).add_to(m)

        for index, row in statA[:50].iterrows():
            x, y = row["gps_lat"], row["gps_lng"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='red', icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2028 <= year < 2036:

        mean_lat = statB["lat"].mean()
        mean_lon = statB["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in statB[:30].iterrows():
            x, y = row["lat"], row["lon"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='green', icon="glyphicon-flash")).add_to(m)

        for index, row in statA[:100].iterrows():
            x, y = row["gps_lat"], row["gps_lng"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='red', icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

    if 2036 <= year < 2040:

        mean_lat = statB["lat"].mean()
        mean_lon = statB["lon"].mean()

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6.3)
        for index, row in statB[:80].iterrows():
            x, y = row["lat"], row["lon"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='green', icon="glyphicon-flash")).add_to(m)

        for index, row in statA[:148].iterrows():
            x, y = row["gps_lat"], row["gps_lng"]
            folium.Marker([x, y], tooltip=type, popup=type,
                          icon=folium.Icon(color='red', icon="glyphicon-flash")).add_to(m)

        out = st_folium(m, width=1200, height=800)

