import warnings

import streamlit as st
from models.stations import Stations
from streamlit_folium import st_folium
import pyproj
import folium

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="H2 Stations Plan",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("deprecation.showPyplotGlobalUse", False)

st.title("Display of stations best positions")
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "Location of Stations";
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

if "big" not in st.session_state:
    st.session_state.big = 1
if "medium" not in st.session_state:
    st.session_state.medium = 1
if "small" not in st.session_state:
    st.session_state.small = 1


def change_big_sate():
    st.session_state.big = large


def change_medium_state():
    st.session_state.medium = medium


def change_small_state():
    st.session_state.small = small


@st.cache_data
def calculate_output(department, large, medium, small):
    stations = Stations()
    return stations.solution(department, large, medium, small)


departments = list(Stations().shapefile["depPrF"].unique())

st.header("Select a department")

with st.expander("Set parameters"):
    st.header("Department:")

    department = st.selectbox(
        label="Department",
        options=departments,
        index=0,
        label_visibility="collapsed",
    )
    st.header("Number of stations:")

    col1, col2, col3 = st.columns(3)
    with col1:
        small = st.number_input(
            "Small", min_value=0, value=1, step=1, on_change=change_big_sate
        )
    with col2:
        medium = st.number_input(
            "Medium", min_value=0, value=1, step=1, on_change=change_medium_state
        )
    with col3:
        large = st.number_input(
            "Large", min_value=0, value=1, step=1, on_change=change_small_state
        )


output = calculate_output(department, large, medium, small)
lambert93 = pyproj.Proj("+init=EPSG:2154")
wgs84 = pyproj.Proj("+init=EPSG:4326")
# Convert the Lambert 93 coordinates to latitude and longitude
output["lon"], output["lat"] = pyproj.transform(
    lambert93, wgs84, output["coords"].str[0].values, output["coords"].str[1].values
)

# Calculate the mean latitude and longitude values
mean_lat = output["lat"].mean()
mean_lon = output["lon"].mean()

m = folium.Map(location=[mean_lat, mean_lon], zoom_start=7.5)
for index, row in output.iterrows():
    x, y, type = row["lat"], row["lon"], row["type"]
    if type == "big":
        color = "red"
    elif type == "medium":
        color = "blue"
    else:
        color = "green"
    folium.Marker([x, y], tooltip=type, popup=type, color=color).add_to(m)


out = st_folium(m, width=1200, height=500)
