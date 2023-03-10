import warnings

import streamlit as st
from models.stations import Stations
from streamlit_folium import st_folium
import pyproj

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
        small = st.number_input("Small", min_value=0, value=0, step=1, on_change=None)
    with col2:
        medium = st.number_input("Medium", min_value=0, value=0, step=1, on_change=None)
    with col3:
        large = st.number_input("Large", min_value=0, value=0, step=1, on_change=None)


stations = Stations()
output = stations.solution(department, small, medium, large)

# Define Lambert 93 and WGS84 coordinate systems
lambert93 = pyproj.Proj("+init=EPSG:2154")
wgs84 = pyproj.Proj("+init=EPSG:4326")
# Convert from Lambert 93 to WGS84
lon, lat = pyproj.transform(lambert93, wgs84, x, y)


# with st.echo():
#         m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
#         tooltip = "Liberty Bell"
#         folium.Marker(
#             [39.949610, -75.150282], popup="Liberty Bell", tooltip=tooltip
#         ).add_to(m)
