import warnings

import pyproj
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.load_data import *
from utils import generate_regional_breakdown
from utils import size_the_network
from utils.helpers import LoadData


warnings.filterwarnings("ignore")
st.set_page_config(page_title='H2 Stations Plan', page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Prediction of regional stations repartition')
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "Deployment Scenario";
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

# Load the data
stations = LoadData.load_stations()
# rrn_vsmap = load_rrn_vsmap()
# rrn_bornage = load_rrn_bornage()
regions = LoadData.load_regions()
regions = regions.rename(columns={'nomnewregi': 'region'})
traffic = LoadData.load_traffic(clean_data=False)
depreg = LoadData.load_depreg(drop_name=True)
# airesPL = load_airesPL()

# PARAMETERS

manufacturers_desc = {
    'man_1': {'name': 'Daimler Truck & Volvo', 'prototype': 'GenH2 Truck', 'technology': 'Hydrogen fuel cell',
              'tank_size': 80, 'autonomy': 1000, 'type_of_PL': 'long-distance'}
    , 'man_2': {'name': 'DAF', 'prototype': 'XF, XG and XG+', 'technology': 'Internal combustion', 'tank_size': 15,
                'autonomy': 150, 'type_of_PL': 'short-distance'}
    , 'man_3': {'name': 'Iveco & Nikola & Hyundai', 'prototype': 'Nikola TRE', 'technology': 'Hydrogen fuel cell',
                'tank_size': 32, 'autonomy': 400, 'type_of_PL': 'long-distance'}
}

stations_desc = {
    'small': {'capex': 3 * 10 ** 6, 'depreciation': 15, 'yearly_opex': 0.10, 'storage_onsite': 2000,
              'construction_time': 1, 'footprint': 650, 'profitability_threshold': 0.9, 'nb_terminals': 2},
    'medium': {'capex': 5 * 10 ** 6, 'depreciation': 15, 'yearly_opex': 0.08, 'storage_onsite': 3000,
               'construction_time': 1, 'footprint': 900, 'profitability_threshold': 0.8, 'nb_terminals': 3},
    'large': {'capex': 8 * 10 ** 6, 'depreciation': 15, 'yearly_opex': 0.07, 'storage_onsite': 4000,
              'construction_time': 1, 'footprint': 1200, 'profitability_threshold': 0.6, 'nb_terminals': 4}
}

parameters = {
    'year': 2030
    , 'nb_trucks': 10000
    , 'manufacturers_desc': manufacturers_desc
    , 'split_manufacturer': {'man_1': 0.4, 'man_2': 0.4, 'man_3': 0.2}
    , 'activation_rate': 0.8
    , 'avg_daily_km': {'short-distance': 290, 'long-distance': 458}
    , 'stations_desc': stations_desc
    , 'split_station_type': {'small': 1 / 2, 'medium': 1 / 3, 'large': 1 / 6}
    , 'average_tank_filling_rate_before_refill': 0.2
    , 'security_buffer': 0.2
    , 'strategic_positioning_index': 1
    , 'prefered_order_of_station_type': ['small', 'medium', 'large']
}

st.header("Select a strategy")
strategy = st.selectbox(label=' ',
                        options=['Slow deployment', 'Matching demand', 'Driving the change'],
                        index=0,
                        label_visibility="collapsed")

with st.expander("Set parameters"):
    st.header("Number of expected trucks:")
    nb_trucks = st.slider(' ', min_value=10000, max_value=60000, value=10000, step=100, on_change=None,
                          label_visibility="visible")

    st.header("Percentage of station type:")
    col1, col2, col3 = st.columns(3)
    with col1:
        small = st.slider('Small', min_value=0.0, max_value=1.0, value=0.3, step=0.1, on_change=None)
    with col2:
        medium = st.slider('Medium', min_value=0.0, max_value=1 - small, value=0.3, step=0.1, on_change=None)
    with col3:
        large = st.slider('Large', min_value=0.0, max_value=1 - small - medium, value=1 - small - medium, step=0.1,
                          on_change=None)

    st.header("Repartition of manufacturers:")
    col1, col2, col3 = st.columns(3)
    with col1:
        man1 = st.slider('Manufacturer 1', min_value=0.0, max_value=1.0, value=0.3, step=0.1, on_change=None)
    with col2:
        man2 = st.slider('Manufacturer 2', min_value=0.0, max_value=1 - man1, value=0.3, step=0.1, on_change=None)
    with col3:
        man3 = st.slider('Manufacturer 3', min_value=0.0, max_value=1 - man1 - man2, value=1 - man1 - man2, step=0.1,
                         on_change=None)

if strategy == "Slow deployment":
    parameters["strategic_positioning_index"] = 0.5
elif strategy == "Matching demand":
    parameters["strategic_positioning_index"] = 1
else:
    parameters["strategic_positioning_index"] = 1.2

parameters["split_station_type"]["small"] = small
parameters["split_station_type"]["medium"] = medium
parameters["split_station_type"]["large"] = large

parameters["split_manufacturer"]["man_1"] = man1
parameters["split_manufacturer"]["man_2"] = man2
parameters["split_manufacturer"]["man_3"] = man3

parameters["nb_trucks"] = nb_trucks

region_breakdown = generate_regional_breakdown.create_region_breakdown(traffic, depreg)

regional_strategies = size_the_network.define_best_regional_strategies(parameters=parameters,
                                                                       manufacturers_desc=manufacturers_desc,
                                                                       stations_desc=stations_desc,
                                                                       region_breakdown=region_breakdown,
                                                                       verbose=False)


data2 = regional_strategies.drop(12, axis=0)

final = regions.merge(data2, on='region')

final.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

fig = px.choropleth(final,
                    color_continuous_scale="Emrld",
                    geojson=final.geometry,
                    locations=final.index,
                    color="total",
                    projection="mercator")
fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

st.header("Number of stations per region")
st.plotly_chart(fig)

st.header('Plan & Costs analysis')

## Calculate the costs & footprint
data2["total_cost"] = data2["small"] * 3 * 10 ** 6 + data2["medium"] * 5 * 10 ** 6 + data2["large"] * 8 * 10 ** 6
data2["footprint"] = data2["small"] * 650 + data2["medium"] * 900 + data2["large"] * 1200
data2["yearly_cost"] = data2["small"] * 3 * 10 ** 6 * 0.10 + data2["medium"] * 5 * 10 ** 6 * 0.08 + data2[
    "large"] * 8 * 10 ** 6 * 0.07

co1, co2 = st.columns(2)
with co1:
    st.markdown('Quantity of H2 to distribute')
    fig2 = px.pie(data2, values='quantity_h2_to_reach', names='region',
                  color_discrete_sequence=px.colors.sequential.Emrld)
    fig2.update(layout_showlegend=True)
    fig2.update_traces(hoverinfo='label+percent', textinfo='value')
    fig2.update_layout(legend=dict(yanchor="middle", y=0.5, xanchor="left", x=-0.8))
    st.plotly_chart(fig2, use_container_width=True)
with co2:
    st.markdown('Costs & Footprint')
    fig3 = go.Figure(data=[
        go.Bar(name='Total Cost', x=data2.region, y=data2.total_cost, yaxis='y', marker_color='darkcyan',
               offsetgroup=1),
        go.Bar(name='Yearly Cost', x=data2.region, y=data2.yearly_cost, yaxis='y', marker_color='palegreen',
               offsetgroup=1),
        go.Bar(name='Footprint', x=data2.region, y=data2.footprint, yaxis='y2', marker_color='darkgreen', offsetgroup=2)
    ],
        layout={
            'yaxis': {'title': 'Total & Yearly cost'},
            'yaxis2': {'title': 'Footprint', 'overlaying': 'y', 'side': 'right'}
        })

    fig3.update_layout(barmode='group',
                       legend=dict(title=None, orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"))
    fig3.update(layout_showlegend=True)
    st.plotly_chart(fig3, use_container_width=True)
