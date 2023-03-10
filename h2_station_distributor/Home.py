import warnings

import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title='H2 Stations Plan', page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('How does this app work ?')

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "H2 Trucks stations";
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

part_id = st.selectbox('Select a project part:',
                       options=['Deployment Scenarios', 'Location of Stations', 'Competition Scenarios',
                                'Production Plants Location'],
                       index=0)

if part_id == 'Deployment Scenarios':
    st.header('1 - Modeling the number of stations per region')
    st.markdown('''This part aims to have a first general idea of the deployment strategy we want to follow.
&nbsp;

### Options selection :


**You can select if you want to match the predicted demand, be safer or bolder in the global strategy.**
This will impact the coverage of needed stations given trucks fleet prediction.

> For slow deployment we implement a coverage of 50%. 
>
> For matching demand we take a coverage of 100%.
>
> For driving the change we cover 120% of the needed stations.
 
&nbsp;
&nbsp;
 
**Then you can choose several options for parameters impacting the number of stations to deploy.**

> The number of trucks predicted in the overall fleet.
>
> - 10000 predicted in 2030
> - 60000 predicted in 2040
> - Possibility to see what should be planned if the number exceeds 60000.
>
>
> The repartition of manufacturers. This has a direct impact on the characteristics of the trucks.
>
>
> The repartition of stations type. Large, medium or small depending on capacity. This impacts the yearly and construction costs as well as number of trucks served.
 
&nbsp;

### Results :


1. Heatmap display of the number of stations per region.
2. Charts on costs and other analysis (footprint & H2 quantity).

''')
elif part_id == 'Location of Stations':
    st.header('2 - Locating the optimal H2 stations')
    st.markdown(''' This part aims to optimize the location (Lon & Lat) of future H2 charging stations for heavy trucks.
&nbsp;

### Visualisation map :


Here you can see the stations on the map & select a station to see its informations. 

You can also input filters about the size of stations, the most important ones, the most costly ones. 

&nbsp;

### Statistical dashboard


Display of information about stations given the filters chosen.

> - The importance coefficient of the top n stations
> - The Cost/Benefits analysis of the overall \& top n stations.
> - The implementation timeline of the stations.

''')
