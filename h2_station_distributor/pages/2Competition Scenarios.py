import warnings

import streamlit as st


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

