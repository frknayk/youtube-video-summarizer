import streamlit as st

from frontend.page_main import page_main
# from pages_app.page_testing import page_testing
# from pages_app.page_training import page_training

st.set_page_config(
    page_title="User Interface of ML App",
    page_icon="🕹️",
    layout="wide",
)


page_names_to_funcs = {
    "Main Page": page_main,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()