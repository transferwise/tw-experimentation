import streamlit as st
from streamlit_utils import PullAndMatchData
from streamlit_utils import _coming_from_other_page, initalise_session_states

st.session_state.update(st.session_state)
st.set_page_config(layout="wide")


CURRENT_PAGE = "Main"


initalise_session_states()

st.session_state["last_page"] = CURRENT_PAGE


st.title("TW Experimentation: AB Testing")

st.markdown(
    """ 
    ### How to use it
    - All images that appear dynamic have a download option. If you hover over the top right corner of the image, click on the camera symbol ("Download plot as a png")
    - If you would like to copy a displayed table, you can simple select all the relevant entries and copy-paste them to outside of the application. 
    Note however that it is currently not possible to also copy the column names :( 
    - Some tables do not display nicely immediately. You can expand them to full-screen on the top right corner of the table.
    ### What can this do for you?

    The experimentation library can help you with:
    - Sample Size Calculator  
    - Integrity Checks 
    - Evaluation (Frequentist A/B testing and Wise-pizza) 
    - Evaluation (Bayesian A/B testing) 

    #### 1. Designing experiments
    By using **TW Experimentation** you can design your experiments, choose sample size, evaluate the experiment and calculate features and metrics.


    #### 2. Evaluating results
    You can use different statistical tests and causal inference techniques.
    """
)
