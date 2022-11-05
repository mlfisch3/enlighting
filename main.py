import streamlit as st

title = f'InLight'
st.set_page_config(page_title=title, layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


import gc
import sys
import datetime
from copy import copy
from utils import session
from app import run_app
from utils.logging import timestamp, log_memory


def main():

    total_start = datetime.datetime.now()

    print('\n')
    print('▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼')
    log_memory('main|run_app|B')
    session.initialize_session()
    session.report_runs('main.py|32')
    ss = st.session_state
    
    incomplete_main_runs = st.session_state.total_main_runs - st.session_state.completed_main_runs
    if incomplete_main_runs > st.session_state.incomplete_main_runs:
        st.session_state.last_run_exited_early = True
        st.session_state.incomplete_main_runs = copy(incomplete_main_runs)
    else:
        st.session_state.last_run_exited_early = False

    st.session_state.total_main_runs += 1

    query_params = st.experimental_get_query_params()
    for k,v in query_params.items():
        ss.query_params[k] = v[0]
        ss.query_params.setdefault(k,v[0])

    if 'cache' in query_params:
        st.session_state.cache_clearance = query_params['cache'][0]
    else:
        st.session_state.cache_clearance = False

    if 'resources' in query_params:
        st.session_state.show_resource_usage = query_params['resources'][0]
    else:
        st.session_state.show_resource_usage = False

    if 'console' in query_params:
        st.session_state.show_console = query_params['console'][0]
    else:
        st.session_state.show_console = False

    if 'debug' in query_params:
        st.session_state.debug = query_params['debug'][0]
    else:
        st.session_state.debug = False

    session.report_runs('main.py|69')

    run_app()
    
    gc.collect()
    log_memory('main|run_app|E')
    total_end = datetime.datetime.now()
    total_process_time = (total_end - total_start).total_seconds()

    print(f'[{timestamp()}]  Total processing time: {total_process_time:.5f} s')
    ss.completed_main_runs += 1
    session.report_runs('main.py|80')
    print(st.session_state.state_history)
    print('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲')
    print('\n')

    sys.stdout.flush()

if __name__ == '__main__':
    
    main()
