import streamlit as st

VERSION = '5'
title = f'Enlighting{VERSION}'
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
import utils.session
from utils import session
from app import run_app
from utils.logging import timestamp, log_memory


def main():

    session.initialize_session()
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

    if 'console' in query_params:
        st.session_state.show_console = query_params['console'][0]
    else:
        st.session_state.show_console = False

    print(f'[{timestamp()}] st.session_state.show_console: {st.session_state.show_console}')
    print(f"[{timestamp()}] st.session_state.query_params.console: {st.session_state.query_params['console']}")
    print(f'[{timestamp()}] ------------ QUERY PARAMS (local) -----------------')
    print(f'[{timestamp()}] {query_params}')
    print(f'[{timestamp()}] ------------ QUERY PARAMS (global) -----------------')
    print(f'[{timestamp()}] {ss.query_params}')

    total_start = datetime.datetime.now()
    log_memory('main|run_app|B')
    print(f'[{timestamp()}] total_main_runs: {ss.total_main_runs}')
    print(f'[{timestamp()}] completed_main_runs: {ss.completed_main_runs}')
    print(f'[{timestamp()}] incomplete_main_runs (global): {ss.incomplete_main_runs}')
    print(f'[{timestamp()}] incomplete_main_runs (local): {incomplete_main_runs}')
    print(f'[{timestamp()}] auto_reloads: {ss.auto_reloads}')
    print(f'[{timestamp()}] total_app_runs: {ss.total_app_runs}')
    print(f'[{timestamp()}] completed_app_runs: {ss.completed_app_runs}')

    # try:
    #     assert run_app(), f'[{timestamp()}] Process was stopped before completing.  Re-running ...'
    # except AttributeError as msg:
    #     print(msg)
    #     st.experimental_rerun()


    gc.collect()
    log_memory('main|run_app|E')
    total_end = datetime.datetime.now()
    total_process_time = (total_end - total_start).total_seconds()
    print(f'[{timestamp()}]  Total processing time: {total_process_time:.5f} s')

    ss.completed_main_runs += 1
    sys.stdout.flush()

if __name__ == '__main__':
    
    main()
