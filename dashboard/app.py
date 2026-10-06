import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import demo, theme, wording

TECH_PARAM = "tech"


def remember_technical_choice():
    if st.session_state.get("tech"):
        st.query_params[TECH_PARAM] = "1"
    elif TECH_PARAM in st.query_params:
        del st.query_params[TECH_PARAM]


demo_mode = demo.enabled()
st.set_page_config(page_title="mlplat bakehouse",
                   page_icon=demo.favicon() if demo_mode else ":material/bakery_dining:", layout="wide")
if "tech" not in st.session_state:
    st.session_state["tech"] = st.query_params.get(TECH_PARAM) == "1"
theme.inject()

pages = []
if demo_mode:
    pages += [
        st.Page("views/start.py", title=wording.DEMO_PAGE_TITLES["start"], icon=":material/waving_hand:",
                default=True),
        st.Page("views/tour.py", title=wording.DEMO_PAGE_TITLES["tour"], icon=":material/tour:", url_path="tour"),
        st.Page("views/how.py", title=wording.DEMO_PAGE_TITLES["how"], icon=":material/schema:", url_path="how"),
    ]
pages += [
    st.Page("views/home.py", title=wording.PAGE_TITLES["home"], icon=":material/storefront:", default=not demo_mode,
            url_path="where" if demo_mode else None),
    st.Page("views/story.py", title=wording.PAGE_TITLES["story"], icon=":material/menu_book:", url_path="run"),
    st.Page("views/compare.py", title=wording.PAGE_TITLES["compare"], icon=":material/compare_arrows:",
            url_path="compare"),
    st.Page("views/runs.py", title=wording.PAGE_TITLES["runs"], icon=":material/list:", url_path="runs"),
]
navigation = st.navigation(pages)
with st.sidebar:
    st.toggle(wording.TECH_TOGGLE, key="tech", help=wording.TECH_TOGGLE_HELP, on_change=remember_technical_choice)
theme.header()
if demo_mode:
    demo.note(demo.facts()[0])
navigation.run()
theme.footer()
