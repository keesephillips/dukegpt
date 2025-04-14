import streamlit as st
from PIL import Image
import model

def submit():
    with spinner_container:
        with st.spinner("Generating an answer..."):
            user_query = st.session_state.widget
            if user_query.strip():
                question, answer = model.ask_question(user_query)
                st.session_state.history.append((question, answer))
                st.session_state.widget = ""

def header_component():
    st.set_page_config(
        page_title="DukeGPT",
        page_icon=Image.open('assets/favicon-96x96.png'),
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;700&family=Open+Sans:wght@400;700&display=swap');
        h1, h2, h3, h4, h5, h6 {
            font-family: 'EB Garamond', Garamond, Georgia, 'Times New Roman', Times, serif;
        }
        body, p, div, span, input, button {
            font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

    st.image("assets/Duke-University-Logo-RGB-Navy.png", use_container_width=True)
    st.title("DukeGPT")
    st.markdown("### Disclaimer:") 
    st.text("DukeGPT is chatbot for current and prospective students and aims to answer questions through Duke's own source material. DukeGPT cannot answer questions outside of the Duke ecosystem.")

def model_component():
    st.markdown("""
        <style>
        /* Override the default styling for st.button */
        div.stButton > button:first-child {
            background-color: #001A57 !important; /* Duke Blue */
            color: #ffffff !important;            /* White text */
            border: none;
            border-radius: 0.25em;
            cursor: pointer;
        }
        /* Hover state */
        div.stButton > button:first-child:hover {
            background-color: #003366 !important; 
            color: #ffffff !important;
        }
        /* Focus (clicked/focused) state */
        div.stButton > button:first-child:focus:not(:disabled),
        div.stButton > button:first-child:active:not(:disabled),
        div.stButton > button:first-child:focus:active:not(:disabled) {
            background-color: #001A57 !important;
            color: #ffffff !important;
            box-shadow: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    global answer
    answer = ""

    if "history" not in st.session_state:
        st.session_state.history = []

    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_area(
            "Ask me questions about Duke University:",
            key="widget",
            on_change=submit,  
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Submit", on_click=submit)

    global spinner_container
    spinner_container = st.empty()
    
    if st.session_state.history:
        st.markdown("## Question:")
        st.markdown(st.session_state.history[-1][0])
        st.markdown("## Answer:")
        st.markdown(st.session_state.history[-1][1])

    st.markdown("""### Limitations:  
- Can only answer general questions about Duke University  
- Can only answer general questions about Pratt School of Engineering  
- Can only answer general questions about the AIPI Program  
- Can only answer questions about university events within the next 30 days  
    """)
    
    with st.sidebar:
        st.image('assets/Duke-Logo-RGB-Navy.png')
        st.markdown("# Conversation History")
        st.markdown("----")
        if st.session_state.history:
            for i, (q, a) in enumerate(reversed(st.session_state.history), start=1):
                with st.expander(q[:20]):
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {a}")
        else:
            st.write("No conversation yet. Ask DukeGPT something!")

if __name__ == "__main__":
    model = model.Model()
    header_component()
    model_component()
