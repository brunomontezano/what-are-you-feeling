import sys
import os
import streamlit as st
import plotly.graph_objects as go

# HACK: This is needed for Python to understand I'm referring to the model
# directory in the project root: https://stackoverflow.com/a/49039555/17012515
sys.path.insert(0, os.path.abspath("."))
from model.predict import predict_status, predict_proba  # noqa: E402

st.set_page_config(page_title="What are you feeling?", page_icon="🧠")

st.title("What are you feeling? 🔮")
st.markdown("`By: Bruno Montezano and Aline Zimerman. 2024.`")

# NOTE: Declare the messages list
if "messages" not in st.session_state:
    st.session_state.messages = []

# NOTE: For each message, add who sent it and the content of the message in
# the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("You can write in this box..."):
    with st.chat_message("👨"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "👨", "content": prompt})

    # NOTE: Join prediction and call for the spider web chart
    response = f"{predict_status(prompt)} You can see a summary of \
    my prediction below:"

    # NOTE: This part is regarding the dictionary of predictions
    prob_dict = predict_proba(prompt)

    categories = list(prob_dict.keys())
    values = [prob * 100 for prob in prob_dict.values()]

    values += values[:1]
    categories += categories[:1]

    # NOTE: Here, I build the spider web/radar chart. It is basically a
    # bar chart using polar coordinates
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            line=dict(color="blue"),
        )
    )

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            )
        ),
        showlegend=False,
    )

    # NOTE: Lastly, I print the predicted status and the chart below
    # in the chat for the user
    with st.chat_message("🤖"):
        st.markdown(response)
        st.plotly_chart(fig)
    st.session_state.messages.append({"role": "🤖", "content": response})
