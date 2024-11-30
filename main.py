import streamlit as st
import agent as at

with st.sidebar:
    db_uri=st.text_input("Database URI", "sqlite:///salaries.db")


st.title("💬 Chat with DB")

if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
      if not db_uri:
          st.info("Please add your database uri to continue.")
          st.stop()

      inputs={
        "query": prompt,
      }
      st.session_state.messages.append({"role": "user", "content": prompt})
      st.chat_message("user").write(prompt)
      #crew = tasks(inputs)
      #msg = crewkickoff(inputs=inputs)
      st.session_state.messages.append({"role": "assistant", "content": msg})
      st.chat_message("assistant").write(msg)