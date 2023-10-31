import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from utils import (
    intent_classifier,
    semantic_search,
    ensure_fit_tokens,
    get_page_contents,
)
from prompts import human_template, system_message
from render import user_msg_container_html_template, bot_msg_container_html_template
import openai


# Set OpenAI API key

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


st.header("RaisoniInfoBot: Chatting with RaisoniInfoBot")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load the Buffett and Branson databases
LocalDB = Chroma(
    persist_directory=os.path.join("db", "Local"), embedding_function=embeddings
)
Local_retriever = LocalDB.as_retriever(search_kwargs={"k": 3})


# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []


# Construct messages from chat history
def construct_messages(history):
    messages = [{"role": "system", "content": system_message}]

    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})

    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)

    return messages


# Define handler functions for each category


def Local_handler(query):
    print("Working on non-advisory query ...")
    # Get relevant documents from Local database
    relevant_docs = Local_retriever.get_relevant_documents(query)

    if not relevant_docs:
        print(
            "Data not available in Local database. Fetching answer from ChatGPT..."
        )
        # Fetch an answer from ChatGPT using the provided query
        answer = chatgpt_fetch_answer(query)
        print("Fetched answer from ChatGPT:", answer)

        # Return the answer in the appropriate message format
        return {"role": "assistant", "content": answer}

    print("Data fetched from Local database.")

    # Use the provided function to prepare the context
    context = get_page_contents(relevant_docs)

    # Prepare the prompt for GPT-3.5-turbo with the context
    query_with_context = human_template.format(query=query, context=context)

    return {"role": "user", "content": query_with_context}


def chatgpt_fetch_answer(query):
    # Make an API call to ChatGPT to get a response
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the appropriate engine for your task
        prompt=query,
        max_tokens=50,  # Adjust the max_tokens as needed
    )
    answer = response.choices[0].text.strip()
    return answer





# Function to route query to correct handler based on category
def route_by_category(query):
    if query :
        return Local_handler(query)

    else:
        raise ValueError("Invalid category")


# Function to generate response
def generate_response():
    # Append user's query to history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(
        {"message": st.session_state.prompt, "is_user": True}
    )

 

    # Route the query based on category
    new_message = route_by_category(st.session_state.prompt)

    # Construct messages from chat history
    messages = construct_messages(st.session_state.history)

    # Add the new_message to the list of messages before sending it to the API
    messages.append(new_message)

    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)

    # Call the Chat Completions API with the messages
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    # Extract the assistant's message from the response
    assistant_message = response["choices"][0]["message"]["content"]

    # Append assistant's message to history
    st.session_state.history.append({"message": assistant_message, "is_user": False})


# Take user input
st.text_input(
    "Enter your prompt:",
    key="prompt",
    placeholder="e.g. 'How can i help you ?'",
    on_change=generate_response,
)

# Display chat history
for message in st.session_state.history:
    if message["is_user"]:
        st.write(
            user_msg_container_html_template.replace("$MSG", message["message"]),
            unsafe_allow_html=True,
        )
    else:
        st.write(
            bot_msg_container_html_template.replace("$MSG", message["message"]),
            unsafe_allow_html=True,
        )
