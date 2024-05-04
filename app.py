import pysqlite3
import sys, os
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint
import streamlit as st

HF_TOKEN = st.secrets["HF_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


@st.cache_resource()
def retrieve_documents():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")
        #api_key=HF_TOKEN, model_name="local:BAAI/bge-m3")
    db = Chroma(persist_directory="./db",
            embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs = {"k":1})
    return retriever

@st.cache_resource()
def create_chain(_retriever):
    template = """
    User: You are an AI Assistant that follows instructions well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

    Keep in mind, you will lose the job, if you answer out of CONTEXT questions

    CONTEXT: {context}
    Query: {question}

    Remember only return AI answer
    Assistant:
    """

    llm = HuggingFaceEndpoint(
    endpoint_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=512,
    top_k=1,
    top_p=0.9,
    temperature=0.5,
    repetition_penalty=1.1,
    streaming=False,
    )

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = ({
        "context": _retriever.with_config(run_name="Docs"),
        "question":RunnablePassthrough()
        }
        | prompt
        | llm
        | output_parser
        )
    return chain


def main():
    st.title("My LLM based smart CV")
    st.header("Ask what ever about sungwon's CV.")
    prompt = st.text_input("Enter your question")
    text_container = st.empty()
    text_debugger = st.empty()
    full_text = ""
    chain = create_chain(retrieve_documents())
    chunk = chain.invoke(prompt)
    text_container.write(chunk)


if __name__ == "__main__":
    main()
