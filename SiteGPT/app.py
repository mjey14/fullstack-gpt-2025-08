from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT        
    """
)

with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

if openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=openai_api_key
    )
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"
FILTER_URLS = [
    "https://developers.cloudflare.com/workers-ai/",
    "https://developers.cloudflare.com/vectorize/",
    "https://developers.cloudflare.com/ai-gateway/",
]

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Always respond strictly in the same language as the user's question.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = (inputs["question"] or "")[:1000]
    answers_chain = answers_prompt | llm
    limited_docs = docs[:3] if isinstance(docs, list) else docs
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": (doc.page_content or "")[:1200],
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in limited_docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Always format citations exactly as: (Source: URL) and use plain URLs (no markdown links).

            Always respond strictly in the same language as the user's question.

            Answers:
            {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = (inputs["question"] or "")[:1000]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{(answer['answer'] or '')[:800]}\n\n(Source: {answer['source']})\n"
        for answer in answers[:3]
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading Cloudflare docs...")
def load_website():
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = SitemapLoader(
        SITEMAP_URL,
        filter_urls=FILTER_URLS,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    docs = docs[:200]
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(openai_api_key=openai_api_key, batch_size=16),
    )
    return vector_store.as_retriever(search_kwargs={"k": 2})





retriever = load_website()
query = st.text_input("Ask a question about the docs.")
if query:
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = chain.invoke(query)
    st.markdown(result.content.replace("$", "\\$"))