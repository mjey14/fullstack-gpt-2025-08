import json
import streamlit as st

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from langchain.schema.runnable import RunnableLambda



class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, level):
    questions_chain = create_questions_chain(level)
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def prepare_inputs(docs, level):
    return {
        "context": format_docs(docs),
        "level": level,
    }


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)
st.title("QuizGPT")



with st.sidebar:
    docs = None
    topic = None
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    level = st.selectbox(
        "Choose your level.",
        (
            "Beginner",
            "Expert",
        ),
    )
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if openai_api_key:
    llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=openai_api_key
    )
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()



questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 3 (THREE) questions to test the user's knowledge about the text.

    The difficulty level is: {level}. 
    
    If the level is "Beginner":
    - Create very simple, straightforward questions about basic facts and concepts
    - Use simple vocabulary and clear, direct questions
    - Focus on who, what, when, where type questions
    - Avoid complex analysis, comparisons, or deep reasoning
    
    If the level is "Expert":
    - Create challenging questions that require deep understanding and analysis
    - Ask about complex relationships, implications, and advanced concepts
    - Include questions about cause-and-effect, critical thinking, and synthesis
    - Use sophisticated vocabulary and require expert-level knowledge

    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples for Beginner level:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital of Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
    Question examples for Expert level:
    
    Question: How did the fall of the Roman Republic influence modern democratic institutions?
    Answers: It had no influence|It established the concept of checks and balances(o)|It created the first constitution|It invented voting systems
    
    Question: What are the economic implications of climate change on global trade patterns?
    Answers: No significant impact|Disruption of supply chains and trade routes(o)|Increased trade opportunities|Complete trade cessation
    
    Question: Analyze the relationship between quantum mechanics and classical physics in modern technology.
    Answers: They are completely separate|Quantum mechanics extends classical physics(o)|Classical physics replaces quantum mechanics|They are identical concepts
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

def create_questions_chain(level):
    def prepare_inputs_with_level(docs):
        return prepare_inputs(docs, level)
    return RunnableLambda(prepare_inputs_with_level) | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    # Initialize session state for quiz tracking
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'correct_count' not in st.session_state:
        st.session_state.correct_count = 0
    if 'show_retry' not in st.session_state:
        st.session_state.show_retry = False
    if 'current_level' not in st.session_state:
        st.session_state.current_level = level
    
    # Check if level has changed and reset quiz
    if st.session_state.current_level != level:
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.session_state.correct_count = 0
        st.session_state.show_retry = False
        st.session_state.current_level = level
        st.cache_data.clear()  # Clear cache to generate new quiz
    
    response = run_quiz_chain(docs, topic if topic else file.name, level)
    
    if not st.session_state.quiz_submitted:
        with st.form("questions_form"):
            user_answers = {}
            
            for i, question in enumerate(response["questions"]):
                st.write(f"**Q{i+1}.** {question['question']}")
                value = st.radio(
                    "Select an answer",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"question_{i}"
                )
                user_answers[i] = value
                
                # Show immediate feedback if answer is selected
                if value is not None:
                    if {"answer": value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                    else:
                        st.error("Wrong!")
            
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                # Calculate correct answers
                correct_count = 0
                for i, question in enumerate(response["questions"]):
                    user_answer = user_answers.get(i)
                    if user_answer and {"answer": user_answer, "correct": True} in question["answers"]:
                        correct_count += 1
                
                st.session_state.quiz_submitted = True
                st.session_state.user_answers = user_answers
                st.session_state.correct_count = correct_count
                st.session_state.show_retry = (correct_count < len(response["questions"]))
                
                # Show results
                st.markdown("## Result")
                st.write(f"Score: {correct_count}/{len(response['questions'])}")

                if correct_count == len(response["questions"]):
                    st.balloons()
                    st.success("Well done!")
                else:
                    st.warning(f"You missed {len(response['questions']) - correct_count} question(s). Try again!")
                    st.session_state.show_retry = True
                st.rerun()
    
    else:
        # Show results after submission
        st.markdown("## Result")
        st.write(f"Score: {st.session_state.correct_count}/{len(response['questions'])}")

        if st.session_state.correct_count == len(response["questions"]):
            st.balloons()
            st.success("Great job!")
        else:
            st.warning(f"You got {len(response['questions']) - st.session_state.correct_count} wrong. Give it another try!")
    
    # Try Again button - always show after submission
    if st.session_state.quiz_submitted:
        st.markdown("---")
        if st.button("🔄 Try Again", type="primary"):
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.session_state.correct_count = 0
            st.session_state.show_retry = False
            st.rerun()








