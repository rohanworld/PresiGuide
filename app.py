import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

# Datasources
pdf_url = "https://presidencyuniversity.in/wp-content/uploads/2024/08/Student-Handbook-2024_Website-upload.pdf"
web_urls = [
    "https://presidencyuniversity.in/about-us/about-presidency/",
    "https://presidencyuniversity.in/",
    "https://presidencyuniversity.in/academic-programmes/",
    "https://presidencyuniversity.in/university/academic-council/",
    "https://presidencyuniversity.in/faculty_cat/civil-engineering-department/",
    "https://presidencyuniversity.in/faculty_cat/chemistry-department/",
    "https://presidencyuniversity.in/faculty_cat/computer-science/",
    "https://presidencyuniversity.in/faculty_cat/languages/",
    "https://presidencyuniversity.in/faculty_cat/electrical-electronics/",
    "https://presidencyuniversity.in/faculty_cat/electronics-communication/",
    "https://presidencyuniversity.in/faculty_cat/learning-development/",
    "https://presidencyuniversity.in/faculty_cat/physics/",
    "https://presidencyuniversity.in/faculty_cat/mathematics/",
    "https://presidencyuniversity.in/faculty_cat/management/",
    "https://presidencyuniversity.in/faculty_cat/school-of-law/",
    "https://presidencyuniversity.in/faculty_cat/petroleum/",
    "https://presidencyuniversity.in/faculty_cat/design/",
    "https://presidencyuniversity.in/faculty_cat/commerce-and-economics/",
    "https://presidencyuniversity.in/faculty_cat/media-studies/",
    "https://en.wikipedia.org/wiki/Presidency_University,_Bangalore"
]


def load_pdf(url):
    loader = PyPDFLoader(url)
    return loader.load() 
def load_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

pdf_data = load_pdf(pdf_url)
web_data = []
for url in web_urls:
    web_text = load_web_page(url)
    web_data.append(web_text)  
web_data_combined = "\n".join(web_data)

pdf_data_combined = "\n".join(doc.page_content for doc in pdf_data)

# Combine PDF data and web page data
combined_data = pdf_data_combined + "\n" + web_data_combined

# Create One Whole Document objects from combined data
documents = [Document(page_content=combined_data, metadata={"source": "combined_source"})]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)


# Create LLM, Embeddings and VectorDB
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    max_tokens=500,
    google_api_key="AIzaSyApA57OGOk-Bcppba2eY3Hj0l9s5z3_uP8"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyApA57OGOk-Bcppba2eY3Hj0l9s5z3_uP8"
)
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})


system_prompt = (
    "You are PresiGuide, a knowledgeable virtual assistant for answering questions based on the student handbook and additional information about Presidency University. "
    "I’m here to help you find information about university policies, guidelines, academic programmes, and other important details. "
    "Provide the answer to the user query in your very first sentence of response and then follow up description. If asked for a list or names, provide the information in a point-wise manner. "
    "Use the retrieved context from the handbook and web pages to provide accurate and concise answers. If you do not have sufficient information "
    "to answer the question, respond politely and inform the user that you cannot provide an answer. Limit your responses to 8 sentences. "
    "When user asks the question which is not related to presidency university, or other sort of questions, you should politely decline the same with a proper message."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answer chain and RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

import gradio as gr

def respond_to_query(user_input, history):
    history = history or []

    if user_input.lower() in ["who are you?", "what are you?", "tell me about yourself"]:
        bot_response = "Hello! I’m PresiGuide, your virtual assistant for the student handbook of Presidency University, created by Mallesh C N. " \
                       "I’m here to help you find information about university policies, guidelines, rules, and other important details. " \
                       "Just ask me anything related to the student handbook, and I’ll do my best to provide you with accurate and helpful answers."
    elif user_input.lower() in ["who created you", "who is your creator", "who built you"]:
        bot_response = "I was created by Mallesh C N, a recent CSE graduate from Presidency University."
    else:
        response = rag_chain.invoke({"input": user_input})
        bot_response = response["answer"]

    # Update history
    history.append((user_input, bot_response))

    # Convert history to a chats UI
    chat_history = [(u, b) for u, b in history]
    return chat_history, history, ""  # Clear the input box by returning an empty string


custom_css = """
#message-box {
    width: 300px;  /* Adjust width as needed */
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as block:  
    gr.Markdown("""<h1><center>PresiGuide - Presidency University ChatBot</center></h1>
    <center>Ask questions about the University and get answers based on the official handbook.</center>
    """)

    with gr.Row():  
        chatbot = gr.Chatbot()

    with gr.Row():  
        message = gr.Textbox(
            placeholder="Enter your question here...",
            lines=2,
            elem_id="message-box"  
        )
        submit = gr.Button("SEND")

    state = gr.State()

    submit.click(
        fn=respond_to_query,
        inputs=[message, state],
        outputs=[chatbot, state, message] 
    )

block.launch(debug=True)

