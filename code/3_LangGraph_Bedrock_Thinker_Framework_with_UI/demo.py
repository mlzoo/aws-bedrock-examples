import streamlit as st

# Set page layout
st.set_page_config(layout="wide")

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Use FAISS instead of Chroma
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any
# For Python versions below 3.12, use the following code instead
# from typing_extensions import TypedDict
# from typing import List, Dict, Any

import os

# 1. AWS Configuration and Initialization
region_name = 'us-west-2'
chat_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
max_tokens = 2048
temperature = 0.9
embedding_model_id = "cohere.embed-multilingual-v3"

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)

llm = ChatBedrock(
    model_id=chat_model_id,
    client=bedrock,
    model_kwargs={
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
)

embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id=embedding_model_id
)

# 2. Define State Type
class AgentState(TypedDict):
    question: str
    understanding: str
    retrieved_docs: List[str]
    current_answer: str
    final_answer: str

# 3. Agent Perception Phase
def understand_question(state: AgentState) -> AgentState:
    print("==================== Understanding Phase ====================")
    teacher_prompt = """You are now a grammar teacher. Please analyze the learner's question and identify:
    1. The main English grammar points involved
    2. Potential points of confusion for the learner
    3. Key aspects to consider when answering this question
    
    Question: {question}
    """
    
    student_prompt = """You are now an English learner. Please evaluate if the teacher's analysis helps understand the question:
    
    Original question: {question}
    Teacher's analysis: {teacher_analysis}
    """
    
    # Teacher analysis
    teacher_response = llm.invoke(teacher_prompt.format(
        question=state["question"]
    ))
    print("teacher_response:", teacher_response.content)
    # Student evaluation
    student_response = llm.invoke(student_prompt.format(
        question=state["question"],
        teacher_analysis=teacher_response.content
    ))
    print("student_response:", student_response.content)

    # Generate understanding summary
    state["understanding"] = f"{teacher_response.content}\n---\n{student_response.content}"
    return state

# 4. Knowledge Retrieval and Answer Generation Phase
class KnowledgeBase:
    def __init__(self, docs_path: str):
        # Load documents
        loader = TextLoader(docs_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.db = FAISS.from_documents(texts, embeddings)
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        return self.db.similarity_search(query, k=k)

def retrieve_and_generate(state: AgentState) -> AgentState:
    print("==================== Retrieval Phase ====================")
    # Combine question and understanding for retrieval
    query = f"{state['question']}\n{state['understanding']}"
    kb = KnowledgeBase("uploaded_document.txt")
    docs = kb.retrieve(query)
    
    # Limit the length of retrieved documents
    max_doc_length = 500  # Maximum characters per document
    truncated_docs = []
    for doc in docs:
        if len(doc.page_content) > max_doc_length:
            truncated_docs.append(doc.page_content[:max_doc_length] + "...")
        else:
            truncated_docs.append(doc.page_content)
    
    state["retrieved_docs"] = truncated_docs
    
    # Batch process answer generation
    generation_prompt = """Based on the following reference materials and question understanding, generate a detailed answer:
    
    Question: {question}
    Question Understanding: {understanding}
    Reference Materials: {docs}
    """
    
    # Split reference materials into smaller chunks
    docs_text = "\n".join(truncated_docs)
    max_chunk_size = 1000
    docs_chunks = [docs_text[i:i+max_chunk_size] for i in range(0, len(docs_text), max_chunk_size)]
    
    # Process chunks and merge results
    responses = []
    for chunk in docs_chunks:
        response = llm.invoke(generation_prompt.format(
            question=state["question"],
            understanding=state["understanding"][:500],  # Limit understanding length
            docs=chunk
        ))
        responses.append(response.content)
    
    state["current_answer"] = "\n".join(responses)
    return state

# 5. Answer Reflection Phase
def reflect_and_improve(state: AgentState) -> AgentState:
    print("==================== Reflection Phase ====================")
    reflection_prompt = """Please evaluate and improve the current answer from the following three aspects:
    1. Content reasonableness
    2. Content completeness
    3. Practicality and helpfulness
    
    Current answer: {current_answer}
    Reference materials: {docs}
    """
    
    reflection = llm.invoke(reflection_prompt.format(
        current_answer=state["current_answer"],
        docs="\n".join(state["retrieved_docs"])
    ))
    print("reflection:", reflection.content)
    improvement_prompt = """Based on the above reflection, please generate an improved final answer:
    
    Reflection results: {reflection}
    Current answer: {current_answer}
    """
    
    final_answer = llm.invoke(improvement_prompt.format(
        reflection=reflection,
        current_answer=state["current_answer"]
    ))
    print("final_answer:", final_answer.content)
    state["final_answer"] = final_answer.content
    return state

# 6. Build Workflow
def create_thinker_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("understand", understand_question)
    workflow.add_node("retrieve_generate", retrieve_and_generate)
    workflow.add_node("reflect", reflect_and_improve)
    
    # Set workflow
    workflow.set_entry_point("understand")
    workflow.add_edge("understand", "retrieve_generate")
    workflow.add_edge("retrieve_generate", "reflect")
    
    # Compile workflow
    return workflow.compile()

# 7. Usage Example
def answer_question(question: str, conversation_history: List[str]) -> str:
    print("==================== Start ====================")
    thinker = create_thinker_agent()
    
    # Use conversation history as context
    context = "\n".join(conversation_history)
    question_with_context = f"{context}\n{question}"
    
    initial_state = AgentState(
        question=question_with_context,
        understanding="",
        retrieved_docs=[],
        current_answer="",
        final_answer=""
    )
    
    final_state = thinker.invoke(initial_state)
    return final_state["final_answer"]

# Streamlit Application
def main():
    st.title("Multi-turn Dialogue QA System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    if uploaded_file is not None:
        # Save uploaded file
        with open("uploaded_document.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    # Display chat history
    for message in st.session_state.conversation:
        if message.startswith("User: "):
            with st.chat_message("user"):
                st.write(message[6:])
        elif message.startswith("System: "):
            with st.chat_message("assistant"):
                st.write(message[8:])
    
    # User input
    user_input = st.chat_input("Please enter your question")
    
    if user_input:
        # Check if file is uploaded
        if not os.path.exists("uploaded_document.txt"):
            st.error("Please upload a file first!")
        else:
            try:
                # Display user input
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Answer question
                with st.spinner("Thinking..."):
                    answer = answer_question(user_input, st.session_state.conversation)
                
                # Display system response
                with st.chat_message("assistant"):
                    st.write(answer)
                
                # Update conversation state
                st.session_state.conversation.append(f"User: {user_input}")
                st.session_state.conversation.append(f"System: {answer}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()