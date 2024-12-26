# pip install langchain langgraph chromadb langchain-community langchain-core boto3
# pip install -U langchain-aws

# 1. AWS Configuration and Initialization

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_core.messages import HumanMessage

region_name = 'us-west-2'
chat_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
max_tokens = 2048
temperature = 0.9

embedding_model_id="cohere.embed-multilingual-v3"
# Initialize Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)

# Configure Claude model
llm = ChatBedrock(
    model_id=chat_model_id,
    client=bedrock,
    model_kwargs={
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
)

# Configure embedding model
embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id=embedding_model_id
)


# 2. Document Processing and Vector Storage
# Load documents
loader = TextLoader("english-grammar.txt")
documents = loader.load()

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
splits = text_splitter.split_documents(documents)

# Create vector database
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)


# 3. Define Workflow Types and States
class AgentState(TypedDict):
    messages: list[str]
    next: str
    question: str
    context: str
    response: str


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Determine if conversation should continue
    if "Need more information" in last_message:
        return "continue"
    return "end"


# 4. Build Workflow Nodes
def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant documents"""
    question = state["question"]
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join(
        [doc.page_content for doc in docs]
        )
    state["context"] = context
    return state


def generate_response(state: AgentState) -> AgentState:
    """Generate answer"""
    question = state["question"]
    context = state["context"]

    prompt = f"""
    Answer the question based on the context below. If the answer cannot be found in the context, please explicitly state so.

    Context: {context}

    Question: {question}
    """

    print("Prompt:", prompt)
    # Correct way to call ChatBedrock
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])

    # Get response content
    response_content = response.content

    state["response"] = response_content
    state["messages"].append(response_content)
    return state


# 5. Build Workflow Graph
# Create workflow graph
workflow = StateGraph(AgentState)


def end_node(state: AgentState) -> AgentState:
    """End node, return current state directly"""
    return state


# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate_response)
workflow.add_node("end", end_node)  # Add end node

# Add edges and conditions
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "continue": "retrieve",
        "end": "end"
    }
)

# Set entry and exit points
workflow.set_entry_point("retrieve")
workflow.set_finish_point("end")


# 6. Usage Example
# Compile workflow
chain = workflow.compile()

# Process query
response = chain.invoke({
    "question": "What is Lesson 2 talking about?",
    "messages": [],
    "context": "",
    "response": "",
    "next": ""
})

print(f"Answer: {response['response']}")