# pip install langchain langgraph chromadb langchain-community langchain-core boto3
# pip install -U langchain-aws

# 1. AWS 配置和初始化

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # 更改导入路径
from langchain_community.document_loaders import TextLoader  # 更改导入路径
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_core.messages import HumanMessage

region_name = 'us-west-2'
chat_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
max_tokens = 2048
temperature = 0.9

embedding_model_id="cohere.embed-multilingual-v3"
# 初始化 Bedrock 客户端
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)

# 配置 Claude 模型
llm = ChatBedrock(
    model_id=chat_model_id,
    client=bedrock,
    model_kwargs={
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
)

# 配置嵌入模型
embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id=embedding_model_id
)


# 2. 文档处理和向量存储
# 加载文档
loader = TextLoader("english-grammar.txt")
documents = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
splits = text_splitter.split_documents(documents)

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)


# 3. 定义工作流类型和状态
class AgentState(TypedDict):
    messages: list[str]
    next: str
    question: str
    context: str
    response: str


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # 判断是否需要继续对话
    if "需要更多信息" in last_message:
        return "continue"
    return "end"


# 4. 构建工作流节点
def retrieve(state: AgentState) -> AgentState:
    """检索相关文档"""
    question = state["question"]
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join(
        [doc.page_content for doc in docs]
        )
    state["context"] = context
    return state


def generate_response(state: AgentState) -> AgentState:
    """生成答案"""
    question = state["question"]
    context = state["context"]

    prompt = f"""
    基于以下上下文回答问题。如果无法从上下文中找到答案，请明确说明。

    上下文：{context}

    问题：{question}
    """

    print("Prompt:", prompt)
    # 使用 ChatBedrock 的正确调用方式
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])

    # 获取响应内容
    response_content = response.content

    state["response"] = response_content
    state["messages"].append(response_content)
    return state


# 5. 构建工作流图
# 创建工作流图
workflow = StateGraph(AgentState)


# 定义结束节点
def end_node(state: AgentState) -> AgentState:
    """结束节点，直接返回当前状态"""
    return state


# 添加节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate_response)
workflow.add_node("end", end_node)  # 添加结束节点

# 添加边和条件
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "continue": "retrieve",
        "end": "end"
    }
)

# 设置入口和出口
workflow.set_entry_point("retrieve")
workflow.set_finish_point("end")


# 6. 使用示例
# 编译工作流
chain = workflow.compile()

# 处理查询
response = chain.invoke({
    "question": "Lesson 2 说了什么？",
    "messages": [],
    "context": "",
    "response": "",
    "next": ""
})

print(f"回答: {response['response']}")
