import streamlit as st

# 设置页面布局
st.set_page_config(layout="wide")

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any
# 如果是python3.12以下，需要改用以下代码
# from typing_extensions import TypedDict
# from typing import List, Dict, Any

import os

# 1. AWS配置和初始化
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

# 2. 定义状态类型
class AgentState(TypedDict):
    question: str
    understanding: str
    retrieved_docs: List[str]
    current_answer: str
    final_answer: str

# 3. Agent感知阶段
def understand_question(state: AgentState) -> AgentState:
    print("==================== 理解阶段 ====================")
    teacher_prompt = """你现在是一位语法教师。请分析以下学习者的问题，识别：
    1. 涉及的主要英语语法知识点
    2. 学习者可能的困惑点
    3. 回答这个问题需要考虑的关键方面
    
    问题: {question}
    """
    
    student_prompt = """你现在是一位英语学习者。请评估教师的分析是否有助于理解问题：
    
    原始问题: {question}
    教师分析: {teacher_analysis}
    """
    
    # 教师分析
    teacher_response = llm.invoke(teacher_prompt.format(
        question=state["question"]
    ))
    print("teacher_response:", teacher_response.content)
    # 学生评估
    student_response = llm.invoke(student_prompt.format(
        question=state["question"],
        teacher_analysis=teacher_response.content
    ))
    print("student_response:", student_response.content)

    # 生成理解总结
    state["understanding"] = f"{teacher_response.content}\n---\n{student_response.content}"
    return state

# 4. 知识检索和答案生成阶段
class KnowledgeBase:
    def __init__(self, docs_path: str):
        # 加载文档
        loader = TextLoader(docs_path)
        documents = loader.load()
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)
        
        # 创建向量存储
        self.db = FAISS.from_documents(texts, embeddings)
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        return self.db.similarity_search(query, k=k)

def retrieve_and_generate(state: AgentState) -> AgentState:
    print("==================== 检索阶段 ====================")
    # 结合问题和理解进行检索
    query = f"{state['question']}\n{state['understanding']}"
    kb = KnowledgeBase("uploaded_document.txt")
    docs = kb.retrieve(query)
    state["retrieved_docs"] = [doc.page_content for doc in docs]
    
    # 生成初步答案
    generation_prompt = """基于以下参考资料和问题理解，生成一个详细的答案：
    
    问题: {question}
    问题理解: {understanding}
    参考资料: {docs}
    """
    
    response = llm.invoke(generation_prompt.format(
        question=state["question"],
        understanding=state["understanding"],
        docs="\n".join(state["retrieved_docs"])
    ))
    print("retrieve_and_generate:", response.content)
    state["current_answer"] = response.content

    return state

# 5. 答案反思阶段
def reflect_and_improve(state: AgentState) -> AgentState:
    print("==================== 反思阶段 ====================")
    reflection_prompt = """请从以下三个方面评估和改进当前答案：
    1. 内容合理性
    2. 内容完整性
    3. 实用性和帮助性
    
    当前答案: {current_answer}
    参考资料: {docs}
    """
    
    reflection = llm.invoke(reflection_prompt.format(
        current_answer=state["current_answer"],
        docs="\n".join(state["retrieved_docs"])
    ))
    print("reflection:", reflection.content)
    improvement_prompt = """基于上述反思，请生成改进后的最终答案：
    
    反思结果: {reflection}
    当前答案: {current_answer}
    """
    
    final_answer = llm.invoke(improvement_prompt.format(
        reflection=reflection,
        current_answer=state["current_answer"]
    ))
    print("final_answer:", final_answer.content)
    state["final_answer"] = final_answer.content
    return state

# 6. 构建工作流
def create_thinker_agent():
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("understand", understand_question)
    workflow.add_node("retrieve_generate", retrieve_and_generate)
    workflow.add_node("reflect", reflect_and_improve)
    
    # 设置流程
    workflow.set_entry_point("understand")
    workflow.add_edge("understand", "retrieve_generate")
    workflow.add_edge("retrieve_generate", "reflect")
    
    # 编译工作流
    return workflow.compile()

# 7. 使用示例
def answer_question(question: str, conversation_history: List[str]) -> str:
    print("==================== 开始 ====================")
    thinker = create_thinker_agent()
    
    # 将对话历史作为上下文
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

# Streamlit 应用
def main():
    st.title("多轮对话问答系统")
    
    # 文件上传
    uploaded_file = st.file_uploader("上传文本文件", type=["txt"])
    if uploaded_file is not None:
        # 保存上传的文件
        with open("uploaded_document.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("文件上传成功！")
    
    # 初始化会话状态
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    # 显示聊天历史
    for message in st.session_state.conversation:
        if message.startswith("用户: "):
            with st.chat_message("user"):
                st.write(message[4:])
        elif message.startswith("系统: "):
            with st.chat_message("assistant"):
                st.write(message[4:])
    
    # 用户输入
    user_input = st.chat_input("请输入您的问题")
    
    if user_input:
        # 检查是否上传了文件
        if not os.path.exists("uploaded_document.txt"):
            st.error("请先上传文件！")
        else:
            try:
                # 显示用户输入
                with st.chat_message("user"):
                    st.write(user_input)
                
                # 回答问题
                with st.spinner("思考中..."):
                    answer = answer_question(user_input, st.session_state.conversation)
                
                # 显示系统回答
                with st.chat_message("assistant"):
                    st.write(answer)
                
                # 更新会话状态
                st.session_state.conversation.append(f"用户: {user_input}")
                st.session_state.conversation.append(f"系统: {answer}")
                
            except Exception as e:
                st.error(f"发生错误：{e}")

if __name__ == "__main__":
    main()