import boto3
from typing import List, Dict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from datetime import datetime
import random

# AWS Bedrock配置
region_name = 'us-west-2'
chat_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
max_tokens = 2048
temperature = 0.9

# 初始化 Bedrock 客户端
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)


# 1. 定义天气查询工具
@tool
def weather_query(city: str) -> Dict:
    """查询指定城市的天气情况

    Args:
        city: 城市名称 (支持: '深圳', '北京', '上海')

    Returns:
        包含天气信息的字典
    """
    # Mock天气数据
    weather_data = {
        '深圳': {
            'temperature': random.randint(20, 28),
            'humidity': random.randint(60, 80),
            'condition': random.choice(['晴朗', '多云', '小雨']),
            'wind': random.choice(['微风', '东南风', '西北风'])
        },
        '北京': {
            'temperature': random.randint(5, 25),
            'humidity': random.randint(30, 50),
            'condition': random.choice(['晴朗', '阴天', '多云']),
            'wind': random.choice(['北风', '西北风', '微风'])
        },
        '上海': {
            'temperature': random.randint(15, 30),
            'humidity': random.randint(50, 70),
            'condition': random.choice(['多云', '阴天', '小雨']),
            'wind': random.choice(['东风', '东南风', '微风'])
        }
    }

    if city not in weather_data:
        return {
            "error": f"暂不支持查询{city}的天气。支持的城市包括：深圳、北京、上海"
        }

    weather = weather_data[city]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "city": city,
        "time": current_time,
        "temperature": f"{weather['temperature']}°C",
        "humidity": f"{weather['humidity']}%",
        "condition": weather['condition'],
        "wind": weather['wind']
    }


# 2. 设置工具节点
tools = [weather_query]
tool_node = ToolNode(tools=tools)

# 3. 配置 Claude 模型
llm = ChatBedrock(
    model_id=chat_model_id,
    client=bedrock,
    model_kwargs={
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
).bind_tools(tools)


# 4. 定义工作流函数
def should_continue(state: MessagesState) -> str:
    """决定是否继续执行工作流"""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState) -> Dict:
    """调用LLM模型"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# 5. 构建工作流图
workflow = StateGraph(MessagesState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 添加边
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# 编译工作流
app = workflow.compile()


# 6. 处理查询的函数
def process_query(question: str) -> List:
    """处理用户查询"""
    system_message = """You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: 深圳 (Shenzhen), 北京 (Beijing), 上海 (Shanghai).
    Please provide weather information in a clear and friendly manner.
    When responding in Chinese, make the response natural and conversational."""

    messages = [
        HumanMessage(content=system_message),
        HumanMessage(content=question)
    ]

    try:
        result = app.invoke({"messages": messages})
        return result["messages"]
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return []


# 测试代码
if __name__ == "__main__":
    # 测试案例
    test_questions = [
        "深圳今天天气怎么样？",
        "北京的天气情况如何？",
        "上海现在什么天气？能详细告诉我吗？",
        "请分别告诉我深圳和北京的天气对比。"
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        try:
            responses = process_query(question)
            print("\n回答:")
            for msg in responses:
                print(f"{msg.type}: {msg.content}")
        except Exception as e:
            print(f"处理问题时发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
