[上一篇文章](./1_LangGraph_Bedrock_RAG_Workflow.md)中讲了使用LangGraph+AWSBedrock做RAG，这篇文章继续讲解使用LangGraph的ReAct Agent来做自动化的tool call。

这次是构建一个自动调用天气预报API的智能天气查询助手。我们将通过带代码来展示怎么创建tool、构建workflow以及处理用户查询。

[完整代码](../code/2_LangGraph_Bedrock_Tool_Call/demo_cn.py)

## **1. 环境准备**

首先，我们需要安装必要的依赖：

```bash
pip install langchain langgraph chromadb langchain-community langchain-core boto3 langchain-aws
```

确保你有正确的 AWS 凭证配置和开通了Bedrock access。

## 2. 导入所需的库及配置LLM

这里使用Bedrock最新的的Claude Sonnet 3.5 v2模型，依然是us-west-2 region，这里的quota最多。

```python
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
```

## **3. 核心组件实现**

### **3.1 天气查询工具**

接着，我们创建一个模拟的天气查询工具：

```python
@tool
def weather_query(city: str) -> Dict:
    """查询指定城市的天气情况"""
    weather_data = {
        '深圳': {
            'temperature': random.randint(20, 28),
            'humidity': random.randint(60, 80),
            'condition': random.choice(['晴朗', '多云', '小雨']),
            'wind': random.choice(['微风', '东南风', '西北风'])
        },
        # 其他城市配置...
    }
    
    if city not in weather_data:
        return {"error": f"暂不支持查询{city}的天气"}
    
    weather = weather_data[city]
    return {
        "city": city,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": f"{weather['temperature']}°C",
        "humidity": f"{weather['humidity']}%",
        "condition": weather['condition'],
        "wind": weather['wind']
    }
```

### **3.2 配置 AWS Bedrock**

设置 Claude 3 模型：

```python
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=bedrock,
    model_kwargs={
        "max_tokens": 2048,
        "temperature": 0.9,
    }
).bind_tools([weather_query])
```

## **4. 工作流构建**

### **4.1 定义状态管理**

```python
def should_continue(state: MessagesState) -> str:
    """决定工作流是否继续"""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END
```

### **4.2 构建工作流图**

```python
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
```

## **5. 运行效果展示**

让我们看几个实际的查询效果：

```bash
问题: 深圳今天天气怎么样？

回答:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: 深圳 (Shenzhen), 北京 (Beijing), 上海 (Shanghai).
    Please provide weather information in a clear and friendly manner.
    When responding in Chinese, make the response natural and conversational.
human: 深圳今天天气怎么样？
ai: 
tool: {"city": "深圳", "time": "2024-12-26 17:10:28", "temperature": "23°C", "humidity": "65%", "condition": "多云", "wind": "微风"}
ai: 亲爱的朋友,让我为您报告一下深圳今天的天气情况吧。根据最新天气数据,今天深圳多云,气温23摄氏度,相对湿度65%,风力较小为微风天气。希望这些信息对您的生活作息和规划有所帮助。祝您今天愉快!需要查询其他城市天气信息随时告诉我哦。

问题: 北京的天气情况如何？

回答:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: 深圳 (Shenzhen), 北京 (Beijing), 上海 (Shanghai).
    Please provide weather information in a clear and friendly manner.
    When responding in Chinese, make the response natural and conversational.
human: 北京的天气情况如何？
ai: 
tool: {"city": "北京", "time": "2024-12-26 17:10:38", "temperature": "6°C", "humidity": "47%", "condition": "阴天", "wind": "北风"}
ai: 北京今天天气阴沉,温度有些寒冷,只有6摄氏度。不过湿度适中,47%还可以接受。北风吹拂,希望您外出多加一件外套保暖哦。总的来说,北京这几天的天气状况一般,不算太差。希望这些天气信息对您有帮助。还有什么需要了解的吗?

问题: 上海现在什么天气？能详细告诉我吗？

回答:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: 深圳 (Shenzhen), 北京 (Beijing), 上海 (Shanghai).
    Please provide weather information in a clear and friendly manner.
    When responding in Chinese, make the response natural and conversational.
human: 上海现在什么天气？能详细告诉我吗？
ai: 
tool: {"city": "上海", "time": "2024-12-26 17:10:43", "temperature": "23°C", "humidity": "56%", "condition": "多云", "wind": "东南风"}
ai: 你好,我很乐意为你查询上海当前的天气情况!根据最新数据显示,现在上海是多云天气,温度23摄氏度,相对湿度为56%。风向来自东南,属于微风的风力。总的来说,天气还算不错,既不太热也不太冷。如果你需要外出的话,建议穿一件舒适的长袖上衣就可以了。希望这些信息对你有帮助。如有任何其他疑问,欢迎随时告诉我!

问题: 请分别告诉我深圳和北京的天气对比。

回答:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: 深圳 (Shenzhen), 北京 (Beijing), 上海 (Shanghai).
    Please provide weather information in a clear and friendly manner.
    When responding in Chinese, make the response natural and conversational.
human: 请分别告诉我深圳和北京的天气对比。
ai: 
tool: {"city": "深圳", "time": "2024-12-26 17:10:50", "temperature": "20°C", "humidity": "75%", "condition": "晴朗", "wind": "东南风"}
ai: 
tool: {"city": "北京", "time": "2024-12-26 17:10:52", "temperature": "19°C", "humidity": "33%", "condition": "阴天", "wind": "西北风"}
ai: 好的,根据查询结果:

深圳当前气温20摄氏度,天气晴朗,湿度较高达75%,有东南风。而北京当前较深圳略凉一些,气温19摄氏度,天空阴沉,空气干燥湿度仅33%,吹西北风。所以深圳比北京暖和多云,湿度也更高。两地天气有一些区别,深圳较为晴朗温暖,北京则偏阴冷干燥些。总的来说,这个时节深圳的天气对户外活动会更加舒适一些。希望这个对比解答能给你一个直观的了解!
```

## **结语**

直接复制这个实例的代码，加上简单的修改，就能把你自己的代码跑起来，快试试吧～