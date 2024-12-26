Here's the translation of your content into natural English:

In the [previous article](./1_LangGraph_Bedrock_RAG_Workflow.md), we discussed using LangGraph + AWS Bedrock for RAG. This article continues by explaining how to use LangGraph's ReAct Agent for automated tool calls.

We'll build an intelligent weather query assistant that automatically calls weather forecast APIs. We'll demonstrate how to create tools, build workflows, and handle user queries through code examples.

[Full code](../code/2_LangGraph_Bedrock_Tool_Call/demo.py)

## **1. Environment Setup**

First, let's install the necessary dependencies:

```bash
pip install langchain langgraph chromadb langchain-community langchain-core boto3 langchain-aws
```

Make sure you have proper AWS credentials configured and Bedrock access enabled.

## **2. Import Required Libraries and Configure LLM**

We'll use Bedrock's latest Claude Sonnet 3.5 v2 model in the us-west-2 region, which has the highest quota.

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

# AWS Bedrock Configuration
region_name = 'us-west-2'
chat_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
max_tokens = 2048
temperature = 0.9

# Initialize Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)
```



## **3. Core Component Implementation**



```python
@tool
def weather_query(city: str) -> Dict:
    """Query weather information for a specified city

    Args:
        city: City name (supported: 'New York', 'London', 'Shanghai')

    Returns:
        Dictionary containing weather information
    """
    # Mock weather data
    weather_data = {
        'New York': {
            'temperature': random.randint(-5, 25),
            'humidity': random.randint(40, 70),
            'condition': random.choice(['Sunny', 'Cloudy', 'Rain']),
            'wind': random.choice(['Light breeze', 'North wind', 'Southeast wind'])
        },
        'London': {
            'temperature': random.randint(5, 20),
            'humidity': random.randint(60, 85),
            'condition': random.choice(['Cloudy', 'Foggy', 'Light rain']),
            'wind': random.choice(['West wind', 'Southwest wind', 'Light breeze'])
        },
        'Shanghai': {
            'temperature': random.randint(15, 30),
            'humidity': random.randint(50, 70),
            'condition': random.choice(['Cloudy', 'Overcast', 'Light rain']),
            'wind': random.choice(['East wind', 'Southeast wind', 'Light breeze'])
        }
    }

    if city not in weather_data:
        return {
            "error": f"Weather query for {city} is not supported. Supported cities include: New York, London, Shanghai"
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
```





## **4. Workflow Construction**

### **4.1 Define State Management**



```python
def should_continue(state: MessagesState) -> str:
    """Decide whether to continue the workflow"""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END
```



### **4.2 Build Workflow Graph**



```python
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Add edges
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



## **5. Demo Results**

Let's look at some actual query results:



```python
Question: How's the weather in New York today?

Response:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: New York, London, Shanghai.
    Please provide weather information in a clear and friendly manner.
human: How's the weather in New York today?
ai: 
tool: {"city": "New York", "time": "2024-12-26 17:26:26", "temperature": "23°C", "humidity": "66%", "condition": "Sunny", "wind": "North wind"}
ai: Hello! I'd be happy to tell you about the weather in New York today. 

Currently in New York:
- Temperature: 23°C
- Weather Condition: Sunny
- Humidity: 66%
- Wind: Coming from the North

It's a lovely sunny day in New York with comfortable temperatures! Perfect weather for outdoor activities. The moderate humidity level and north wind should make it feel quite pleasant outside.

Is there anything else you'd like to know about the weather?

Question: What's the weather like in London?

Response:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: New York, London, Shanghai.
    Please provide weather information in a clear and friendly manner.
human: What's the weather like in London?
ai: 
tool: {"city": "London", "time": "2024-12-26 17:26:32", "temperature": "6°C", "humidity": "78%", "condition": "Foggy", "wind": "West wind"}
ai: Hello! I'll help you with the weather information for London. Here's what it's like:

Temperature: 6°C
Condition: Foggy
Humidity: 78%
Wind: West wind

It's quite a foggy day in London with cool temperatures. Make sure to dress warmly and perhaps bring a light jacket. The fog might affect visibility, so take care if you're planning to go out!

Is there anything else you'd like to know about the weather in London or any other city I can help you with?

Question: Can you tell me the weather conditions in Shanghai in detail?

Response:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: New York, London, Shanghai.
    Please provide weather information in a clear and friendly manner.
human: Can you tell me the weather conditions in Shanghai in detail?
ai: 
tool: {"city": "Shanghai", "time": "2024-12-26 17:26:39", "temperature": "18°C", "humidity": "57%", "condition": "Overcast", "wind": "Southeast wind"}
ai: Let me provide you with the current weather details for Shanghai:

Temperature: It's currently 18°C in Shanghai
Condition: The sky is Overcast
Humidity: The humidity level is at 57%
Wind: There's a Southeast wind blowing

It's a relatively mild day in Shanghai, though the overcast conditions mean it's somewhat cloudy. The humidity is at a moderate level, making it fairly comfortable. The southeast wind could bring some maritime influence from the East China Sea.

Is there anything specific about Shanghai's weather conditions you'd like me to explain further?

Question: Please compare the weather between New York and London.

Response:
human: You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: New York, London, Shanghai.
    Please provide weather information in a clear and friendly manner.
human: Please compare the weather between New York and London.
ai: 
tool: {"city": "New York", "time": "2024-12-26 17:26:56", "temperature": "21°C", "humidity": "66%", "condition": "Sunny", "wind": "Light breeze"}
tool: {"city": "London", "time": "2024-12-26 17:26:56", "temperature": "6°C", "humidity": "77%", "condition": "Cloudy", "wind": "Light breeze"}
ai: Let me compare the weather conditions between New York and London for you:

New York:
- Temperature: 21°C
- Condition: Sunny
- Humidity: 66%
- Wind: Light breeze

London:
- Temperature: 6°C
- Condition: Cloudy
- Humidity: 77%
- Wind: Light breeze

Comparison:
- New York is significantly warmer today, with a temperature of 21°C compared to London's 6°C (a 15°C difference)
- While New York is enjoying sunny conditions, London is experiencing cloudy weather
- London has slightly higher humidity (77%) compared to New York (66%)
- Both cities are experiencing light breeze conditions

Overall, New York is having a warmer and sunnier day, while London is experiencing a cooler, cloudy day, which is quite typical for these cities!
```



## **Conclusion**

You can get started by simply copying this example code and making minor modifications to run your own implementation. Give it a try!

