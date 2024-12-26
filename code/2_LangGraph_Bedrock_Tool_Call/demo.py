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


# 1. Define weather query tool
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


# 2. Set up tool node
tools = [weather_query]
tool_node = ToolNode(tools=tools)

# 3. Configure Claude model
llm = ChatBedrock(
    model_id=chat_model_id,
    client=bedrock,
    model_kwargs={
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
).bind_tools(tools)


# 4. Define workflow functions
def should_continue(state: MessagesState) -> str:
    """Decide whether to continue the workflow"""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState) -> Dict:
    """Call the LLM model"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# 5. Build workflow graph
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

# Compile workflow
app = workflow.compile()


# 6. Query processing function
def process_query(question: str) -> List:
    """Process user queries"""
    system_message = """You are a helpful weather assistant that can check weather conditions for different cities.
    Always use the weather_query tool to get weather information.
    The weather tool supports these cities: New York, London, Shanghai.
    Please provide weather information in a clear and friendly manner."""

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


# Test code
if __name__ == "__main__":
    # Test cases
    test_questions = [
        "How's the weather in New York today?",
        "What's the weather like in London?",
        "Can you tell me the weather conditions in Shanghai in detail?",
        "Please compare the weather between New York and London."
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            responses = process_query(question)
            print("\nResponse:")
            for msg in responses:
                print(f"{msg.type}: {msg.content}")
        except Exception as e:
            print(f"Error occurred while processing question: {str(e)}")
            import traceback
            print(traceback.format_exc())