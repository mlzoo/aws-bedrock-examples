
# UI-Enabled LLM Knowledge Base + Bedrock: Multi-Stage Reasoning (Teacher & Student RAG-Reflection) + Multi-Turn Dialogue

This article demonstrates a LLM-based multi-stage analysis and comprehension system with RAG for question answering.

## System Architecture

The system consists of three core stages:

1. Question Understanding Stage (Agent Perception)
2. Knowledge Retrieval and Answer Generation Stage
3. Answer Reflection and Improvement Stage

The system utilizes LangGraph and the Claude model provided by AWS Bedrock platform as the underlying language model, combined with vector databases for knowledge retrieval augmentation.

## Detailed Implementation

### 1. Question Understanding Stage

In this stage, the system takes on two roles - "teacher" and "student" - to provide insights from both perspectives:

```python
def understand_question(state: AgentState):
    teacher_prompt = """You are now an English grammar teacher. Please analyze the learner's question and identify:
    1. Main English grammar points involved
    2. Potential areas of confusion for learners
    3. Key aspects to consider when answering this question
    
    Question: {question}
    """
    
    student_prompt = """You are now an English learner. Please evaluate if the teacher's analysis helps understand the question:
    
    Original question: {question}
    Teacher's analysis: {teacher_analysis}
    """
    
    # Teacher's perspective analysis
    teacher_response = llm.invoke(teacher_prompt.format(
        question=state["question"]
    ))
    # Student's perspective evaluation
    student_response = llm.invoke(student_prompt.format(
        question=state["question"],
        teacher_analysis=teacher_response.content
    ))
    # Generate comprehensive understanding
    state["understanding"] = f"{teacher_response}\n{student_response}"
```

[Rest of the implementation sections follow the same pattern...]

## Workflow Orchestration

Using langgraph to build the complete processing flow:

```python
def create_thinker_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("understand", understand_question)
    workflow.add_node("retrieve_generate", retrieve_and_generate)
    workflow.add_node("reflect", reflect_and_improve)
    # Set up flow
    workflow.set_entry_point("understand")
    workflow.add_edge("understand", "retrieve_generate")
    workflow.add_edge("retrieve_generate", "reflect")
    return workflow.compile()
```

## Demo

The system provides detailed answers to knowledge base queries. For example, when uploading an English grammar document and asking questions like "what words end in ly",

The system will provide accurate and easy-to-understand answers through multiple rounds of analysis and improvement.

![](../pics/3_thinker_demo_en.png)

## Summary

[Code repository](../code/3_LangGraph_Bedrock_Thinker_Framework_with_UI/demo.py)