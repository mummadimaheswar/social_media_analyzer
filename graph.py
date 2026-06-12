import os
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langfuse.langchain import CallbackHandler

# Define the Expected Output Schema
class PostAnalysis(BaseModel):
    tone: str = Field(description="The emotional tone of the post (e.g., Positive, Negative, Enthusiastic, Serious).")
    intent: str = Field(description="The primary purpose of the post (e.g., Promotion, Complaint, Informational, Engagement).")
    communication_style: str = Field(description="The style of communication (e.g., Announcement, Storytelling, Direct, Conversational).")
    summary: str = Field(description="A brief summary explaining the tone, intent, and style.")

# Define the State for LangGraph
class GraphState(TypedDict):
    post_text: str
    analysis: dict

# Define the Node Function
def analyze_post_node(state: GraphState) -> GraphState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Enforce structured output
    structured_llm = llm.with_structured_output(PostAnalysis)
    
    prompt = PromptTemplate.from_template(
        "Analyze the following social media post text and classify its tone, intent, and communication style.\n\n"
        "Post:\n{post_text}"
    )
    
    chain = prompt | structured_llm
    
    # Initialize Langfuse CallbackHandler
    langfuse_handler = CallbackHandler()
    
    result = chain.invoke(
        {"post_text": state["post_text"]},
        config={"callbacks": [langfuse_handler]}
    )
    
    analysis_data = result.model_dump()
    try:
        # Langfuse CallbackHandler >= 2.0.0
        analysis_data["trace_url"] = langfuse_handler.get_trace_url()
    except AttributeError:
        # Langfuse instance is available via langfuse_handler.auth_check inside _langfuse_client maybe?
        try:
            analysis_data["trace_url"] = langfuse_handler._langfuse_client.get_trace_url(trace_id=langfuse_handler.last_trace_id)
        except Exception:
            analysis_data["trace_url"] = None
    
    return {"analysis": analysis_data}

# Build the LangGraph Workflow
workflow = StateGraph(GraphState)

workflow.add_node("analyze", analyze_post_node)

workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", END)

# Compile the graph
app = workflow.compile()

def run_analysis(post_text: str) -> dict:
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
    
    inputs = {"post_text": post_text}
    result = app.invoke(inputs)
    return result["analysis"]
