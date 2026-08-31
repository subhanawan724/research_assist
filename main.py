

import os
import re
import json
import uuid
import time
from huggingface_hub import InferenceClient
from tavily import TavilyClient

from sklearn.cluster import KMeans

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langsmith import Client as LangSmithClient
from langchain_core.tracers import LangChainTracer
from typing import Annotated
from langgraph.prebuilt import InjectedState





HF_TOKEN = os.environ["HF_TOKEN"]
TAVILY_KEY = os.environ["TAVILY_KEY"]
GROQ_KEY = os.environ["GROQ_KEY"]
LANGSMITH_KEY = os.environ["LANGSMITH_KEY"]

client = InferenceClient(provider="featherless-ai", token=HF_TOKEN)
tavily = TavilyClient(api_key=TAVILY_KEY)


langsmith_client = LangSmithClient(api_key=LANGSMITH_KEY)
tracer = LangChainTracer(project_name="academic-research-assistant", client=langsmith_client)

chat_model = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=GROQ_KEY, temperature=0)
embedding_client = InferenceClient(token=HF_TOKEN)   
def get_embedding(text: str) -> list:
    result = embedding_client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
    return result



@tool
def search_web(query: str) -> str:
    """Search the web for papers, articles, or information on a topic. 
    Returns titles, URLs, and short snippets. Use this FIRST to discover 
    candidates, then use fetch_full_content on promising results before 
    treating them as reliable sources — snippets alone are not enough 
    for analysis or comparison."""
    results = tavily.search(query=query, max_results=3)
    formatted = "\n\n".join(
        [f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}" for r in results["results"]]
    )
    return formatted



@tool
def fetch_full_content(url: str) -> str:
    """Fetch the full text content of a specific webpage/paper given its URL. 
    You MUST use this before including any paper in your final analysis or 
    comparison — never rely on search snippets alone for judgments."""
    try:
        result = tavily.extract(urls=[url])
        if result["results"]:
            content = result["results"][0]["raw_content"]
            return content[:2500]
        else:
            return "Could not extract content from this URL."
    except Exception as e:
        return f"Error fetching content: {e}"


@tool
def cluster_papers(state: Annotated[dict, InjectedState]) -> str:
    """Cluster all papers fetched so far via fetch_full_content in this session. 
    Call this with no arguments — it automatically reads everything you've fetched."""

    messages = state.get("messages", [])

    fetched_content=[]

    for msg in messages:
        if hasattr(msg,"name") and msg.name =="fetch_full_content" and msg.status =="success":
            fetched_content.append(msg.content)
    
    if len(fetched_content) < 2:
        return "Not enough papers to cluster (need at least 2)."

    embeddings = [get_embedding(content) for content in fetched_content]
    num_clusters = min(3, len(fetched_content))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    groups = {}
    for label, entry in zip(labels, fetched_content):
        groups.setdefault(int(label), []).append(entry)

    result_parts = []
    for gid, full_entries in groups.items():
        titles_only = [e.split("\n")[0] for e in full_entries]
        block = f"Cluster {gid} ({len(full_entries)} papers) — Titles: {', '.join(titles_only)}\n\nFull content of each paper in this cluster:\n" + "\n---\n".join(full_entries)
        result_parts.append(block)

    
    return f"Clustered {len(fetched_content)} papers total.\n\n" + "\n\n=====\n\n".join(result_parts)



@tool
def compare_content(cluster_data: str) -> str:
    """Analyze the full content of papers within a cluster (or across clusters) to find 
    genuine research gaps — not just by counting papers, but by examining what specific 
    angles, methods, or applications are covered vs missing. Input should be the output 
    from cluster_papers (or any block of paper content). Returns a detailed gap analysis."""
    prompt = f"""Below are groups of papers with their full content:

{cluster_data}

Analyze the actual content (not just counting papers) to identify genuine research gaps:
- What specific angles, methods, or applications appear well-covered (multiple papers, deep discussion)?
- What specific angles are mentioned only briefly or not at all?
- Are there contradictions or unexplored combinations between clusters?

Give a detailed, specific analysis — avoid vague statements. Reference actual content from the papers."""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()



@tool
def write_report(query: str, findings_and_gaps: str) -> str:
    """Write a final, well-structured research report for the user based on all 
    collected findings and identified gaps. Use this as your LAST step, only after 
    you have gathered sufficient papers (with full content) and analyzed them for gaps. 
    Input should include the original query and a summary of everything found/analyzed 
    as a single text string (join multiple points into one string, not a list)."""
    prompt = f"""User's original request: {query}

All findings and gap-analysis so far:
{findings_and_gaps}

Write a clear, well-organized final report for the user with these sections:
1. Summary of papers found (brief, with titles)
2. Key themes covered
3. Identified research gaps (be specific)
4. Suggested next steps for someone wanting to research these gaps

Write in clear prose, not just bullet dumps."""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()



@tool
def ask_user(question: str) -> str:
    """Ask the user a clarifying question when you need their input to proceed 
    effectively — for example, if the topic is very broad and could go multiple 
    directions, or if you're unsure about scope. Use sparingly — only when genuinely 
    needed, not for every minor decision. Returns the user's response."""
    user_response = interrupt(question)
    return user_response


tools = [search_web, fetch_full_content, cluster_papers, compare_content, write_report, ask_user]
model_with_tools = chat_model.bind_tools(tools)






class AgentState(MessagesState):
    max_steps: int
    steps_taken: int



SYSTEM_PROMPT="""You are an autonomous research assistant. Your job is to process the user's research request thoroughly and efficiently using the tools available to you, while dynamically adapting to the type of input provided.

CRITICAL RULES & EXECUTION MODES:

1. INTENT & ROUTING (CHECK FIRST):
   - MODE A (Web Investigation): If the user provides a general research topic or question without source text/URLs, you must execute the full search-and-fetch pipeline.
   - MODE B (Direct Analysis / Provided Data): If the user provides pre-written text, notes, extracted papers, or specific URLs, SKIP `search_web` and `fetch_full_content` entirely. Do not invent or search for new data; process what is explicitly given.
   - MODE C (Partial Task): If the user explicitly asks for a single operation (e.g., "just fetch this URL" or "just cluster this"), perform ONLY that requested action and return the result without forcing a full report.

2. TOOL DISCIPLINE (FOR MODE A ONLY):
   - Call ONLY ONE tool at a time. NEVER call multiple tools in a single turn — wait for each tool's real result before deciding your next action.
   - MAXIMUM SEARCHES: A maximum of 3 times per session.
   - MAXIMUM FETCHES: A maximum of 3 times IN TOTAL across the entire session.
   - CANDIDATE POOL STRATEGY: After completing your searches, review results to form a pool of 5 to 6 relevant URLs, pick the absolute best 3, and call `fetch_full_content` strictly on those top 3.
   - Never judge, compare, or include a paper based on a snippet alone; always fetch full content for real URLs returned by search.

3. PROCESSING & SYNTHESIS (MODES A & B):
   - Once full content is fetched (Mode A) or provided (Mode B), IMMEDIATELY move to clustering: use `cluster_papers` to group them thematically using actual titles and content.
   - When calling `cluster_papers`, include the complete extracted data of ALL papers without omitting anything.
   - Use `compare_content` with the ACTUAL `cluster_papers` output.

4. FINALIZATION:
   - Only call `ask_user` if the request is genuinely ambiguous or too broad.
   - Call `write_report` ONLY as the final step of a full research session, using real collected findings — never fabricate data or findings.
   - Be thorough, avoid redundant tool calls, and maintain strict factual fidelity to the provided or retrieved data."""

def agent_brain(state: AgentState) -> dict:
    messages = state["messages"]

    original_human_message = state["messages"][0]

    

    #search_already_used = any(
        #hasattr(m, "tool_calls") and m.tool_calls and
        #any(tc["name"] == "search_web" for tc in m.tool_calls)
        #for m in state["messages"]
   # )

    if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    if len(messages) > 11:
            messages = [messages[0], original_human_message] + messages[-10:]


    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model_with_tools.invoke(messages, config={"callbacks": [tracer]})
            break
        
        except Exception as e:
          error_str = str(e).lower()
          if "too large" in error_str:
            raise  # retry se fayda nahi, seedha fail hone do (ya better: aur trim karke retry)
          elif "rate_limit" in error_str and attempt < max_retries - 1:
            time.sleep(25)
          else:
            raise

    

    
    if (not hasattr(response, "tool_calls") or not response.tool_calls) and response.content:
        match = re.search(r'\{[\s\S]*"name"\s*:\s*"(\w+)"[\s\S]*"arguments"\s*:\s*(\{[\s\S]*?\})[\s\S]*\}', response.content)
        if match:
            tool_name = match.group(1)
            try:
                tool_args = json.loads(match.group(2))
                response = AIMessage(
                    content="",
                    tool_calls=[{"name": tool_name, "args": tool_args, "id": f"manual_{state.get('steps_taken', 0)}", "type": "tool_call"}]
                )
            except json.JSONDecodeError:
                pass

    
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 1:
        response.tool_calls = [response.tool_calls[0]]

    
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            for key, value in tc["args"].items():
                if isinstance(value, list):
                    tc["args"][key] = "\n".join(str(v) for v in value)

    
    #if not search_already_used and hasattr(response, "tool_calls") and response.tool_calls:
        #if response.tool_calls[0]["name"] != "search_web":
            #response.tool_calls = [{
                #"name": "search_web",
                #"args": {"query": state["messages"][0].content},
                #"id": f"forced_search_{state.get('steps_taken', 0)}",
                #"type": "tool_call"
            #}]

    return {
        "messages": [response],
        "steps_taken": state.get("steps_taken", 0) + 1,
    }



def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if state.get("steps_taken", 0) >= state.get("max_steps", 15):
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


tool_node = ToolNode(tools)

agent_builder = StateGraph(AgentState)
agent_builder.add_node("brain", agent_brain)
agent_builder.add_node("tools", tool_node)

agent_builder.set_entry_point("brain")

agent_builder.add_conditional_edges(
    "brain",
    should_continue,
    {"continue": "tools", "end": END}
)
agent_builder.add_edge("tools", "brain")

checkpointer = MemorySaver()
agent_graph = agent_builder.compile(checkpointer=checkpointer)


def run_research_agent(query: str, thread_id: str = None) -> str:
   
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer]
    }

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "steps_taken": 0,
        "max_steps":15,
    }

    result = agent_graph.invoke(initial_state, config=config)

   
    if "__interrupt__" in result:
        return {
            "status": "waiting_for_input",
            "question": result["__interrupt__"][0].value,
            "thread_id": thread_id,
        }

    return {
        "status": "done",
        "answer": result["messages"][-1].content,
        "thread_id": thread_id,
    }





def resume_research_agent(user_response: str, thread_id: str) -> dict:
    """Jab ask_user ne poocha ho, is function se user ka jawab wapas agent ko do."""
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer]
    }

    result = agent_graph.invoke(Command(resume=user_response), config=config)

    if "__interrupt__" in result:
        return {
            "status": "waiting_for_input",
            "question": result["__interrupt__"][0].value,
            "thread_id": thread_id,
        }

    return {
        "status": "done",
        "answer": result["messages"][-1].content,
        "thread_id": thread_id,
    }





from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Academic Research Assistant")


class QueryRequest(BaseModel):
    query: str


class ResumeRequest(BaseModel):
    user_response: str
    thread_id: str


@app.post("/api/research")
def start_research(request: QueryRequest):
    result = run_research_agent(request.query)
    return result


@app.post("/api/resume")
def resume_research(request: ResumeRequest):
    result = resume_research_agent(request.user_response, request.thread_id)
    return result


app.mount("/", StaticFiles(directory="static", html=True), name="static")


