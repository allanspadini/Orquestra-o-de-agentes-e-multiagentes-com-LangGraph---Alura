# new_backend.py
import warnings
warnings.filterwarnings("ignore", message=".*TqdmWarning.*")

import os
from dotenv import load_dotenv

from typing import TypedDict, Annotated, List
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains import create_structured_output_runnable
from tavily import TavilyClient

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Define as variáveis de ambiente
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')


# Define o estado do agente (AgentState)
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# Define o modelo Pydantic para a saída estruturada
class Queries(BaseModel):
    queries: List[str]


# Inicializa o banco de dados para os checkpoints
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Inicializa o modelo de linguagem
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Cria um Runnable para a saída estruturada (forma correta para Gemini)
structured_model = model.with_structured_output(Queries)


# Prompts
PLAN_PROMPT = """Você é um escritor especialista com a tarefa de criar um esboço de alto nível para uma redação. \
Escreva esse esboço para o tópico fornecido pelo usuário. Apresente um plano da redação junto com quaisquer notas \
ou instruções relevantes para as seções."""

WRITER_PROMPT = """Você é um assistente de redação com a tarefa de escrever excelentes redações de 5 parágrafos. \
Gere a melhor redação possível para a solicitação do usuário e o esboço inicial. \
Se o usuário fornecer críticas, responda com uma versão revisada das suas tentativas anteriores. \
Utilize todas as informações abaixo conforme necessário:

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

# Inicializa o cliente Tavily
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# Definição dos nós do LangGraph
def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    queries = structured_model.invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 0) + 1
    }


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState):
    queries = structured_model.invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


# Construção do Grafo
builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer=memory)

# Exemplo de como rodar o grafo (o Gradio fará isso por você)
# thread = {"configurable": {"thread_id": "1"}}
# for s in graph.stream({
#     'task': "Qual é a diferença entre o langchain e langsmith",
#     "max_revisions": 2,
#     "revision_number": 0,
#     "content": [],
# }, thread):
#    print(s)