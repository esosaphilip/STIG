# core/sourcing_agent.py
from typing import TypedDict
from langgraph.graph import StateGraph
from core.documentIngestion import Documentloader
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from core.llm_agent import StigLLMAgent
import json
from requests.exceptions import HTTPError
import time

load_dotenv()  # Load environment variables from .env file

# Step 1 — Define the State Schema
class PoliticianState(TypedDict):
    name: str
    date_of_birth: str
    date_of_death: str
    state_of_origin: str
    party: str
    offices_held: list[str]
    tenure_dates: list[str]
    scandals: list[str]
    court_cases: list[str]
    associations: list[str]
    allies: list[str]
    enemies: list[str]
    business_partners: list[str]
    notable_statements: list[str]
    social_media: dict[str, str]
    old_politicians_newspaper_appearances: list[str]
    any_other_interesting_facts: list[str]
    news_mentions: list[str]
    raw_wikipedia_text: str    # Stores extracted Wikipedia text
    raw_news_text: str         # Stores raw news text
    sources: list[str]         # Stores all source URLs


# Step 2 — Define Node 1 (Wikipedia search)
def search_wikipedia(state: PoliticianState) -> dict:
    politician_name = state["name"]
    
    try:
        loader = Documentloader(source_type="wikipedia", source_path=politician_name)
        docs = loader.load_documents()
        extracted_text = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
        new_sources = [
            doc.metadata.get("source", f"https://en.wikipedia.org/wiki/{politician_name.replace(' ', '_')}")
            for doc in docs
        ]
    except HTTPError as e:
        # rate limited — return empty so agent can continue with web search
        print(f"Wikipedia rate limited: {e}")
        extracted_text = ""
        new_sources = []
    
    return {
        "raw_wikipedia_text": extracted_text,
        "sources": state.get("sources", []) + new_sources
    }


# Step 3 — define Node 2 (Tavily web search)
def search_web(state: PoliticianState) -> dict:
    """ this node will take the state, 
    perform a web search for the politician using Tavily,
    and update the state with the results """
    
    
    # 1. get name from state
    politician_name = state["name"]
    # 2. create Tavily client using env variable
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # 3. search for politician scandals and news
    scandal_query = f"{politician_name} Nigeria politician scandal court"
    news_query = f"{politician_name} Nigeria politician news"
    # search query should be: "{name} Nigeria politician scandal court"
    response = tavily_client.search(query=f"{politician_name} Nigeria politician scandal court", limit=10)
    
    # 4. extract content from results
    # result["results"] is a list
    # each item has "content" and "url"
    extracted_contented = [result["content"] for result in response.get("results", [])]
    # 5. join all content into raw_news_text
    raw_news_text = "\n\n".join(extracted_contented)
    # 6. collect all URLs
    sources = [result["url"] for result in response.get("results", [])]
    # 7. return updated fields only
    return {
        "raw_news_text": raw_news_text,
        "sources": state.get("sources", []) + sources
    }


def extract_profile(state: PoliticianState) -> dict:
    # 1. combine all raw text
    all_text = state.get("raw_wikipedia_text", "") + "\n\n" + state.get("raw_news_text", "")
    
    # 2. build a prompt asking LLM to extract fields as JSON
    prompt = f"""
    Extract information about this politician from the text below.
    Return ONLY a valid JSON object with these fields:
    - name
    - date_of_birth
    - date_of_death
    - state_of_origin
    - party: political party name, or "Military" if military ruler, or "Independent" if none
    - offices_held (list)
    - scandals (list)
    - court_cases (list)
    - allies(people who supported, worked with, or were loyal to this politician) (list)
    - enemies (list)
    - notable_statements (list)
    - any_other_interesting_facts (list)
    - business_partners (list)
    - notable_statements (list)
    - social_media: (dict with keys as platform names and values as handles)
    - old_politicians_newspaper_appearances: (list)

    Text:
    {all_text[:5000]}
    
    Return only JSON. No explanation.
    """
    
    # 3. call the LLM
    llm_agent = StigLLMAgent(model_name="mistral")
    # use OllamaLLM with mistral
    # invoke the prompt
    response = llm_agent.answer(question=prompt, context_chunks=[])
    # parse the JSON response
    try:
        extracted_data = json.loads(response)
    except json.JSONDecodeError:
        extracted_data = {}
    
    # 4. return updated state fields
    return extracted_data

# Step 5 — build the graph
def create_sourcing_agent():
    """Build and compile the politician sourcing agent graph"""
    
    # 1. create the graph with our state schema
    graph = StateGraph(PoliticianState)
    
    # 2. add nodes — each node is a function
    graph.add_node("search_wikipedia", search_wikipedia)
    graph.add_node("search_web", search_web)
    graph.add_node("extract_profile", extract_profile)
    
    # 3. add edges — define the flow
    # set entry point — which node runs first?
    graph.set_entry_point("search_wikipedia")
    
    # connect nodes in sequence
    graph.add_edge("search_wikipedia", "search_web")
    graph.add_edge("search_web", "extract_profile")
    
    # set finish point — which node runs last?
    graph.set_finish_point("extract_profile")
    
    # 4. compile and return
    return graph.compile()


if __name__ == "__main__":
    # create the agent
    agent = create_sourcing_agent()
    
    # run it with a starting state
    result = agent.invoke({
        "name": "Sani Abacha",
        "date_of_birth": "",
        "date_of_death": "",
        "state_of_origin": "",
        "party": "",
        "offices_held": [],
        "tenure_dates": [],
        "scandals": [],
        "court_cases": [],
        "associations": [],
        "allies": [],
        "enemies": [],
        "business_partners": [],
        "notable_statements": [],
        "social_media": {},
        "old_politicians_newspaper_appearances": [],
        "any_other_interesting_facts": [],
        "news_mentions": [],
        "raw_wikipedia_text": "",
        "raw_news_text": "",
        "sources": []
    })
    
    print(result)