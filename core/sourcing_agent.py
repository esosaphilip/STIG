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
    # Optional country hint. "" means a genuinely global, unbiased web search;
    # a concrete value narrows the search to disambiguate common names.
    country: str
    date_of_birth: str
    date_of_death: str
    state_of_origin: str
    party: str
    offices_held: list[str]
    tenure_dates: list[str]
    # The five "claim" fields carry specific, reputationally damaging statements
    # about a real, often-living person, so each entry is a {"claim", "source"}
    # object rather than a bare string — see verify_citations for why.
    scandals: list[dict]
    court_cases: list[dict]
    associations: list[str]
    allies: list[str]
    enemies: list[dict]
    business_partners: list[dict]
    notable_statements: list[dict]
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
        # Tag each chunk with its own source URL inline, instead of tracking URLs
        # only separately in state["sources"]. extract_profile needs to see which
        # URL states which claim so a citation can point at a specific block; a
        # flat URL list loses that link. Same "[SOURCE: ...]" shape llm_agent.py
        # already uses to build /ask's context.
        blocks = []
        new_sources = []
        for doc in docs:
            url = doc.metadata.get(
                "source",
                f"https://en.wikipedia.org/wiki/{politician_name.replace(' ', '_')}",
            )
            blocks.append(f"[SOURCE: {url}]\n{doc.page_content}")
            new_sources.append(url)
        extracted_text = "\n\n".join(blocks)
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
    # 2. build the query. Without a country we search globally and stay unbiased;
    # with one we append it to narrow a common name. Previously "Nigeria" was
    # hardcoded into every query, which biased every lookup toward Nigeria and
    # broke non-Nigerian subjects.
    country = state.get("country", "")
    if country:
        query = f'"{politician_name}" {country} politician scandal history'
    else:
        query = f'"{politician_name}" politician scandal history'
    # 3. create Tavily client using env variable
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # 4. search for politician scandals and news
    response = tavily_client.search(
    query=query,
    limit=10
    )
    # 5. extract content from results
    # result["results"] is a list; each item has "content" and "url"
    results = response.get("results", [])
    # 6. tag each result's content with its own URL inline (not one merged blob),
    # so a claim can be traced back to the exact source that stated it. The URLs
    # are still collected separately below for the code-level citation check.
    raw_news_text = "\n\n".join(
        f"[SOURCE: {result['url']}]\n{result['content']}" for result in results
    )
    # 7. collect all URLs
    sources = [result["url"] for result in results]
    # 8. return updated fields only
    return {
        "raw_news_text": raw_news_text,
        "sources": state.get("sources", []) + sources
    }


# The five fields whose entries are specific, reputationally damaging claims about
# a real (often living) named person. These are the ones where a hallucination
# does the most harm, so they are the ones we require a verifiable citation for.
CLAIM_FIELDS = [
    "scandals",
    "court_cases",
    "enemies",
    "business_partners",
    "notable_statements",
]


def verify_citations(extracted: dict, real_sources: list[str]) -> dict:
    """
    Deterministically drop any claim whose cited source isn't real.

    Why this exists, and why the prompt alone isn't enough: a local model can still
    hallucinate a citation — invent a plausible-looking URL, or cite a real URL from
    the sources list for a claim that URL never actually made. Asking nicely in the
    prompt only reduces how often that happens; this is the code-level check that
    actually catches it when it does. This mirrors the same "AI extracts, code
    verifies" split used elsewhere (tolerance check in confirmAI, priority parsing
    in DringlichAI).

    `real_sources` is the list of URLs we genuinely collected from Wikipedia/Tavily,
    not whatever the model claimed. Anything citing a URL outside that set is dropped
    and reported, because "the model cited a source that wasn't in the actual search
    results" is a finding worth surfacing, not hiding.

    No LLM call and no network here — plain Python, so it's unit-testable without
    Ollama running.
    """
    real_source_set = set(real_sources)
    cleaned = dict(extracted)

    for field in CLAIM_FIELDS:
        claims = extracted.get(field, [])
        # A malformed field (model returned a string instead of a list) is treated
        # as "nothing verifiable" rather than crashing the whole extraction.
        if not isinstance(claims, list):
            print(
                f"[verify_citations] Field '{field}' was not a list; "
                f"dropping its contents. Got: {claims!r}"
            )
            cleaned[field] = []
            continue

        kept = []
        for claim in claims:
            # Anything not shaped as {"claim", "source"} can't be verified against a
            # source, so it's dropped rather than trusted as-is.
            if not isinstance(claim, dict):
                print(
                    f"[verify_citations] Dropped unverifiable entry in '{field}' "
                    f"(not a claim/source object): {claim!r}"
                )
                continue
            source = claim.get("source")
            if source in real_source_set:
                kept.append(claim)
            else:
                print(
                    f"[verify_citations] Dropped claim in '{field}': model cited a "
                    f"source that wasn't in the actual search results -> {source!r}"
                )
        cleaned[field] = kept

    return cleaned


def extract_profile(state: PoliticianState) -> dict:
    # 1. combine all raw text
    all_text = state.get("raw_wikipedia_text", "") + "\n\n" + state.get("raw_news_text", "")
    
    # 2. build a prompt asking LLM to extract fields as JSON.
    # Two things matter here that the old prompt lacked:
    # (a) a grounding instruction — a local model with no such instruction will
    #     happily fill fields from its own training data, which for scandals/court
    #     cases means presenting a fabricated accusation as fact; and
    # (b) a per-claim citation requirement for the risky fields — asking the model
    #     to point at the exact [SOURCE: ...] block it used gives us something the
    #     code can later verify (see verify_citations), rather than a flat URL list
    #     that nothing is tied to.
    prompt = f"""
    Extract information about this politician from the text below.

    GROUNDING RULES (follow exactly):
    - Only include information that is explicitly stated in the text below.
    - Do not use outside knowledge.
    - If something is not mentioned in the text, leave it out — do not guess or infer.

    CITATION RULES for scandals, court_cases, enemies, business_partners and notable_statements:
    - Each entry MUST be an object: {{"claim": "<the claim>", "source": "<the exact [SOURCE: ...] URL from the text above that states this claim>"}}
    - The "source" value must be copied exactly from one of the [SOURCE: ...] lines above.
    - If you cannot point to one specific [SOURCE: ...] URL above that states the claim, do not include it.

    Return ONLY a valid JSON object with these fields:
    - name
    - date_of_birth
    - date_of_death
    - state_of_origin
    - party: political party name, or "Military" if military ruler, or "Independent" if none
    - offices_held (list)
    - scandals (list of {{"claim", "source"}} objects, per the citation rules)
    - court_cases (list of {{"claim", "source"}} objects, per the citation rules)
    - allies(people who supported, worked with, or were loyal to this politician) (list)
    - enemies (list of {{"claim", "source"}} objects, per the citation rules)
    - notable_statements (list of {{"claim", "source"}} objects, per the citation rules)
    - any_other_interesting_facts (list)
    - business_partners (list of {{"claim", "source"}} objects, per the citation rules)
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

    # 4. verify citations in code, not just by asking the model nicely. Any risky
    # claim whose cited source isn't one we actually collected gets dropped here.
    extracted_data = verify_citations(extracted_data, state.get("sources", []))

    # 5. return updated state fields
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
        "country": "",
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