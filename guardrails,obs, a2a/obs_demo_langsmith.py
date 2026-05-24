from typing import TypedDict, Dict, Any
import numpy as np
import os
import time
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import openai

# --- LangSmith Imports ---
from langsmith import Client
from langsmith.run_helpers import traceable
from langchain_core.tracers.context import tracing_v2_enabled
import langchain

# ------------------- Setup -------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "observability_demo"

# Initialize LangSmith client
langsmith_client = Client()


# ------------------- State Definition -------------------
class AgentState(TypedDict):
    query: str
    response: str | None
    cache_hit_type: str | None
    similarity: float | None
    duration: float | None
    total_cost: float | None
    total_tokens: int | None


# ------------------- Personality Agent -------------------
class PersonalityAgent:
    def __init__(self, similarity_threshold=0.85):
        self.cache: Dict[str, str] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.client = openai.OpenAI()
        self.similarity_threshold = similarity_threshold

    def _update_run_metadata(self, metadata: Dict[str, Any]):
        """Helper to update current run metadata"""
        try:
            # LangSmith automatically captures metadata through the traceable decorator
            # We return the metadata so it can be used in the function return
            return metadata
        except Exception:
            return metadata

    @traceable(
        name="get_embedding", 
        run_type="embedding",
        metadata={"model": "text-embedding-3-small"}
    )
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for semantic similarity"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        usage = getattr(response, "usage", None)
        total_tokens = usage.total_tokens if usage else 0

        # Calculate cost for embeddings (text-embedding-3-small: $0.02 per 1M tokens)
        embedding_cost = (total_tokens / 1_000_000) * 0.02

        print(f"[LangSmith] Embedding tokens: {total_tokens}, Cost: ${embedding_cost:.6f}")

        embedding = np.array(response.data[0].embedding)
        
        # Return both embedding and metadata
        return embedding

    @traceable(name="find_semantic_cache_match", run_type="tool")
    def find_semantic_cache_match(self, query: str):
        """Find most similar key in cache"""
        if not self.embeddings_cache:
            return None

        query_embedding = self.get_embedding(query)
        max_similarity = 0
        best_match = None

        for cached_key, cached_embedding in self.embeddings_cache.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                cached_embedding.reshape(1, -1)
            )[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = cached_key

        if max_similarity >= self.similarity_threshold:
            return best_match, max_similarity
        
        return None

    @traceable(
        name="call_openai_model", 
        run_type="llm",
        metadata={"model": "gpt-3.5-turbo", "max_tokens": 100}
    )
    def call_openai_model(self, name: str) -> Dict[str, Any]:
        """Call OpenAI model and return response with metadata"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Give exactly 2 concise lines about this person's personality and achievements."},
                    {"role": "user", "content": f"Tell me about {name}"}
                ],
                max_tokens=100
            )

            usage = getattr(response, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            # Calculate cost (GPT-3.5-turbo: $0.0015/1K prompt tokens, $0.002/1K completion tokens)
            cost_usd = (prompt_tokens / 1000 * 0.0015) + (completion_tokens / 1000 * 0.002)

            content = response.choices[0].message.content.strip()

            print(f"[LangSmith] LLM tokens: {total_tokens}, Cost: ${cost_usd:.6f}")

            # Return both content and metadata
            return {
                "content": content,
                "metadata": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost_usd,
                    "model": "gpt-3.5-turbo"
                }
            }

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "content": f"Error: {e}",
                "metadata": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0,
                    "error": str(e)
                }
            }

    @traceable(name="record_event", run_type="tool")
    def record_event(self, event_name: str, metadata: dict):
        """Record events as spans in LangSmith"""
        print(f"Event: {event_name}, Metadata: {metadata}")


# ------------------- LangGraph Flow Definition -------------------
def build_graph(agent: PersonalityAgent):
    from langgraph.graph import StateGraph, END
    
    graph = StateGraph(AgentState)

    @traceable(name="check_exact_cache", run_type="chain")
    def check_exact_cache(state: AgentState):
        key = state["query"].lower().strip()
        if key in agent.cache:
            agent.record_event("cache_hit_exact", {"query": key})
            state["response"] = f"⚡ EXACT CACHE HIT: {agent.cache[key]}"
            state["cache_hit_type"] = "exact"
            state["total_cost"] = 0.0
            state["total_tokens"] = 0
        else:
            state["cache_hit_type"] = None
        return state
    
    graph.add_node("check_exact_cache", check_exact_cache)

    @traceable(name="check_semantic_cache", run_type="chain")
    def check_semantic_cache(state: AgentState):
        if state["cache_hit_type"] == "exact":
            return state
            
        query = state["query"]
        match = agent.find_semantic_cache_match(query)
        if match:
            matched_key, sim = match
            agent.record_event("cache_hit_semantic", {
                "query": query, "matched_key": matched_key, "similarity": sim
            })
            state["response"] = f"⚡ SEMANTIC CACHE HIT (similarity: {sim:.2f}): {agent.cache[matched_key]}"
            state["cache_hit_type"] = "semantic"
            state["similarity"] = sim
            state["total_cost"] = 0.0  # Only embedding cost, which is minimal
            state["total_tokens"] = 0
        return state
    
    graph.add_node("check_semantic_cache", check_semantic_cache)

    @traceable(name="call_api", run_type="chain")
    def call_api(state: AgentState):
        if state["cache_hit_type"] in ["exact", "semantic"]:
            return state
            
        query = state["query"]
        agent.record_event("api_call_triggered", {"query": query})
        start = time.time()
        
        # Call OpenAI and get response with metadata
        llm_result = agent.call_openai_model(query)
        duration = time.time() - start

        key = query.lower().strip()
        agent.cache[key] = llm_result["content"]
        agent.embeddings_cache[key] = agent.get_embedding(key)

        agent.record_event("api_response_cached", {
            "query": query,
            "duration_sec": round(duration, 2)
        })

        state["response"] = f"FRESH (took {duration:.2f}s): {llm_result['content']}"
        state["duration"] = duration
        state["cache_hit_type"] = "fresh"
        state["total_cost"] = llm_result["metadata"]["cost_usd"]
        state["total_tokens"] = llm_result["metadata"]["total_tokens"]
        
        return state
    
    graph.add_node("call_api", call_api)

    graph.add_edge("check_exact_cache", "check_semantic_cache")
    graph.add_edge("check_semantic_cache", "call_api")
    graph.add_edge("call_api", END)
    graph.set_entry_point("check_exact_cache")

    return graph.compile()


# ------------------- Demo -------------------
@traceable(
    name="personality_query", 
    run_type="chain",
    metadata={"project": "observability_demo", "version": "1.0"}
)
def run_query(graph, query: str) -> Dict[str, Any]:
    """Wrapper function to track entire query execution"""
    result = graph.invoke({"query": query})
    
    # Print summary to console
    print(f"\n=== QUERY SUMMARY ===")
    print(f"Query: {query}")
    print(f"Cache Hit: {result.get('cache_hit_type', 'none')}")
    print(f"Tokens: {result.get('total_tokens', 0)}")
    print(f"Cost: ${result.get('total_cost', 0):.6f}")
    print(f"Response: {result.get('response', '')[:100]}...")
    print("====================\n")
    
    return result


# ------------------- Main Execution -------------------
if __name__ == "__main__":
    # Enable LangSmith tracing
    with tracing_v2_enabled():
        agent = PersonalityAgent()
        graph = build_graph(agent)

        queries = [
            "Elon Musk",
            "Elon Musk",  # Should be exact cache hit
            "Tell me about Elon",  # Should be semantic cache hit
            "Marie Curie",
            "marie curie",  # Should be exact cache hit
            "Info on Curie",  # Should be semantic cache hit
            "Albert Einstein"
        ]

        total_cost = 0.0
        total_tokens = 0

        for i, q in enumerate(queries, 1):
            print(f"\n  Processing query {i}/{len(queries)}: '{q}'")
            
            result = run_query(graph, q)
            
            # Accumulate totals
            total_cost += result.get('total_cost', 0)
            total_tokens += result.get('total_tokens', 0)
            
            # Add a small delay between queries
            if i < len(queries):
                time.sleep(1)

        print(f"\n SESSION SUMMARY")
        print(f"Total Queries: {len(queries)}")
        print(f"Total Tokens: {total_tokens}")
        print(f"Total Cost: ${total_cost:.6f}")
        print(f"Cache Size: {len(agent.cache)} entries")