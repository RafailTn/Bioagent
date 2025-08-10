from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os
from Bio import Entrez
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()

llm = ChatOllama(model="qwen3:1.7b")

# --- 1. Set up your environment ---
# IMPORTANT: Add your OpenAI API key to your environment variables
# os.environ["OPENAI_API_KEY"] = "sk-..." 

# Provide your email to NCBI (required for Entrez)
Entrez.email = "rafailadam46@gmail.com"


# --- 2. Define the core search function (the engine of our tool) ---
def pubmed_search(keywords: str, year: int, pnum:str) -> dict:
    """
    Searches PubMed for papers with specific keywords from a given year.

    Args:
        keywords (str): The search terms (e.g., 'crispr gene editing').
        year (int): The publication year to filter by.
        pnum (str): The maximum number of papers to search for within ""

    Returns:
        A dictionary containing a list of paper titles and their IDs.
    """
    print(f"🛠️ Agent is searching PubMed for '{keywords}' from {year}...")
    
    # Construct the search query for Entrez
    search_term = f"({keywords}) AND {year}[pdat]"
    
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=pnum, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    
    if not record["IdList"]:
        return {"results": "No papers found matching the criteria."}
        
    # Fetch summaries for the found IDs
    id_list = record["IdList"]
    handle = Entrez.esummary(db="pubmed", id=",".join(id_list))
    summaries = Entrez.read(handle)
    handle.close()
    
    # Format the output
    results = [
        f"Title: {summary['Title']} (PMID: {summary['Id']})"
        for summary in summaries
    ]
    return {"papers": results}

memory = MemorySaver() 
agent = create_react_agent(
    model = llm,
    tools=[pubmed_search],
    checkpointer=memory
)
config = {"configurable": {"thread_id": "1"}}
message = [HumanMessage(content="Find me 5 SVA (SINE-VNTR-Alu) element related papers for the year 2024 and summarize their findings. If you don't find enough papers from that year only, you are allowed to search for papers from the next and/or the previous year as well and summarize those instead. Each summary should be at least 200 words.")]
result = agent.invoke({"messages":message}, config)
print(result)

