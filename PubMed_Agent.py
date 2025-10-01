import argparse
import datetime
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
from langchain_tavily import TavilySearch
import xml.etree.ElementTree as ET
load_dotenv()

Entrez.email = "rafailadam46@gmail.com"
pi_llm = ChatOllama(model="llama3.1:8b")

def pubmed_search_detailed(keywords: str, year: int, pnum: str) -> dict:
    """
    Search PubMed for papers and retrieve metadata + abstract/full text if available.
    """
    print(f"🔍 Searching PubMed for '{keywords}' from {year}...")

    handle = Entrez.esearch(
        db="pubmed",
        term=f"({keywords}) AND {year}[pdat]",
        retmax=pnum,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()

    id_list = record["IdList"]
    if not id_list:
        return {"results": "No papers found matching the criteria."}

    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
    articles = Entrez.read(handle)["PubmedArticle"]
    handle.close()

    results = []
    for article in articles:
        pmid = article['MedlineCitation']['PMID']
        title = article['MedlineCitation']['Article']['ArticleTitle']
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

        abstract = "No abstract available."
        if 'Abstract' in article['MedlineCitation']['Article']:
            abstract_list = article['MedlineCitation']['Article']['Abstract']['AbstractText']
            abstract = " ".join(abstract_list)

        # --- Try to fetch full text from PubMed Central ---
        full_text = None
        try:
            # LinkOut / PMC check
            elink_handle = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pmc")
            elink_record = Entrez.read(elink_handle)
            elink_handle.close()
            if elink_record and "LinkSetDb" in elink_record[0] and elink_record[0]["LinkSetDb"]:
                pmcid = elink_record[0]["LinkSetDb"][0]["Link"][0]["Id"]
                pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
                try:
                    r = requests.get(pmc_url)
                    if r.status_code == 200:
                        # crude full-text extraction
                        full_text = r.text
                except Exception:
                    pass
        except Exception:
            pass

        results.append({
            "title": title,
            "pmid": pmid,
            "abstract": abstract,
            "full_text": full_text if full_text else "Not available.",
            "link": link
        })

    return {"papers": results}

# --- Memory and Agent ---
memory = MemorySaver()

system_prompt = SystemMessage(content=(
    "You are a research assistant specialized in biomedical literature.\n"
    "You have access to PubMed search. For each query:\n"
    "- You can use TavilySearch to search for today's date."
    "- Search for relevant papers (at least 5) prioritizing ones closer to the today's date, unless the user asks for publications within a specific year.\n"
    "- Retrieve their abstracts.\n"
    "- Attempt to fetch and summarize the full text if available in PubMed Central. If not, provide an as detailed as possible summary of the abstract, taking into account only the retrieved information.\n"
    "- Always provide the PubMed ID (PMID) and link.\n"
    "- If summarizing full text, highlight insights not in the abstract.\n"
    "- Maintain conversation memory across turns, so user can refine search or ask follow-ups.\n"
    "- Be concise but detailed in scientific explanation.\n"
    "- Do not fabricate PMIDs or descriptions of articles, if you are not certain in the information you received from a paper, state so."
))

agent = create_react_agent(
    model=pi_llm,
    tools=[pubmed_search_detailed, TavilySearch()],
    checkpointer=memory
)

# --- CLI Loop ---
def main():
    print("🧬 PubMed Research Assistant (CLI)")
    print("Type 'exit' to quit.\n")
    config = {"configurable": {"thread_id": "cli-thread"}}
    # initialize with system prompt
    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break

            messages = [system_prompt, HumanMessage(content=user_input)]
            result = agent.invoke({"messages": messages}, config)
            ai_message = result['messages'][-1].content
            print(f"\n🤖 AI: {ai_message}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
