import argparse
from datetime import date
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
from Bio import Entrez
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_tavily import TavilySearch
import re
import requests
import xml.etree.ElementTree as ET
# RAG Components
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
import json
from typing import List, Optional

load_dotenv()

Entrez.email = "rafailadam46@gmail.com"
pi_llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="FremyCompany/BioLORD-2023")

# ============================================
# RAG SETUP - FIXED PERSISTENCE
# ============================================

os.makedirs("./geo_rag_db", exist_ok=True)

# GEO vectorstore (separate database)
geo_vectorstore = Chroma(
    persist_directory="./geo_rag_db",
    embedding_function=embeddings,
    collection_name="geo_datasets",
    collection_metadata={"hnsw:space": "cosine"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ============================================
# NEW FUNCTION: Search and store GEO datasets
# ============================================

def geo_search_and_store(keywords: str, max_results: int = 10) -> str:
    """
    Search GEO (Gene Expression Omnibus) for datasets and store in separate database.
    
    Args:
        keywords: Search keywords for GEO
        max_results: Maximum number of results to retrieve
    
    Returns:
        Summary of GEO datasets found and stored
    """
    print(f"Searching GEO for: '{keywords}'...")
    
    try:
        # Search GEO database
        handle = Entrez.esearch(db="gds", term=keywords, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        if not id_list:
            return f"No GEO datasets found for '{keywords}'"
        
        print(f"Found {len(id_list)} GEO dataset(s)")
        
        # Fetch GEO dataset details
        handle = Entrez.esummary(db="gds", id=",".join(id_list))
        summaries = Entrez.read(handle)
        handle.close()
        
        stored_datasets = []
        
        for summary in summaries:
            try:
                # Extract GEO information
                geo_id = summary.get('Accession', 'Unknown')
                title = summary.get('title', 'No title')
                summary_text = summary.get('summary', 'No summary available')
                platform = summary.get('GPL', 'Unknown platform')
                samples = summary.get('n_samples', 'Unknown')
                dataset_type = summary.get('entryType', 'Unknown type')
                organism = summary.get('taxon', 'Unknown organism')
                pub_date = summary.get('PDAT', 'Unknown date')
                
                # Extract supplementary info if available
                supplementary_type = summary.get('suppFile', '')
                ftplink = summary.get('FTPLink', '')
                
                # Create content for vector store
                content = f"GEO Accession: {geo_id}\n"
                content += f"Title: {title}\n"
                content += f"Type: {dataset_type}\n"
                content += f"Platform: {platform}\n"
                content += f"Organism: {organism}\n"
                content += f"Number of Samples: {samples}\n"
                content += f"Publication Date: {pub_date}\n"
                content += f"Summary: {summary_text}\n"
                
                if supplementary_type:
                    content += f"Supplementary Files: {supplementary_type}\n"
                if ftplink:
                    content += f"FTP Link: {ftplink}\n"
                
                # Also try to get associated PubMed IDs if available
                pmids = summary.get('PubMedIds', [])
                if pmids:
                    content += f"Associated PubMed IDs: {', '.join(map(str, pmids))}\n"
                
                # Split and store in GEO database
                chunks = text_splitter.split_text(content)
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "geo_id": geo_id,
                            "title": title,
                            "dataset_type": dataset_type,
                            "platform": platform,
                            "organism": organism,
                            "samples": str(samples),
                            "pub_date": pub_date,
                            "chunk_index": i,
                            "source": "geo",
                            "associated_pmids": ','.join(map(str, pmids)) if pmids else ""
                        }
                    )
                    for i, chunk in enumerate(chunks)
                ]
                
                geo_vectorstore.add_documents(documents)
                stored_datasets.append(f"{title} (GEO:{geo_id})")
                print(f"  ✓ Stored GEO dataset: {geo_id}")
                
            except Exception as e:
                print(f"  ❌ Error processing GEO dataset: {e}")
                continue
        
        # Summary
        summary = f"GEO SEARCH COMPLETE:\n"
        summary += f"Total datasets stored: {len(stored_datasets)}\n"
        summary += f"Datasets:\n"
        for dataset in stored_datasets:
            summary += f"  - {dataset}\n"
        
        return summary
        
    except Exception as e:
        return f"Error searching GEO: {e}"


def search_geo_database(query: str, num_results: int = 10) -> str:
    """
    Search the GEO RAG database for stored datasets.
    
    Args:
        query: Search query (can be GEO ID or keywords)
        num_results: Number of results to return
    
    Returns:
        Formatted search results from GEO database
    """
    print(f"Searching GEO database for: '{query}'")
    
    # Check if this is a GEO ID query
    is_geo_id = query.upper().startswith(('GSE', 'GDS', 'GSM', 'GPL'))
    
    try:
        if is_geo_id:
            # Direct GEO ID lookup
            geo_id = query.upper()
            # Try to get all chunks for this specific dataset
            results = geo_vectorstore.similarity_search(f"GEO Accession: {geo_id}", k=20)
            # Filter to ensure we only get the right dataset
            results = [r for r in results if r.metadata.get('geo_id', '').upper() == geo_id]
            
            if not results:
                return f"No dataset found with GEO ID: {geo_id}"
            
            # Sort by chunk index
            results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
        else:
            # Keyword search
            results = geo_vectorstore.similarity_search(query, k=num_results)
        
        if not results:
            return "No relevant GEO datasets found in database."
        
        # Group results by GEO ID
        datasets_dict = {}
        for doc in results:
            geo_id = doc.metadata.get("geo_id", "unknown")
            if geo_id not in datasets_dict:
                datasets_dict[geo_id] = {
                    "geo_id": geo_id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "type": doc.metadata.get("dataset_type", "Unknown"),
                    "platform": doc.metadata.get("platform", "Unknown"),
                    "organism": doc.metadata.get("organism", "Unknown"),
                    "samples": doc.metadata.get("samples", "Unknown"),
                    "pub_date": doc.metadata.get("pub_date", "Unknown"),
                    "pmids": doc.metadata.get("associated_pmids", ""),
                    "chunks": []
                }
            datasets_dict[geo_id]["chunks"].append(doc.page_content)
        
        # Format output
        output = [f"Found {len(datasets_dict)} GEO dataset(s):\n"]
        
        for idx, (geo_id, info) in enumerate(datasets_dict.items(), 1):
            # Combine chunks for complete content
            full_content = "\n".join(info['chunks'])
            
            output.append(
                f"\n--- DATASET {idx} ---\n"
                f"GEO ID: {geo_id}\n"
                f"Title: {info['title']}\n"
                f"Type: {info['type']}\n"
                f"Platform: {info['platform']}\n"
                f"Organism: {info['organism']}\n"
                f"Samples: {info['samples']}\n"
                f"Publication Date: {info['pub_date']}\n"
            )
            
            if info['pmids']:
                output.append(f"Associated PubMed IDs: {info['pmids']}\n")
            
            # Add content preview or full content depending on single dataset
            if is_geo_id and len(datasets_dict) == 1:
                output.append(f"\nFull Information:\n{full_content}\n")
            else:
                output.append(f"\nSummary: {full_content[:500]}...\n")
        
        return "".join(output)
        
    except Exception as e:
        return f"Error searching GEO database: {e}"


def get_geo_database_stats() -> str:
    """Get statistics about the GEO database."""
    try:
        collection = geo_vectorstore._collection
        count = collection.count()
        
        if count == 0:
            return "GEO database is empty (0 chunks, 0 datasets)"
        
        # Get sample results to count unique datasets
        sample_results = geo_vectorstore.similarity_search("", k=min(100, count))
        unique_geo_ids = set()
        unique_titles = set()
        
        for doc in sample_results:
            geo_id = doc.metadata.get("geo_id")
            title = doc.metadata.get("title")
            if geo_id:
                unique_geo_ids.add(geo_id)
            if title:
                unique_titles.add(title[:50])
        
        output = f"GEO Database Statistics:\n"
        output += f"- Total chunks: {count}\n"
        output += f"- Estimated datasets: {len(unique_geo_ids)}\n"
        
        if unique_titles:
            output += f"- Sample datasets:\n"
            for title in list(unique_titles)[:3]:
                output += f"  • {title}...\n"
        
        return output
        
    except Exception as e:
        return f"Error getting GEO stats: {e}"

geo_search_and_store_tool = tool(
    geo_search_and_store,
    description="Search GEO for datasets and store them in the GEO database"
)

search_geo_database_tool = tool(
    search_geo_database,
    description="Search the GEO database for stored datasets"
)

get_geo_stats_tool = tool(
    get_geo_database_stats,
    description="Get statistics about the GEO database"
)

# ============================================
# ENHANCED AGENT SETUP
# ============================================

memory = MemorySaver()

# Enhanced system prompt to handle both PubMed and GEO
system_prompt = '''You are a biomedical research assistant with access to GEO datasets through a local RAG database.

GEO OPERATIONS:
1. Search and store datasets: Use 'geo_search_and_store_tool'
2. Search stored datasets: Use 'search_geo_database_tool'
3. Get stats: Use 'get_geo_stats_tool'

GEO SEARCH RULES:
- When asked about gene expression, microarray, RNA-seq datasets, use GEO tools
- GEO IDs start with GSE, GDS, GSM, or GPL

WORKFLOW:
1. Check the RAG database first
2. If not found, search and store new data
3. Retrieve and present results

RESPONSE FORMAT:
- Use appropriate citations (GEO ID for datasets)
- When presenting both, clearly separate the sections
'''

# Create agent with all tools
tools = [
    # GEO tools
    geo_search_and_store_tool,
    search_geo_database_tool,
    get_geo_stats_tool,
]

geo_agent = create_react_agent(
    model=pi_llm,
    tools=tools,
    checkpointer=memory,
    prompt=system_prompt
)

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("Biomedical Research Assistant (GEO)")
    print("Commands: 'exit' to quit, 'geo-stats' for GEO info, 'clear' to clear screen\n")
    
    print("GEO Database:")
    geo_stats = get_geo_database_stats()
    print(f"{geo_stats}\n")
    
    config = {"configurable": {"thread_id": "cli-thread"}}
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            if user_input.lower() == "geo-stats":
                print(f"\n{get_geo_database_stats()}\n")
                continue
            
            if user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
           
            if user_input.lower().startswith("test geo"):
                parts = user_input[8:].strip()
                keywords = parts if parts else "cancer RNA-seq"
                print(f"\nTesting GEO search: '{keywords}'...")
                result = geo_search_and_store(keywords, max_results=5)
                print(result)
                print(f"\n{get_geo_database_stats()}\n")
                continue
            
            # Regular agent interaction
            print("\nProcessing...")
            
            # Invoke agent with increased recursion limit for complex operations
            result = geo_agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "cli-thread"}, "recursion_limit": 25}
            )
            
            # Print response
            for msg in reversed(result['messages']):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\nAI: {msg.content}\n")
                    break

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
