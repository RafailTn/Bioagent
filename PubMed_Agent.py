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

Entrez.email = "your-email"
pi_llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# ============================================
# RAG SETUP - FIXED PERSISTENCE
# ============================================

os.makedirs("./pubmed_rag_db", exist_ok=True)

vectorstore = Chroma(
    persist_directory="./pubmed_rag_db",
    embedding_function=embeddings,
    collection_name="pubmed_papers",
    collection_metadata={"hnsw:space": "cosine"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_authors(article):
    """Extract authors from a PubMed article."""
    authors = []
    try:
        if 'AuthorList' in article['MedlineCitation']['Article']:
            author_list = article['MedlineCitation']['Article']['AuthorList']
            for author in author_list:
                if 'CollectiveName' in author:
                    authors.append(author['CollectiveName'])
                else:
                    name_parts = []
                    if 'LastName' in author:
                        name_parts.append(author['LastName'])
                    if 'ForeName' in author:
                        name_parts.append(author['ForeName'])
                    elif 'Initials' in author:
                        name_parts.append(author['Initials'])
                    if name_parts:
                        authors.append(' '.join(name_parts))
    except (KeyError, TypeError):
        pass
    return '; '.join(authors) if authors else "No authors listed"


def pubmed_search_and_store_multi_year(keywords: str, years: str = None, pnum: int = 10) -> str:
    """
    Enhanced version that handles multiple years.
    
    Args:
        keywords: Search keywords
        years: Can be a single year ("2024"), multiple years ("2022,2023,2024"), 
               or a range ("2022-2024")
        pnum: Number of papers per year
    
    Returns:
        Summary of all papers found and stored
    """
    # Parse years input
    year_list = []
    
    if not years:
        year_list = [str(date.today().year)]
    elif "-" in years:
        # Handle range like "2022-2024"
        start_year, end_year = years.split("-")
        year_list = [str(y) for y in range(int(start_year.strip()), int(end_year.strip()) + 1)]
    elif "," in years:
        # Handle comma-separated like "2022,2023,2024"
        year_list = [y.strip() for y in years.split(",")]
    else:
        # Single year
        year_list = [years.strip()]
    all_results = []
    total_stored = 0
    total_full_text = 0
    all_papers = [] 
    for year in year_list:
        print(f"\nSearching PubMed for '{keywords}' in year {year}...")
        # Build search term
        search_term = f"({keywords}) AND {year}[pdat]"
        # Search PubMed
        try:
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=pnum, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            all_results.append(f"Year {year}: Search failed - {e}")
            continue
        id_list = record["IdList"]
        if not id_list:
            all_results.append(f"Year {year}: No papers found")
            continue
        # Fetch and process papers
        try:
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
            articles = Entrez.read(handle)["PubmedArticle"]
            handle.close()
        except Exception as e:
            all_results.append(f"Year {year}: Fetch failed - {e}")
            continue
        year_papers = []
        year_full_text = 0
        for article in articles:
            try:
                pmid = str(article['MedlineCitation']['PMID'])
                title = article['MedlineCitation']['Article']['ArticleTitle']
                authors = extract_authors(article)
                # Extract abstract
                abstract = "No abstract available."
                if 'Abstract' in article['MedlineCitation']['Article']:
                    abstract_list = article['MedlineCitation']['Article']['Abstract']['AbstractText']
                    abstract = " ".join(str(a) for a in abstract_list)
                # Try to fetch full text
                full_text = None
                try:
                    handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc")
                    record = Entrez.read(handle)
                    handle.close()
                    if record and record[0].get("LinkSetDb"):
                        pmc_id_list = [link['Id'] for link in record[0]['LinkSetDb'][0]['Link']]
                        if pmc_id_list:
                            pmc_id = pmc_id_list[0]
                            handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
                            xml_data = handle.read()
                            handle.close()
                            if xml_data:
                                root = ET.fromstring(xml_data)
                                text_parts = [p.text for p in root.findall('.//body//p') if p.text]
                                if not text_parts:
                                    body = root.find('.//body')
                                    if body is not None:
                                        text_parts = [text for text in body.itertext() if text.strip()]
                                if text_parts:
                                    full_text = ' '.join(text_parts)
                                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                                    year_full_text += 1
                                    print(f"  Full text retrieved for PMID:{pmid}")
                except Exception:
                    pass
                # Create content for vector store
                content = f"PMID: {pmid}\nTitle: {title}\n"
                content += f"Authors: {authors}\nYear: {year}\n"
                if full_text:
                    content += f"\nFull Text: {full_text}"
                else:
                    content += f"\nAbstract: {abstract}"
                # Split and store
                chunks = text_splitter.split_text(content)
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "pmid": pmid,
                            "title": title,
                            "authors": authors,
                            "year": year,
                            "chunk_index": i,
                            "source": "pubmed",
                            "has_full_text": bool(full_text)
                        }
                    )
                    for i, chunk in enumerate(chunks)
                ]
                vectorstore.add_documents(documents)
                year_papers.append(f"{title}-(PMID:{pmid})")
                all_papers.append(f"{title}-(PMID:{pmid})\n")
            except Exception as e:
                print(f"  ❌ Error processing article: {e}")
                continue
        # Summary for this year
        if year_papers:
            all_results.append(
                f"Year {year}: Found {len(year_papers)} papers, "
                f"{year_full_text} with full text"
            )
            total_stored += len(year_papers)
            total_full_text += year_full_text
    # Final summary
    summary = f"SEARCH COMPLETE:\n"
    summary += f"Total papers stored: {total_stored}\n"
    summary += f"Total with full text: {total_full_text}\n\n"
    summary += f"Titles with PMIDS: {all_papers}\n\n"
    return summary


# Original function kept for backward compatibility
def pubmed_search_and_store(keywords: str, year: str = None, pnum: int = 10) -> str:
    """Original single-year search function."""
    return pubmed_search_and_store_multi_year(keywords, year, pnum)


# Create tools
pubmed_search_and_store_tool = tool(
    pubmed_search_and_store_multi_year,
    description="Search PubMed for papers and store them. Supports single year, multiple years (comma-separated), or year ranges (e.g., 2022-2024)"
)

def search_rag_database(query: str, num_results: int = 10) -> str:
    """
    Enhanced search that better handles specific paper requests.
    
    Args:
        query: Can be:
            - A PMID (e.g., "12345" or "PMID:12345")
            - Multiple PMIDs (e.g., "12345,67890")
            - A paper title or partial title
            - General keywords for topic search
        num_results: Number of results (ignored for PMID lookups)
    
    Returns:
        Formatted search results with appropriate detail level
    """
    print(f"🔎 Searching RAG database for: '{query}'")
    # Clean up query - remove "PMID:" prefix if present
    clean_query = query.replace("PMID:", "").replace("pmid:", "").strip()
    # Check if this is a PMID query
    is_pmid_query = all(part.strip().isdigit() for part in clean_query.split(','))
    # Check if this might be a title search (contains quotes or looks like a full title)
    is_title_search = ('"' in query or 
                      len(query.split()) > 5 or 
                      any(word in query.lower() for word in ['title:', 'paper:', 'article:']))
    if is_pmid_query:
        # Direct PMID lookup - return FULL content
        pmids = [p.strip() for p in clean_query.split(',')]
        results = []
        for pmid in pmids:
            # Get ALL chunks for this specific PMID
            pmid_filter = {"pmid": pmid}
            # Try to use filter if available, otherwise search and filter manually
            try:
                pmid_results = vectorstore.similarity_search(
                    f"PMID: {pmid}", 
                    k=50,  # Get many chunks to ensure complete paper
                    filter=pmid_filter
                )
            except:
                # Fallback if filter not supported
                pmid_results = vectorstore.similarity_search(f"PMID: {pmid}", k=50)
                pmid_results = [r for r in pmid_results if r.metadata.get('pmid') == pmid]
            # Sort by chunk index to maintain order
            pmid_results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
            results.extend([(doc, 1.0) for doc in pmid_results])
        if not results:
            return f"No paper found with PMID: {clean_query}"
    elif is_title_search:
        # Title search - try to find specific paper and return full content
        clean_title = query.replace('"', '').replace('title:', '').replace('paper:', '').strip()
        # Search by title
        results = vectorstore.similarity_search_with_score(clean_title, k=30)
        if results:
            # Find the best matching paper by title similarity
            papers_by_title = {}
            for doc, score in results:
                title = doc.metadata.get("title", "")
                pmid = doc.metadata.get("pmid", "unknown")
                # Calculate title similarity
                title_lower = title.lower()
                query_lower = clean_title.lower()
                # Check for exact or near-exact title match
                if query_lower in title_lower or title_lower in query_lower:
                    if pmid not in papers_by_title:
                        papers_by_title[pmid] = {
                            "docs": [],
                            "title": title,
                            "similarity": score
                        }
                    papers_by_title[pmid]["docs"].append((doc, score))
            # If we found a good title match, get ALL chunks for that paper
            if papers_by_title:
                best_pmid = min(papers_by_title.keys(), 
                               key=lambda x: papers_by_title[x]["similarity"])
                # Get all chunks for this specific paper
                all_chunks = vectorstore.similarity_search(
                    f"PMID: {best_pmid}", 
                    k=50
                )
                all_chunks = [c for c in all_chunks if c.metadata.get('pmid') == best_pmid]
                all_chunks.sort(key=lambda x: x.metadata.get('chunk_index', 0))
                results = [(doc, 1.0) for doc in all_chunks]
    else:
        # General keyword search - return summaries of multiple papers
        results = vectorstore.similarity_search_with_score(query, k=num_results)
    if not results:
        return "No relevant papers found in local database."
    # Group results by PMID
    papers_dict = {}
    for doc, score in results:
        pmid = doc.metadata.get("pmid", "unknown")
        if pmid not in papers_dict:
            papers_dict[pmid] = {
                "pmid": pmid,
                "title": doc.metadata.get("title", "Unknown"),
                "authors": doc.metadata.get("authors", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "has_full_text": doc.metadata.get("has_full_text", False),
                "chunks": [],
                "scores": []
            }
        papers_dict[pmid]["chunks"].append(doc.page_content)
        papers_dict[pmid]["scores"].append(score)
    # Format output based on query type
    output = []
    # Determine detail level based on number of papers and query type
    is_single_paper_request = (is_pmid_query and len(papers_dict) == 1) or \
                             (is_title_search and len(papers_dict) == 1)
    if is_single_paper_request:
        # Single paper requested - provide FULL content
        output.append(f"📄 DETAILED PAPER INFORMATION:\n")
        for pmid, info in papers_dict.items():
            content_type = "FULL TEXT" if info['has_full_text'] else "ABSTRACT"
            # Combine ALL chunks for complete content
            full_content = "\n\n".join(info['chunks'])
            # Remove duplicate information that might appear across chunks
            lines = full_content.split('\n')
            seen_lines = set()
            clean_lines = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and line_stripped not in seen_lines:
                    seen_lines.add(line_stripped)
                    clean_lines.append(line)
            full_content = '\n'.join(clean_lines)
            output.append(
                f"\n{'='*60}\n"
                f"PMID: {pmid}\n"
                f"Title: {info['title']}\n"
                f"Authors: {info['authors']}\n"
                f"Year: {info['year']}\n"
                f"Content Type: {content_type}\n"
                f"{'='*60}\n\n"
                f"COMPLETE CONTENT:\n\n{full_content}\n"
                f"{'='*60}\n"
            )
    elif len(papers_dict) <= 3:
        # Few papers - provide moderate detail
        output.append(f"Found {len(papers_dict)} relevant paper(s):\n")
        for idx, (pmid, info) in enumerate(papers_dict.items(), 1):
            content_type = "full text" if info['has_full_text'] else "abstract"
            # Provide first 2-3 chunks for moderate detail
            combined_content = "\n\n".join(info['chunks'][:3])
            output.append(
                f"\n--- PAPER {idx} ---\n"
                f"PMID: {pmid}\n"
                f"Title: {info['title']}\n"
                f"Authors: {info['authors']}\n"
                f"Year: {info['year']}\n"
                f"Content Type: {content_type}\n"
                f"\nContent Preview:\n{combined_content}\n"
            )
    else:
        # Many papers - provide summaries
        output.append(f"Found {len(papers_dict)} relevant papers:\n")
        for idx, (pmid, info) in enumerate(papers_dict.items(), 1):
            content_type = "full text" if info['has_full_text'] else "abstract"
            # Just show a brief excerpt
            excerpt = info['chunks'][0][:500] + "..."
            output.append(
                f"\n[{idx}] PMID: {pmid} | {info['title'][:80]}...\n"
                f"    Authors: {info['authors'][:50]}...\n"
                f"    Year: {info['year']} | Type: {content_type}\n"
                f"    Preview: {excerpt[:200]}...\n"
            )
        output.append("\n💡 Tip: To see full details of a specific paper, search by its PMID.")
    return "".join(output)

search_rag_database_tool = tool(search_rag_database)

def check_rag_for_topic(keywords: str) -> str:
    """Checks if papers on this topic exist in the RAG database."""
    try:
        collection = vectorstore._collection
        count = collection.count()
        if count == 0:
            return "Database is empty. No papers stored yet."
        results = vectorstore.similarity_search(keywords, k=5)
        if not results:
            return f"No papers found for '{keywords}' in database ({count} total chunks)."
        papers_info = {}
        for doc in results:
            pmid = doc.metadata.get("pmid")
            if pmid and pmid not in papers_info:
                papers_info[pmid] = {
                    "title": doc.metadata.get("title"),
                    "year": doc.metadata.get("year")
                }
        output = f"Found {len(papers_info)} relevant papers in database:\n"
        for pmid, info in papers_info.items():
            output += f"- PMID: {pmid} | {info['title'][:60]}... ({info['year']})\n"
        return output
    except Exception as e:
        return f"Error checking database: {e}"

check_rag_for_topic_tool = tool(check_rag_for_topic)

def get_database_stats() -> str:
    """Get statistics about the RAG database."""
    try:
        collection = vectorstore._collection
        count = collection.count()
        if count == 0:
            return "Database is empty (0 chunks, 0 papers)"
        sample_results = vectorstore.similarity_search("", k=min(100, count))
        unique_pmids = set()
        unique_titles = set()
        for doc in sample_results:
            pmid = doc.metadata.get("pmid")
            title = doc.metadata.get("title")
            if pmid:
                unique_pmids.add(pmid)
            if title:
                unique_titles.add(title[:50])
        output = f"Database Statistics:\n"
        output += f"- Total chunks: {count}\n"
        output += f"- Estimated papers: {len(unique_pmids)}\n"
        if unique_titles:
            output += f"- Sample papers:\n"
            for title in list(unique_titles)[:3]:
                output += f"  • {title}...\n"
        return output
    except Exception as e:
        return f"Error getting stats: {e}"

get_database_stats_tool = tool(get_database_stats)

# ============================================
# ENHANCED AGENT SETUP
# ============================================

memory = MemorySaver()

# Enhanced system prompt to handle specific paper requests better
system_prompt = '''You are a PubMed research assistant with a local RAG database.

IMPORTANT PAPER RETRIEVAL RULES:

1. SPECIFIC PAPER REQUESTS:
   - When user asks about a SPECIFIC paper (by PMID or title), retrieve ONLY that paper
   - Use the exact PMID if provided (e.g., "tell me about PMID 12345" → search "12345")
   - Use the paper title if that's what they reference
   - The search will return COMPLETE content for single papers

2. MULTIPLE PAPERS:
   - When user asks about a topic, retrieve multiple relevant papers
   - Provide summaries and comparisons across papers

3. SEARCH STRATEGIES:
   - For "tell me more about [specific paper]": Use PMID or title for targeted retrieval
   - For "what does [paper] say about X": First retrieve the specific paper by PMID
   - For general topics: Use keyword search

WORKFLOW:
1. Check local database with 'check_rag_for_topic_tool'
2. For existing papers:
   - Use PMID for specific paper requests
   - Use keywords for topic searches
3. For new searches:
   - Use 'pubmed_search_and_store_tool' first
   - Then retrieve with 'search_rag_database_tool'
   - Ask the user for next steps 

CRITICAL RULES:
- When discussing a specific paper, cite ONLY from that paper's content
- Never mix information from different papers unless explicitly comparing
- Always use PMIDs when referring to specific papers
- When a user asks for "more details" or "full information" about a paper they've mentioned, retrieve it by PMID
- If a pubmed search does not return anything, ask the user for next steps 

RESPONSE FORMAT:
- For single paper: Provide comprehensive analysis using all available content
- For multiple papers: Provide structured comparisons
- Always cite with format: "According to [Title] (PMID: [number])..."
'''

# Create agent with enhanced tools
tools = [
    pubmed_search_and_store_tool,
    search_rag_database_tool,
    check_rag_for_topic_tool,
    get_database_stats_tool,
]

agent = create_react_agent(
    model=pi_llm,
    tools=tools,
    checkpointer=memory,
    prompt=system_prompt
)

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("PubMed Research Assistant")
    print("Commands: 'exit' to quit, 'stats' for database info, 'clear' to clear screen\n")
    
    stats_result = get_database_stats()
    print(f"{stats_result}\n")
    
    config = {"configurable": {"thread_id": "cli-thread"}}
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            if user_input.lower() == "stats":
                print(f"\n{get_database_stats()}\n")
                continue
            
            if user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # Test command for direct multi-year search
            if user_input.lower().startswith("test multi"):
                parts = user_input[10:].strip().split()
                keywords = parts[0] if parts else "CRISPR"
                years = parts[1] if len(parts) > 1 else "2022-2024"
                print(f"\nTesting multi-year search: '{keywords}' for years {years}...")
                result = pubmed_search_and_store_multi_year(keywords, years, pnum=3)
                print(result)
                print(f"\n{get_database_stats()}\n")
                continue
            
            # Regular agent interaction
            print("\nProcessing...")
            
            # Invoke agent with increased recursion limit for complex multi-year searches
            result = agent.invoke(
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

