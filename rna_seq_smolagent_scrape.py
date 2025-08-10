"""
LINE-1 RNA-seq Database Monitor using smolagents
Searches GEO and ENA databases for new LINE-1 retrotransposon studies
Saves results to TSV file instead of sending emails
"""

import os
import csv
import json
import requests
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool, tool
from smolagents.utils import encode_image_base64, make_image_url
from smolagents import OpenAIServerModel
from dotenv import load_dotenv
from Bio import Entrez
import sys

# Load environment variables
load_dotenv()

@tool
def search_omics_data(query: str, species: str, year: int, email: str, max_results: int = 20):
    """
    Searches GEO and SRA for studies matching a query, species, and year.

    Args:
        query (str): The main search term (e.g., "breast cancer", "neurodevelopment").
        species (str): The scientific name of the organism (e.g., "Homo sapiens", "Mus musculus").
        year (int): The publication year to filter by.
        email (str): Your email address (required by NCBI).
        max_results (int): The maximum number of results to return per database. Defaults to 20.

    Returns:
        dict: A dictionary containing lists of results for 'GEO' and 'SRA'.
              Returns None if an error occurs.
    """
    Entrez.email = email
    
    # Construct a precise search term using NCBI's query syntax
    search_term = f'({query}) AND "{species}"[Organism] AND "{year}"[Publication Date]'
    print(f"🔬 Using search term: {search_term}\n")

    results = {"GEO": [], "SRA": []}
    
    try:
        # --- 1. Search GEO Datasets (gds) ---
        print("--- Searching GEO Datasets... ---")
        handle_geo = Entrez.esearch(db="gds", term=search_term, retmax=max_results, sort="relevance")
        record_geo = Entrez.read(handle_geo)
        handle_geo.close()
        
        geo_ids = record_geo["IdList"]
        if geo_ids:
            print(f"Found {len(geo_ids)} GEO datasets. Fetching summaries...")
            summary_handle_geo = Entrez.esummary(db="gds", id=",".join(geo_ids))
            summary_geo = Entrez.read(summary_handle_geo)
            summary_handle_geo.close()
            # Format the GEO results
            for summary in summary_geo:
                results["GEO"].append({
                    "ID": summary["Id"],
                    "Title": summary["title"],
                    "Summary": summary["summary"],
                    "PubDate": summary["pubdate"],
                })
        else:
            print("No matching GEO datasets found.")

        # --- 2. Search Sequence Read Archive (SRA) ---
        print("\n--- Searching SRA Experiments... ---")
        handle_sra = Entrez.esearch(db="sra", term=search_term, retmax=max_results, sort="relevance")
        record_sra = Entrez.read(handle_sra)
        handle_sra.close()
        
        sra_ids = record_sra["IdList"]
        if sra_ids:
            print(f"Found {len(sra_ids)} SRA experiments. Fetching summaries...")
            summary_handle_sra = Entrez.esummary(db="sra", id=",".join(sra_ids))
            summary_sra = Entrez.read(summary_handle_sra)
            summary_handle_sra.close()
            # Format the SRA results
            for summary in summary_sra:
                results["SRA"].append({
                    "Accession": summary["Accession"],
                    "Title": summary["Title"],
                    "Platform": summary["Platform"],
                    "Study": summary["Study"],
                })
        else:
            print("No matching SRA experiments found.")
            
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return None
        
    return results

model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # Free model
        # Alternative free models you can try:
        # model_id="google/flan-t5-base",
        # model_id="microsoft/Phi-3-mini-4k-instruct",
        # model_id="HuggingFaceH4/zephyr-7b-beta",
        # model_id="mistralai/Mistral-7B-Instruct-v0.1",
        provider='together',
        token=os.getenv("HF_TOKEN_FINE_GRAINED"),  # Optional, helps with rate limits
        timeout=120  # Increase timeout for free tier
        )

web_agent = CodeAgent(
    model=model,
    tools=[
        search_omics_data, 
        VisitWebpageTool(),
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)

def check_reasoning_and_output(final_answer, agent_memory, output_type="plot"):
    """
    Evaluates the reasoning and output (plot or table) of a multi-agent system for finding LINE-1 related RNA-seq experiments.
    
    Args:
        final_answer: The final output (e.g., study accessions or metadata) from the multi-agent system.
        agent_memory: The agent's memory containing reasoning steps (e.g., search queries, filtering logic).
        output_type: Type of output to evaluate ("plot" for px.scatter_map, "table" for study metadata).
    
    Returns:
        bool: True if the output passes, raises Exception if it fails.
    """
    # Use Qwen2-VL-7B-Instruct instead of OpenAIServerModel
    multimodal_model = InferenceClientModel(model_id="Qwen/Qwen2-VL-7B-Instruct", api_key=os.environ["HF_TOKEN_FINE_GRAINED"])
    
    # Initialize variables for output evaluation
    content_to_evaluate = []
    prompt_parts = []

    # Handle different output types (plot or table)
    if output_type == "plot":
        filepath = "saved_map.png"
        assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
        image = Image.open(filepath)
        content_to_evaluate.append({
            "type": "image_url",
            "image_url": {"url": make_image_url(encode_image_base64(image))},
        })
        output_description = "a scatter map plot created using px.scatter_map"
    elif output_type == "table":
        filepath = "saved_results.json"  # Assume results are saved as JSON
        assert os.path.exists(filepath), "Make sure to save the results under saved_results.json!"
        with open(filepath, 'r') as f:
            table_data = json.load(f)
        content_to_evaluate.append({
            "type": "text",
            "text": f"Table data: {json.dumps(table_data, indent=2)}",
        })
        output_description = "a table summarizing LINE-1 related RNA-seq experiments"
    else:
        raise ValueError("output_type must be 'plot' or 'table'")

    # Construct the prompt
    prompt = (
        f"The task is to find LINE-1 related RNA-seq experiments for the Homo sapiens taxon from year 2021 until the year 2025 from the European Nucleotide Archive (ENA) and Gene Expression Omnibus (GEO). "
        f"The multi-agent system executed the following steps: {agent_memory.get_succinct_steps()}. "
        f"The final output is {output_description}. "
        f"Please evaluate if the reasoning process and output correctly address the task. Specifically, check:\n"
        f"1. Does the output include RNA-seq experiments explicitly related to LINE-1 for the Homo sapiens taxon? (e.g., studies mentioning LINE-1, retrotransposons, or transposable elements)?\n"
        f"2. Are the experiments sourced from ENA or GEO, with valid study accessions or metadata (e.g., geographic location, collection date)?\n"
        f"3. For plots, was px.scatter_map used to visualize experiment locations or metadata? For tables, does the data include relevant study details (e.g., accession, title, sample count)?\n"
        f"4. Is the reasoning logical, with clear search queries and filtering for LINE-1 relevance?\n"
        f"List reasons why the output is correct or incorrect, then provide a final decision: PASS in caps lock if satisfactory, FAIL if not. "
        f"Don't be harsh: if the output mostly solves the task (e.g., retrieves relevant experiments), it should pass."
    )
    prompt_parts.append({"type": "text", "text": prompt})

    # Combine prompt and output content
    messages = [
        {
            "role": "user",
            "content": prompt_parts + content_to_evaluate,
        }
    ]

    # Get model output
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    
    # Check for failure and raise exception
    if "FAIL" in output:
        raise Exception(f"Output failed validation: {output}")
    
    return True

manager_agent = CodeAgent(
    model=InferenceClientModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096),
    tools=[],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_output],
    max_steps=15,
)

manager_agent.run("""
Find me 5 bulk RNA-seq experiments on non-diseased tissues related to LINE-1 retrotransposons in the Homo sapiens taxon for the years 2022-2025 from the GEO and/or the ENA database 
The samples should allow me to perform differential gene expression analysis between at least two tissues within the same study
""")
