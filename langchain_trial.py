import token
from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, WebSearchTool
import os
from dotenv import load_dotenv

load_dotenv()
# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=InferenceClientModel())
# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

from smolagents import tool
import numpy as np
import time
import datetime

agent = CodeAgent(tools=[WebSearchTool()], model=InferenceClientModel(), additional_authorized_imports=['datetime'])

# agent.run(
#     """
#     Alfred needs to prepare for the party. Here are the tasks:
#     1. Prepare the drinks - 30 minutes
#     2. Decorate the mansion - 60 minutes
#     3. Set up the menu - 45 minutes
#     4. Prepare the music and playlist - 45 minutes
#     If we start right now, at what time will the party be ready?
#     """
# )

# hf_key = os.environ.get("HF_TOKEN")
# hf_key_fg = os.environ.get("HF_TOKEN_FINE_GRAINED")
# hf_key_read = os.environ.get("HF_TOKEN_READ")
# # Change to your username and repo name
# host = os.environ.get("LANGFUSE_HOST")
# pkey = os.environ.get("LANGFUSE_PUBLIC_KEY")
# skey = os.environ.get("LANGFUSE_SECRET_KEY")
# login()

# # agent.push_to_hub('Rafail01/AlfredAgent')

# from langfuse import get_client
 
# langfuse = get_client()
 
# # Verify connection
# if langfuse.auth_check():
#     print("Langfuse client is authenticated and ready!")
# else:
#     print("Authentication failed. Please check your credentials and host.")


# from openinference.instrumentation.smolagents import SmolagentsInstrumentor

# SmolagentsInstrumentor().instrument()

# # Change to your username and repo name
# # alfred_agent = agent.from_hub('Rafail01/AlfredAgent', trust_remote_code=True)

agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")  

