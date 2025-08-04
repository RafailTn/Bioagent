from huggingface_hub import login
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

# Authenticate with your Hugging Face write token
login(token=os.environ["HF_TOKEN"])

# Define the agent
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=InferenceClientModel(),
    additional_authorized_imports=['datetime']
)

