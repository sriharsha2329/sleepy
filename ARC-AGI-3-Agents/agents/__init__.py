"""Minimal agent registry — registers OUR MyAgent (+ random) without the heavy template imports
(langgraph/smolagents/llm) that aren't installed. main.py reads AVAILABLE_AGENTS from here. Original kept as
agents/__init__.py.orig."""
from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.random_agent import Random
from .my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: "dict[str, Type[Agent]]" = {"random": Random, "myagent": MyAgent}
