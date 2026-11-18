"""Wire the trained model, Pinecone retrieval, and Claude reasoning together."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from .activity_ml.inference import HARPredictor
from .agent import build_agent
from .pipeline import RecognitionPipeline

DEFAULT_CHECKPOINT = Path(__file__).parent / "assets" / "best_model.pt"


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in the environment or .env")
    return value


def connect_store(*, create_index: bool = False) -> PineconeVectorStore:
    from pinecone import ServerlessSpec

    load_dotenv()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=required_env("OPENAI_API_KEY"), max_retries=2,
    )
    pinecone = Pinecone(api_key=required_env("PINECONE_API_KEY"))
    name = os.getenv("PINECONE_INDEX", "har-agent-rag")
    if create_index and not pinecone.has_index(name):
        pinecone.create_index(
            name=name, dimension=1536, metric="cosine", timeout=120,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return PineconeVectorStore(
        index=pinecone.Index(name), embedding=embeddings,
        namespace=os.getenv("PINECONE_NAMESPACE", "agentic-har"),
    )


def load_predictor() -> HARPredictor:
    return HARPredictor.from_checkpoint(os.getenv("HAR_CHECKPOINT", str(DEFAULT_CHECKPOINT)))


def build_live_pipeline() -> RecognitionPipeline:
    load_dotenv()
    llm = ChatAnthropic(
        model=required_env("ANTHROPIC_MODEL"),
        api_key=required_env("ANTHROPIC_API_KEY"), timeout=30, max_retries=2,
    )
    return RecognitionPipeline(load_predictor(), build_agent(llm, connect_store()))
