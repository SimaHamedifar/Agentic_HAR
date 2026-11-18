"""Publish the MuRAL knowledge corpus to a configured Pinecone namespace."""

import argparse
from pathlib import Path

from langchain_core.documents import Document

from ..activity_ml.preprocessing import load_activity_lookup, load_house_frame
from ..pipeline import describe_event
from ..runtime import connect_store
from ..schemas import HistoricalEvent
from .general import general_documents
from .household import HOUSEHOLD_ID, household_documents


def historical_documents(mural_dir: Path) -> list[Document]:
    frame = load_house_frame(mural_dir / "02", load_activity_lookup(mural_dir), "02")
    fields = list(HistoricalEvent.model_fields)
    lines = [
        describe_event(HistoricalEvent.model_validate(row))
        for row in frame[fields].to_dict("records")
    ]
    prefix = "Historical example of sensor events and activities from a weekend outing return.\n"
    chunks = []
    chunk = prefix
    for line in lines:
        if len(chunk) + len(line) + 1 > 8000:
            chunks.append(chunk)
            chunk = prefix
        chunk += line + "\n"
    if chunk != prefix:
        chunks.append(chunk)
    return [
        Document(page_content=chunk, metadata={
            "knowledge_type": "historical_pattern",
            "household_id": HOUSEHOLD_ID,
            "category": "weekend",
        })
        for chunk in chunks
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mural-dir", type=Path, required=True)
    parser.add_argument("--create-index", action="store_true")
    args = parser.parse_args()
    groups = {
        "general": general_documents,
        "h002-household": household_documents,
        "h002-history": historical_documents(args.mural_dir),
    }
    store = connect_store(create_index=args.create_index)
    for prefix, docs in groups.items():
        ids = [f"{prefix}-{i}" for i in range(len(docs))]
        store.add_documents(docs, ids=ids)
        print(f"Upserted {len(docs)} documents for {prefix}")


if __name__ == "__main__":
    main()
