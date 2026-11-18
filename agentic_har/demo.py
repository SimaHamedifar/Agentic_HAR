"""Offline example using synthetic sensor events and the trained checkpoint."""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from .agent import build_agent
from .knowledge.household import household_documents
from .labels import ACTIVITY_MAP
from .pipeline import RecognitionPipeline
from .schemas import RecognitionRequest

EXAMPLE = RecognitionRequest.model_validate({
    "household_id": "H002",
    "history": [
        {"time": f"19:{minute:02}:00", "location": "couch",
         "sensor_status": "ON" if minute % 2 else "OFF"}
        for minute in range(10)
    ],
    "event": {"time": "19:10:00", "appliance": "TV", "sensor_status": "ON"},
})


class DemoStore:
    def similarity_search(self, query, *, k, filter):
        documents = [*household_documents, Document(
            page_content="Synthetic example: TV use often follows couch motion.",
            metadata={"household_id": "H002", "knowledge_type": "historical_pattern"},
        )]
        return [doc for doc in documents if all(doc.metadata.get(key) == value
                for key, value in filter.items())][:k]


class DemoReasoner:
    def with_structured_output(self, schema, *, method):
        def classify(prompt):
            text = prompt.messages[-1].content
            current_event = text.split("Preceding sensor events:", 1)[0]
            name = "watching TV" if "TV activated" in current_event else "others"
            return schema(activity=next(key for key, label in ACTIVITY_MAP.items() if label == name))

        return RunnableLambda(classify)


def build_demo_pipeline(predictor=None) -> RecognitionPipeline:
    if predictor is None:
        from .runtime import load_predictor

        predictor = load_predictor()
    return RecognitionPipeline(predictor, build_agent(DemoReasoner(), DemoStore()))
