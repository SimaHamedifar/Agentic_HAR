# Agentic HAR

Recognize a smart-home resident’s current activity from MuRAL sensor events. The workflow follows the original `agent/har_agent/main2.ipynb` experiment. A BiLSTM with self-attention scores the ten preceding events. LangGraph retrieves household context and historical patterns from Pinecone, then Claude selects one of 22 activity IDs using the current sensor event, the history, and the model’s top three scores.

The bundled checkpoint comes from the original project. Activity IDs and label spellings remain compatible with MuRAL. The model was trained to classify the last event inside its own window. Its scores are supporting evidence about the immediate past when the agent classifies the next event.

## Run the walkthrough

From this folder with Python 3.11 or later:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
jupyter lab walkthrough.ipynb
```

The default example runs the real checkpoint and graph with synthetic sensor events, local retrieval, and a deterministic stand-in for Claude. Optional cells replay the original MuRAL CSV and can call the configured live services. The original dataset stays in `Agentic_AI_HAR_Sima/data/MuRAL`.

## Deploy the API

Copy `.env.example` to `.env` and fill in the provider keys and an Anthropic model your account supports that accepts native JSON-schema output. Use a Pinecone index compatible with 1,536-dimensional cosine embeddings. To create the index if needed and seed the configured namespace, run this once with the original MuRAL folder path:

```sh
python -m agentic_har.knowledge.seed --mural-dir /path/to/Agentic_AI_HAR_Sima/data/MuRAL --create-index
```

The seed includes the original general corpus plus H002 layout and history. The live graph searches layout and historical patterns for the requested household, following the original retrieval configuration. The supplied corpus includes household-specific documents for H002. Choose the appropriate household ID and seed its data before serving other households. Seeding sends document text to OpenAI for embedding and stores it in Pinecone. Live requests send sensor history to OpenAI for embedding and sensor evidence and retrieved context to Anthropic for classification.

Start locally or use the CPU container:

```sh
uvicorn agentic_har.api:app
# Or build and run the container
docker build -t agentic-har .
docker run --env-file .env -p 127.0.0.1:8000:8000 agentic-har
```

The example payload uses synthetic events:

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/recognize \
  -H 'Content-Type: application/json' --data-binary @example_request.json
```

Open [API docs](http://127.0.0.1:8000/docs) for the request and response schemas. Supply ten preceding events from the same household in chronological order and one current event. Each event uses a local `HH:MM:SS` time and MuRAL sensor values. Historical events may include previously recognized activity names and precomputed `time_delta` values. The pipeline computes omitted deltas from adjacent timestamps, using zero for the first history row. Numeric activity IDs, display labels, and model scores are returned as JSON. Place the service behind an authenticated gateway if exposing it beyond localhost.

## Project layout

| Path | Purpose |
| --- | --- |
| `agentic_har/activity_ml/` | MuRAL preprocessing, training, and BiLSTM inference |
| `agentic_har/pipeline.py` | Format events, compute features, and combine model and graph output |
| `agentic_har/agent.py` | Retrieve household evidence and select a structured activity ID |
| `agentic_har/knowledge/` | Knowledge corpus and explicit indexing command |
| `agentic_har/dataset.py`, `metrics.py` | Chronological replay and evaluation |
| `agentic_har/runtime.py`, `api.py` | Provider wiring and HTTP service |

For optional model training, install `.[training]`, copy `training.example.json`, and set its MuRAL folder path. Run `python -m agentic_har.activity_ml.classifier --config training.example.json`. Set `HAR_CHECKPOINT` to the resulting checkpoint to serve it.
