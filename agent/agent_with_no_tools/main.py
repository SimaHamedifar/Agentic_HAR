import argparse
import os
import pandas as pd
import dotenv
from langchain_anthropic import ChatAnthropic

from agent.agent_with_no_tools.config import SENSOR_TYPES, ACTIVITY_MAP
from data import load_data, get_train_test_split, slice_data, format_event_sequences
from agent import build_agent
from metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM-based Human Activity Recognition")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV data")
    parser.add_argument("--output_path", type=str, default="results/Agent_predictions.csv", help="Path to save predictions")
    parser.add_argument("--window_length", type=int, default=10, help="Sliding window length")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples to process (for debugging)")
    parser.add_argument("--model", type=str, default="claude-3-sonnet-20240229", help="Anthropic model name to use")
    return parser.parse_args()

def main():
    args = parse_args()

    dotenv.load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    
    llm = ChatAnthropic(model_name=args.model, api_key=api_key)
    agent = build_agent(llm)

    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    
    ratio = 0.5 if len(df) < 50 else 0.95
    _, test_df = get_train_test_split(df, train_ratio=ratio)
    print(f"Test split size: {len(test_df)}")

    X, x, y = slice_data(test_df, window_length=args.window_length)

    print("Formatting event sequences...")
    event_sequence_txt, event_txt = format_event_sequences(X, x, SENSOR_TYPES)

    if args.limit:
        event_sequence_txt = event_sequence_txt[:args.limit]
        event_txt = event_txt[:args.limit]
        y = y[:args.limit]
        x = x[:args.limit]

    print(f"Running predictions on {len(event_sequence_txt)} sequences...")
    rows = []
    for n_event_seq, event_seq in enumerate(event_sequence_txt):
        if n_event_seq % 10 == 0:
            print(f"Processed {n_event_seq}/{len(event_sequence_txt)}")
            
        result = agent.invoke({
            'event_sequence': event_seq,
            'event': event_txt[n_event_seq]
        })
        rows.append({
            'event_id': n_event_seq, 
            'true_activity': y[n_event_seq],
            'predicted_activity': result["detected_activity"]
        })

    predictions = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    predictions.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")

    y_true = predictions['true_activity']
    y_pred = predictions['predicted_activity']
    
    y_pred_labels = y_pred.astype(str).map(ACTIVITY_MAP)
    
    invalid_or_error_rate = float(y_pred_labels.isna().mean())
    n_tested = len(predictions)
    
    metrics = compute_metrics(y_true, y_pred_labels, n_tested, invalid_or_error_rate)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Total Tested: {metrics['n_tested']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Invalid/Error Rate: {metrics['invalid_or_error_rate']:.4f}")

if __name__ == "__main__":
    main()