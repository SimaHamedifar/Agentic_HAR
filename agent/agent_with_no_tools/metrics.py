import pandas as pd

def _classification_summary(y_true, y_pred, labels):
    records = []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        records.append({'activity': label, 'precision': precision, 'recall': recall,
                        'f1': f1, 'support': int((y_true == label).sum())})
    return pd.DataFrame(records)

def compute_metrics(y_true, y_pred_labels, n_tested: int, invalid_or_error_rate: float):
    labels = sorted(y_true.dropna().unique().tolist())
    report = _classification_summary(y_true, y_pred_labels, labels)
    
    metrics = {
        'n_tested': n_tested,
        'accuracy': float((y_true == y_pred_labels).mean()),
        'macro_f1': float(report['f1'].mean()),
        'invalid_or_error_rate': invalid_or_error_rate,
        'report': report,
    }
    return metrics