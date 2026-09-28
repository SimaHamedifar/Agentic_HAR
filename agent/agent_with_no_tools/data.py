import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Loads and returns the dataframe, dropping the Unnamed: 0 column if it exists."""
    df = pd.read_csv(filepath)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def get_train_test_split(df: pd.DataFrame, train_ratio: float = 0.95):
    """Splits data into train and test without shuffling."""
    train_size = int(train_ratio * len(df))
    train_df = df.iloc[:train_size]
    test_df = df[train_size:].reset_index(drop=True)
    return train_df, test_df

def slice_data(df: pd.DataFrame, output: str = "activity", window_length: int = 10):
    """Slices data into sliding windows."""
    X = []
    x = []
    y = []
    for i in range(len(df) - window_length):
        window = df.iloc[i:i + window_length]
        X.append(window)
        x.append(df.iloc[i + window_length])
        y.append(df.iloc[i + window_length][output])
    return X, x, y

def format_event_sequences(X, x, sensor_types: list[str]):
    """Formats event sequences into string representations."""
    event_sequence_txt = []
    
    for window_id, window in enumerate(X):
        event_seq_txt = ""
        for event_id, event in window.iterrows():
            activated_sensor_type = "unknown_sensor"
            for sensor_type in sensor_types:
                if event[sensor_type] != 'unknown':
                    activated_sensor_type = sensor_type
            
            event_seq_txt += (
                f"At time {event['time']}, Activated sensor: {activated_sensor_type}: {event.get(activated_sensor_type, 'unknown')} with the status of {event['sensor_status']}. "
                f"Detected activity: {event['activity']}.\n"
            )
        event_sequence_txt.append(event_seq_txt)

    event_txt = []
    for i, event in enumerate(x): 
        # For the target event, we also need to figure out which sensor triggered it
        activated_sensor_type = "unknown_sensor"
        for sensor_type in sensor_types:
            if event[sensor_type] != 'unknown':
                activated_sensor_type = sensor_type
        
        event_txt.append(
            f"At time {event['time']}, Activated sensor: {activated_sensor_type}: {event.get(activated_sensor_type, 'unknown')} with the status of {event['sensor_status']}. "
        )

    return event_sequence_txt, event_txt