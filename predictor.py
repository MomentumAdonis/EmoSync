import math
import pandas as pd
import numpy as np
import json

def map_va_to_emotion(valence, arousal, level1=0.2, level2=0.6):
    """
    Maps a given valence and arousal value to an emotion label using circular segmentation.
    
    The function first converts (valence, arousal) into polar coordinates (radius and angle).
    If the radius is very small (i.e., below level1), the emotion is classified as "Neutral".
    Otherwise, based on the radius, the angle is divided into bins:
      - If radius is between level1 and level2, it uses 8 bins (45° each) for level2 emotions.
      - If radius is >= level2, it uses 12 bins (30° each) for level3 emotions.
    
    Parameters
    ----------
    valence : float
        The valence value (expected range: -1.0 to +1.0).
    arousal : float
        The arousal value (expected range: -1.0 to +1.0).
    level1 : float, optional
        Radius threshold for "Neutral/Indifferent" emotion. Default is 0.2.
    level2 : float, optional
        Radius threshold separating level2 and level3 emotions. Default is 0.6.
        
    Returns
    -------
    emotion_label : str
        The emotion label corresponding to the input (valence, arousal) pair.
    """
    # Calculate the Euclidean distance (radius) from the origin.
    radius = math.sqrt(valence**2 + arousal**2)
    # Calculate the angle in degrees using arctan2, which handles quadrant determination.
    angle = math.degrees(math.atan2(arousal, valence))
    if angle < 0:
        angle += 360  # Normalize negative angles to be within [0, 360)

    # If the radius is below the first threshold, classify as Neutral.
    if radius < level1:
        return "Neutral"

    # Define the emotion labels for the middle ring (level2): 8 bins of 45° each.
    level2_labels = [
        "Cheerful",
        "Joyous",
        "Impatient",
        "Stressed",
        "Gloomy",
        "Fatigued",
        "Calm",
        "Hopeful"
    ]
    # Define the emotion labels for the outer ring (level3): 12 bins of 30° each.
    level3_labels = [
        "Happy",
        "Delighted",
        "Excited",
        "Tense",
        "Angry",
        "Frustrated",
        "Depressed",
        "Bored",
        "Tired",
        "Peaceful",
        "Relaxed",
        "Content"
    ]

    # Determine the bin based on the radius.
    if radius < level2:
        # For radius between level1 and level2, use 8 bins (each of 45 degrees)
        bin_size = 360 / 8  
        bin_index = int(angle // bin_size)
        return level2_labels[bin_index]
    else:
        # For radius >= level2, use 12 bins (each of 30 degrees)
        bin_size = 360 / 12  
        bin_index = int(angle // bin_size)
        return level3_labels[bin_index]


def map_song_va_to_emotion(va_df, level1=0.2, level2=0.6):
    """
    Maps the valence and arousal predictions for each time step in a song to an emotion label.
    
    Parameters
    ----------
    va_df : pd.DataFrame
        DataFrame with at least:
          - 'time_s': float, timestamp in seconds
          - 'arousal_pred': float, predicted arousal
          - 'valence_pred': float, predicted valence
          
    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: 'time_s' and 'emotion', where 'emotion' is the
        label derived using map_va_to_emotion() for each row.
    """
    # Initialize an empty list to store emotion mapping for each row.
    emotion_list = []
    
    # Iterate over each row in the DataFrame.
    for index, row in va_df.iterrows():
        time_value = row["time_s"]
        arousal_value = row["arousal_pred"]
        valence_value = row["valence_pred"]
        # Map the (valence, arousal) pair to an emotion using the helper function.
        emotion = map_va_to_emotion(valence_value, arousal_value, level1=level1, level2=level2)
        # Append the result as a dictionary.
        emotion_list.append({"time_s": time_value, "emotion": emotion})
    
    # Convert the list of dictionaries to a DataFrame.
    df_emotion = pd.DataFrame(emotion_list)
    return df_emotion


def compress_playback_sequence(prepped_df):
    """
    Removes consecutive duplicate emotions from a playback sequence DataFrame.
    
    The input DataFrame should have columns: ['start_time', 'emotion'] in ascending order.
    Only the first occurrence in a sequence of identical emotions is retained.
    
    Parameters
    ----------
    prepped_df : pd.DataFrame
        DataFrame containing the playback sequence.
        
    Returns
    -------
    pd.DataFrame
        A new DataFrame with consecutive duplicate emotions removed.
    """
    if prepped_df.empty:
        return prepped_df.copy()

    compressed_rows = []
    last_emotion = None

    # Iterate over each row and add the row if its emotion is different from the last seen.
    for idx, row in prepped_df.iterrows():
        current_emotion = row['emotion']
        if current_emotion != last_emotion:
            compressed_rows.append((row['start_time'], current_emotion))
            last_emotion = current_emotion

    compressed_df = pd.DataFrame(compressed_rows, columns=['start_time', 'emotion'])
    return compressed_df


def prep_playback_sequence(emotion_df, minDuration=3.0):
    """
    Converts a detailed row-by-row timeline into a minimal sequence of emotion changes.
    
    This function condenses consecutive rows with the same emotion into a single segment.
    If a segment's duration is shorter than minDuration, consecutive segments are merged and
    the dominant emotion (by time duration) is selected for the merged segment.
    Finally, a row indicating "Completed" is appended at the end.
    
    Parameters
    ----------
    emotion_df : pd.DataFrame
        DataFrame with columns:
          - 'time_s': float, timestamp
          - 'emotion': str, emotion label at that timestamp.
    minDuration : float
        Minimum duration for a segment to be considered valid.
        
    Returns
    -------
    final_df : pd.DataFrame
        A DataFrame with columns ['start_time', 'emotion'] representing the condensed playback sequence.
    """
    # Sort the DataFrame by time.
    df_sorted = emotion_df.sort_values(by='time_s').reset_index(drop=True)
    if df_sorted.empty:
        return pd.DataFrame(columns=['start_time', 'emotion'])

    # Build raw segments by grouping consecutive rows with the same emotion.
    raw_segments = []
    current_emotion = df_sorted.loc[0, 'emotion']
    segment_start = df_sorted.loc[0, 'time_s']
    prev_time = segment_start
    prev_emotion = current_emotion

    n_rows = len(df_sorted)
    for i in range(1, n_rows):
        row_time = df_sorted.loc[i, 'time_s']
        row_emo = df_sorted.loc[i, 'emotion']

        if row_emo != prev_emotion:
            # End the current segment at the previous time and record it.
            raw_segments.append((segment_start, prev_time, current_emotion))
            # Start a new segment.
            segment_start = row_time
            current_emotion = row_emo

        prev_time = row_time
        prev_emotion = row_emo

    # Append the final segment.
    last_time = df_sorted.loc[n_rows-1, 'time_s']
    raw_segments.append((segment_start, last_time, current_emotion))

    # Create a DataFrame of raw segments and compute their durations.
    raw_df = pd.DataFrame(raw_segments, columns=['start_time', 'end_time', 'emotion'])
    raw_df['duration'] = raw_df['end_time'] - raw_df['start_time']

    # Merge segments that are shorter than minDuration.
    merged_segments = []
    i = 0
    n_segs = len(raw_df)

    while i < n_segs:
        seg_start = raw_df.loc[i, 'start_time']
        seg_end = raw_df.loc[i, 'end_time']
        chunk_emotions = [(raw_df.loc[i, 'start_time'], raw_df.loc[i, 'end_time'], raw_df.loc[i, 'emotion'])]
        chunk_duration = raw_df.loc[i, 'duration']
        i_next = i + 1

        # Merge subsequent segments until the total duration reaches minDuration.
        while chunk_duration < minDuration and i_next < n_segs:
            chunk_emotions.append((
                raw_df.loc[i_next, 'start_time'],
                raw_df.loc[i_next, 'end_time'],
                raw_df.loc[i_next, 'emotion']
            ))
            seg_end = raw_df.loc[i_next, 'end_time']
            chunk_duration += raw_df.loc[i_next, 'duration']
            i_next += 1

        # Determine the dominant emotion in the merged chunk.
        emo_time_dict = {}
        for (st, en, emo) in chunk_emotions:
            dur = en - st
            emo_time_dict[emo] = emo_time_dict.get(emo, 0.0) + dur
        dominant_emo = max(emo_time_dict, key=emo_time_dict.get)

        merged_segments.append((seg_start, seg_end, dominant_emo))
        i = i_next

    # Convert merged segments to a minimal sequence (only start_time and emotion).
    final_rows = [(seg_start, emo) for seg_start, seg_end, emo in merged_segments]
    # Append a final row to indicate playback completion.
    final_end = merged_segments[-1][1]
    final_rows.append((final_end, "Completed"))

    final_df = pd.DataFrame(final_rows, columns=['start_time', 'emotion'])
    # Remove any consecutive duplicate entries.
    final_df = compress_playback_sequence(final_df)

    return final_df


def jsonify_playback_sequence(prepped_df, song_id=None):
    """
    Converts a playback sequence DataFrame into a JSON string.
    
    The JSON structure includes the song ID (if provided) and a sequence list of time stamps
    with corresponding emotions.
    
    Parameters
    ----------
    prepped_df : pd.DataFrame
        DataFrame with columns ['start_time', 'emotion'] in ascending time order.
    song_id : int or str, optional
        Song identifier to include in the JSON.
        
    Returns
    -------
    json_str : str
        A JSON string with the following structure:
        {
          "song_id": <song_id or null>,
          "sequence": [
             {"start_time": 15.0, "emotion": "Neutral/Indifferent"},
             {"start_time": 28.5, "emotion": "Impatient"},
             ...
          ]
        }
    """
    # Build a list of dictionaries from each row in the DataFrame.
    sequence_list = []
    for _, row in prepped_df.iterrows():
        sequence_list.append({
            "start_time": float(row["start_time"]),
            "emotion": row["emotion"]
        })

    # Create the final JSON structure.
    output = {
        "song_id": song_id,
        "sequence": sequence_list
    }
    
    # Convert the structure to a JSON string.
    json_str = json.dumps(output)
    return json_str


def smoothen_prediction(df_results, smoothFactor=5):
    """
    Applies a simple moving-average smoothing to the 'arousal_pred' and 'valence_pred'
    columns in the given DataFrame.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame with columns 'time_s', 'arousal_pred', and 'valence_pred'.
    smoothFactor : int
        Window size for the moving average. Larger values result in more smoothing.
        
    Returns
    -------
    df_smoothed : pd.DataFrame
        A copy of the input DataFrame with the specified columns smoothed.
    """
    # Create a copy of the DataFrame and sort it by time.
    df_smoothed = df_results.copy()
    df_smoothed.sort_values(by='time_s', inplace=True)
    
    # Apply a centered rolling average to smooth 'arousal_pred' and 'valence_pred'.
    df_smoothed['arousal_pred'] = (
        df_smoothed['arousal_pred']
        .rolling(window=smoothFactor, min_periods=1, center=True)
        .mean()
    )
    df_smoothed['valence_pred'] = (
        df_smoothed['valence_pred']
        .rolling(window=smoothFactor, min_periods=1, center=True)
        .mean()
    )
    
    return df_smoothed


def amplify_prediction(df_results, factor=1.0):
    """
    Amplifies the predicted arousal and valence values by multiplying them by a given factor.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame with columns 'arousal_pred' and 'valence_pred'.
    factor : float
        Multiplicative factor to increase the magnitude of the predictions.
        
    Returns
    -------
    df_amplified : pd.DataFrame
        A copy of the input DataFrame with the specified columns amplified.
    """
    df_amplified = df_results.copy()
    df_amplified['arousal_pred'] *= factor
    df_amplified['valence_pred'] *= factor
    return df_amplified


if __name__ == "__main__":
    # --- Testing the processing pipeline with sample data ---
    # Create a sample DataFrame simulating predictions over time.
    sample_data = [
        {"time_s": 15.0, "arousal_pred": 0.0, "valence_pred": 0.0},
        {"time_s": 15.5, "arousal_pred": -0.05, "valence_pred": -0.05},
        {"time_s": 16.0, "arousal_pred": -0.1, "valence_pred": -0.1},
        {"time_s": 16.5, "arousal_pred": -0.1, "valence_pred": -0.1},
        {"time_s": 17.0, "arousal_pred": -0.1, "valence_pred": -0.1},
        {"time_s": 17.5, "arousal_pred": -0.15, "valence_pred": -0.15},
        {"time_s": 18.0, "arousal_pred": -0.15, "valence_pred": -0.15},
        {"time_s": 18.5, "arousal_pred": -0.15, "valence_pred": -0.15},
        {"time_s": 19.0, "arousal_pred": -0.15, "valence_pred": -0.15},
        {"time_s": 19.5, "arousal_pred": -0.15, "valence_pred": -0.15}
    ]
    df_sample = pd.DataFrame(sample_data)
    print("Sample DataFrame (valence/arousal predictions):")
    print(df_sample)

    # Test the mapping function to assign emotion labels.
    df_emotion = map_song_va_to_emotion(df_sample)
    print("\nMapped emotions:")
    print(df_emotion)

    # Test the playback sequence preparation function.
    df_prepped = prep_playback_sequence(df_emotion, minDuration=1.0)
    print("\nFinal playback sequence:")
    print(df_prepped)

    # Test conversion of the playback sequence into JSON format.
    json_result = jsonify_playback_sequence(df_prepped, song_id="TestSong")
    print("\nJSON string:")
    print(json_result)
