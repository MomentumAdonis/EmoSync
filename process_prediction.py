import math
import pandas as pd
import numpy as np
import json

def map_va_to_emotion(valence, arousal, level1=0.2, level2=0.6):
    """
    Maps a given valence and arousal value to an emotion label based on a circular segmentation.

    Parameters
    ----------
    valence : float
        The valence value (typically in the range -1.0 to +1.0).
    arousal : float
        The arousal value (typically in the range -1.0 to +1.0).
    level1 : float, optional
        The threshold below which any value is considered "Neutral/Indifferent". Default is 0.2.
    level2 : float, optional
        The threshold that separates level 2 and level 3. For radius between level1 and level2, use 8 bins; 
        for radius >= level2, use 12 bins. Default is 0.6.

    Returns
    -------
    emotion_label : str
        The emotion label corresponding to the (valence, arousal) pair.
    """
    # Calculate the polar coordinates
    radius = math.sqrt(valence**2 + arousal**2)
    angle = math.degrees(math.atan2(arousal, valence))
    if angle < 0:
        angle += 360  # Normalize angle to [0, 360)

    # Level 1: Very close to center means neutral
    if radius < level1:
        return "Neutral/Indifferent"

    # Define labels for level 2 (8 groups, each 45°)
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
    # Define labels for level 3 (12 groups, each 30°)
    level3_labels = [
        "Happy/Elated",
        "Delighted",
        "Excited",
        "Tense/Afraid",
        "Angry/Annoyed",
        "Frustrated/Bitter/Upset",
        "Depressed/Miserable",
        "Bored",
        "Tired/Sleepy",
        "Peaceful",
        "Relaxed",
        "Content/Pleased"
    ]

    if radius < level2:
        bin_size = 360 / 8  # 45 degrees per bin
        bin_index = int(angle // bin_size)
        return level2_labels[bin_index]
    else:
        bin_size = 360 / 12  # 30 degrees per bin
        bin_index = int(angle // bin_size)
        return level3_labels[bin_index]
    


def map_song_va_to_emotion(va_df, level1=0.2, level2=0.6):
    """
    Maps valence and arousal predictions for each time step in a song to an emotion label.

    Parameters
    ----------
    va_df : pd.DataFrame
        A DataFrame with at least three columns:
            - 'time_s': float, time in seconds
            - 'arousal_pred': float, predicted arousal value
            - 'valence_pred': float, predicted valence value

    Returns
    -------
    pd.DataFrame
        A new DataFrame with two columns: 'time_s' and 'emotion',
        where 'emotion' is the string returned by map_va_to_emotion() for each row.
    """
    # List to hold the mapping results
    emotion_list = []
    
    # Iterate over each row in the dataframe
    for index, row in va_df.iterrows():
        time_value = row["time_s"]
        arousal_value = row["arousal_pred"]
        valence_value = row["valence_pred"]
        # Use the previously defined function to map to an emotion
        emotion = map_va_to_emotion(valence_value, arousal_value, level1=level1, level2=level2)
        emotion_list.append({"time_s": time_value, "emotion": emotion})
    
    # Create a new DataFrame from the results
    df_emotion = pd.DataFrame(emotion_list)
    return df_emotion



def compress_playback_sequence(prepped_df):
    """
    Removes consecutive duplicates in the 'emotion' column from the playback sequence.
    
    The input DataFrame is assumed to have columns: ['start_time', 'emotion'] in ascending time order.
    Returns a new DataFrame with the same columns, but consecutive duplicates are removed.
    """
    if prepped_df.empty:
        return prepped_df.copy()

    compressed_rows = []
    last_emotion = None

    for idx, row in prepped_df.iterrows():
        current_emotion = row['emotion']
        if current_emotion != last_emotion:
            # Keep this row if the emotion differs from the previous row
            compressed_rows.append((row['start_time'], current_emotion))
            last_emotion = current_emotion

    compressed_df = pd.DataFrame(compressed_rows, columns=['start_time','emotion'])
    return compressed_df



def prep_playback_sequence(emotion_df, minDuration=3.0):
    """
    Converts a row-by-row timeline (time_s, emotion) into a minimal sequence of changes:
      - Only store (start_time, emotion) for each change.
      - If multiple small segments (< minDuration) occur in succession, 
        merge them and pick the majority emotion for that entire merged chunk.
      - At the end, add one final row for 'Playback Completed' at the final time.

    Parameters
    ----------
    emotion_df : pd.DataFrame
        Must have columns:
          - 'time_s' : float, time in ascending or any order
          - 'emotion': str, the emotion at that time
    minDuration : float
        Minimum duration for a valid segment. 
        Segments below this threshold are merged with subsequent segments, 
        and the final merged chunk's emotion is whichever has the largest time share in that chunk.

    Returns
    -------
    final_df : pd.DataFrame
        Columns: ['start_time', 'emotion']
        - The last row is something like (final_time, 'Playback Completed').

    Notes
    -----
    1) The “dominant emotion” is determined by how many seconds each emotion occupies
       within that short chunk. We assume the chunk is row-based with consecutive times. 
    2) If the entire final chunk is still < minDuration, we still finalize it, 
       picking the majority anyway.
    """

    # 0) Sort input by time_s
    df_sorted = emotion_df.sort_values(by='time_s').reset_index(drop=True)
    if df_sorted.empty:
        # If no data, return an empty DataFrame
        return pd.DataFrame(columns=['start_time','emotion'])

    # 1) Build "raw segments" of consecutive identical emotions
    #    Each segment: (start_time, end_time, emotion)
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
            # End the previous segment at the previous row_time
            raw_segments.append((segment_start, prev_time, current_emotion))
            # Start a new segment
            segment_start = row_time
            current_emotion = row_emo

        prev_time = row_time
        prev_emotion = row_emo

    # Add the final raw segment
    last_time = df_sorted.loc[n_rows-1, 'time_s']
    raw_segments.append((segment_start, last_time, current_emotion))

    # Convert to DataFrame with durations
    raw_df = pd.DataFrame(raw_segments, columns=['start_time','end_time','emotion'])
    raw_df['duration'] = raw_df['end_time'] - raw_df['start_time']

    # 2) Merge small segments (< minDuration) in succession and pick majority emotion
    merged_segments = []
    i = 0
    n_segs = len(raw_df)

    while i < n_segs:
        seg_start = raw_df.loc[i, 'start_time']
        seg_end = raw_df.loc[i, 'end_time']
        # We'll keep a small buffer of (time ranges, emotion)
        # to measure the total time each emotion occupies in this chunk.
        chunk_emotions = [(raw_df.loc[i, 'start_time'], raw_df.loc[i, 'end_time'], raw_df.loc[i, 'emotion'])]
        chunk_duration = raw_df.loc[i, 'duration']
        i_next = i + 1

        # While total chunk duration < minDuration and we haven't exhausted segments
        while chunk_duration < minDuration and i_next < n_segs:
            # Add the next segment
            chunk_emotions.append((
                raw_df.loc[i_next, 'start_time'],
                raw_df.loc[i_next, 'end_time'],
                raw_df.loc[i_next, 'emotion']
            ))
            seg_end = raw_df.loc[i_next, 'end_time']
            chunk_duration += raw_df.loc[i_next, 'duration']
            i_next += 1

        # Now we have a chunk that is >= minDuration, or we reached the last segment.
        # Determine the "dominant emotion" in this chunk by total time for each emotion.
        # We'll store times in a dictionary {emotion: total_time}
        emo_time_dict = {}
        for (st, en, emo) in chunk_emotions:
            dur = en - st
            emo_time_dict[emo] = emo_time_dict.get(emo, 0.0) + dur

        # Pick the emotion with the largest time share
        dominant_emo = max(emo_time_dict, key=emo_time_dict.get)

        # The final chunk goes from seg_start to seg_end, with emotion=dominant_emo
        merged_segments.append((seg_start, seg_end, dominant_emo))

        # Move index to i_next
        i = i_next

    # 3) Convert merged_segments to a final DataFrame but we only want start_time & emotion.
    #    Then add a final row at the end with 'Playback Completed'.
    final_rows = []
    for idx, (start_t, end_t, emo) in enumerate(merged_segments):
        final_rows.append((start_t, emo))
    # Add final row for "Playback Completed"
    final_end = merged_segments[-1][1]  # end_t of last segment
    final_rows.append((final_end, "Playback Completed"))

    final_df = pd.DataFrame(final_rows, columns=['start_time','emotion'])

    final_df = compress_playback_sequence(final_df)

    return final_df



def jsonify_playback_sequence(prepped_df, song_id=None):
    """
    Converts a playback sequence DataFrame into a JSON string.

    Parameters
    ----------
    prepped_df : pd.DataFrame
        Expected columns: ['start_time', 'emotion'] (in ascending time order).
    song_id : int or str, optional
        If provided, it will be included in the JSON structure.

    Returns
    -------
    json_str : str
        A JSON string with the structure:
        {
          "song_id": <song_id or null>,
          "sequence": [
             {"start_time": 15.0, "emotion": "Neutral/Indifferent"},
             {"start_time": 28.5, "emotion": "Impatient"},
             ...
          ]
        }
    """

    # Build a list of dicts for each row
    sequence_list = []
    for _, row in prepped_df.iterrows():
        sequence_list.append({
            "start_time": float(row["start_time"]),  # ensure it's a float
            "emotion": row["emotion"]
        })

    # Build the final JSON structure
    output = {
        "song_id": song_id,
        "sequence": sequence_list
    }

    # Convert to JSON string
    json_str = json.dumps(output)
    return json_str



def smoothen_prediction(df_results, smoothFactor=5):
    """
    Smoothens the arousal_pred and valence_pred columns in df_results
    using a simple moving-average approach.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        Must contain at least:
            - 'time_s' (float): the timestamps
            - 'arousal_pred' (float): predicted arousal values
            - 'valence_pred' (float): predicted valence values
    smoothFactor : int
        The window size for the moving average. A larger number produces 
        more smoothing. Typically 3-9 is reasonable for mild smoothing.
        
    Returns
    -------
    df_smoothed : pd.DataFrame
        A copy of df_results, sorted by 'time_s' with 'arousal_pred' and 
        'valence_pred' replaced by their smoothed values. All other columns 
        are preserved as-is.
    """

    # 1) Make a copy and sort by time
    df_smoothed = df_results.copy()
    df_smoothed.sort_values(by='time_s', inplace=True)
    
    # 2) Apply rolling average to the predicted columns
    #    Use center=True to align the window around each point
    #    min_periods=1 ensures we get a value even at the edges
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

    # 3) Return to original row order if needed
    #    (If you want to preserve the original ordering, re-sort or re-index here)
    #    For example:
    # original_index = df_results.index
    # df_smoothed = df_smoothed.reindex(original_index)
    
    # 4) Return the smoothed DataFrame
    return df_smoothed



def amplify_prediction(df_results, factor=1.0):
    """
    Amplifies the arousal_pred and valence_pred columns in df_results
    by a given factor.

    Parameters
    ----------
    df_results : pd.DataFrame
        Must contain at least:
            - 'arousal_pred' (float): predicted arousal values
            - 'valence_pred' (float): predicted valence values
    factor : float
        The multiplication factor to apply. For example, 1.5 would
        increase the magnitude by 50%.

    Returns
    -------
    df_amplified : pd.DataFrame
        A copy of df_results with 'arousal_pred' and 'valence_pred'
        multiplied by the factor.
    """
    # Make a copy to avoid modifying the original DataFrame
    df_amplified = df_results.copy()

    # Multiply the prediction columns by the factor
    df_amplified['arousal_pred'] *= factor
    df_amplified['valence_pred'] *= factor

    return df_amplified



if __name__ == "__main__":
    # --- Test the functions ---
    # Create a sample DataFrame for a song's valence/arousal predictions over time
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

    # Test mapping function
    df_emotion = map_song_va_to_emotion(df_sample)
    print("\nMapped emotions:")
    print(df_emotion)

    # Test preparation of final playback sequence
    df_prepped = prep_playback_sequence(df_emotion, minDuration=1.0)
    print("\nFinal playback sequence:")
    print(df_prepped)

    # Test JSON conversion
    json_result = jsonify_playback_sequence(df_prepped, song_id="TestSong")
    print("\nJSON string:")
    print(json_result)

    

