import soundfile as sf
import json
import os
import glob

def create_audio_clips(audio_file_path, json_file_path):
    """Split a WAV audio file into multiple sequential clips based on transcript timestamps.

    This function reads a transcript JSON to extract start and end timestamps for each 
    utterance, extracts the corresponding audio frames from the source WAV file, and 
    saves each segment as a separate audio clip in a temporary directory.

    Args:
        audio_file_path (str): The absolute or relative path to the source WAV audio file.
        json_file_path (str): The path to the JSON transcript file containing utterance timestamps.

    Returns:
        tuple[str, list[str]]: A tuple containing:
            - str: The path to the temporary directory where the split clips are stored.
            - list[str]: A list of file paths to the generated WAV audio clips.
    """
    
    clip_temp_dir_path = '/home/kanta/llm_annotation_app/data/audio/audio_temp'
    os.makedirs(clip_temp_dir_path, exist_ok=True)

    with open(json_file_path, "r", encoding="utf-8") as f:
        input_transcript = json.load(f)["utterances"]

    start_time_sec = input_transcript[0]['start']
    end_times_list = [item["end"] for item in input_transcript]

    data, framerate = sf.read(audio_file_path)
    start_frame = int(start_time_sec * framerate)

    for i, end_time_sec in enumerate(end_times_list):
        end_frame = int(end_time_sec * framerate)
        
        clip_data = data[start_frame:end_frame]

        clip_name = f'clip_{i}.wav'
        output_path = os.path.join(clip_temp_dir_path, clip_name)
        
        sf.write(output_path, clip_data, framerate, subtype='PCM_16')

    clip_paths = sorted(glob.glob(os.path.join(clip_temp_dir_path, "*")))
        
    return clip_temp_dir_path, clip_paths