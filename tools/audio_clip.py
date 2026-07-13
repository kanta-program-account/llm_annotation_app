import wave
import json
import os

def create_audio_clips(audio_file_path, json_file_path):
    
    clips_path = '/home/kanta/llm_annotation_app/data/audio/audio_temp'
    os.makedirs(clips_path, exist_ok=True)

    with open(json_file_path, "r", encoding="utf-8") as f:
        input_transcript = json.load(f)

    start_time_sec = input_transcript[0]['start']
    end_times_list = [item["end"] for item in input_transcript]

    clip_audio_list = []

    with wave.open(audio_file_path, 'rb') as wf:
        params = wf.getparams()
        framerate = wf.getframerate()
        start_frame = int(start_time_sec * framerate)

        for i, end_time_sec in enumerate(end_times_list):
            wf.setpos(start_frame)

            end_frame = int(end_time_sec * framerate)
            num_frames = end_frame - start_frame

            clip_audio = wf.readframes(num_frames)

            clip_name = f'clip_{i}.wav'
            output_path = os.path.join(clips_path, clip_name)
            with wave.open(output_path, 'wb') as wf_out:
                wf_out.setparams(params)
                wf_out.writeframes(clip_audio)



