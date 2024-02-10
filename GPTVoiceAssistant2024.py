"""
VoiceGPT Interaction Script

Author: Dan Woodward
Version: 0.1
Last updated: 10th February 2024

Description:
This script enables an interactive conversation with GPT-4 using voice commands. It performs the following steps:
1) Listens and records audio input from the user.
2) Transcribes the recorded voice input into text using OpenAI's Whisper model.
3) Sends the transcribed text as input to the GPT-4 API for processing.
4) Receives a response from GPT-4 and converts this text into speech, playing it back to the user.

Requirements:
- OpenAI API key for accessing Whisper and GPT-4 models
- PyAudio for recording audio input
- NumPy for audio signal processing
- Wave for handling WAV file operations
- Pygame for playing the text-to-speech audio

Usage:
Ensure you have the required libraries installed and an OpenAI API key set up.
Run the script, speak into your microphone, and engage in a conversation with GPT-4.
Say 'quit' or 'exit' to end the interaction.

"""

import numpy
import pyaudio
import collections
import wave
from openai import OpenAI
import os
# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import contextlib
import sys
import time
import pygame

# Initialize the OpenAI client in your Control Panel > Edit the system environment variables with OPENAI_API_KEY 
# Else set this to client = OpenAI(api_key="YOUR_KEY_HERE") 
client = OpenAI()

# ANSI color codes
COLORS = {
    "blue": "\033[94m",
    "bright pink": "\033[95m",
    "end": "\033[0m",
}

def print_colored(text, color):
    if color in COLORS:
        sys.stdout.write(COLORS[color] + text + COLORS["end"] + '\n')
    else:
        sys.stdout.write(text + '\n')

def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level

threshold = 500  # Adjust based on your environment or microphone gain. 
threshold_end = 300
min_record_time = 2  # Minimum record time in seconds so it doesn't send off after 0.1 seconds of noise 

def record_voice(filename='output.wav'):
    print("\nListening... Speak into the microphone.")
    audio = pyaudio.PyAudio()
    stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    frames = []
    voice_activity_detected = False
    last_voice_time = time.time()

    while True:
        data = stream.read(512, exception_on_overflow=False)
        pegel, _, _ = get_levels(data, 0, 0)

        if pegel > threshold:
            voice_activity_detected = True
            frames.append(data)
            last_voice_time = time.time()
        elif voice_activity_detected and (time.time() - last_voice_time) > min_record_time:
            if (time.time() - last_voice_time) >= 2:
                break

        if not voice_activity_detected:
            frames.append(data)  # Keep appending data to capture the start of speech

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(file_path):
    absolute_file_path = os.path.abspath(file_path)
    with open(absolute_file_path, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
    # Assuming the response might be directly a string or a dictionary containing 'text'
    if isinstance(transcript_response, dict) and 'text' in transcript_response:
        transcribed_text = transcript_response['text'].strip()
    elif isinstance(transcript_response, str):
        transcribed_text = transcript_response.strip()
    else:
        print("Unexpected response format.")
        transcribed_text = ""
    
    return transcribed_text

def chat_with_openai(transcribed_text):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. English speaking only. Succinct and direct. As you are giving spoken answers with an audio output, you need only give around two sentence answers as a time where appropriate."},
            {"role": "user", "content": transcribed_text},
        ],
        stream=False,
    )
    return response.choices[0].message.content

def text_to_speech(text, filename='response.mp3'):
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except Exception as e:
            print(f"Error removing file: {e}")
    response = client.audio.speech.create(model="tts-1", voice="fable", input=text)
    with open(filename, "wb") as f:
        f.write(response.content)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

if __name__ == "__main__":
    pygame.mixer.init()
    while True:
        record_filename = os.path.join(os.getcwd(), 'output.wav')
        record_voice(record_filename)
        transcribed_text = transcribe_audio(record_filename)
        if transcribed_text:
            if 'quit' in transcribed_text.lower() or 'exit' in transcribed_text.lower():
                print_colored("Exiting the program.", "blue")
                sys.exit()
            else:
                print_colored(f"\nYou: {transcribed_text}", "blue")
                response_text = chat_with_openai(transcribed_text)
                print_colored(f"ChatGPT: {response_text}", "bright pink")
                response_filename = os.path.join(os.getcwd(), 'response.mp3')
                text_to_speech(response_text, response_filename)
        else:
            print_colored("\nNo spoken words detected, ready for next recording.", "blue")


