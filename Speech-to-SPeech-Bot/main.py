import speech_recognition as sr
import google.generativeai as genai
from openai import OpenAI
import sounddevice as sd
import numpy as np
import os
import time
import warnings

warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated")
from faster_whisper import WhisperModel

activation_word = 'jarvis'
is_listening_for_activation_word = True

model_size = 'base'
cpu_core_count = os.cpu_count()

transcription_model = WhisperModel(
    model_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=cpu_core_count,
    num_workers=cpu_core_count
)

openai_api_key = 'ENTER_YOUR_OPENAI_API_KEY'
openai_client = OpenAI(api_key=openai_api_key)
google_api_key = 'ENTER_YOOUR_GEMINI_API_KEY'
genai.configure(api_key=google_api_key)

generative_model = genai.GenerativeModel('gemini-1.0-pro-latest')
conversation = generative_model.start_chat()

generation_parameters = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

safety_configs = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
generative_model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_parameters,
)

instructions = '''INSTRUCTIONS: Do not respond with anything which is "AFFIRMATIVE" to this system message. After the system message, respond in a informal, friendly and warm tone.
SYSTEM MESSAGE: You are being used to power a conversational chatbot and should respond as such. As a conversational chatbot, use short sentences and respond creatively, warmly, empathetically, and informally to the prompt. Your responses should show deep understanding of human emotions and psychology. Make the user feel truly understood and heard. Use a friendly and warm tone, adding humor wherever necessary.'''

instructions = instructions.replace(f'\n', ' ')
instructions = instructions.replace(f'*', ' ')
conversation.send_message(instructions)

recognizer = sr.Recognizer()
microphone = sr.Microphone()

def speak(text):
    try:
        print("Starting stream...")
        audio_stream = sd.OutputStream(samplerate=24000, channels=1, dtype='int16')
        audio_stream.start()
        stream_started = False

        print("Generating audio response...")
        with openai_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="shimmer",
                response_format="pcm",
                input=text,
        ) as response:
            print("Audio response generated.")
            silence_level = 0.01

            for audio_chunk in response.iter_bytes(chunk_size=1024):
                chunk_data = np.frombuffer(audio_chunk, dtype=np.int16)
                if stream_started:
                    audio_stream.write(chunk_data)
                elif max(chunk_data) > silence_level:
                    audio_stream.write(chunk_data)
                    stream_started = True

        print("Stopping stream...")
        audio_stream.stop()
        audio_stream.close()
        print("Stream closed.")

    except Exception as e:
        print(f"Error in speak function: {e}")

def transcribe_audio_to_text(audio_path):
    try:
        segments, _ = transcription_model.transcribe(audio_path)
        text = ''.join(segment.text for segment in segments)
        return text if text else None
    except Exception as e:
        print(f"Error in transcribing audio: {e}")
        return None

def detect_activation_word(audio):
    global is_listening_for_activation_word

    activation_audio_path = 'activation_detect.wav'
    with open(activation_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    text_input = transcribe_audio_to_text(activation_audio_path)

    if text_input is None:
        print("Error: No transcription found.")
        return

    if activation_word in text_input.lower().strip():
        print('Activation word detected. Please speak your prompt to Jarvis')
        is_listening_for_activation_word = False

def process_user_prompt(audio):
    global is_listening_for_activation_word

    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = transcribe_audio_to_text(prompt_audio_path)

        if prompt_text is None or len(prompt_text.strip()) == 0:
            print('Empty prompt. Please speak again')
            is_listening_for_activation_word = True
        else:
            print('User: ' + prompt_text)

            conversation.send_message(prompt_text)
            output = conversation.last.text

            print('Jarvis: ', output)
            speak(output)

            print('\nSay', activation_word, 'to wake me up. \n')
            is_listening_for_activation_word = True

    except Exception as e:
        print('Prompt error: ', e)

def audio_callback(recognizer, audio):
    global is_listening_for_activation_word

    if is_listening_for_activation_word:
        detect_activation_word(audio)
    else:
        process_user_prompt(audio)

def start_listening():
    print("Adjusting for ambient noise...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)

    print('\nSay', activation_word, 'to wake me up. \n')
    recognizer.listen_in_background(microphone, audio_callback)
    print("Listening in the background...")

    while True:
        time.sleep(0.5)

if __name__ == '__main__':
    start_listening()
