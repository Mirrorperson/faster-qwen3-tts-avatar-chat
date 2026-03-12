import os
import json
import requests
import numpy as np
import gradio as gr
import re
import io
import grpc
import scipy.io.wavfile
import sys
# Add faster_liveportrait so our stubs can be found
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "faster_liveportrait"))
import avatar_pb2
import avatar_pb2_grpc
from faster_qwen3_tts import FasterQwen3TTS

# --- CONFIG ---
OLLAMA_URL = os.getenv("OLLAMA_Service_URL", "http://host.docker.internal:11434/api/chat")
OLLAMA_MODEL = "qwen3.5:9b"
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Ryan" 

print(f"Loading 0.6B CustomVoice Model...")
tts = FasterQwen3TTS.from_pretrained(TTS_MODEL_ID)
print("TTS Ready.")

# --- gRPC CONFIG ---
# host.docker.internal allows this docker container to route out to the host Windows 
# machine, which then routes it back into the FasterLivePortrait docker container that 
# has port 50051 mapped to it.
FLP_SERVER_ADDRESS = 'host.docker.internal:50051'

def trim_to_sentence_limit(text, limit=5):
    # This regex finds sentence endings (. ! or ?) followed by a space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > limit:
        return " ".join(sentences[:limit])
    return text

def chat_and_speak(user_text, history, src_image_path):
    if not user_text: return history, None
    history = history or []
    
    # 1. THE DEEP SCRUB (Only send role/content to Ollama)
    clean_messages = [{"role": "system", "content": "You are a concise voice assistant. Your responses MUST be 2 to 3 sentences long. Never exceed 5 sentences. Be direct and conversational."}]
    
    for msg in history:
        # Check if it's a dictionary (New Gradio)
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            clean_messages.append({"role": msg["role"], "content": str(msg["content"])})
        # Check if it's a tuple (Old Gradio)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            clean_messages.append({"role": "user", "content": str(msg[0])})
            clean_messages.append({"role": "assistant", "content": str(msg[1])})

    clean_messages.append({"role": "user", "content": user_text})

    # PRINT RAW JSON FOR DEBUGGING
    print("--- OLLAMA REQUEST START ---")
    print(json.dumps(clean_messages, indent=2))
    print("--- OLLAMA REQUEST END ---")

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, 
            "messages": clean_messages, 
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.6,
                "repeat_penalty": 1.2,
                "num_predict": 150
            }
        })
        
        if r.status_code != 200:
            print(f"Ollama Error Status: {r.status_code}")
            print(f"Ollama Error Body: {r.text}")
            r.raise_for_status()

        reply = r.json()["message"]["content"].strip()
        
        reply = trim_to_sentence_limit(reply, limit=5)
    except Exception as e:
        print(f"Exception during Ollama call: {e}")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, None

    # --- TTS GENERATION ---
    print(f"Synthesizing: {reply}")
    try:
        audio_list, sr = tts.generate_custom_voice(text=reply + " ", language="English", speaker=DEFAULT_SPEAKER)
        audio_data = np.concatenate(audio_list)
        audio_int16 = (audio_data * 32767).astype(np.int16) # int16 is much more stable
    except Exception as e:
        audio_int16 = np.zeros(100, dtype=np.int16)
        sr = 24000
        print(f"TTS Error: {e}")

    # --- AVATAR VIDEO GENERATION (gRPC) ---
    print("Sending audio and image to FasterLivePortrait Server...")
    video_out_path = None
    if src_image_path:
        try:
            # 1. Read the source image bytes
            with open(src_image_path, "rb") as f_img:
                img_bytes = f_img.read()
                
            # 2. Convert raw np audio array to standard WAV bytes in memory
            wav_io = io.BytesIO()
            scipy.io.wavfile.write(wav_io, sr, audio_int16)
            wav_bytes = wav_io.getvalue()
            
            # 3. Connect via gRPC with unlimited message size (videos can be large)
            GRPC_OPTIONS = [
                ('grpc.max_send_message_length', -1),    # unlimited
                ('grpc.max_receive_message_length', -1), # unlimited
            ]
            channel = grpc.insecure_channel(FLP_SERVER_ADDRESS, options=GRPC_OPTIONS)
            stub = avatar_pb2_grpc.AvatarServiceStub(channel)
            
            request = avatar_pb2.AvatarRequest(
                source_image=img_bytes,
                driving_audio=wav_bytes
            )
            
            response = stub.AnimateAvatar(request, timeout=300)  # 5 min timeout
            channel.close()
            
            # 4. Save returned MP4 to a temp file for Gradio
            if response.video_output:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_vid:
                    f_vid.write(response.video_output)
                    video_out_path = f_vid.name
                print(f"Video saved to: {video_out_path}")
            else:
                print("Received empty video output from server.")
                
        except grpc.RpcError as rpc_error:
            print(f"gRPC Error: {rpc_error}")
        except Exception as e:
            print(f"Avatar Generation Error: {e}")
    else:
        print("No source image provided. Skipping video generation.")

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    
    # Return video file path if successful, else fallback to None (or audio could be added as fallback)
    return history, video_out_path

with gr.Blocks(title="Qwen3 CustomVoice Avatar Chat") as demo:
    gr.Markdown(f"# 🗣️ Qwen3 Voice Chat + FasterLivePortrait")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_out = gr.Video(label="AI Avatar Video", autoplay=True)
            src_image = gr.Image(type="filepath", label="Avatar Source Image (Face)", sources=["upload"])
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Type your message here...")
    
    msg.submit(chat_and_speak, [msg, chatbot, src_image], [chatbot, video_out])
    msg.submit(lambda: "", None, msg)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)