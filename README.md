---
title: faster-qwen3-tts-avatar
emoji: 🎙🧑
tags: [text-to-speech, avatar, live-portrait, cuda-graphs]
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Faster Qwen3-TTS + Live Avatar Chat

A real-time voice chat demo combining **Qwen3-TTS** for speech synthesis with **FasterLivePortrait** for audio-driven avatar animation.

---
### 🌟 Credits & Foundations
This project is built upon the amazing work of:
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) by [Andi Marafioti](https://github.com/andimarafioti)
- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) by [warmshao](https://github.com/warmshao)
- [Ollama](https://ollama.com)
---

### 🎥 Demo
https://github.com/user-attachments/assets/67b532ce-ed97-46b4-87e4-8cb0675f58da
---

```
User message → Qwen3 LLM (Ollama) → Qwen3-TTS → WAV audio
                                                       ↓
                                         FasterLivePortrait (gRPC)
                                                       ↓
                                           Lip-synced avatar video
```

## 🚀 Beginner's Quickstart Guide

This guide ensures everything works by running the services in three separate terminals. This approach is the most reliable for Windows/GPU environments.

### Prerequisites

1.  **NVIDIA GPU**: You need a modern NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2.  **Ollama**: Install locally from [ollama.com](https://ollama.com).
3.  **Docker Desktop**: Ensure it is running.

---

### Step 1: Start the AI Brain (Terminal 1)
Open a terminal and run:
```powershell
# 1. Download the brain
ollama pull qwen3.5:9b

# 2. Start it
set OLLAMA_HOST=0.0.0.0
ollama serve
```

---

### Step 2: Start the Avatar Engine (Terminal 2)
Open another terminal, navigate to your repo, and run:
```powershell
# 1. Build the engine image (This handles all the pip installs automatically)
docker build -t my-avatar-engine -f Dockerfile.avatar .

# 2. Start it instantly
docker rm -f faster_liveportrait 2>nul & docker run -it --rm --gpus all --name faster_liveportrait -v %cd%\faster_liveportrait:/root/FasterLivePortrait -p 50051:50051 my-avatar-engine
```
*Wait for: `FasterLivePortrait gRPC Server is listening on port 50051...`*

---

### Step 3: Start the Chat UI (Terminal 3)
Open one last terminal and run:
```powershell
# 1. Build the app image
docker build -t faster-qwen3-tts-demo .

# 2. Clear any old app container and launch
docker rm -f tts_app 2>nul & docker run -it --rm --name tts_app --gpus all -p 7860:7860 -v %cd%\app.py:/app/app.py -v %cd%\faster_liveportrait:/app/faster_liveportrait -v %cd%\hf_cache:/hf_cache -e HF_HOME=/hf_cache -e OLLAMA_Service_URL=http://host.docker.internal:11434/api/chat --dns 8.8.8.8 --add-host host.docker.internal:host-gateway faster-qwen3-tts-demo bash -c "pip install grpcio grpcio-tools scipy && python3 /app/app.py"
```

---

### Step 4: Start Chatting!
1.  Open Chrome/Edge to: **http://localhost:7860**
2.  Upload a clear face photo on the left.
3.  Type a message and watch your AI avatar come to life!

---

## Architecture Overview

- **`app.py`**: The main brain that combines TTS, the UI, and the gRPC connection.
- **`flp_grpc_server.py`**: The specialized animation server.
- **`host.docker.internal`**: The bridge connecting your containers to Ollama and each other.
