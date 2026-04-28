# 📄 Chat with PDF — RAG Project

Ask questions about any PDF using a local AI. **Free, private, and accessible from any device.**
Your PDF data never leaves your machine.

---

## 🚀 Quick Start (One Command)

### Mac / Linux
```bash
# First time only — make the script executable
chmod +x launch.sh

# Every time — starts everything automatically
./launch.sh
```

### Windows
```
Double-click launch.bat
```

That's it. The script will:
1. ✅ Check Ollama, Streamlit, and Ngrok are installed
2. ✅ Start Ollama (and pull the model if needed)
3. ✅ Start Streamlit
4. ✅ Start Ngrok and print your **public URL**

---

## 📋 One-Time Setup

### 1. Install Ollama
Download from **https://ollama.com** and install it.

### 2. Install Ngrok
Download from **https://ngrok.com/download**, install it, then:
- Sign up for a free account at ngrok.com
- Run this once to link your account:
  ```bash
  ngrok config add-authtoken YOUR_TOKEN_HERE
  ```
  (Find your token at: https://dashboard.ngrok.com/get-started/your-authtoken)

### 3. Install Python packages
```bash
pip install -r requirements.txt
```

---

## 🌐 Accessing Your App

After running the launch script, you'll see:

```
✅ Everything is running!

   🌐 Your public URL:
   https://abc123.ngrok-free.app

   Share this link to access your app from
   any device. Only you know this URL.
   Your PDF data never leaves this machine.

   Local URL:       http://localhost:8501
   Ngrok dashboard: http://localhost:4040
```

- **From your computer** → use the Local URL
- **From your phone or another laptop** → use the Ngrok URL
- **Ctrl+C** → cleanly shuts down Streamlit and Ngrok

> ⚠️ The free Ngrok URL changes every time you restart. Upgrade to Ngrok's free static domain (one free static domain per account) to get a permanent URL.

---

## 🔒 Privacy & Data Flow

```
Your Machine Only
┌────────────────────────────────────────┐
│                                        │
│  PDF → Chunks → Embeddings → ChromaDB  │
│                    ↓                   │
│              Ollama LLM                │
│                    ↓                   │
│             Streamlit App              │
│                    ↓                   │
└──────────────── Ngrok Tunnel ──────────┘
                      ↓
              Your Phone/Laptop
              (encrypted HTTPS)
```

Ngrok acts as an **encrypted tunnel** — it routes traffic to your machine but cannot read the content (HTTPS encrypted end-to-end). Your PDF text and embeddings are stored only in RAM on your machine.

---

## 📁 Project Structure

```
chat-with-pdf/
├── launch.sh           # 🚀 Mac/Linux one-command launcher
├── launch.bat          # 🚀 Windows one-command launcher
├── app.py              # Streamlit UI
├── pdf_processor.py    # Extracts + chunks PDF text
├── rag_engine.py       # RAG logic: embed, retrieve, answer
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ollama not found` | Install from https://ollama.com |
| `ngrok not found` | Install from https://ngrok.com/download |
| `ngrok: authentication failed` | Run `ngrok config add-authtoken YOUR_TOKEN` |
| URL not showing | Open http://localhost:4040 manually |
| Slow first run | Embedding model is downloading (~90MB), one-time only |
| Bad answers | Reduce chunk size or increase top-K in the sidebar |

---

## 💡 Want a Permanent URL?

Ngrok gives you **one free static domain**:
1. Go to https://dashboard.ngrok.com/domains
2. Claim your free static domain (e.g. `yourname.ngrok-free.app`)
3. Update `launch.sh` line to:
   ```bash
   ngrok http $STREAMLIT_PORT --domain=yourname.ngrok-free.app > /tmp/ngrok.log 2>&1 &
   ```

Now your URL never changes! ✨