# 🎥 YouTube Video Summarizer and Q&A Chatbot

A powerful AI-powered Streamlit application that transforms YouTube videos into interactive conversations. Get instant summaries and ask questions about any YouTube video content using advanced RAG (Retrieval Augmented Generation) technology.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 📄 Video Summarization

- **Quick Summary**: Generate concise 3-5 bullet point summaries in seconds
- **Detailed Summary**: Get comprehensive analysis with main topics, key points, and takeaways
- **Smart Processing**: Automatically handles transcripts up to 8000 characters

### 💬 Interactive Q&A

- **Natural Conversations**: Ask questions about video content in plain language
- **Conversation Memory**: Context-aware responses that remember previous exchanges
- **Follow-up Questions**: Ask related questions without repeating context
- **Source Verification**: AI answers only from video content, no hallucinations

### ⏱️ Timestamp Integration

- **Relevant Timestamps**: Get clickable timestamps for specific video segments
- **Direct Navigation**: Click timestamps to jump to exact moments in the video
- **Context Mapping**: See exactly where information appears in the video

### 🌍 Multi-Language Support

- English transcripts
- Hindi (हिंदी) transcripts
- Marathi (मराठी) transcripts
- Automatic language detection

### 🎨 Modern UI

- **Dual Theme Support**: Optimized for both light and dark modes
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Intuitive Interface**: Clean, user-friendly design with tab navigation
- **Real-time Updates**: Instant feedback with loading indicators

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))
- Internet connection for YouTube access

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd project8-yt-chatbot
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**

   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows (Command Prompt):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```

### Running the Application

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Upload a YouTube Video

1. Open the app in your browser
2. In the sidebar, paste a YouTube URL (supports multiple formats):
   - `https://www.youtube.com/watch?v=VIDEO_ID`
   - `https://youtu.be/VIDEO_ID`
   - `https://youtube.com/embed/VIDEO_ID`
3. Click **"🚀 Process Video"**
4. Wait for the transcript to be fetched and processed (usually 10-30 seconds)

### Step 2: Generate Summaries

1. Navigate to the **"📄 Summarize Video"** tab
2. Choose your summary type:
   - **Quick Summary**: Fast, concise overview (3-5 bullet points)
   - **Detailed Summary**: Comprehensive analysis with structure
3. Click **"✨ Generate Summary"**
4. View your AI-generated summary instantly

### Step 3: Ask Questions

1. Switch to the **"💬 Ask Questions"** tab
2. Type your question in the input field
3. Click **"🚀 Get Answer"**
4. View the answer along with relevant timestamps
5. Ask follow-up questions - the AI remembers context!

### Example Questions

```
- What is this video about?
- Explain the main concept in simple terms
- What did the speaker say about [topic]?
- Summarize the section about [specific topic]
- Can you give me more details on that?
- What are the key takeaways?
```

## 🏗️ Architecture

### Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **LLM**: Google Gemini 2.5 Flash (via LangChain)
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Transcript API**: YouTube Transcript API
- **Framework**: LangChain (RAG implementation)

### How It Works

```
┌─────────────────┐
│  YouTube Video  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fetch Transcript│ (YouTube Transcript API)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Split Text     │ (RecursiveCharacterTextSplitter)
└────────┬────────┘    Chunk Size: 1000, Overlap: 200
         │
         ▼
┌─────────────────┐
│ Create Embeddings│ (HuggingFace Embeddings)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Store in FAISS  │ (Vector Database)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  User Query → RAG Pipeline       │
│  1. Retrieve relevant chunks     │
│  2. Add conversation context     │
│  3. Generate answer with Gemini  │
└─────────────────────────────────┘
```

### Key Components

#### 1. Transcript Processing

```python
fetch_youtube_transcript(url)
├── Extract video ID
├── Fetch transcript with timestamps
├── Format timestamps (MM:SS or HH:MM:SS)
└── Return text + timestamp data
```

#### 2. Text Chunking

```python
split_text(text)
├── Chunk size: 1000 characters
├── Overlap: 200 characters
└── Separators: ["\n\n", "\n", ".", " "]
```

#### 3. Vector Storage

```python
create_vectorstore(chunks)
├── Model: all-MiniLM-L6-v2
├── Dimension: 384
└── Index: FAISS
```

#### 4. Conversational Chain

```python
get_conversational_chain(vectorstore, chat_history)
├── Retriever: Top 4 similar chunks
├── Context: Last 3 Q&A exchanges
├── LLM: Gemini 2.5 Flash
└── Output: Answer + source documents
```

## 📁 Project Structure

```
project8-yt-chatbot/
│
├── app.py                  # Main Streamlit application
├── main.py                 # Standalone backend script
├── main.ipynb             # Jupyter notebook for testing
│
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── .env                   # Environment variables (not in repo)
├── .env.example           # Example environment file
│
├── README.md              # This file
├── README_APP.md          # Additional app documentation
│
└── .venv/                 # Virtual environment (not in repo)
```

## ⚙️ Configuration

### Environment Variables

| Variable         | Description           | Required | Default |
| ---------------- | --------------------- | -------- | ------- |
| `GOOGLE_API_KEY` | Google Gemini API key | Yes      | -       |

### Customizable Parameters

In `app.py`, you can modify:

```python
# Text Chunking
chunk_size = 1000          # Size of each text chunk
chunk_overlap = 200        # Overlap between chunks

# LLM Configuration
model = "gemini-2.5-flash" # Gemini model version
temperature = 0.3          # Creativity (0.0-1.0)

# Retrieval
search_type = "similarity" # Search algorithm
k = 4                      # Number of chunks to retrieve

# Conversation Memory
history_context = 3        # Number of previous exchanges to include

# Summary Limits
quick_summary_chars = 4000 # Characters for quick summary
detailed_summary_chars = 8000 # Characters for detailed summary
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "Failed to fetch transcript" Error

**Possible causes:**

- Video doesn't have captions/subtitles
- Captions not available in supported languages
- Invalid YouTube URL
- Video is private or restricted

**Solutions:**

- Verify the video has captions (check on YouTube)
- Try videos with auto-generated captions
- Ensure URL is correct and complete
- Use public videos only

#### 2. "Module not found" Error

**Solution:**

```bash
pip install -r requirements.txt
```

#### 3. API Key Errors

**Solutions:**

- Verify `.env` file exists in project root
- Check API key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
- Ensure no extra spaces in API key
- Restart the app after adding API key

#### 4. Slow Processing

**Causes & Solutions:**

- Large videos: Normal, wait for completion
- First run: Downloading embedding model (one-time)
- Multiple requests: API rate limiting, wait a moment

#### 5. Memory Issues

**Solution:**

```bash
# Clear session and reload
# Click "Clear Video" button in sidebar
```

## 🚧 Limitations

- Requires videos with available transcripts/captions
- Transcript accuracy depends on YouTube's auto-generation quality
- Limited to supported languages (English, Hindi, Marathi)
- API rate limits apply based on Google Gemini quota
- Large videos may take time to process
- Conversation context limited to last 3 exchanges

## 📊 Performance

- **Processing Time**: 10-30 seconds for average video
- **Memory Usage**: ~500MB for typical video
- **Chunk Processing**: 1000 chars per chunk with 200 overlap
- **Retrieval Speed**: ~1-2 seconds per query
- **Response Time**: 2-5 seconds for answer generation

---

**Built with ❤️ using LangChain, Streamlit, and Google Gemini**

_Last Updated: October 30, 2025_


