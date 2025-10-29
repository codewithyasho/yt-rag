'''
YouTube Video Summarizer and Q&A Chatbot
Powered by LangChain & Groq
'''

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import re
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer and Q&A",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI - works in both light and dark mode
st.markdown("""
    <style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        border-radius: 4px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    
    /* Message boxes - auto adapt to theme */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #ff4b4b;
    }
    .bot-message {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
    }
    
    /* Force white text in dark mode for message boxes */
    [data-testid="stAppViewContainer"][data-theme="dark"] .chat-message,
    [data-testid="stAppViewContainer"][data-theme="dark"] .chat-message * {
        color: #ffffff !important;
    }
    
    /* Force black text in light mode for message boxes */
    [data-testid="stAppViewContainer"][data-theme="light"] .chat-message,
    [data-testid="stAppViewContainer"]:not([data-theme="dark"]) .chat-message {
        color: #000000 !important;
    }
    
    [data-testid="stAppViewContainer"][data-theme="light"] .chat-message *,
    [data-testid="stAppViewContainer"]:not([data-theme="dark"]) .chat-message * {
        color: #000000 !important;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 2px solid rgba(100, 150, 200, 0.3);
        background-color: rgba(100, 150, 200, 0.05);
    }
    
    .info-box h3 {
        margin-top: 0;
        color: #ff4b4b;
    }
    
    /* Ensure text is always visible */
    .stMarkdown, .stText {
        color: inherit;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = None
if 'transcript_with_timestamps' not in st.session_state:
    st.session_state.transcript_with_timestamps = None
if 'video_url' not in st.session_state:
    st.session_state.video_url = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_transcript(video_url: str):
    """Fetch YouTube transcript with timestamps."""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            raise ValueError(
                "Invalid YouTube URL. Please check the URL and try again.")

        # Fetch transcript with timestamps using the correct API method
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=['en', 'hi', 'mr'])

        # Create two versions: one with timestamps, one without
        transcript_text = " ".join([entry.text for entry in transcript_list])

        # Format with timestamps for reference
        transcript_with_timestamps = []
        for entry in transcript_list:
            timestamp = format_timestamp(entry.start)
            transcript_with_timestamps.append({
                'timestamp': timestamp,
                'time_seconds': entry.start,
                'text': entry.text
            })

        return transcript_text, transcript_with_timestamps, video_id

    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {str(e)}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def split_text(text: str):
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.create_documents([text])
    return chunks


def create_vectorstore(chunks):
    """Create FAISS vectorstore from text chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def generate_summary(transcript_text: str, summary_type: str = "quick"):
    """Generate video summary using LLM."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        if summary_type == "quick":
            prompt = f"""
            You are a helpful assistant that creates concise video summaries.
            
            Provide a brief, quick summary of the following video transcript in 3-5 bullet points.
            Focus on the main topics and key takeaways.
            
            Transcript:
            {transcript_text[:4000]}
            
            Quick Summary:
            """
        else:  # detailed
            prompt = f"""
            You are a helpful assistant that creates comprehensive video summaries.
            
            Provide a detailed summary of the following video transcript with:
            1. Main Topic/Title
            2. Key Points (in bullet points)
            3. Important Details
            4. Conclusion/Takeaways
            
            Transcript:
            {transcript_text[:8000]}
            
            Detailed Summary:
            """

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error generating summary: {str(e)}"


def get_conversational_chain(vectorstore, chat_history):
    """Create conversational retrieval chain with chat history context."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Create a simple prompt template for Q&A with context
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        
        # Build conversation history for context
        history_text = ""
        if chat_history:
            history_text = "\n\nPrevious conversation:\n"
            for item in chat_history[-3:]:  # Last 3 exchanges for context
                history_text += f"Q: {item['question']}\nA: {item['answer']}\n"
        
        prompt = ChatPromptTemplate.from_template(
            f"""
            You are a helpful assistant answering questions about a YouTube video.
            Use the following context from the video transcript to answer the question.
            If you cannot find the answer in the context, say "I cannot find that information in the video."
            {history_text}
            
            Context from video:
            {{context}}
            
            Question: {{input}}
            
            Answer:
            """
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)

        return qa_chain

    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None


def find_relevant_timestamps(question: str, answer: str, transcript_data):
    """Find relevant timestamps based on the answer content."""
    try:
        # Extract key phrases from answer
        answer_words = set(answer.lower().split())

        relevant_segments = []
        for entry in transcript_data:
            text_words = set(entry['text'].lower().split())
            # Calculate overlap
            overlap = len(answer_words.intersection(text_words))
            if overlap > 2:  # If there's significant overlap
                relevant_segments.append(entry)

        # Return top 3 most relevant timestamps
        return relevant_segments[:3]

    except Exception as e:
        return []


# ==================== MAIN APP ====================

st.title("🎥 YouTube Video Summarizer and Q&A")
st.markdown("**Powered by LangChain & Groq**")

# Sidebar for video URL input
with st.sidebar:
    st.header("📹 Upload Video")
    video_url_input = st.text_input(
        "Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    process_button = st.button(
        "🚀 Process Video", type="primary", use_container_width=True)

    if process_button and video_url_input:
        with st.spinner("⏳ Fetching transcript..."):
            try:
                # Fetch transcript
                transcript_text, transcript_timestamps, video_id = fetch_youtube_transcript(
                    video_url_input)

                # Split and embed
                chunks = split_text(transcript_text)
                vectorstore = create_vectorstore(chunks)

                # Store in session state
                st.session_state.transcript_text = transcript_text
                st.session_state.transcript_with_timestamps = transcript_timestamps
                st.session_state.vectorstore = vectorstore
                st.session_state.video_url = video_url_input
                st.session_state.video_id = video_id
                st.session_state.chat_history = []

                st.success("✅ Video processed successfully!")
                st.markdown(f"**Video chunks:** {len(chunks)}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Show video info if loaded
    if st.session_state.video_url:
        st.markdown("---")
        st.markdown("### 📊 Current Video")
        st.video(st.session_state.video_url)

        if st.button("🗑️ Clear Video", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.transcript_text = None
            st.session_state.transcript_with_timestamps = None
            st.session_state.video_url = None
            st.session_state.chat_history = []
            st.rerun()


# Main content area with tabs
if st.session_state.vectorstore is None:
    st.markdown("""
        <div class='info-box'>
            <h3>👋 Welcome!</h3>
            <p>Please upload a YouTube video URL from the sidebar to get started.</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>🎯 Generate quick or detailed video summaries</li>
                <li>💬 Ask questions about the video with conversation memory</li>
                <li>⏱️ Get timestamps for relevant video segments</li>
                <li>🔄 Follow-up questions with context</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
else:
    # Create tabs
    tab1, tab2 = st.tabs(["📄 Summarize Video", "💬 Ask Questions"])

    # ==================== SUMMARY TAB ====================
    with tab1:
        st.header("Generate Video Summary")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**Select summary type:**")

        summary_type = st.radio(
            "Summary Type",
            ["Quick Summary (Faster)", "Detailed Summary"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            with st.spinner("🤔 Generating summary..."):
                summary_mode = "quick" if "Quick" in summary_type else "detailed"
                summary = generate_summary(
                    st.session_state.transcript_text, summary_mode)

                st.markdown("### 📝 Summary")
                st.markdown(f"""
                    <div class='chat-message bot-message'>
                        {summary}
                    </div>
                """, unsafe_allow_html=True)

    # ==================== Q&A TAB ====================
    with tab2:
        st.header("Ask Questions About Your Video")

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Conversation History")
            for i, chat in enumerate(st.session_state.chat_history):
                # User question
                st.markdown(f"""
                    <div class='chat-message user-message'>
                        <strong>🙋 You:</strong><br>{chat['question']}
                    </div>
                """, unsafe_allow_html=True)

                # Bot answer
                st.markdown(f"""
                    <div class='chat-message bot-message'>
                        <strong>🤖 Assistant:</strong><br>{chat['answer']}
                    </div>
                """, unsafe_allow_html=True)

                # Show timestamps if available
                if chat.get('timestamps'):
                    with st.expander("⏱️ Relevant Timestamps"):
                        for ts in chat['timestamps']:
                            timestamp_link = f"https://www.youtube.com/watch?v={st.session_state.video_id}&t={int(ts['time_seconds'])}s"
                            st.markdown(
                                f"**[{ts['timestamp']}]({timestamp_link})** - {ts['text']}")

        # Question input
        st.markdown("---")
        st.markdown("### 🔍 Enter your question:")

        user_question = st.text_input(
            "Question",
            placeholder="What is this video about?",
            label_visibility="collapsed",
            key="question_input"
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button(
                "🚀 Get Answer", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if ask_button and user_question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get conversational chain with chat history
                    qa_chain = get_conversational_chain(
                        st.session_state.vectorstore,
                        st.session_state.chat_history
                    )

                    if qa_chain:
                        # Get answer using invoke instead of __call__
                        response = qa_chain.invoke({"input": user_question})
                        answer = response['answer']

                        # Find relevant timestamps
                        timestamps = find_relevant_timestamps(
                            user_question,
                            answer,
                            st.session_state.transcript_with_timestamps
                        )

                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': answer,
                            'timestamps': timestamps
                        })

                        # Rerun to display new message
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with ❤️ using LangChain, Streamlit, and Google Gemini</p>
    </div>
""", unsafe_allow_html=True)
