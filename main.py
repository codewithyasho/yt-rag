'''
Youtube Transcript Chatbot using RAG (Retrieval Augmented Generation)
youtube summarizer + question answering bot
'''

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()


# Fetch YouTube ID, get transcript and convert to single string
def fetch_youtube_transcript(video_url: str) -> str:
    """Fetch and return YouTube transcript as a single string."""
    try:
        # Extract video ID properly
        import re
        video_id = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        video_id = video_id.group(1)
        # fetching the transcript
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'hi', 'mr'])

        # flatten the transcript into a single string
        transcript_text = " ".join([t.text for t in fetched_transcript])

        return transcript_text
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {str(e)}")


# 2. Splitting the Text into Chunks
def split_text(text):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    ).create_documents([text])

    return chunks


# 3. Creating Embeddings and Storing in Vector Database
def embedd_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


if __name__ == "__main__":
    url = "https://youtu.be/YSHSHJKNi_U?si=wOmr9A3LZdF1HEsO"
    text = fetch_youtube_transcript(url)
    chunks = split_text(text)
    vectorstore = embedd_and_store(chunks)


    # 4. Creating retriever and Chat Model
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

    prompt = ChatPromptTemplate.from_template(
    """
    You're a helpful assistant. Based on the following context, answer the following question only from provided context. if answer is not found in context, simply say 'answer not found'.
     
    Context:
    {context}

    Question: {input}

    Answer:
    """
    )


    ## BUILDING FINAL RAG CHAIN WITH RETRIEVER & PROMPT

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(retriever, document_chain)


    query = "वीडियो में प्रशिक्षक का नाम क्या है?"


    response = rag_chain.invoke(
    input={"input": query}
    )

    print("\nRAG Response:")
    print(response["answer"])

