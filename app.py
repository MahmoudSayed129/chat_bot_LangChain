import os
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from langdetect import detect
from deep_translator import GoogleTranslator
from pinecone import Pinecone
from openai import OpenAI
import tempfile

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---------- Config ----------
DATASET_PATH = "data/coaching_millionaer_dataset.json"
load_dotenv(override=True)

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ebook"

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})

# ---------- Global Clients / Chains ----------
client = None          # raw OpenAI client (Whisper, TTS, etc.)
retriever = None       # LangChain retriever
vectorstore = None     # LangChain Pinecone vectorstore
rag_chain = None       # LangChain RAG chain (book-based)
fallback_chain = None  # LangChain fallback chain (no book context)

# ---------- LLM Client Setup (raw OpenAI client) ----------
try:
    if not OPENAI_API_KEY:
        raise ValueError("⚠️ Missing OPENAI_API_KEY in environment variables")
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ Using OpenAI API for all tasks (Whisper, GPT, TTS)")
except Exception as e:
    print(f"❌ Failed to initialize LLM client: {e}")
    client = None

# ---------- LangChain: Embeddings + Pinecone + RAG ----------
try:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY missing in environment variables")

    # Pinecone client + index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Embeddings for BOTH:
    # - building the Pinecone index (you do this in a separate ingestion script)
    # - querying via LangChain
    #
    # Make sure your ingestion uses the SAME model: "text-embedding-3-small".
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    # Wrap existing Pinecone index in a LangChain vectorstore
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

    # LangChain retriever (semantic search)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    # LangChain chat LLM for RAG + fallback
    chat_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )

    # --------- Prompts (reuse your persona text) ---------

    def system_prompt_book_only() -> str:
        return (
            "Du bist **Javid Niazi-Hoffmann**, Gründer von J&P Mentoring. "
            "Sprich immer auf **Deutsch**, egal in welcher Sprache der Nutzer schreibt. "
            "Antworte natürlich, empathisch und selbstbewusst – so, als würdest du den Nutzer persönlich coachen. "
            "Nutze den bereitgestellten Kontext nur als Hintergrundwissen, "
            "aber erwähne niemals, woher die Informationen stammen. "
            "Beziehe dich nicht auf Bücher, Kapitel oder Seiten. "
            "Gib deine Ratschläge direkt in deiner eigenen Stimme – klar, inspirierend und menschlich. "
            "Sei authentisch und unterstützend, als würdest du dich wirklich um das Wachstum des Nutzers kümmern. "
        )

    def system_prompt_fallback() -> str:
        return (
            "Du bist **Javid Niazi-Hoffmann**, erfolgreicher Unternehmer und Mentor bei J&P Mentoring. "
            "Antworte immer auf **Deutsch**, unabhängig von der Sprache der Nutzeranfrage. "
            "Sprich direkt und natürlich, als würdest du in einem echten Mentoring-Gespräch mit dem Nutzer sprechen. "
            "Vermeide es, wie ein Assistent zu klingen oder externe Quellen zu erwähnen. "
            "Dein Ton ist praktisch, empathisch und selbstbewusst – motivierend, aber bodenständig. "
            "Bleibe menschlich und authentisch in deiner Ausdrucksweise."
        )

    def system_prompt_youtube_script() -> str:
        return (
            "Du bist **Javid Niazi-Hoffmann**, erfolgreicher Unternehmer und Mentor bei J&P Mentoring. "
            "Du erstellst **starke YouTube-Video-Skripte auf Deutsch**. "
            "Sprich immer auf **Deutsch**, sei klar, inspirierend und bodenständig. "
            "Schreibe so, dass der Text direkt vom Teleprompter abgelesen werden kann – "
            "mit natürlicher Sprache, kurzen Sätzen und klaren Übergängen. "
            "Nutze Du-Ansprache, sei motivierend und ergebnisorientiert."
            "Do not return Headlines like [Hook],[CTA] ...etc"
        )

    # Expose these prompts globally as well (for routes outside this block)
    globals()["system_prompt_book_only"] = system_prompt_book_only
    globals()["system_prompt_fallback"] = system_prompt_fallback
    globals()["system_prompt_youtube_script"] = system_prompt_youtube_script

    # ---- RAG Prompt: book persona + retrieved context ----
    book_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt_book_only()
                + "\n\nNutze den folgenden Kontext nur, wenn er hilfreich ist:\n\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    # Chain that stuffs retrieved docs into the prompt
    qa_chain = create_stuff_documents_chain(
        llm=chat_llm,
        prompt=book_prompt,
    )

    # Full retrieval chain: retriever → qa_chain
    rag_chain = create_retrieval_chain(
        retriever,
        qa_chain,
    )

    # Fallback chain without retrieval (no book context)
    fallback_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_fallback()),
            ("human", "{input}"),
        ]
    )
    fallback_chain = fallback_prompt | chat_llm

    print("✅ LangChain RAG chain initialized successfully.")

except Exception as e:
    print("❌ LangChain / Pinecone initialization failed:", e)
    traceback.print_exc()
    # define dummy prompt functions if above failed, so routes don't crash
    def system_prompt_book_only() -> str:
        return "Du bist ein hilfreicher deutscher Assistent."

    def system_prompt_fallback() -> str:
        return "Du bist ein hilfreicher deutscher Assistent."

    def system_prompt_youtube_script() -> str:
        return (
            "Du erstellst starke YouTube-Video-Skripte auf Deutsch. "
            "Do not return Headlines like [Hook],[CTA] ...etc"
        )

    globals()["system_prompt_book_only"] = system_prompt_book_only
    globals()["system_prompt_fallback"] = system_prompt_fallback
    globals()["system_prompt_youtube_script"] = system_prompt_youtube_script


# ---------- Translator ----------
def translate_text(text: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text


# ---------- Helpers ----------
def detect_language(question: str) -> str:
    try:
        return detect(question)
    except Exception:
        return "unknown"


def normalize_language(lang: str, text: str) -> str:
    if lang == "nl" and any(
        word in text.lower() for word in ["wer", "was", "wie", "javid", "coaching"]
    ):
        return "de"
    return lang


def format_answers(question: str, answer: str, results):
    pages = [f"Seite {r.get('page', '')}" for r in results if r.get("page")]
    source = ", ".join(pages) if pages else "No source"
    top_score = max([r.get("score", 0.0) for r in results], default=0.0)
    return {
        "answers": [
            {
                "question": question,
                "answer": answer,
                "source": source,
                "bm25_score": top_score,
            }
        ]
    }


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "running",
            "retriever_ready": bool(retriever),
            "hf_key_loaded": bool(HF_TOKEN),
            "pinecone_key_loaded": bool(PINECONE_API_KEY),
            "index_name": PINECONE_INDEX_NAME,
        }
    )


@app.route("/youtube-script", methods=["POST", "OPTIONS"])
def youtube_script():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return ("", 204)

    if client is None:
        return jsonify({"error": "⚠️ No language model initialized."}), 500

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    # Expecting these keys from the frontend
    topic = (data.get("topic") or "").strip()
    duration_minutes = (data.get("duration_minutes") or "").strip()
    tone = (data.get("tone") or "").strip()
    target_audience = (data.get("target_aience") or data.get("target_audience") or "").strip()
    userName = (data.get("userName") or "").strip()

    if not topic:
        return jsonify({"error": "Video topic is required."}), 400

    # Fallback defaults
    if not userName:
        userName = ""
    if not duration_minutes:
        duration_minutes = "10"
    if not tone:
        tone = "inspirierend, klar, authentisch"
    if not target_audience:
        target_audience = "Menschen, die finanziell und persönlich wachsen wollen"

    # Build user prompt for GPT
    user_prompt = f"""
    Erstelle ein ausführliches YouTube-Video-Skript auf Deutsch.
    
    Thema: {topic}
    Ziel-Videolänge: ca. {duration_minutes} Minuten
    Tonfall: {tone}
    Zielgruppe: {target_audience}
    Speaker: {userName}
    
    Struktur des Skripts:
    1. Starker Hook in den ersten 5–10 Sekunden (sofortige Aufmerksamkeit, großes Versprechen).
    3. Klar strukturierter Hauptteil mit mehreren Abschnitten:
       - Erkläre das Thema verständlich.
       - Nutze Beispiele, Metaphern oder kurze Stories.
       - Gib konkrete Tipps oder Schritte.
    4. Übergänge zwischen den Abschnitten, damit das Skript natürlich fließt.
    5. Starker Call-to-Action am Ende
       (z.B. Kanal abonnieren, Kommentar schreiben, kostenloses Erstgespräch, Link in der Beschreibung).
    
    Format:
    - Schreibe den Text als gesprochenes Skript in der Du-Form.
    - Kein Fließtext-Roman, sondern gut lesbare Absätze.
    - Do not return Headlines like [Hook],[CTA] ...etc
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_youtube_script()},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=10000,
        )
        script_text = response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"⚠️ LLM call failed: {e}"}), 500

    # Response for the frontend
    return jsonify(
        {
            "topic": topic,
            "duration_minutes": duration_minutes,
            "tone": tone,
            "target_audience": target_audience,
            "script": script_text,
        }
    ), 200


@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        data = request.get_json(force=True) or {}
        question = (data.get("question") or "").strip()
    except Exception:
        return jsonify(format_answers("", "Invalid JSON request", [])), 200

    if not question:
        return jsonify(format_answers("", "Please enter a question.", [])), 200

    print(f"\n--- User Question ---\n{question}")

    user_lang = normalize_language(detect_language(question), question)
    print(f"Detected language: {user_lang}")

    if rag_chain is None:
        return jsonify(
            format_answers(question, "⚠️ RAG chain not initialized.", [])
        ), 200

    try:
        # Run LangChain RAG chain
        rag_result = rag_chain.invoke({"input": question})
        answer = rag_result["answer"]
        docs = rag_result.get("context", []) or []

        # If no docs came back, fall back to non-book persona chain
        if not docs and fallback_chain is not None:
            fb = fallback_chain.invoke({"input": question})
            answer = getattr(fb, "content", str(fb))
            docs = []

    except Exception as e:
        traceback.print_exc()
        return jsonify(
            format_answers(question, f"⚠️ RAG chain failed: {e}", [])
        ), 200

    # Convert LangChain Documents into the structure expected by format_answers
    results = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        results.append(
            {
                "context": getattr(d, "page_content", ""),
                "page": meta.get("page"),
                "score": 0.0,  # we don't have explicit scores from the chain
            }
        )

    return jsonify(format_answers(question, answer, results))


# ---------- Voice Chat ----------
@app.route("/voice", methods=["POST"])
def voice_chat():
    try:
        audio = request.files.get("audio")
        if not audio:
            return jsonify({"error": "No audio file uploaded"}), 400

        if client is None:
            return jsonify({"error": "No language model initialized"}), 500

        # Save temporary audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            audio.save(tmp.name)
            audio_path = tmp.name

        # Step 1️⃣: Transcribe user speech to text using Whisper
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(audio_path, "rb"),
        )
        text = transcription.text.strip()
        print(f"🎤 Transcribed: {text}")

        if not text:
            return jsonify({"error": "Transcription failed or empty"}), 400

        # Step 2️⃣: Use the same LangChain RAG chain as /ask
        if rag_chain is None:
            return jsonify({"error": "RAG chain not initialized"}), 500

        try:
            rag_result = rag_chain.invoke({"input": text})
            answer_text = rag_result["answer"]
            docs = rag_result.get("context", []) or []

            # If RAG didn't return docs, fall back
            if not docs and fallback_chain is not None:
                fb = fallback_chain.invoke({"input": text})
                answer_text = getattr(fb, "content", str(fb))
                docs = []
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"RAG chain failed: {e}"}), 500

        # Extract page sources for the response
        source_pages = [
            d.metadata.get("page") for d in docs if getattr(d, "metadata", None)
        ]

        # Step 3️⃣: Generate voice reply with GPT TTS
        try:
            speech_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=answer_text,
            ) as speech:
                speech.stream_to_file(speech_file.name)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"TTS failed: {e}"}), 500

        # Step 4️⃣: Return transcript + answer + audio
        return jsonify(
            {
                "transcript": text,
                "answer": answer_text,
                "audio_url": f"https://mahmous-chatbot3.hf.space/audio/{os.path.basename(speech_file.name)}",
                "source": source_pages,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_file(
        os.path.join(tempfile.gettempdir(), filename), mimetype="audio/mpeg"
    )


# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Server started on port {port}")
    app.run(host="0.0.0.0", port=port)
