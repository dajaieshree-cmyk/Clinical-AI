
import streamlit as st
import re
import csv
import os
import pandas as pd
from collections import Counter
from datetime import datetime
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Clinical NLP Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEEDBACK_FILE = "feedback.csv"

# Minimum similarity score for a sentence to be returned as evidence.
MIN_SCORE = 0.08

# ---------------- CUSTOM UI DESIGN ----------------

st.markdown("""
<style>
/* Main page */
.stApp {
    background-color: #F4F7FB;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #123047;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}

/* Subtitle */
.subtitle {
    color: #607487;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

/* Section headings */
h2, h3 {
    color: #173B52 !important;
}

/* Buttons */
.stButton > button {
    background-color: #087F8C;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #066873;
    color: white;
    border: none;
}

/* Input fields */
.stTextInput input,
.stTextArea textarea {
    border-radius: 9px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px dashed #9ABBC4;
    border-radius: 12px;
    padding: 1rem;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Horizontal dividers */
hr {
    border-color: #DCE5ED;
}

/* Mobile-friendly title */
@media (max-width: 640px) {
    .main-title {
        font-size: 1.9rem;
    }

    .block-container {
        padding-top: 1rem;
    }
}

/* Sidebar design */
[data-testid="stSidebar"] {
    background-color: #123047;
}

[data-testid="stSidebar"] * {
    color: #F4F7FB;
}

[data-testid="stSidebar"] hr {
    border-color: #456174;
}

/* Sidebar branding */
.sidebar-brand {
    font-size: 1.4rem;
    font-weight: 800;
    color: #65D6C5;
    padding: 0.5rem 0;
}

.sidebar-description {
    font-size: 0.9rem;
    color: #D0DFE9;
    line-height: 1.6;
}

/* ---------------- HERO HEADER ---------------- */
.hero {
    background: linear-gradient(120deg, #0B1B2B 0%, #123047 55%, #0B4A50 100%);
    border-radius: 20px;
    padding: 1.7rem 1.9rem;
    margin-bottom: 1rem;
    box-shadow: 0 20px 45px -22px rgba(11, 27, 43, 0.65);
    border: 1px solid rgba(101, 214, 197, 0.16);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -70px;
    top: -90px;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(101, 214, 197, 0.22) 0%,
        rgba(101, 214, 197, 0) 70%
    );
}
.hero-badge {
    display: inline-block;
    background: rgba(101, 214, 197, 0.14);
    color: #65D6C5;
    border: 1px solid rgba(101, 214, 197, 0.3);
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero .main-title {
    color: #F4F7FB;
    font-size: 2.3rem;
    margin-bottom: 0.25rem;
}
.hero .subtitle {
    color: #B9CDDB;
    margin-bottom: 0;
    max-width: 780px;
}
.hero-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}
.hero-pill {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #CFE0EC;
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Section accent */
.block-container h2,
.block-container h3 {
    border-left: 4px solid #087F8C;
    padding-left: 0.65rem;
    letter-spacing: -0.2px;
}

/* ---------------- EVALUATION CONSOLE ---------------- */
.eval-console {
    background:
        radial-gradient(circle at 88% 12%, rgba(101, 214, 197, 0.16) 0%, rgba(101, 214, 197, 0) 42%),
        linear-gradient(135deg, #0B1B2B 0%, #123047 48%, #0A3A44 100%);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin: 0.7rem 0 1.3rem 0;
    box-shadow: 0 22px 50px -24px rgba(11, 27, 43, 0.6);
    border: 1px solid rgba(101, 214, 197, 0.18);
}
.eval-top {
    display: flex;
    flex-wrap: wrap;
    gap: 1.6rem;
    align-items: center;
    justify-content: space-between;
}
.eval-top-text { flex: 1 1 320px; }
.eval-title {
    color: #F4F7FB;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.3px;
}
.eval-sub {
    color: #9FB6C7;
    font-size: 0.9rem;
    margin-top: 0.3rem;
    line-height: 1.5;
}
.eval-tags { margin-top: 0.8rem; }
.eval-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #CFE0EC;
    border-radius: 999px;
    padding: 0.26rem 0.7rem;
    font-size: 0.74rem;
    font-weight: 700;
    margin-right: 0.4rem;
    margin-bottom: 0.35rem;
}
.eval-kpis {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.35rem;
}
.eval-kpi {
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.035));
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 14px;
    padding: 0.85rem 1rem 0.9rem 1rem;
    position: relative;
    overflow: hidden;
}
.eval-kpi::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--accent, #65D6C5);
}
.eval-kpi-label {
    color: #9FB6C7;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 800;
}
.eval-kpi-value {
    color: #FFFFFF;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.2;
    margin-top: 0.2rem;
}
.eval-kpi-note {
    color: #8FA6B7;
    font-size: 0.76rem;
    margin-top: 0.1rem;
}
.eval-progress {
    height: 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.10);
    margin-top: 1.2rem;
    overflow: hidden;
}
.eval-progress > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #34D399, #65D6C5);
}
.eval-ring {
    width: 128px;
    height: 128px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: conic-gradient(
        var(--ring-color, #34D399) calc(var(--pct, 0) * 1%),
        rgba(255, 255, 255, 0.10) 0
    );
    flex: 0 0 auto;
}
.eval-ring-inner {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: #0E2437;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.eval-ring-value {
    color: #FFFFFF;
    font-size: 1.45rem;
    font-weight: 800;
    line-height: 1;
}
.eval-ring-label {
    color: #8FA6B7;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-top: 0.28rem;
}
.eval-cases {
    display: grid;
    gap: 0.6rem;
    margin-top: 1.25rem;
}
.eval-case {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-left: 4px solid var(--status, #34D399);
    border-radius: 12px;
    padding: 0.75rem 0.95rem;
}
.eval-case-body { flex: 1 1 auto; min-width: 0; }
.eval-case-q {
    color: #EAF2F8;
    font-weight: 700;
    font-size: 0.95rem;
}
.eval-case-line {
    color: #A9C0D0;
    font-size: 0.84rem;
    margin-top: 0.3rem;
}
.eval-case-ev {
    color: #D6E4EE;
    font-size: 0.86rem;
    margin-top: 0.25rem;
    padding-left: 0.6rem;
    border-left: 2px solid rgba(255, 255, 255, 0.15);
}
.eval-case-scores {
    color: #8FA6B7;
    font-size: 0.78rem;
    margin-top: 0.4rem;
    font-weight: 600;
}
.eval-badge {
    flex: 0 0 auto;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.9px;
    padding: 0.26rem 0.6rem;
    border-radius: 999px;
    margin-top: 0.1rem;
}
.eval-badge.pass {
    background: rgba(52, 211, 153, 0.16);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.45);
}
.eval-badge.check {
    background: rgba(251, 191, 36, 0.16);
    color: #FBBF24;
    border: 1px solid rgba(251, 191, 36, 0.45);
}
.eval-note {
    color: #8FA6B7;
    font-size: 0.78rem;
    margin-top: 1rem;
    line-height: 1.5;
}

/* ---------------- RESULT TABLE ---------------- */
.result-card {
    background: #FFFFFF;
    border: 1px solid #DCE5ED;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: 0 10px 26px -20px rgba(11, 27, 43, 0.5);
    margin-bottom: 0.5rem;
}
.result-heading {
    font-weight: 800;
    color: #123047;
    font-size: 1rem;
    margin-bottom: 0.2rem;
}
.result-sub {
    color: #607487;
    font-size: 0.82rem;
    margin-bottom: 0.6rem;
}
.results-table {
    font-size: 0.86rem;
    border: 1px solid #E3EAF1;
    border-radius: 12px;
    overflow-x: auto;
    background: #FFFFFF;
}
.rt-head, .rt-row {
    display: grid;
    grid-template-columns: 1.25fr 1.25fr 1.25fr 84px 66px 66px 100px;
    gap: 0.55rem;
    align-items: center;
    padding: 0.6rem 0.85rem;
    min-width: 720px;
}
.rt-head {
    background: #F1F6FA;
    color: #33556B;
    font-weight: 800;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}
.rt-row { border-top: 1px solid #EDF2F6; }
.rt-row:hover { background: #F8FBFD; }
.rt-cell { color: #2C3E50; line-height: 1.45; }
.rt-q { font-weight: 700; color: #123047; }
.rt-ev { color: #56707F; }
.rt-num {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    color: #123047;
    text-align: right;
}
.rt-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.7px;
}
.rt-pill.pass { background: rgba(16, 185, 129, 0.15); color: #047857; }
.rt-pill.check { background: rgba(245, 158, 11, 0.20); color: #B45309; }
.rt-f1 { display: flex; align-items: center; gap: 0.45rem; justify-content: flex-end; }
.rt-bar {
    width: 44px;
    height: 6px;
    border-radius: 999px;
    background: #E3EAF1;
    overflow: hidden;
    flex: 0 0 auto;
}
.rt-bar > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #087F8C, #34D399);
}

@media (max-width: 900px) {
    .eval-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero .main-title { font-size: 1.8rem; }
}

</style>
""", unsafe_allow_html=True)

# ---------------- PAGE HEADER ----------------

st.markdown(
    '<div class="hero">'
    '<span class="hero-badge">Semantic search · TF-IDF · Negation aware</span>'
    '<div class="main-title">🩺 Clinical NLP Assistant</div>'
    '<div class="subtitle">'
    'Evidence-grounded clinical document question answering '
    'powered by Natural Language Processing'
    '</div>'
    '<div class="hero-strip">'
    '<span class="hero-pill">📄 TXT / PDF ingestion</span>'
    '<span class="hero-pill">🔎 Sentence retrieval</span>'
    '<span class="hero-pill">📊 Retrieval evaluation</span>'
    '<span class="hero-pill">👍 Human feedback loop</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "Educational prototype only. Not for medical diagnosis "
    "or treatment decisions. Verify all information with "
    "a qualified healthcare professional."
)


# ---------------- DOCUMENT PROCESSING ----------------

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )

    
def split_sentences(text):
    text = re.sub(r"\b(Sample )?Clinical Note\b", "", text, flags=re.IGNORECASE)
    sentences = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        # Protect dots in common abbreviations
        line = re.sub(
            r"\b(Dr|Mr|Mrs|Ms|Pt|vs|approx|b\.i\.d|t\.i\.d|q\.i\.d)\.",
            lambda m: m.group(0).replace(".", "<DOT>"),
            line,
            flags=re.IGNORECASE,
        )

        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.replace("<DOT>", ".").strip()
            if len(part) > 2:
                sentences.append(part)

    return sentences


# ---------------- NLP PREPROCESSING ----------------

STOP_WORDS = {
    "what", "which", "who", "when", "where", "why", "how",
    "is", "are", "was", "were", "the", "a", "an", "of",
    "in", "on", "to", "for", "does", "do", "did", "patient",
    "mentioned", "reported", "tell", "me", "about", "any",
    "and", "or", "please", "describe", "explain", "document",
    "according", "based", "this", "that", "has", "have", "state", "states",
    "had", "he", "she", "his", "her", "they", "their", "it"
}

NEGATION_TERMS = {
    "no", "not", "denies", "denied", "without", "never",
    "negative", "absent", "none", "neither"
}


def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


from nltk.stem import PorterStemmer
_stemmer = PorterStemmer()

SYNONYM_MAP = {
    r"\b(drug |medication |medicine |food )?allerg(y|ies|ic)\b": "allergy",
    r"\bno known allergy\b": "no allergy",
    r"\b(short(ness)? of breath|dyspn(o)?ea|breathless(ness)?|sob)\b": "dyspnea",
    r"\b(fever|febrile|pyrexia|high temperature)\b": "fever",
    r"\b(head ?ache|cephalalgia)\b": "headache",
    r"\b(nausea|nauseous|queasy|feeling sick)\b": "nausea",
    r"\b(vomit(ing)?|emesis|throwing up)\b": "vomiting",
    r"\b(hypertension|high blood pressure|htn)\b": "hypertension",
    r"\b(cardiac|heart)\b": "heart",
    r"\b(drugs?|medications?|medicines?|meds)\b": "medication",
    r"\b(myocardial infarction|heart attack)\b": "mi",
}


def normalize(text):
    text = text.lower()
    for pattern, replacement in SYNONYM_MAP.items():
        text = re.sub(pattern, replacement, text)
    return text


def content_tokens(text):
    return {
        _stemmer.stem(w)
        for w in tokenize(normalize(text))
        if w not in STOP_WORDS
    }


def has_negation(text):
    words = tokenize(text)
    return any(word in NEGATION_TERMS for word in words)


# ---------------- EVIDENCE RETRIEVAL ----------------

NAME_QUESTION = re.compile(r"\b(name|who is the patient|called)\b", re.IGNORECASE)
NAME_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
)


def find_patient_name_sentence(document, question):
    if not NAME_QUESTION.search(question):
        return []
    for sentence in split_sentences(document):
        if NAME_PATTERN.search(sentence):
            return [(sentence, 1.0)]
    return []


def find_relevant_sentences_tfidf(document, question, top_k=3):
    sentences = split_sentences(document)

    if not sentences or not question.strip():
        return []

    # Remove sentences that merely repeat the question
    sentences = [
        sentence for sentence in sentences
        if sentence.lower().strip(" .!?")
        != question.lower().strip(" .!?")
    ]

    if not sentences:
        return []

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            [normalize(s) for s in sentences] + [normalize(question)]
        )

        scores = cosine_similarity(
            matrix[-1], matrix[:-1]
        ).flatten()

    except ValueError:
        scores = [0.0] * len(sentences)

    # Keyword overlap supplements TF-IDF for exact terms
    # Keyword overlap: ignore common words that don't carry meaning
        # Keyword overlap supplements TF-IDF for exact terms
    question_words = content_tokens(question)

    
    for i, sentence in enumerate(sentences):
        sentence_words = content_tokens(sentence)

        if question_words:
            overlap = len(question_words & sentence_words)
            scores[i] += 0.15 * overlap / len(question_words)

        # Give stronger preference to sentences containing
        # the important terms from the question.
            if overlap == 0:
                 scores[i] = 0.0


    # Handle questions about absence or negation
    asks_about_absence = any(
        word in tokenize(question)
        for word in [
            "absent", "denies", "denied",
            "without", "negative"
        ]
    )

    if asks_about_absence:
        question_negated = True

        for i, sentence in enumerate(sentences):
            if has_negation(sentence) != question_negated:
                scores[i] *= 0.5

    # Rank sentences by score
    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:
        score = float(scores[index])
        
            # Keep only sentences sharing important words with the question
        sentence_words = content_tokens(sentences[index])
        meaningful_overlap = question_words & sentence_words
        
        if score >= MIN_SCORE and meaningful_overlap:
            results.append((sentences[index], score))

    return results


EMB_MIN_SCORE = 0.35

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def find_relevant_sentences_embedding(document, question, top_k=3):
    sentences = split_sentences(document)

    if not sentences or not question.strip():
        return []

    sentences = [
        s for s in sentences
        if s.lower().strip(" .!?") != question.lower().strip(" .!?")
    ]

    if not sentences:
        return []

    model = load_embedding_model()
    sentence_vectors = model.encode(sentences, normalize_embeddings=True)
    question_vector = model.encode([question], normalize_embeddings=True)[0]

    scores = sentence_vectors @ question_vector

    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)

    return [
        (sentences[i], float(scores[i]))
        for i in ranked[:top_k]
        if scores[i] >= EMB_MIN_SCORE
    ]


def find_relevant_sentences(document, question, top_k=3):
    name_hit = find_patient_name_sentence(document, question)
    if name_hit:
        return name_hit[:top_k]

    if st.session_state.get("retriever") == "Embeddings (MiniLM)":
        return find_relevant_sentences_embedding(document, question, top_k)
    return find_relevant_sentences_tfidf(document, question, top_k)



YES_NO_STARTERS = {"does", "did", "do", "is", "are", "was", "were",
                   "has", "have", "had", "can"}

def generate_simple_answer(question, evidence):
    if not question or not question.strip():
        return "Please enter a question."
    if not evidence:
        return "No relevant evidence was found in the document."

    sentence, score = evidence[0]
    if score < (EMB_MIN_SCORE if st.session_state.get("retriever") == "Embeddings (MiniLM)" else MIN_SCORE):
        return ("I couldn't find sufficiently relevant evidence "
                "to answer this question in the document.")

    words = tokenize(question)
    if words and words[0] in YES_NO_STARTERS:
        verdict = "No" if has_negation(sentence) else "Yes"
        return f"**{verdict}.** The document states: {sentence}"

    return f"Based on the document: {sentence}"

# ---------------- FEEDBACK STORAGE ----------------

def save_feedback(question, evidence, rating, correction, filename):
    file_exists = os.path.exists(filename)

    fields = [
        "timestamp",
        "document_name",
        "question",
        "retrieved_evidence",
        "rating",
        "correction"
    ]

    with open(
        filename,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "document_name": st.session_state.get(
                "document_name", "Pasted text"
            ),
            "question": question,
            "retrieved_evidence": evidence,
            "rating": rating,
            "correction": correction
        })


# ---------------- DOCUMENT INPUT ----------------

st.header("1. Provide a clinical document")

uploaded_file = st.file_uploader(
    "Upload a TXT or PDF file",
    type=["txt", "pdf"]
)

sample_text = st.text_area(
    "Or paste clinical text here",
    height=160,
    placeholder=(
        "Example: The patient reports a headache for two days. "
        "The patient also reports nausea since this morning. "
        "The patient denies having a fever."
    )
)

# ---------------- DOCUMENT STATUS CARD ----------------

if uploaded_file is not None:
    st.success(
        f"📄 Document selected: {uploaded_file.name}"
    )
elif sample_text.strip():
    st.info("✍️ Pasted clinical text is ready.")
else:
    st.caption(
        "📂 No document selected yet. "
        "Upload a TXT/PDF file or paste text above."
    )

st.divider()


document = sample_text.strip()
document_name = "Pasted text"

if uploaded_file is not None:
    document_name = uploaded_file.name

    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            document = read_pdf(uploaded_file)
        else:
            document = uploaded_file.getvalue().decode(
                "utf-8", errors="replace"
            ).strip()

    except Exception as error:
        st.error(f"Could not read the document: {error}")
        document = ""

st.session_state["document_name"] = document_name


# ---------------- DOCUMENT PREVIEW ----------------

if document:
    st.header("2. Document preview")

    with st.expander("View document text", expanded=False):
        st.write(document[:5000])

        if len(document) > 5000:
            st.caption("Preview shortened. Retrieval uses the full text.")

    st.caption(
        f"Document length: {len(document)} characters | "
        f"Sentences: {len(split_sentences(document))}"
    )

        # ---------------- QUESTION ANSWERING ----------------

    st.markdown("## 3. Ask a question")
    st.caption(
        "Ask a question about your document to find "
        "relevant supporting evidence."
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g. What symptoms are mentioned?",
        key="question_input",
        help="Use the clinical document currently loaded above."
    )

    if st.button("Find Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            results = find_relevant_sentences(
                document, question, top_k=3
            )

            st.session_state["last_question"] = question
            st.session_state["last_results"] = results
            st.session_state["last_document"] = document
            st.session_state["last_document_name"] = document_name
            st.session_state.pop("feedback_submitted", None)

    # Display results only for the current document
    if (
        st.session_state.get("last_document") == document
        and st.session_state.get("last_question")
    ):
        results = st.session_state.get("last_results", [])

        st.markdown("## 4. Simple Answer")

        if results:
            simple_answer = generate_simple_answer(
                st.session_state["last_question"],
                results
            )

            with st.container(border=True):
                st.markdown("#### 💬 Answer summary")
                st.write(simple_answer)
                st.caption(
                    "Answer generated from retrieved document evidence. "
                    "Verify the supporting sentence before relying on it."
                )
        else:
            with st.container(border=True):
                st.warning(
                    "No relevant evidence was found. "
                    "Try rephrasing your question."
                )

        st.header("5. Retrieved evidence")

        if results:
            st.success(
                f"Retrieved {len(results)} relevant sentence(s)."
            )

            for index, (sentence, score) in enumerate(results):
                label = (
                    "Most relevant evidence"
                    if index == 0 else f"Evidence {index + 1}"
                )

                with st.container(border=True):
                    st.markdown(f"#### 📄 {label}")
                    st.write(sentence)
                    st.progress(min(max(float(score), 0.0), 1.0))

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.metric(
                            "Similarity score",
                            f"{score:.3f}"
                        )

                    with col2:
                        st.caption(
                            "Higher scores indicate text similarity, "
                            "not medical accuracy or answer confidence."
                        )

        else:
            st.info(
                "No matching evidence was found. Try rephrasing "
                "the question using terms present in the document."
            )

        # ---------------- HUMAN FEEDBACK ----------------

        st.header("6. Human feedback")

        st.write(
            "Evaluate whether the retrieved evidence answers "
            "your question."
        )

        with st.form("feedback_form"):
            rating = st.radio(
                "Was the retrieved evidence helpful?",
                ["Correct", "Incorrect"],
                horizontal=True
            )

            correction = st.text_area(
                "Suggested correction (optional)",
                placeholder="Explain what should be improved."
            )

            submitted = st.form_submit_button("Save Feedback")

        if submitted:
            evidence = " | ".join(
                sentence for sentence, _ in results
            )

            try:
                save_feedback(
                    st.session_state["last_question"],
                    evidence,
                    rating,
                    correction,
                    FEEDBACK_FILE
                )

                st.session_state["feedback_submitted"] = True
                st.success(
                    "Feedback saved successfully to feedback.csv."
                )

            except OSError as error:
                st.error(
                    f"Could not save feedback. "
                    f"Check folder permissions: {error}"
                )

        if st.session_state.get("feedback_submitted"):
            st.caption("Your latest feedback was saved locally.")

st.divider()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🩺 Clinical NLP</div>',
        unsafe_allow_html=True
    )

    st.caption("DOCUMENT INTELLIGENCE")

    st.divider()

    st.radio(
        "Retriever",
        ["TF-IDF", "Embeddings (MiniLM)"],
        key="retriever"
    )
    st.divider()

    st.markdown("### About the system")

    st.markdown(
        '<div class="sidebar-description">'
        'An NLP-based assistant that retrieves relevant '
        'evidence from uploaded clinical documents.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Technologies")

    st.markdown(
        "- Python\n"
        "- Streamlit\n"
        "- TF-IDF\n"
        "- Cosine Similarity\n"
        "- Scikit-learn\n"
        "- NLTK"
    )

    st.divider()

    st.caption("Version 1.1 · Academic prototype")

st.header("7. Feedback summary")

if os.path.exists(FEEDBACK_FILE):
    try:
        with open(
            FEEDBACK_FILE, "r", newline="", encoding="utf-8-sig"
        ) as file:
            feedback_rows = list(csv.DictReader(file))

        if feedback_rows:
            total = len(feedback_rows)
            correct = sum(
                row.get("rating") == "Correct"
                for row in feedback_rows
            )
            incorrect = sum(
                row.get("rating") == "Incorrect"
                for row in feedback_rows
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total feedback entries", total)
            col2.metric("Marked correct", correct)
            col3.metric("Marked incorrect", incorrect)

            with st.expander("View feedback history"):
                st.dataframe(
                    feedback_rows,
                    width="stretch"
                )

            with open(FEEDBACK_FILE, "rb") as feedback_file:
                feedback_export = feedback_file.read()

            st.download_button(
                "Download feedback CSV",
                data=feedback_export,
                file_name="feedback_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No feedback has been recorded yet.")

    except (OSError, csv.Error) as error:
        st.error(f"Could not read feedback history: {error}")
else:
    st.info(
        "Feedback history will appear here after you submit "
        "your first rating."
    )

# ---------------- NLP EVALUATION ----------------

st.header("📊 NLP Evaluation Dashboard")
st.write(
    "Generate draft evaluation questions from the clinical document currently loaded, "
    "download them as a CSV, or upload a manually verified evaluation CSV."
)

# ---------------- EVALUATION SUPPORT ----------------

# Column names accepted in a hand prepared evaluation CSV.
EVALUATION_QUESTION_COLUMNS = (
    "question",
    "questions",
    "question text",
    "query",
)

EVALUATION_EVIDENCE_COLUMNS = (
    "relevant_evidence",
    "relevant evidence",
    "expected_evidence",
    "expected evidence",
    "evidence",
)

# Words that carry no clinical meaning. They are ignored when comparing
# the content of two sentences, so that the sentence template ("The patient
# reports a ... for ...") does not make two different findings look alike.
EVIDENCE_TEMPLATE_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "patient", "report", "reports", "reported", "reporting", "state",
    "states", "stated", "note", "notes", "clinical", "document", "deny",
    "denies", "denied", "also", "there", "this", "that", "any", "from",
    "by", "at", "as", "it", "his", "her", "their", "s", "t", "not",
}

# Thresholds used when two sentences are not literally the same text but
# still describe the same finding.
EVIDENCE_MATCH_MIN_RECALL = 0.8
EVIDENCE_MATCH_MIN_PRECISION = 0.6

# Sentences that end with these words are document headings, not evidence.
HEADING_TAIL_WORDS = {
    "questions", "question", "note", "notes", "history",
    "examination", "exam", "assessment", "plan", "findings",
    "summary", "complaints", "medications", "allergies",
}


def evidence_tokens(text):
    """Normalized word tokens used for comparing two pieces of evidence."""

    if not text:
        return []

    return re.findall(r"\b\w+\b", text.lower())


def topic_tokens(text):
    """Clinical content words of a sentence.

    Stop words, the sentence template and negation words are left out, so
    that only the medical content (symptom, duration, finding) remains.
    """

    return {
        word
        for word in tokenize(text)
        if word not in EVIDENCE_TEMPLATE_WORDS
        and word not in NEGATION_TERMS
    }


def contains_token_sequence(short, long):
    """True when `short` occurs as a contiguous run of words inside `long`."""

    if not short or len(short) > len(long):
        return False

    for start in range(len(long) - len(short) + 1):
        if long[start:start + len(short)] == short:
            return True

    return False


def polarity_conflict(expected, retrieved):
    """True when two sentences state the same fact with opposite polarity.

    "The patient denies shortness of breath" and "The patient has shortness
    of breath" share every topic word but disagree about the finding, so
    they must never be counted as the same piece of evidence.
    """

    expected_topic = topic_tokens(expected)

    if not expected_topic:
        return False

    # Only compare polarity when the retrieved sentence covers the same
    # topic as the expected evidence.
    if not expected_topic.issubset(topic_tokens(retrieved)):
        return False

    return has_negation(expected) != has_negation(retrieved)


def evidence_matches(
    expected,
    retrieved,
    min_recall=EVIDENCE_MATCH_MIN_RECALL,
    min_precision=EVIDENCE_MATCH_MIN_PRECISION
):
    """Check whether retrieved evidence answers the expected evidence.

    Differences in punctuation, capitalisation and whitespace are ignored,
    and a slightly longer sentence (for example one with an extra clause)
    is still accepted. A genuinely different sentence, or the same sentence
    with the opposite clinical polarity, is never accepted.
    """

    expected_tokens = evidence_tokens(expected)
    retrieved_tokens = evidence_tokens(retrieved)

    if not expected_tokens or not retrieved_tokens:
        return False

    # Opposite polarity is always a mismatch.
    if polarity_conflict(expected, retrieved):
        return False

    # Identical word sequence (ignoring case, punctuation, whitespace).
    if expected_tokens == retrieved_tokens:
        return True

    # One sentence contains the other: minor formatting difference.
    if contains_token_sequence(expected_tokens, retrieved_tokens):
        return True

    if contains_token_sequence(retrieved_tokens, expected_tokens):
        return True

    # Otherwise compare the clinical content of the two sentences.
    expected_content = topic_tokens(expected)
    retrieved_content = topic_tokens(retrieved)

    if not expected_content or not retrieved_content:
        return False

    overlap = len(expected_content & retrieved_content)

    recall = overlap / len(expected_content)
    precision = overlap / len(retrieved_content)

    return recall >= min_recall and precision >= min_precision


def overlap_scores(expected, retrieved):
    """Word overlap precision, recall and F1 for two pieces of evidence."""

    expected_words = evidence_tokens(expected)
    retrieved_words = evidence_tokens(retrieved)

    if not expected_words or not retrieved_words:
        return 0.0, 0.0, 0.0

    expected_counts = Counter(expected_words)
    retrieved_counts = Counter(retrieved_words)

    overlap = sum((expected_counts & retrieved_counts).values())

    precision = overlap / len(retrieved_words)
    recall = overlap / len(expected_words)

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return precision, recall, f1


def resolve_column(columns, candidates):
    """Find a column by name, ignoring case, spaces and underscores."""

    if columns is None:
        return None

    lookup = {
        re.sub(r"[^a-z0-9]", "", str(name).strip().lower()): name
        for name in columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())

        if key in lookup:
            return lookup[key]

    return None


def clean_cell(value):
    """Return a trimmed string for a CSV cell, treating blanks as empty."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


# ---------------- EVALUATION CASE GENERATION ----------------

def is_heading(sentence):
    """True for short heading-like sentences such as 'Suggested questions'."""

    text = re.sub(r"\s+\d+\s*$", "", sentence).strip().strip(".:;")

    if not text:
        return True

    words = tokenize(text)

    if not words:
        return True

    return words[-1] in HEADING_TAIL_WORDS and len(words) <= 5


def build_question_for_sentence(sentence):
    """Build a natural question for one sentence of the clinical note.

    Returns a tuple of (question, disambiguating_words). The question always
    contains content words taken from the sentence, so the retriever is able
    to find that sentence again from the question alone.
    """

    cleaned = sentence.strip().rstrip(".!?").strip()

    # 1. Duration statements, e.g. "reports a headache for two days".
    duration_match = re.search(
        r"\b(?:reports?|has had|have had|experiences?|experienced|"
        r"presents with)\s+(?:having\s+)?((?:a|an|the)\s+)?"
        r"([^,.!?]+?)\s+(?:for|since)\s+([^,.!?]+)",
        cleaned,
        flags=re.IGNORECASE
    )
    if duration_match:
        symptom = " ".join(
            part.strip()
            for part in duration_match.group(1, 2)
            if part and part.strip()
        )

        if symptom:
            return (
                f"How long has the patient had {symptom}?",
                symptom.split()
            )

    # 2. Explicit denials, e.g. "denies shortness of breath".
    denial_match = re.search(
        r"\b(?:denies|denied|has no|does not have|does not report)\s+"
        r"(?:having\s+|any\s+)?(.+)",
        cleaned,
        flags=re.IGNORECASE
    )
    if denial_match:
        topic = re.split(
            r"\b(?:but|however|although|while)\b|[,;:]",
            denial_match.group(1)
        )[0].strip(" .!?")

        if topic:
            return (
                f"Does the patient have {topic}?",
                topic.split()
            )

    # 3. Reported intake or actions, e.g. "reports drinking warm water".
    intake_match = re.search(
        r"\b(?:reports?|has been|is)\s+"
        r"(drinking|eating|taking|using)\b\s*(.*)",
        cleaned,
        flags=re.IGNORECASE
    )
    if intake_match:
        action = intake_match.group(1).lower()
        remainder = intake_match.group(2).strip(" .!?")

        return (
            f"What does the patient report {action}?",
            remainder.split() if remainder else []
        )

    # 4. A stated finding, e.g. "has a dry cough".
    finding_match = re.search(
        r"\b(?:has|have|presents with)\s+((?:a|an|the)\s+)?([^,.!?]+)",
        cleaned,
        flags=re.IGNORECASE
    )
    if finding_match:
        finding = " ".join(
            part.strip()
            for part in finding_match.group(1, 2)
            if part and part.strip()
        ).strip(" .!?")

        if finding:
            return (
                f"Does the patient have {finding}?",
                finding.split()
            )

    # 5. A reported symptom, e.g. "reports feeling tired".
    reported_match = re.search(
        r"\breports?\s+(?:a|an|the)?\s*([^,.!?]+)",
        cleaned,
        flags=re.IGNORECASE
    )
    if reported_match:
        symptom = reported_match.group(1).strip(" .!?")

        if symptom:
            return (
                f"Does the patient report {symptom}?",
                symptom.split()
            )

    # 6. Fallback: keep the meaningful words of the sentence so that the
    #    question can still retrieve it.
    words = [
        word for word in tokenize(cleaned)
        if word not in STOP_WORDS
        and word not in NEGATION_TERMS
        and word not in {"patient", "reports", "reported", "states", "state"}
    ]

    if not words:
        return None, []

    topic = " ".join(words[:5])

    return (
        f"What does the note state about {topic}?",
        words[:5]
    )


def generate_evaluation_cases(source_document):
    """Create natural, rule-based test questions grounded in source sentences.

    Every question contains content words taken from the sentence it was
    built from, so the retrieval step can find that sentence again using
    the question as the query.
    """

    cases = []
    used_questions = set()

    for sentence in split_sentences(source_document):
        cleaned = sentence.strip()

        # Questions and headings are not evidence.
        if not cleaned or cleaned.endswith("?"):
            continue

        if is_heading(cleaned):
            continue

        question, extra_words = build_question_for_sentence(cleaned)

        if not question:
            continue

        key = question.lower()

        if key in used_questions:
            # The same question was produced for a different sentence, so
            # add the distinguishing words of this sentence.
            distinguishing = [
                word for word in extra_words
                if word.lower() not in question.lower()
            ]

            if not distinguishing:
                continue

            question = (
                f"{question.rstrip('?')} "
                f"({' '.join(distinguishing[:5])})?"
            )
            key = question.lower()

            if key in used_questions:
                continue

        used_questions.add(key)

        cases.append({
            "question": question,
            "relevant_evidence": cleaned
        })

    return cases


# ---------------- EVALUATION ----------------

def evaluate_cases(source_document, test_data):
    """Evaluate retrieval using each question, not its expected answer."""

    if test_data is None or len(test_data) == 0:
        return pd.DataFrame()

    question_column = resolve_column(
        test_data.columns, EVALUATION_QUESTION_COLUMNS
    )
    evidence_column = resolve_column(
        test_data.columns, EVALUATION_EVIDENCE_COLUMNS
    )

    if question_column is None or evidence_column is None:
        return pd.DataFrame()

    rows = []

    for _, row in test_data.iterrows():
        question = clean_cell(row[question_column])
        expected = clean_cell(row[evidence_column])

        if not question:
            continue

        # Important: the evidence is retrieved using the QUESTION.
        # The expected evidence is only used to score the result.
        retrieved = find_relevant_sentences(
            source_document,
            question,
            top_k=3
        )

        retrieved_text = retrieved[0][0] if retrieved else ""

        unanswerable = not expected
        if unanswerable:
            expected = "(not answerable from the note)"
            match = not retrieved
            rank = 1 if match else 0
            score_value = 1.0 if match else 0.0
            precision = recall = f1 = score_value
        else:
            match = evidence_matches(expected, retrieved_text)
            rank = next(
                (i + 1 for i, (s, _) in enumerate(retrieved)
                 if evidence_matches(expected, s)),
                0,
            )
            precision, recall, f1 = overlap_scores(expected, retrieved_text)

        rows.append({
            "Question": question,
            "Expected Evidence": expected,
            "Retrieved Evidence": (
                retrieved_text or "No evidence retrieved"
            ),
            "Evidence Match": (
                "PASS" if match else "CHECK"
            ),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
            "Rank": rank,
            "Hit@3": int(rank > 0),
            "RR": round(1 / rank, 3) if rank else 0.0
        })

    return pd.DataFrame(rows)


# ---------------- EVALUATION CONSOLE (UI) ----------------

def build_evaluation_console(
    results,
    title="Retrieval evaluation",
    subtitle="",
    tags=None,
):
    """Build the HTML for the evaluation console.

    Shows KPI tiles, a match-rate ring and one card per evaluated case.
    The numbers come straight from the evaluation results, nothing is
    hard coded.
    """

    from html import escape

    total = len(results)
    passed = int((results["Evidence Match"] == "PASS").sum())
    rate = (passed / total * 100) if total else 0.0

    avg_f1 = float(results["F1 Score"].mean()) if total else 0.0
    avg_precision = float(results["Precision"].mean()) if total else 0.0
    avg_recall = float(results["Recall"].mean()) if total else 0.0
    hit3 = float(results["Hit@3"].mean()) * 100 if total else 0.0
    mrr = float(results["RR"].mean()) if total else 0.0

    if rate >= 80:
        ring_color = "#34D399"
    elif rate >= 50:
        ring_color = "#FBBF24"
    else:
        ring_color = "#FB7185"

    kpis = [
        ("test-cases", "Test cases", str(total),
         "questions evaluated", "#65D6C5"),
        ("evidence-matches", "Evidence matches", f"{passed}/{total}",
         "expected evidence retrieved", ring_color),
        ("match-rate", "Match rate", f"{rate:.1f}%",
         "retrieval consistency", ring_color),
        ("average-f1", "Average F1", f"{avg_f1:.3f}",
         "word overlap F1 score", "#A78BFA"),
        ("hit-at-3", "Hit@3", f"{hit3:.1f}%",
         "answer in top 3", "#65D6C5"),
        ("mrr", "MRR", f"{mrr:.3f}",
         "mean reciprocal rank", "#A78BFA"),
    ]

    parts = []

    parts.append('<div class="eval-console">')
    parts.append('<div class="eval-top">')

    parts.append('<div class="eval-top-text">')
    parts.append(
        f'<div class="eval-title">{escape(title)}</div>'
    )
    parts.append(
        f'<div class="eval-sub">{escape(subtitle)}</div>'
    )

    if tags:
        parts.append('<div class="eval-tags">')
        for tag in tags:
            parts.append(
                f'<span class="eval-tag">{escape(str(tag))}</span>'
            )
        parts.append('</div>')

    parts.append('</div>')

    parts.append(
        '<div class="eval-ring" '
        f'style="--pct: {rate:.2f}; --ring-color: {ring_color};">'
        '<div class="eval-ring-inner">'
        f'<div class="eval-ring-value">{rate:.0f}%</div>'
        '<div class="eval-ring-label">Match rate</div>'
        '</div></div>'
    )

    parts.append('</div>')

    parts.append('<div class="eval-kpis">')
    for key, label, value, note, accent in kpis:
        parts.append(
            f'<div class="eval-kpi" data-metric="{key}" '
            f'style="--accent: {accent};">'
            f'<div class="eval-kpi-label">{escape(label)}</div>'
            f'<div class="eval-kpi-value">{escape(value)}</div>'
            f'<div class="eval-kpi-note">{escape(note)}</div>'
            '</div>'
        )
    parts.append('</div>')

    parts.append(
        '<div class="eval-progress">'
        f'<span style="width: {rate:.2f}%;"></span>'
        '</div>'
    )

    if total:
        parts.append('<div class="eval-cases">')
        for _, row in results.iterrows():
            is_pass = str(row["Evidence Match"]).strip().upper() == "PASS"
            status = "#34D399" if is_pass else "#FBBF24"
            badge = "PASS" if is_pass else "CHECK"
            badge_class = "pass" if is_pass else "check"

            parts.append(
                f'<div class="eval-case" style="--status: {status};">'
                f'<span class="eval-badge {badge_class}">{badge}</span>'
                '<div class="eval-case-body">'
                f'<div class="eval-case-q">'
                f'{escape(str(row["Question"]))}</div>'
                f'<div class="eval-case-line">'
                f'Expected evidence: {escape(str(row["Expected Evidence"]))}'
                '</div>'
                f'<div class="eval-case-ev">'
                f'Retrieved: {escape(str(row["Retrieved Evidence"]))}'
                '</div>'
                f'<div class="eval-case-scores">'
                f'Precision {float(row["Precision"]):.3f} · '
                f'Recall {float(row["Recall"]):.3f} · '
                f'F1 {float(row["F1 Score"]):.3f}</div>'
                '</div></div>'
            )
        parts.append('</div>')

    parts.append(
        '<div class="eval-note">'
        f'Average precision {avg_precision:.3f} · '
        f'average recall {avg_recall:.3f} · '
        f'average F1 {avg_f1:.3f}. Evidence matching ignores punctuation, '
        'capitalisation and whitespace, and never accepts a sentence with '
        'the opposite clinical polarity.'
        '</div>'
    )

    parts.append('</div>')

    return "".join(parts)


def build_results_table(results):
    """Build the HTML for the detailed results table.

    The table is drawn with HTML instead of ``st.dataframe`` so that the
    verdict and the score bar keep their colours: Streamlit renders
    dataframes on a canvas, where pandas styling is not applied.
    """

    from html import escape

    headings = [
        ("Question", ""),
        ("Expected evidence", ""),
        ("Retrieved evidence", ""),
        ("Verdict", ""),
        ("Precision", "rt-num"),
        ("Recall", "rt-num"),
        ("F1", "rt-num"),
    ]

    parts = ['<div class="results-table">']

    parts.append('<div class="rt-head">')
    for label, css_class in headings:
        parts.append(
            f'<div class="{css_class}">{escape(label)}</div>'
        )
    parts.append('</div>')

    for _, row in results.iterrows():
        is_pass = str(row["Evidence Match"]).strip().upper() == "PASS"
        badge = "PASS" if is_pass else "CHECK"
        badge_class = "pass" if is_pass else "check"

        precision = float(row["Precision"])
        recall = float(row["Recall"])
        f1 = float(row["F1 Score"])

        parts.append('<div class="rt-row">')
        parts.append(
            f'<div class="rt-cell rt-q">{escape(str(row["Question"]))}</div>'
        )
        parts.append(
            f'<div class="rt-cell rt-ev">'
            f'{escape(str(row["Expected Evidence"]))}</div>'
        )
        parts.append(
            f'<div class="rt-cell rt-ev">'
            f'{escape(str(row["Retrieved Evidence"]))}</div>'
        )
        parts.append(
            f'<div><span class="rt-pill {badge_class}">{badge}</span></div>'
        )
        parts.append(f'<div class="rt-num">{precision:.3f}</div>')
        parts.append(f'<div class="rt-num">{recall:.3f}</div>')
        parts.append(
            f'<div class="rt-num rt-f1">'
            f'<span class="rt-bar">'
            f'<span style="width: {max(0.0, min(1.0, f1)) * 100:.0f}%;"></span>'
            f'</span>{f1:.3f}</div>'
        )
        parts.append('</div>')

    parts.append('</div>')

    return "".join(parts)




if not document.strip():
    st.info("Load a PDF/TXT file or paste clinical text above to generate evaluation cases.")
else:
    generated_cases = generate_evaluation_cases(document)

    st.subheader("1. Automatically generated test cases")
    st.caption(
        "These draft questions are generated with simple rules from sentences in the loaded note. "
        "Each question keeps the content words of its sentence, so the retriever can find that "
        "sentence again. Review them before using the scores as a formal evaluation."
    )

    if generated_cases:
        generated_df = pd.DataFrame(generated_cases)

        with st.container(border=True):
            st.markdown(
                '<div class="result-heading">Generated evaluation cases</div>'
                '<div class="result-sub">'
                'Each question is derived from one sentence of the loaded '
                'note, and that sentence is the expected evidence.'
                '</div>',
                unsafe_allow_html=True
            )
            st.dataframe(generated_df, width="stretch", hide_index=True)

        st.download_button(
            "⬇️ Download generated evaluation CSV",
            data=generated_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="generated_clinical_evaluation.csv",
            mime="text/csv",
            key="download_generated_eval"
        )

        st.subheader("2. Evaluate this document")
        if st.button("Run evaluation on generated test cases", type="primary"):
            evaluation_results = evaluate_cases(document, generated_df)
            st.session_state["auto_eval_results"] = evaluation_results
            st.session_state["auto_eval_document"] = document

        if (
            st.session_state.get("auto_eval_document") == document
            and "auto_eval_results" in st.session_state
        ):
            evaluation_results = st.session_state["auto_eval_results"]
            if not evaluation_results.empty:
                st.markdown(
                    build_evaluation_console(
                        evaluation_results,
                        title="Automated retrieval evaluation",
                        subtitle=(
                            "Each question was answered by the retriever "
                            "and the returned sentence was compared with "
                            "the expected evidence from the note."
                        ),
                        tags=[
                            f"Document: {document_name}",
                            f"{len(split_sentences(document))} sentences",
                            f"Retriever: {st.session_state.get('retriever', 'TF-IDF')}",
                            "Query = question only",
                        ],
                    ),
                    unsafe_allow_html=True
                )

                with st.container(border=True):
                    st.markdown(
                        '<div class="result-heading">Detailed results</div>'
                        '<div class="result-sub">'
                        'Verdicts, word overlap precision / recall and F1 '
                        'for every evaluated question.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        build_results_table(evaluation_results),
                        unsafe_allow_html=True
                    )

                st.download_button(
                    "⬇️ Download evaluation results",
                    data=evaluation_results.to_csv(index=False).encode("utf-8-sig"),
                    file_name="clinical_nlp_evaluation_results.csv",
                    mime="text/csv",
                    key="download_auto_eval_results"
                )
            else:
                st.warning("No valid evaluation rows were generated.")
    else:
        st.warning("No sentences were found in the current document.")

    st.subheader("3. Optional: evaluate a manually prepared CSV")
    st.caption(
        "Use this when you have human-verified expected evidence. "
        "Required columns: question, relevant_evidence "
        "(column names are matched without regard to case, spaces or underscores)."
    )
    evaluation_file = st.file_uploader(
        "Upload a custom evaluation CSV (optional)",
        type=["csv"],
        key="evaluation_csv"
    )

    if evaluation_file is not None:
        try:
            custom_data = pd.read_csv(evaluation_file)
            question_column = resolve_column(
                custom_data.columns, EVALUATION_QUESTION_COLUMNS
            )
            evidence_column = resolve_column(
                custom_data.columns, EVALUATION_EVIDENCE_COLUMNS
            )

            if question_column is None or evidence_column is None:
                st.error(
                    "Your CSV must contain columns named question and "
                    "relevant_evidence. Columns found: "
                    f"{list(custom_data.columns)}."
                )
            else:
                custom_results = evaluate_cases(document, custom_data)
                if custom_results.empty:
                    st.warning("The CSV has no usable question/evidence rows.")
                else:
                    st.markdown(
                        build_evaluation_console(
                            custom_results,
                            title="Custom evaluation CSV",
                            subtitle=(
                                "A human-verified question set was uploaded "
                                "and evaluated against the document that is "
                                "currently loaded."
                            ),
                            tags=[
                                f"File: {evaluation_file.name}",
                                f"{len(custom_results)} questions",
                                f"Retriever: {st.session_state.get('retriever', 'TF-IDF')}",
                                "Query = question only",
                            ],
                        ),
                        unsafe_allow_html=True
                    )

                    with st.container(border=True):
                        st.markdown(
                            '<div class="result-heading">Detailed results</div>'
                            '<div class="result-sub">'
                            'Verdicts, word overlap precision / recall and '
                            'F1 for every question in the uploaded CSV.'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            build_results_table(custom_results),
                            unsafe_allow_html=True
                        )
                    st.download_button(
                        "Download custom evaluation results",
                        data=custom_results.to_csv(index=False).encode("utf-8-sig"),
                        file_name="custom_clinical_evaluation_results.csv",
                        mime="text/csv",
                        key="download_custom_eval_results"
                    )
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read the evaluation CSV: {error}")

st.info(
    "Evaluation note: precision, recall, F1, and evidence-match status are based on word overlap "
    "and text matching. Evidence matching ignores differences in punctuation, capitalisation and "
    "whitespace, and never accepts a sentence with the opposite polarity (for example a denial "
    "instead of a finding). Automatically generated expected evidence is copied from the document, "
    "so these checks assess retrieval consistency—not independent medical correctness."
)


# ---------------- PROJECT INFORMATION ----------------

with st.expander("About this project"):
    st.write(
        "This educational project demonstrates Natural Language "
        "Processing through TF-IDF vectorization, cosine similarity, "
        "keyword overlap, sentence retrieval, and human feedback. "
        "It is inspired by research on interactive clinical "
        "question-answering systems. It is not a validated clinical "
        "decision-support system."
    )
