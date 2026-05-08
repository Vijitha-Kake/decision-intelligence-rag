import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Re-ranker import (requires: pip install sentence-transformers)
from reranker import rerank

# ---------- App Config ----------
st.set_page_config(
    page_title="Decision Intelligence Platform for Manufacturing Defect Analysis (RAG)",
    layout="wide"
)
load_dotenv()

FAISS_DIR = "vector_store/faiss_openai"

# ---------- Helpers ----------

def format_context(doc_score_pairs) -> str:
    """Format FAISS (doc, score) pairs into numbered context blocks."""
    blocks = []
    for i, (doc, score) in enumerate(doc_score_pairs, 1):
        md = doc.metadata
        src = (
            f"{md.get('source_file')} | domain={md.get('domain')} "
            f"| page={md.get('page')} | faiss_score={score:.4f}"
        )
        blocks.append(f"[{i}] SOURCE: {src}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def format_context_reranked(triples) -> str:
    """Format re-ranked (doc, faiss_score, rerank_score) triples into context blocks."""
    blocks = []
    for i, (doc, faiss_s, rerank_s) in enumerate(triples, 1):
        md = doc.metadata
        src = (
            f"{md.get('source_file')} | domain={md.get('domain')} "
            f"| page={md.get('page')} "
            f"| faiss={faiss_s:.4f} | rerank={rerank_s:.4f}"
        )
        blocks.append(f"[{i}] SOURCE: {src}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def build_sources_list(doc_score_pairs) -> str:
    """Build a clean numbered source list from FAISS pairs."""
    lines = []
    for i, (doc, score) in enumerate(doc_score_pairs, 1):
        md = doc.metadata
        lines.append(
            f"[{i}] {md.get('source_file')} | domain={md.get('domain')} "
            f"| page={md.get('page')} | faiss_score={score:.4f}"
        )
    return "\n".join(lines)


def build_sources_list_reranked(triples) -> str:
    """Build a clean numbered source list from re-ranked triples."""
    lines = []
    for i, (doc, faiss_s, rerank_s) in enumerate(triples, 1):
        md = doc.metadata
        lines.append(
            f"[{i}] {md.get('source_file')} | domain={md.get('domain')} "
            f"| page={md.get('page')} "
            f"| faiss={faiss_s:.4f} | rerank={rerank_s:.4f}"
        )
    return "\n".join(lines)


def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a manufacturing operations + quality assistant for Toyota Camry body assembly (welding).\n"
         "Rules:\n"
         "1) Use ONLY the provided context blocks.\n"
         "2) If insufficient, say: 'I don't know based on the provided context.'\n"
         "3) Do not invent numbers/procedures.\n"
         "4) In sections B–F, include citations like [1] or [1][2] for every bullet.\n"
         "5) Citations MUST refer to the numbered context blocks.\n"),
        ("human",
         "Context blocks:\n{context}\n\n"
         "Question:\n{question}\n\n"
         "Respond in this exact format:\n"
         "A) Problem summary (1–2 sentences)\n"
         "B) Likely root causes (bullet list with citations like [1])\n"
         "C) Evidence (bullet list with citations like [1], [2])\n"
         "D) Immediate actions (next 24 hours) (bullets with citations)\n"
         "E) Preventive actions (next 2–4 weeks) (bullets with citations)\n"
         "F) Escalation (who + why) (bullets with citations)\n\n"
         "G) Sources (copy exactly the list below)\n"
         "{sources}\n"
         )
    ])


# ---------- Caching ----------

@st.cache_resource
def load_vectorstore():
    """Load embeddings + FAISS once per Streamlit session."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


def build_rag_chain(db, top_k: int, model_name: str, temperature: float):
    def retrieve_with_scores(question: str):
        return db.similarity_search_with_score(question, k=top_k)

    parallel_chain = RunnableParallel({
        "question":   RunnablePassthrough(),
        "doc_scores": RunnableLambda(retrieve_with_scores),
    })

    to_prompt_inputs = RunnableLambda(lambda x: {
        "question":        x["question"],
        "context":         format_context(x["doc_scores"]),
        "sources":         build_sources_list(x["doc_scores"]),
        "_raw_doc_scores": x["doc_scores"],
    })

    prompt = build_prompt()
    model  = ChatOpenAI(model=model_name, temperature=temperature)
    parser = StrOutputParser()

    # debug_chain returns raw doc_scores; rag_chain returns the final answer
    debug_chain = parallel_chain | to_prompt_inputs
    rag_chain   = (
        debug_chain
        | RunnableLambda(lambda x: {k: x[k] for k in ["question", "context", "sources"]})
        | prompt | model | parser
    )

    return debug_chain, rag_chain


# ---------- UI ----------

st.title("Decision Intelligence Platform for Manufacturing Defect Analysis (RAG)")
st.caption("LLM-powered root cause analysis using structured and unstructured manufacturing documents")

# ---------- Sidebar ----------

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "FAISS TOP_K  (docs to retrieve)",
        min_value=5, max_value=30, value=20, step=5,
        help="Retrieve more docs so the re-ranker has better candidates to choose from."
    )
    top_n_rerank = st.slider(
        "Re-ranker TOP_N  (docs passed to LLM)",
        min_value=1, max_value=10, value=5, step=1,
        help="After re-ranking, only the best N chunks go into the prompt."
    )

    use_reranker = st.toggle(
        "🔀 Enable Cross-Encoder Re-Ranker",
        value=True,
        help="Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to re-score each (query, chunk) pair."
    )

    model_name   = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-5"], index=0)
    temperature  = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    show_context = st.checkbox("Show retrieved context blocks", value=True)
    show_comparison = st.checkbox(
        "Show FAISS vs Re-Ranker comparison",
        value=True,
        help="Side-by-side view showing which chunks moved up or down after re-ranking."
    )

    st.divider()
    st.caption(
        "**Re-ranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  \n"
        "Pre-trained on MS MARCO (Bing search relevance). ~80 MB, loads once."
    )

# ---------- Sample Questions ----------

SAMPLE_QUESTIONS = {
    "Welding / Quality": [
        "Why are welding misalignment defects increasing?",
        "What are common indicators and root causes of welding misalignment?",
        "How can we reduce panel gap and flush issues in body assembly?"
    ],
    "Maintenance / Equipment": [
        "What maintenance actions reduce alignment drift over time?",
        "What happens if we skip calibration or delay preventive maintenance?",
        "Which robot or fixture issues most often lead to quality defects?"
    ],
    "Supplier / Materials": [
        "How do supplier batch variations contribute to defects?",
        "What material tolerance issues commonly cause misalignment?",
        "What controls can reduce variability from incoming panels?"
    ],
}

DEFAULT_Q = (
    "Welding misalignment defects are increasing. "
    "What are common causes and what maintenance actions should we take?"
)

if "question" not in st.session_state:
    st.session_state.question = DEFAULT_Q

st.markdown("### Try a sample question (optional)")

categories    = ["— Select a category —"] + list(SAMPLE_QUESTIONS.keys())
selected_cat  = st.selectbox("Category", categories, index=0)

if selected_cat != categories[0]:
    qs         = ["— Select a sample question —"] + SAMPLE_QUESTIONS[selected_cat]
    selected_q = st.selectbox("Sample question", qs, index=0)
    if selected_q != qs[0]:
        st.session_state.question = selected_q
else:
    st.selectbox("Sample question", ["Select a category first"], index=0, disabled=True)

question = st.text_area("Or ask your own question", key="question", height=110)

col1, col2 = st.columns([1, 1])
run_btn   = col1.button("Run RAG", type="primary")
clear_btn = col2.button("Clear")

if clear_btn:
    for key in ["answer", "sources", "context", "doc_scores",
                "top_pairs", "reranked_triples", "use_reranker"]:
        st.session_state.pop(key, None)
    st.rerun()

# ---------- Load DB ----------

try:
    db = load_vectorstore()
except Exception as e:
    st.error(f"Failed to load FAISS index from '{FAISS_DIR}'. Error: {e}")
    st.stop()

debug_chain, rag_chain = build_rag_chain(
    db, top_k=top_k, model_name=model_name, temperature=temperature
)

# ---------- Run ----------

if run_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving → re-ranking → generating answer..."):

            # Step 1: FAISS retrieval (wide net)
            debug_out  = debug_chain.invoke(question)
            doc_scores = debug_out["_raw_doc_scores"]       # List[(doc, faiss_score)]

            # Step 2: Cross-encoder re-ranking (optional)
            if use_reranker:
                reranked_triples = rerank(question, doc_scores, top_n=top_n_rerank)
                top_pairs        = [(doc, fs) for doc, fs, _ in reranked_triples]
                context_text     = format_context_reranked(reranked_triples)
                sources_text     = build_sources_list_reranked(reranked_triples)
            else:
                reranked_triples = None
                top_pairs        = doc_scores[:top_n_rerank]
                context_text     = format_context(top_pairs)
                sources_text     = build_sources_list(top_pairs)

            # Step 3: LLM answer (using re-ranked context)
            answer = rag_chain.invoke(question)

            st.session_state.update({
                "answer":           answer,
                "sources":          sources_text,
                "context":          context_text,
                "doc_scores":       doc_scores,
                "top_pairs":        top_pairs,
                "reranked_triples": reranked_triples,
                "use_reranker":     use_reranker,
            })

# ---------- Output ----------

if "answer" in st.session_state:
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Answer")
        st.write(st.session_state["answer"])

        if show_context:
            label = "Context sent to LLM (re-ranked)" if st.session_state.get("use_reranker") else "Context sent to LLM"
            st.subheader(label)
            st.text(st.session_state["context"])

    with right:
        st.subheader("Sources")
        st.text(st.session_state["sources"])

        # ── FAISS vs Re-Ranker comparison view ──
        if (
            show_comparison
            and st.session_state.get("use_reranker")
            and st.session_state.get("reranked_triples") is not None
        ):
            st.subheader("🔀 FAISS vs Re-Ranker")
            st.caption(
                f"FAISS retrieved **{len(st.session_state['doc_scores'])}** chunks  →  "
                f"Re-ranker kept top **{len(st.session_state['reranked_triples'])}**"
            )

            # Map each doc to its original FAISS rank (1-indexed)
            faiss_order = {
                id(doc): i + 1
                for i, (doc, _) in enumerate(st.session_state["doc_scores"])
            }

            for new_rank, (doc, faiss_s, rerank_s) in enumerate(
                st.session_state["reranked_triples"], 1
            ):
                old_rank = faiss_order.get(id(doc), "?")
                if old_rank != "?" and new_rank < old_rank:
                    arrow = "🟢 ▲ moved up"
                elif old_rank != "?" and new_rank > old_rank:
                    arrow = "🔴 ▼ moved down"
                else:
                    arrow = "⚪ ─ same"

                md = doc.metadata or {}
                st.markdown(
                    f"**[{new_rank}]** {arrow}  *(was FAISS #{old_rank})* \n"
                    f"- `{md.get('source_file')}` · page {md.get('page')} · domain `{md.get('domain')}` \n"
                    f"- FAISS dist: `{faiss_s:.4f}` → Rerank score: `{rerank_s:.4f}`"
                )

        else:
            # Fallback: original metadata display (re-ranker off)
            st.subheader("Top Matches (metadata)")
            for i, (doc, score) in enumerate(st.session_state["top_pairs"], 1):
                md = doc.metadata or {}
                st.markdown(
                    f"**[{i}] score={score:.4f}** \n"
                    f"- file: `{md.get('source_file')}` \n"
                    f"- domain: `{md.get('domain')}` \n"
                    f"- page: `{md.get('page')}`"
                )
