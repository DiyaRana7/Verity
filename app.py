from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from detector import analyze, load_model, load_split_dataset, train_model


st.set_page_config(page_title="Verity | News credibility", page_icon="V", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


def fill_sample_story():
    st.session_state.article_text = (
        "BREAKING!!! A secret report proves that a common kitchen spice cures every illness in 24 hours. "
        "Doctors are hiding the truth. Share this with everyone before the post is deleted!"
    )

st.markdown("""
<style>
:root { --ink:#152238; --muted:#607086; --paper:#f4f6fa; --teal:#19a890; --line:#e1e7ef; --navy:#101d33; }
.stApp, [data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 75% 0%,#eaf3f3 0%,transparent 33%),#f4f6fa; color:var(--ink); }
.block-container { max-width:1080px; padding:1.5rem 2rem 4rem; }
[data-testid="stHeader"] { background:rgba(244,246,250,.95); }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] label,
.stCaption, label, [data-testid="stWidgetLabel"] p { color:#35445a !important; }
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color:#152238; letter-spacing:-.035em; }
[data-testid="stCaptionContainer"] p { color:var(--muted) !important; }
.brand { display:flex; align-items:center; gap:10px; color:var(--ink); font:700 13px 'Segoe UI',sans-serif; letter-spacing:.14em; padding:8px 0 18px; }
.brand::before { content:'V'; display:grid; place-items:center; width:30px; height:30px; border-radius:10px; background:var(--navy); color:#66e0c5; font-weight:800; letter-spacing:0; }
.hero { position:relative; isolation:isolate; overflow:hidden; background:linear-gradient(118deg,#101d33 0%,#172c46 68%,#174353 100%); border:1px solid #243b54; border-radius:24px; padding:clamp(30px,6vw,66px); margin:16px 0 34px; box-shadow:0 18px 48px #172c461c; }
.hero::after { content:''; position:absolute; z-index:-1; width:360px; height:360px; border:1px solid #ffffff19; border-radius:50%; right:-105px; top:-155px; box-shadow:0 0 0 38px #ffffff08,0 0 0 78px #ffffff06; }
.hero-kicker { color:#70e4ca; font:600 11px 'Consolas',monospace; letter-spacing:.17em; text-transform:uppercase; }
.hero h1 { color:#f6f9ff; font-size:clamp(39px,7vw,70px); line-height:1.01; letter-spacing:-.055em; margin:20px 0 16px; max-width:700px; }
.hero p { color:#d5e1ef !important; font-size:17px; line-height:1.75; max-width:650px; }
.hero-chip { display:inline-flex; margin-top:12px; border:1px solid #ffffff30; border-radius:99px; padding:8px 12px; color:#d9e9f1; font-size:12px; }
.analyze-intro { display:flex; align-items:center; gap:16px; background:linear-gradient(110deg,#e9f6f4,#edf1fb 70%,#f8f0e9); border:1px solid #dce8ec; border-radius:17px; padding:18px 20px; margin:10px 0 20px; }
.analyze-icon { display:grid; place-items:center; flex:0 0 48px; height:48px; border-radius:15px; background:#fff; box-shadow:0 4px 14px #13284312; font-size:24px; }
.analyze-intro strong { color:var(--ink); font-size:16px; }
.analyze-intro p { color:var(--muted); margin:3px 0 0; }
.eyebrow { color:#087e74; font:600 10px 'Consolas',monospace; letter-spacing:.15em; text-transform:uppercase; }
.result-card { background:linear-gradient(120deg,#fff6ea,#fffaf4); border:1px solid #f0dfc5; border-radius:18px; padding:26px; margin:20px 0; box-shadow:0 8px 26px #29384c0a; }
.result-card.low-risk { background:linear-gradient(120deg,#eaf7f1,#f3f8f4); border-color:#d2e8dc; }
.result-title { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; color:var(--ink); font-size:clamp(24px,4vw,34px); font-weight:750; letter-spacing:-.035em; margin:12px 0 6px; }
.score-pill { border:1px solid #ead5b6; border-radius:999px; background:#fff; color:#80551d; padding:8px 13px; font:600 12px 'Consolas',monospace; letter-spacing:0; white-space:nowrap; }
.low-risk .score-pill { border-color:#b8dcca; color:#126b52; }
.result-copy { color:#526176; margin:8px 0 17px; }
.score-track { height:9px; overflow:hidden; background:#e2e8ed; border-radius:99px; }
.score-track span { display:block; height:100%; background:linear-gradient(90deg,#f3bd65,#dd8547); border-radius:99px; }
.low-risk .score-track span { background:linear-gradient(90deg,#5cc9a3,#14876c); }
.signal-card { border:1px solid var(--line); border-radius:12px; background:#fff; padding:15px; color:#304158; margin-bottom:8px; }
.stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div { color:var(--ink) !important; background:#fff !important; caret-color:#087e74 !important; border-color:#cbd5e0 !important; border-radius:12px !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:#18a88f !important; box-shadow:0 0 0 3px #19a89024 !important; }
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color:#738198 !important; opacity:1 !important; }
.stButton > button, .stDownloadButton > button { min-height:44px; border-radius:11px; border:1px solid #d6dfe8; background:#fff; color:var(--ink); font-weight:650; transition:all .16s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#19a890; color:#087e74; transform:translateY(-1px); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline:3px solid #19a89055; outline-offset:2px; }
.stButton > button[kind="primary"] { background:#117f72; border-color:#117f72; color:white; box-shadow:0 5px 14px #117f7228; }
.stButton > button[kind="primary"]:hover { background:#0b6b61; border-color:#0b6b61; color:white; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:15px; padding:17px; box-shadow:0 5px 16px #29384c08; }
[data-testid="stDataFrame"], [data-testid="stFileUploader"] section { border-radius:14px; }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #cbd6e2; border-radius:15px; padding:10px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#ffffffc9; border-radius:16px; border-color:#e1e7ef; box-shadow:0 5px 18px #29384c08; }
[data-testid="stSidebar"] { background:#eef2f7; border-right:1px solid #e0e6ed; }
[data-testid="stSidebar"] * { color:#26374e; }
[data-testid="stSidebar"] [data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#e0e6ed; }
footer { visibility:hidden; }
@media(max-width:700px) { .block-container { padding:1rem 1rem 3rem; } .hero { padding:28px 22px; border-radius:19px; } }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "training_metrics" not in st.session_state:
    st.session_state.training_metrics = None
if "page" not in st.session_state:
    st.session_state.page = "Home"
elif st.session_state.page == "Batch check":
    st.session_state.page = "Home"

model = get_model()
st.markdown('<div class="brand">VERITY <span style="color:#718096;font-weight:500;letter-spacing:.08em">NEWS CREDIBILITY STUDIO</span></div>', unsafe_allow_html=True)

pages = ["Home", "Analyze", "Train model", "History"]
nav_columns = st.columns(len(pages))
for nav_column, page_name in zip(nav_columns, pages):
    with nav_column:
        if st.button(page_name, key=f"nav_{page_name}", type="primary" if st.session_state.page == page_name else "secondary", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()

with st.sidebar:
    st.markdown("## Your workspace")
    st.caption("News credibility screening")
    st.divider()
    st.markdown("**CURRENT SCREENING MODE**")
    if model is not None:
        st.success("Custom model active")
        st.caption("TF-IDF + logistic regression")
    else:
        st.info("Language-pattern screening")
        st.caption("Train a model from labeled articles on the Train model page.")
    st.divider()
    st.caption("Screening gives you clues to investigate. Always verify claims with original sources and independent reporting.")

if st.session_state.page == "Home":
    st.markdown("""
    <section class="hero">
      <div class="hero-kicker">A smarter pause in the news cycle</div>
      <h1>Read the story.<br>Question the signal.</h1>
      <p>Screen an article in seconds, understand the language patterns behind the result, and train your own local model from labeled news.</p>
      <span class="hero-chip">PRIVATE BY DESIGN &nbsp;·&nbsp; NO AI API REQUIRED</span>
    </section>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">A CLEAR WAY TO START</div>', unsafe_allow_html=True)
    step_columns = st.columns(3)
    steps = [
        ("01", "Paste a story", "Add a headline, claim, or article on the Analyze page."),
        ("02", "Review the signals", "See the screening score and language patterns behind it."),
        ("03", "Check your sources", "Use the result as a prompt to investigate the original reporting."),
    ]
    for column, (number, title, description) in zip(step_columns, steps):
        with column, st.container(border=True):
            st.markdown(f'<div class="eyebrow">YOUR NEXT {number}</div>', unsafe_allow_html=True)
            st.subheader(title)
            st.write(description)

    st.write("")
    st.markdown('<div class="eyebrow">EXPLORE THE WORKSPACE</div>', unsafe_allow_html=True)
    route_columns = st.columns(3)
    routes = [
        ("Analyze an article", "Screen one story", "Analyze"),
        ("Train a model", "Use your Kaggle data", "Train model"),
        ("View history", "Revisit this session", "History"),
    ]
    for column, (title, description, destination) in zip(route_columns, routes):
        with column, st.container(border=True):
            st.subheader(title)
            st.caption(description)
            if st.button("Open page →", key=f"home_{destination}", use_container_width=True):
                st.session_state.page = destination
                st.rerun()

    st.info("Verity screens text patterns. It cannot establish whether an event happened or a claim is true.")

elif st.session_state.page == "Analyze":
    st.markdown('<div class="eyebrow">01 / SINGLE ARTICLE</div>', unsafe_allow_html=True)
    st.title("Analyze a story")
    st.markdown('<div class="analyze-intro"><div class="analyze-icon">🔎</div><div><strong>Curious about a story?</strong><p>Drop it in below. We’ll point out wording worth a second look.</p></div></div>', unsafe_allow_html=True)
    st.button("✨ Try a sample story", key="try_sample_story", on_click=fill_sample_story)
    with st.container(border=True):
        text = st.text_area("Article or claim", height=220, max_chars=50000, placeholder="Paste a headline, claim, or article text here…", key="article_text", help="Enter at least 20 characters. Maximum 50,000 characters.")
        st.caption(f"{len(text):,} / 50,000 characters")
        run = st.button("🔎 Check this story", type="primary", use_container_width=True)
    if run:
        if len(text.strip()) < 20:
            st.session_state.latest = None
            st.warning("Add at least 20 characters so Verity has enough text to screen.")
        else:
            try:
                with st.spinner("🔎 Checking the wording…"):
                    result = analyze(text, model)
                fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
                entry = {"time": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"), "preview": " ".join(text.split())[:180], "input_fingerprint": fingerprint, **result.to_dict()}
                st.session_state.history.insert(0, entry)
                st.session_state.history = st.session_state.history[:100]
                st.session_state.latest = entry
                st.session_state.scroll_to_result = True
            except ValueError as error:
                st.error(str(error))
    entry = st.session_state.get("latest")
    current_fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if entry and entry.get("input_fingerprint") != current_fingerprint:
        entry = None
    if entry:
        st.markdown('<div id="analysis-result"></div>', unsafe_allow_html=True)
        needs_review = entry["label"] in {"Review recommended", "Higher model score for misleading"}
        verdict = "Worth a closer look" if needs_review else "Fewer warning signs found"
        css_class = "result-card" if needs_review else "result-card low-risk"
        if entry["method"] == "Language-pattern screening":
            simple_signals = entry["signals"]
            method_description = "Checked for common attention-grabbing language patterns"
        else:
            simple_signals = ["The model matched patterns in its labeled training examples."]
            method_description = "Compared with examples from your trained dataset"
        st.markdown(
            f'<section class="{css_class}"><div class="eyebrow">SCREENING SUMMARY</div>'
            f'<div class="result-title">{verdict}<span class="score-pill">Score: {round(entry["score"] * 100)} / 100</span></div>'
            f'<p class="result-copy">{method_description}</p>'
            f'<div class="score-track"><span style="width:{entry["score"]:.0%}"></span></div></section>',
            unsafe_allow_html=True,
        )
        st.markdown("**Why it was flagged**" if needs_review else "**What the screen checked**")
        for signal in simple_signals:
            st.markdown(f'<div class="signal-card">{signal}</div>', unsafe_allow_html=True)
        st.info("Next step: Check the original source and compare the claim with independent, reliable reporting.")
        st.caption("The score is a screening measure, not the probability that a story is true or false. It cannot confirm facts.")
        if st.session_state.pop("scroll_to_result", False):
            components.html("""<script>
            const result = window.parent.document.getElementById('analysis-result');
            if (result) result.scrollIntoView({behavior: 'smooth', block: 'start'});
            </script>""", height=0)

elif st.session_state.page == "Batch check":
    st.markdown('<div class="eyebrow">02 / MULTIPLE ARTICLES</div>', unsafe_allow_html=True)
    st.title("Screen a collection")
    st.write("Run the same screening across a set of stories, then export a review-ready CSV.")
    st.markdown('<div class="analyze-intro"><div class="analyze-icon">▤</div><div><strong>One file, many stories</strong><p>Use a CSV with a text, content, article, news, or title column.</p></div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            cols = {str(c).strip().lower(): c for c in batch.columns}
            text_col = next((cols[n] for n in ("text", "content", "article", "news", "title") if n in cols), None)
            title_col = cols.get("title")
            if not text_col:
                st.error("No text column found. Rename a column to text, content, article, news, or title.")
            else:
                st.caption(f"{len(batch):,} rows found · Text column: {text_col}")
                if st.button("Analyze CSV", type="primary"):
                    if title_col and title_col != text_col:
                        texts = batch[[title_col, text_col]].fillna("").astype(str).agg(" ".join, axis=1)
                    else:
                        texts = batch[text_col].fillna("").astype(str)
                    outputs = []
                    progress = st.progress(0, text="Screening articles…")
                    for index, article in enumerate(texts):
                        try:
                            outputs.append(analyze(article, model).to_dict())
                        except ValueError as error:
                            outputs.append({"label": "Not enough text", "score": 0.0, "method": "Skipped", "signals": [], "explanation": str(error)})
                        if index % 100 == 0 or index + 1 == len(texts):
                            progress.progress((index + 1) / max(len(texts), 1), text=f"Screened {index + 1:,} of {len(texts):,}")
                    results = batch.copy()
                    results["screening_result"] = [item["label"] for item in outputs]
                    results["misleading_score"] = [item["score"] for item in outputs]
                    results["method"] = [item["method"] for item in outputs]
                    results["signals"] = ["; ".join(item["signals"]) for item in outputs]
                    st.session_state.batch_results = results
                if "batch_results" in st.session_state:
                    batch_results = st.session_state.batch_results
                    review_count = int(batch_results["screening_result"].isin(["Review recommended", "Higher model score for misleading"]).sum())
                    metric_columns = st.columns(3)
                    metric_columns[0].metric("Stories screened", f"{len(batch_results):,}")
                    metric_columns[1].metric("Worth reviewing", f"{review_count:,}")
                    metric_columns[2].metric("No strong signal", f"{len(batch_results) - review_count:,}")
                    st.dataframe(batch_results.head(100), use_container_width=True, hide_index=True)
                    st.caption(f"Showing up to 100 rows of {len(batch_results):,}. Download includes the full collection.")
                    st.download_button("Download all results as CSV", batch_results.to_csv(index=False).encode("utf-8-sig"), "verity_screening_results.csv", "text/csv")
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

elif st.session_state.page == "Train model":
    st.markdown('<div class="eyebrow">03 / OPTIONAL CUSTOM CLASSIFIER</div>', unsafe_allow_html=True)
    st.title("Build your custom model")
    st.write("Teach Verity from labeled examples, then use the trained model to screen new stories.")
    st.info("Supported text columns: `text`, `content`, `article`, `news`, or `title`. Supported label columns: `label`, `labels`, `class`, or `target`. Common labels include 0/FAKE and 1/REAL, or fake/real. Check your dataset's label definitions before training.")

    local_dataset_dir = Path(__file__).parent / "datasets"
    local_files = sorted(local_dataset_dir.glob("*.csv")) if local_dataset_dir.is_dir() else []
    local_names = {path.stem.lower() for path in local_files}
    if "fake" in local_names and ({"true", "real"} & local_names):
        with st.container(border=True):
            st.markdown("### Dataset ready to use")
            st.write(" · ".join(f"`{path.name}` · {path.stat().st_size / (1024 * 1024):.1f} MB" for path in local_files))
            st.caption("Labels will be assigned from the filenames. Title and article columns are combined automatically.")
            if st.button("Train from my datasets folder", type="primary"):
                try:
                    with st.spinner("Loading articles and training. Large files may take a few minutes…"):
                        local_frame = load_split_dataset(local_dataset_dir)
                        metrics = train_model(local_frame)
                    st.session_state.training_metrics = metrics
                    get_model.clear()
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not train from the local dataset: {error}")

    st.markdown("### Or bring your own labeled CSV")
    with st.container(border=True):
        dataset = st.file_uploader("Choose a training dataset", type=["csv"], key="training_upload")
    if dataset:
        try:
            frame = pd.read_csv(dataset)
            st.caption(f"{len(frame):,} rows · Columns: {', '.join(map(str, frame.columns))}")
            if st.button("Train and evaluate model", type="primary"):
                with st.spinner("Training and evaluating on a held-out sample…"):
                    metrics = train_model(frame)
                st.session_state.training_metrics = metrics
                get_model.clear()
                st.rerun()
        except Exception as error:
            st.error(f"Could not read or train this CSV: {error}")

    metrics = st.session_state.training_metrics
    if metrics:
        st.divider()
        st.markdown('<div class="eyebrow">YOUR MODEL IS READY</div>', unsafe_allow_html=True)
        st.success(f"Model trained on {metrics['examples']:,} usable articles.")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Held-out accuracy", f"{metrics['accuracy']:.1%}")
        metric_columns[1].metric("Misleading-class F1", f"{metrics['f1']:.1%}")
        metric_columns[2].metric("Examples", f"{metrics['examples']:,}")
        st.caption("Evaluation uses a stratified 80/20 split. Model performance depends on dataset quality and may not generalize to new topics or sources.")

    with st.expander("Tips for a useful training dataset"):
        st.markdown("- Include both real and fake examples with correct labels.\n- Use diverse topics and sources.\n- Remove duplicates where possible to reduce train/test overlap.\n- Check Kaggle's label definitions before using numeric labels.")

elif st.session_state.page == "History":
    st.markdown('<div class="eyebrow">04 / THIS SESSION</div>', unsafe_allow_html=True)
    st.title("Your recent screens")
    st.write("Review and export the articles you’ve analyzed in this session.")
    if st.session_state.history:
        history_df = pd.DataFrame([{"Time": entry["time"], "Result": entry["label"], "Score": f"{entry['score']:.0%}", "Method": entry["method"], "Text preview": entry["preview"]} for entry in st.session_state.history])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button("Export history as CSV", history_df.to_csv(index=False).encode("utf-8-sig"), "verity_history.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Your analyzed articles will show up here during this session.")

st.divider()
st.caption("VERITY · Python-powered news screening · Results should be independently verified.")
