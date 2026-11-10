"""AI Landscape Monitor: verified first-party research, SEC financials, RAG and tool-using agent."""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from research import (
    COMPANIES, KEYWORDS, NEWSROOMS, SEC_CIK, collect_financials, collect_news,
    load_snapshot, make_evidence_answer, retrieve, run_research_agent, topic_matches,
)

st.set_page_config(page_title="AI Landscape | Research Dashboard", page_icon="◈", layout="wide")
st_autorefresh(interval=3_600_000, key="hourly_background_refresh")

st.markdown("""
<style>
.block-container {padding-top: 2.0rem; max-width: 1380px;}
.hero {padding: 1.45rem 1.7rem; border: 1px solid #205268; border-radius: 17px;
       background: linear-gradient(110deg,#0e3042,#112538 60%,#082b34); margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.2rem; color: #f3fbff; margin: 0.15rem 0;}
.hero p {color: #b6d5dd; margin: 0.3rem 0;}
.eyebrow {font-size:.78rem; letter-spacing:.15em; color:#5de2ce; font-weight:700;}
.source-card {background:#142a3a;border:1px solid #26485b;border-radius:13px;padding:1rem 1.15rem;margin:.5rem 0;}
.source-card h4 {margin:0 0 .22rem 0;color:#f3fbff;}
.source-card p {margin: .2rem 0; color:#b6cbd7;font-size:.87rem;}
.smallnote {color:#b5c8d7;font-size:.89rem;}
</style>
""", unsafe_allow_html=True)


def secret(key, default=""):
    value = os.environ.get(key, default)
    try:
        value = st.secrets.get(key, value)
    except Exception:
        pass
    return str(value or "")


def age_hours(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except (ValueError, AttributeError):
        return 1e6


@st.cache_data(ttl=3600, show_spinner="Checking official company sources…")
def get_data(force_live=False, sec_contact_email=""):
    snapshot = load_snapshot()
    if snapshot and not force_live and age_hours(snapshot.get("generated_at", "")) < 12:
        return snapshot, "Scheduled verified-source snapshot"
    news = collect_news(max_per_company=8)
    financials = collect_financials(sec_contact_email)
    if not news["documents"] and snapshot:
        return snapshot, "Saved verified-source snapshot (live sources temporarily unavailable)"
    return {"generated_at": news["collected_at"], "news": news, "financials": financials}, "Live first-party collection"


with st.sidebar:
    st.markdown("### ◈ Research settings")
    companies = st.multiselect("Companies", COMPANIES, default=COMPANIES)
    topic = st.selectbox("Topic", ["All topics"] + list(KEYWORDS))
    search_text = st.text_input("Article search", placeholder="GPU, partnership, model launch…")
    st.markdown("---")
    st.caption("Auto-refresh is hourly while a viewer has the app open. The included GitHub Actions workflow also attempts a source refresh every 6 hours.")
    if st.button("↻ Refresh from official websites", use_container_width=True):
        st.cache_data.clear()
        st.session_state["force_live_once"] = True
        st.rerun()
    st.caption("**Data policy:** first-party corporate announcements and SEC filings only. No invented stock prices, market shares or private-company revenues.")
    st.markdown("### Official newsrooms")
    for company in COMPANIES:
        st.markdown(f"[{company}]({NEWSROOMS[company]})")

force_live = st.session_state.pop("force_live_once", False)
sec_email = secret("SEC_CONTACT_EMAIL")
try:
    bundle, data_mode = get_data(force_live=force_live, sec_contact_email=sec_email)
except Exception as err:
    st.error("Unable to load the research corpus. Try refreshing. " + type(err).__name__)
    st.stop()

news = bundle.get("news", {})
financials = bundle.get("financials", {})
all_docs = news.get("documents", [])
financial_rows = financials.get("rows", [])
shown = [d for d in all_docs if d["company"] in companies]
if topic != "All topics":
    shown = [d for d in shown if topic in topic_matches(d)]
if search_text:
    q = search_text.lower()
    shown = [d for d in shown if q in (d.get("title", "") + " " + d.get("content", "")).lower()]

def display_date(date_str):
    return date_str or "Date not disclosed"


def source_card(doc):
    st.markdown(f"<div class='source-card'><div class='smallnote'>{doc['company']} · "
                f"{display_date(doc.get('published'))} · {doc.get('source_type', '')}</div>"
                f"<h4>{html_escape(doc['title'])}</h4>"
                f"<p>{html_escape(doc.get('content', '')[:280])}…</p></div>",
                unsafe_allow_html=True)
    st.link_button("Read original verified source ↗", doc["url"], use_container_width=False)


def html_escape(text):
    import html
    return html.escape(str(text), quote=True)


st.markdown("""
<div class="hero">
  <div class="eyebrow">AI INDUSTRY INTELLIGENCE · RESEARCH DASHBOARD</div>
  <h1>AI Landscape Monitor</h1>
  <p>Google · OpenAI · Anthropic · NVIDIA · Apple</p>
  <p>Official announcements, SEC filings, source-grounded retrieval and an optional research agent.</p>
</div>
""", unsafe_allow_html=True)

st.caption(f"**Collection mode:** {data_mode}  |  **Snapshot generated (UTC):** {bundle.get('generated_at', 'Unknown')}  |  "
           "**Verification:** URL host must match an approved official corporate domain")

if not all_docs:
    st.warning("No verified articles were accessible. The dashboard will not fabricate news. "
               "Check deployment internet access, the official source links, or run the scheduled refresh workflow.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Verified source pages", len(all_docs))
k2.metric("Companies monitored", sum(bool(news.get("source_counts", {}).get(c, 0)) for c in COMPANIES))
k3.metric("SEC annual observations", len(financial_rows))
k4.metric("Source categories", len(KEYWORDS))

pages = st.tabs(["Overview", "Company comparisons", "Financial evidence", "Research agent + RAG", "Sources & methodology"])

with pages[0]:
    st.subheader("Overview for business leaders and market researchers")
    st.info("**Interpretation:** This dashboard measures the volume and content of *official company announcements*, "
            "not industry market share, business outcomes, or independent claims verification.")
    if shown:
        df = pd.DataFrame([{"Company": d["company"], "Source": d["title"], "Date": d.get("published"),
                            "Topics": ", ".join(topic_matches(d)) or "Other"} for d in shown])
        left, right = st.columns([1, 1.25])
        with left:
            tally = df.groupby("Company").size().reset_index(name="Verified pages")
            fig = px.bar(tally, x="Company", y="Verified pages", color="Company",
                         title="Official sources in current corpus", text="Verified pages")
            fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=55, b=5))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            topics = Counter(tag for d in shown for tag in topic_matches(d))
            if topics:
                rows = [{"Topic": t, "Documents mentioning it": n} for t, n in topics.items()]
                fig = px.bar(pd.DataFrame(rows), x="Documents mentioning it", y="Topic", orientation="h",
                             title="Topic mention signals in official pages")
                fig.update_layout(margin=dict(l=5, r=5, t=55, b=5))
                st.plotly_chart(fig, use_container_width=True)
        dated = [d for d in shown if d.get("published")]
        if dated:
            dated_df = pd.DataFrame([{"Month": d["published"][:7], "Company": d["company"], "Count": 1} for d in dated])
            by_month = dated_df.groupby(["Month", "Company"], as_index=False)["Count"].sum()
            fig = px.line(by_month, x="Month", y="Count", color="Company", markers=True,
                          title="Official publications by month (only sources with dates)")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Recent source-grounded developments")
        for d in shown[:6]:
            with st.expander(f"{d['company']} · {d['title']}", expanded=False):
                st.write(d.get("content", "")[:1400])
                st.caption("Published: " + display_date(d.get("published")))
                st.link_button("View original", d["url"])
    else:
        st.info("No sources match the selected filters. Clear the sidebar filters or refresh.")
    st.markdown("#### Decisions this report supports")
    st.write("**Strategists:** track product and partnership announcements. **Investors:** inspect SEC-reported historical revenue "
             "for the public firms only. **Researchers:** trace claims back to official sources and ask citation-grounded questions.")

with pages[1]:
    st.subheader("Compare competitive activity")
    if shown:
        rows = []
        for d in shown:
            for t in topic_matches(d):
                rows.append({"Company": d["company"], "Topic": t, "Source count": 1})
        if rows:
            grouped = pd.DataFrame(rows).groupby(["Company", "Topic"], as_index=False)["Source count"].sum()
            fig = px.bar(grouped, x="Company", y="Source count", color="Topic", barmode="group",
                         title="Verified corporate source coverage by topic")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Counts depend on the limited collected corpus and source publishing volume; "
                   "they do not measure market share or provide an exhaustive industry ranking.")
    cols = st.columns(2)
    for i, company in enumerate(COMPANIES):
        with cols[i % 2]:
            company_docs = [d for d in shown if d["company"] == company]
            with st.container(border=True):
                st.markdown(f"#### {company}")
                st.caption(f"{len(company_docs)} matching verified records")
                if company_docs:
                    d = company_docs[0]
                    st.write(f"**Latest in selected corpus:** {d['title']}")
                    st.caption(display_date(d.get("published")))
                    st.link_button("Read primary source", d["url"], key=f"co-{company}")
                else:
                    st.write("No matching records; source may be unavailable or filtered out.")

with pages[2]:
    st.subheader("SEC filings: audited-source financial evidence")
    st.write("These figures come from SEC EDGAR structured XBRL annual-revenue disclosures (Form 10-K). "
             "Google is represented by Alphabet. OpenAI and Anthropic are private companies and have "
             "no directly comparable 10-K series here.")
    if financial_rows:
        fdf = pd.DataFrame(financial_rows).copy()
        fdf["Revenue (USD billions)"] = fdf["value_usd"] / 1e9
        fdf["Fiscal period end"] = fdf["end"]
        fig = px.line(fdf.sort_values("Fiscal period end"), x="Fiscal period end", y="Revenue (USD billions)",
                      color="company", markers=True, title="Annual revenue in SEC 10-K filings (fiscal years differ)")
        fig.update_layout(legend_title_text="Company")
        st.plotly_chart(fig, use_container_width=True)
        table = fdf[["company", "Fiscal period end", "Revenue (USD billions)", "filed", "url"]].rename(
            columns={"company": "Company", "filed": "SEC filed on", "url": "Primary SEC source"})
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"Primary SEC source": st.column_config.LinkColumn()})
        st.warning("**Comparability:** fiscal calendars differ, SEC facts may be restated, and the chart is not a stock-price "
                   "or profitability comparison. Values may reflect issuer-specific XBRL tags.")
    else:
        st.info("SEC annual data is not populated. Set `SEC_CONTACT_EMAIL` in Streamlit **Secrets**, "
                "or in GitHub Actions repository secrets, then refresh. Use your real contact email as SEC requires.")
    st.caption("Official SEC API: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json")
    for warning in financials.get("warnings", []):
        st.caption("Source status: " + warning)

with pages[3]:
    st.subheader("Source-grounded research assistant")
    st.markdown("**RAG pipeline:** verified-source ingestion → article text → TF-IDF retrieval → "
                "evidence display → optional LLM tool-calling agent → URL-cited answer.")
    st.caption("The retrieval engine works without a paid key. Generative answers and tool-calling "
               "require an OpenAI API key with API billing enabled, separate from ChatGPT subscription.")
    question = st.text_input("Ask an evidence-based research question", value="How are Google and OpenAI competing in AI models?", key="ask")
    answer_mode = st.radio("Mode", ["Free evidence retrieval", "OpenAI tool-using research agent"], horizontal=True)
    model = secret("OPENAI_MODEL", "gpt-4.1-mini")
    api_key = secret("OPENAI_API_KEY")
    if answer_mode.startswith("OpenAI") and not api_key:
        st.info("To use the actual model-driven agent, add OPENAI_API_KEY to the app's Streamlit Secrets. "
                "Never paste your API key in a public GitHub file.")
    if st.button("Run research", type="primary", key="run_research", disabled=not bool(question.strip())):
        with st.spinner("Retrieving verified evidence…"):
            if answer_mode.startswith("Free"):
                answer, matches = make_evidence_answer(question, shown or all_docs)
                trace = [{"tool": "retrieve_tfidf", "arguments": {"query": question}, "results": len(matches)}]
            elif not api_key:
                answer, matches = "No API key configured; showing evidence-only retrieval instead.", retrieve(question, shown or all_docs)
                trace = []
            else:
                try:
                    answer, trace = run_research_agent(question, shown or all_docs, financial_rows, api_key, model=model)
                    matches = retrieve(question, shown or all_docs)
                except Exception as exc:
                    answer, matches = make_evidence_answer(question, shown or all_docs)
                    answer = (f"The OpenAI agent was unavailable ({type(exc).__name__}). "
                              "Showing extractive evidence instead.\n\n" + answer)
                    trace = []
            st.session_state["research_output"] = (answer, matches, trace)
    if "research_output" in st.session_state:
        answer, matches, trace = st.session_state["research_output"]
        st.markdown("#### Answer / evidence")
        st.text(answer) if answer_mode.startswith("Free") else st.write(answer)
        st.markdown("#### Retrieved primary-source citations")
        for i, d in enumerate(matches, 1):
            st.markdown(f"**[{i}] [{d['title']}]({d['url']})** — {d['company']} · "
                        f"{display_date(d.get('published'))} · similarity {d.get('relevance', 0):.3f}")
        with st.expander("Show agent / retrieval audit trace"):
            st.json(trace)
    st.markdown("#### Questions worth testing")
    st.write("• Which firms announced new AI products recently?  • What partnerships are mentioned in official releases?  "
             "• What can we verify about NVIDIA's infrastructure role?  • How do Alphabet, Apple and NVIDIA annual revenues differ?")

with pages[4]:
    st.subheader("Trust, citations and reproducibility")
    st.markdown("**Scope:** Google (Alphabet), OpenAI, Anthropic, NVIDIA and Apple. "
                "Only approved official corporate domains are indexed; SEC EDGAR provides public-company financial figures. "
                "This verifies *source origin*, not that corporate marketing claims are independently true.")
    st.markdown("**Update mechanism:** GitHub Actions scheduled refresh every 6 hours (subject to GitHub scheduling, "
                "permissions, and inactivity limits); live checks when the dashboard starts and hourly reruns while open. "
                "Source failures are surfaced rather than replaced with fabricated observations.")
    st.markdown("**RAG:** scikit-learn TF-IDF over collected article text. **Agent:** OpenAI Responses API function "
                "calling with source-search and SEC-revenue tools; outputs require citations to URLs. "
                "**Limits:** incomplete media coverage, newsrooms may change markup, corporate-source bias, "
                "and missing private-company data. Financial figures reflect fiscal-year reporting calendars.")
    st.markdown("#### Full source register")
    if shown:
        register = pd.DataFrame([{
            "Company": d["company"], "Title": d["title"], "Published": d.get("published") or "Unspecified",
            "Type": d.get("source_type", ""), "Verified": "Official domain", "URL": d["url"]
        } for d in shown])
        st.dataframe(register, hide_index=True, use_container_width=True, column_config={"URL": st.column_config.LinkColumn()})
        st.download_button("Download current source register (CSV)", register.to_csv(index=False).encode("utf-8"),
                           file_name="verified_ai_sources.csv", mime="text/csv")
    st.markdown("#### Collector health")
    for company in COMPANIES:
        st.write(f"**{company}:** {news.get('source_counts', {}).get(company, 0)} verified article/page records")
    if news.get("warnings"):
        with st.expander("Collection warnings and feed status"):
            for warning in news["warnings"]:
                st.caption(warning)
    st.markdown("#### Research ethics")
    st.caption("AI summaries can be wrong; cross-check cited articles. No claims of independent validation, "
               "exhaustive indexing, or investment advice. For a class presentation, show primary-source links "
               "and the tool audit trace.")

st.divider()
st.caption("Developed as a business analytics research demonstration · Official-source attribution · "
           "Every graph reflects retrieved records, not simulated data")
