"""Verified-source collection, SEC finance data, retrieval, and optional tool-using research agent.
Only official corporate domains are admitted to the research corpus.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import feedparser
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COMPANIES = ["Google", "OpenAI", "Anthropic", "NVIDIA", "Apple"]
DOMAINS = {
    "Google": ["blog.google", "deepmind.google", "google.com", "cloud.google.com", "abc.xyz"],
    "OpenAI": ["openai.com"],
    "Anthropic": ["anthropic.com"],
    "NVIDIA": ["nvidia.com", "blogs.nvidia.com"],
    "Apple": ["apple.com", "machinelearning.apple.com"],
}
RSS_FEEDS = {
    "Google": ["https://blog.google/technology/ai/rss/", "https://deepmind.google/blog/rss.xml"],
    "OpenAI": ["https://openai.com/news/rss.xml"],
    "Anthropic": [],  # Anthropic does not publish an official news RSS feed; use its newsroom.
    "NVIDIA": ["https://blogs.nvidia.com/feed/"],
    "Apple": ["https://www.apple.com/newsroom/rss-feed.rss"],
}
NEWSROOMS = {
    "Google": "https://blog.google/technology/ai/",
    "OpenAI": "https://openai.com/news/",
    "Anthropic": "https://www.anthropic.com/news",
    "NVIDIA": "https://blogs.nvidia.com/",
    "Apple": "https://www.apple.com/newsroom/",
}
REFERENCE_PAGES = {
    "Google": [("Google AI official updates", "https://blog.google/technology/ai/")],
    "OpenAI": [("OpenAI official news", "https://openai.com/news/")],
    "Anthropic": [("Anthropic official news", "https://www.anthropic.com/news")],
    "NVIDIA": [("NVIDIA official AI and company blog", "https://blogs.nvidia.com/")],
    "Apple": [("Apple Newsroom official updates", "https://www.apple.com/newsroom/")],
}
SEC_CIK = {"Google": "0001652044", "NVIDIA": "0001045810", "Apple": "0000320193"}
SEC_FORMS = ("10-K", "10-Q")
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
]
USER_AGENT = "Mozilla/5.0 (compatible; AcademicAILandscapeDashboard/1.0)"
TIMEOUT = 15
KEYWORDS = {
    "Models & products": ["model", "gemini", "claude", "gpt", "apple intelligence", "inference", "platform", "launch"],
    "Partnerships": ["partner", "collaborat", "agreement", "alliance", "joint", "integrat"],
    "Infrastructure & chips": ["gpu", "chip", "data center", "datacenter", "compute", "infrastructure", "accelerat"],
    "Policy & safety": ["safety", "regulat", "privacy", "policy", "security", "responsib"],
    "Financial performance": ["earnings", "revenue", "fiscal", "quarter", "financial", "invest"],
}


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_url(url):
    """Remove fragments and tracking query parameters; only use http(s)."""
    try:
        p = urlsplit(str(url).strip())
        if p.scheme not in ("http", "https") or not p.hostname:
            return ""
        return urlunsplit((p.scheme, p.netloc.lower(), p.path or "/", "", ""))
    except ValueError:
        return ""


def verified_domain(url, company):
    """Domain allowlist. No unverified aggregator articles can become sources."""
    try:
        p = urlsplit(url)
        host = (p.hostname or "").lower()
        return p.scheme == "https" and any(host == d or host.endswith("." + d) for d in DOMAINS[company])
    except (ValueError, KeyError):
        return False


def parse_date(value):
    if not value:
        return None
    try:
        s = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except (ValueError, TypeError):
        try:
            return parsedate_to_datetime(str(value)).date().isoformat()
        except (ValueError, TypeError, IndexError):
            return None


def strip_text(text, limit=5500):
    return re.sub(r"\s+", " ", BeautifulSoup(str(text or ""), "html.parser").get_text(" ")).strip()[:limit]


def get_session(sec_contact_email=None):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/rss+xml,application/xml,application/json"})
    if sec_contact_email:
        session.headers["User-Agent"] = f"Academic AI research dashboard {sec_contact_email}"
    return session


def extract_article(html, fallback_title=""):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("script,style,nav,footer,header,form,aside"):
        tag.decompose()
    title = (soup.find("h1") or soup.find("title"))
    title = strip_text(title.get_text(" ") if title else fallback_title, limit=240) or fallback_title
    node = soup.find("article") or soup.find("main") or soup.body or soup
    parts = []
    for item in node.find_all(["p", "h2", "h3", "li"], limit=180):
        t = strip_text(item.get_text(" "), 500)
        if len(t) > 35 and t not in parts:
            parts.append(t)
    body = " ".join(parts)[:5500]
    description = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    if len(body) < 90 and description:
        body = strip_text(description.get("content", ""), limit=5500)
    date_meta = soup.find("meta", attrs={"property": "article:published_time"})
    date_meta = date_meta or soup.find("meta", attrs={"name": "date"})
    date_meta = date_meta or soup.find("time", attrs={"datetime": True})
    date = parse_date((date_meta.get("content") or date_meta.get("datetime")) if date_meta else None)
    return title, body, date


def make_doc(company, url, title, content, date=None, kind="Newsroom article"):
    url = normalize_url(url)
    if not verified_domain(url, company):
        return None
    return {
        "id": f"{company.lower()}-{abs_hash(url)}",
        "company": company, "title": strip_text(title, 240), "url": url,
        "published": parse_date(date), "content": strip_text(content, 5500),
        "source_type": kind, "domain_verified": True,
    }


def abs_hash(s):
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def fetch_document(session, company, url, title="", published=None, source_type="Newsroom article"):
    url = normalize_url(url)
    if not verified_domain(url, company):
        return None
    try:
        r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        final_url = normalize_url(r.url)
        if not verified_domain(final_url, company):
            return None
        page_title, body, page_date = extract_article(r.text, title)
        if not body:
            return None
        return make_doc(company, final_url, title or page_title, body, published or page_date, source_type)
    except requests.RequestException:
        return None


def feed_candidates(session, company):
    results, warnings = [], []
    for feed_url in RSS_FEEDS[company]:
        try:
            r = session.get(feed_url, timeout=TIMEOUT)
            r.raise_for_status()
            parsed = feedparser.parse(r.content)
            if not parsed.entries:
                warnings.append(f"No entries in official feed: {feed_url}")
            for e in parsed.entries[:18]:
                link = normalize_url(e.get("link", ""))
                if verified_domain(link, company):
                    results.append((link, e.get("title", ""), e.get("published") or e.get("updated")))
        except requests.RequestException as e:
            warnings.append(f"Official feed unavailable: {feed_url} ({type(e).__name__})")
    return results, warnings


def newsroom_candidates(session, company):
    url = NEWSROOMS[company]
    try:
        r = session.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        if not verified_domain(normalize_url(r.url), company):
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        result, seen = [], set()
        for a in soup.find_all("a", href=True):
            link = normalize_url(urljoin(r.url, a["href"]))
            label = strip_text(a.get_text(" "), 240)
            path = urlsplit(link).path.rstrip("/").lower()
            if not verified_domain(link, company) or link in seen or len(label) < 14:
                continue
            if company == "Anthropic":
                valid = "/news/" in path or "/research/" in path
            elif company == "Apple":
                valid = "/newsroom/20" in path
            elif company == "Google":
                valid = "/technology/ai/" in path or "/blog/" in path
            elif company == "NVIDIA":
                valid = path.count("/") >= 2
            else:
                valid = "/index/" in path or "/news/" in path
            if valid and len(path) > 15 and link != normalize_url(url):
                seen.add(link)
                result.append((link, label, None))
            if len(result) >= 22:
                break
        return result
    except requests.RequestException:
        return []


def collect_news(max_per_company=8):
    """Returns verified official article text, with failure telemetry; no invented news."""
    session = get_session()
    docs, warnings, sources = [], [], {}
    for company in COMPANIES:
        candidates, feed_warnings = feed_candidates(session, company)
        warnings.extend(feed_warnings)
        candidates.extend(newsroom_candidates(session, company))
        seen = set()
        company_docs = []
        for url, title, date in candidates:
            if url in seen:
                continue
            seen.add(url)
            doc = fetch_document(session, company, url, title, date)
            if doc and len(doc["content"]) >= 90:
                company_docs.append(doc)
            if len(company_docs) >= max_per_company:
                break
        
        # Use official OpenAI RSS summaries if full articles are blocked.
        if company == "OpenAI" and not company_docs:
            try:
                feed_url = RSS_FEEDS["OpenAI"][0]
                response = session.get(feed_url, timeout=TIMEOUT)
                response.raise_for_status()

                for entry in feedparser.parse(response.content).entries:
                    link = normalize_url(entry.get("link", ""))
                    title = strip_text(entry.get("title", ""), 240)
                    summary = strip_text(entry.get("summary", ""))

                    if not verified_domain(link, "OpenAI"):
                        continue
                    if len(title) < 15 or len(summary) < 35:
                        continue

                    doc = make_doc(
                        "OpenAI", link, title, summary,
                        entry.get("published") or entry.get("updated"),
                        kind="Official publisher RSS excerpt"
                    )
                    if doc:
                        company_docs.append(doc)
                    if len(company_docs) >= max_per_company:
                        break

            except requests.RequestException as exc:
                warnings.append(
                    f"OpenAI RSS fallback failed: {type(exc).__name__}"
                )

        # Reliable primary-source context even if a feed is unavailable.
        if not company_docs:
            for title, url in REFERENCE_PAGES[company]:
                doc = fetch_document(session, company, url, title, source_type="Official reference page")
                if doc:
                    company_docs.append(doc)
        if not company_docs:
            warnings.append(f"No verified content could be fetched for {company}; no results fabricated.")
        sources[company] = len(company_docs)
        docs.extend(company_docs)
    docs.sort(key=lambda d: d.get("published") or "0000-00-00", reverse=True)
    return {"collected_at": now_utc(), "documents": docs, "source_counts": sources, "warnings": warnings}


def fetch_sec_annual_revenue(company, contact_email):
    """USD annual revenue from 10-K Company Facts (not stock market cap)."""
    if company not in SEC_CIK:
        return [], "Private / no comparable public-company 10-K revenue series"
    if not contact_email or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", contact_email):
        return [], "Add SEC_CONTACT_EMAIL in Streamlit secrets to access SEC data."
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{SEC_CIK[company]}.json"
    try:
        r = get_session(contact_email).get(url, timeout=25)
        r.raise_for_status()
        data = r.json()
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        entries = []
        for tag in REVENUE_TAGS:
            values = us_gaap.get(tag, {}).get("units", {}).get("USD", [])
            for item in values:
                try:
                    duration = (datetime.fromisoformat(item["end"]) - datetime.fromisoformat(item["start"])).days
                    if item.get("form") == "10-K" and item.get("fp") == "FY" and 320 <= duration <= 400 and item.get("val") is not None:
                        entries.append({"company": company, "end": item["end"], "fy": int(item["fy"]),
                                        "value_usd": int(item["val"]), "filed": item.get("filed", ""),
                                        "accession": item.get("accn", ""), "tag": tag,
                                        "url": url, "source": "SEC EDGAR XBRL Company Facts (10-K)"})
                except (KeyError, ValueError, TypeError):
                    pass
            if entries:
                break
        # Later-filed facts can restate prior years. Keep newest for each fiscal period end.
        latest = {}
        for e in entries:
            key = e["end"]
            if key not in latest or e["filed"] > latest[key]["filed"]:
                latest[key] = e
        return sorted(latest.values(), key=lambda e: e["end"])[-5:], None
    except (requests.RequestException, ValueError) as e:
        return [], f"SEC request failed for {company}: {type(e).__name__}"


def collect_financials(contact_email):
    rows, warnings = [], []
    for company in SEC_CIK:
        company_rows, warning = fetch_sec_annual_revenue(company, contact_email)
        rows.extend(company_rows)
        if warning:
            warnings.append(warning)
    return {"collected_at": now_utc(), "rows": rows, "warnings": warnings}


def topic_matches(document):
    text = (document.get("title", "") + " " + document.get("content", "")).lower()
    return [topic for topic, terms in KEYWORDS.items() if any(term in text for term in terms)]


def retrieve(query, documents, companies=None, limit=5):
    """TF-IDF retriever: lexical RAG retrieval over explicitly verified primary sources."""
    docs = [d for d in documents if d.get("domain_verified") and (not companies or d.get("company") in companies)]
    if not docs or not str(query).strip():
        return []
    corpus = [(d["title"] + " " + d.get("content", ""))[:6000] for d in docs]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=15000)
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [{**docs[i], "relevance": round(float(scores[i]), 4)} for i in order[:limit] if scores[i] > 0]
    except ValueError:
        return []


def make_evidence_answer(question, documents):
    matches = retrieve(question, documents, limit=5)
    if not matches:
        return "No matching passages found in the currently verified source corpus. Refresh or broaden the query.", []
    snippets = []
    for i, d in enumerate(matches[:4], 1):
        sentence_candidates = re.split(r"(?<=[.!?])\s+", d.get("content", ""))
        terms = set(re.findall(r"\b[a-z]{4,}\b", question.lower()))
        sentences = sorted(sentence_candidates, key=lambda s: sum(t in s.lower() for t in terms), reverse=True)
        excerpt = next((s.strip() for s in sentences if len(s) > 35), d["content"][:290])[:340]
        snippets.append(f"[{i}] {d['company']} — {d['title']}: {excerpt}")
    answer = "EXTRACTIVE EVIDENCE (not an AI-generated conclusion):\n\n" + "\n\n".join(snippets)
    return answer, matches[:4]


def run_research_agent(question, documents, financial_rows, api_key, model="gpt-4.1-mini"):
    """True model-driven tool selection via OpenAI Responses function calls.
    A tool must be called first. Models are forced to ground conclusions in returned records.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=35, max_retries=1)
    tools = [
        {"type": "function", "name": "search_official_sources",
         "description": "Search verified first-party company announcements and extract evidence with URLs.",
         "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                         "company": {"type": "string", "enum": COMPANIES + ["All"]}},
                        "required": ["query", "company"], "additionalProperties": False}},
        {"type": "function", "name": "get_annual_revenue",
         "description": "Get available SEC 10-K annual revenue in USD for the three public firms; no private-company estimates.",
         "parameters": {"type": "object", "properties": {"company": {"type": "string", "enum": COMPANIES + ["All"]}},
                        "required": ["company"], "additionalProperties": False}},
    ]
    instructions = (
        "You are a skeptical AI industry research agent for investors and strategists. "
        "Call the supplied source tools and answer ONLY with grounded facts returned by them. "
        "Link each substantive claim to the actual source URL in returned evidence. "
        "Distinguish issuer claims from independent verification; these are FIRST-PARTY SOURCES. "
        "Company announcements are not independent proof of outcomes. "
        "No unsupported market share, stock prices, token costs, projections, or private revenue. "
        "Explain missing/old data and fiscal-year differences. Keep answers under 450 words."
    )
    input_items = [{"role": "user", "content": question}]
    trace = []
    for iteration in range(4):
        response = client.responses.create(model=model, instructions=instructions, input=input_items,
                                           tools=tools, tool_choice="required" if iteration == 0 else "auto",
                                           max_output_tokens=2400, store=False)
        input_items += response.output
        calls = [x for x in response.output if x.type == "function_call"]
        if not calls:
            return response.output_text or "No response was generated.", trace
        for call in calls:
            args = json.loads(call.arguments)
            if call.name == "search_official_sources":
                target = args.get("company", "All")
                matches = retrieve(args.get("query", question), documents,
                                   companies=None if target == "All" else [target], limit=6)
                payload = [{k: d.get(k) for k in ("company", "title", "published", "url", "relevance")}
                           | {"excerpt": d.get("content", "")[:1100]} for d in matches]
            elif call.name == "get_annual_revenue":
                target = args.get("company", "All")
                payload = [r for r in financial_rows if target == "All" or r["company"] == target]
            else:
                payload = {"error": "Unknown tool"}
            trace.append({"tool": call.name, "arguments": args, "results": len(payload) if isinstance(payload, list) else 0})
            input_items.append({"type": "function_call_output", "call_id": call.call_id,
                                "output": json.dumps(payload, ensure_ascii=False)[:17000]})
    return "Tool-call limit reached; see the evidence and tool trace. No unsupported answer generated.", trace


def load_snapshot(path="data/latest.json"):
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data.get("news", {}).get("documents"), list):
            return None
        return data
    except (OSError, ValueError, TypeError):
        return None


def write_snapshot(path="data/latest.json", sec_contact_email=""):
    data = {"news": collect_news(), "financials": collect_financials(sec_contact_email), "generated_at": now_utc()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data
