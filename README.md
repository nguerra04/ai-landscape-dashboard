# AI Landscape Monitor — MIS research assignment

A deployable **Streamlit business intelligence dashboard** covering **Google, OpenAI, Anthropic, NVIDIA and Apple**. The app gathers only verified official corporate-domain source content, can retrieve SEC annual revenue from EDGAR for the three publicly traded companies, supports **TF-IDF retrieval-augmented research**, and (with an optional paid API key) runs an **OpenAI Responses API tool-using research agent**.

## What it demonstrates

- **Acquisition:** corporate RSS feeds, official newsroom fallback HTML, and SEC XBRL Company Facts API.
- **Source checking:** domain allowlist after redirects, source URLs, dated articles when published dates are available; warnings on failed sources.
- **Analysis:** company/topic coverage, source-level trends, historical SEC annual revenue with explicit comparability warnings.
- **RAG:** real TF-IDF retrieval over the collected article text. No API key required to retrieve/cite passages.
- **Agent:** genuine LLM-selected tools (`search_official_sources`, `get_annual_revenue`) with a visible tool trace. Requires an OpenAI API key and API billing.
- **Refresh:** a GitHub Actions workflow attempts source ingestion every 6 hours, plus Streamlit hourly in-session reload and manual refresh. GitHub may delay/discontinue unattended scheduled jobs on inactive repositories.

## Start in Colab

1. Open the included `AI_Landscape_Colab_Setup.ipynb` in Google Colab.
2. Run the cells in order. This generates the Python files, checks code validity and creates a ZIP for GitHub.
3. No paid API key is needed for the dashboard, source retrieval, or official article search. Optional OpenAI agent calls incur usage costs.

## Deploy with Streamlit Community Cloud

1. Create a **new GitHub repository**, e.g. `ai-landscape-dashboard`.
2. Upload the **contents** of this folder to its root: `app.py`, `research.py`, `requirements.txt`, `refresh_data.py`, `.streamlit/config.toml`, `data/`, and `.github/workflows/refresh.yml`.
3. Go to https://share.streamlit.io/ → **Create app** → **Yup, I have an app**.
4. Choose your GitHub repository, branch `main`, and entrypoint `app.py`.
5. Click **Deploy** and share the `*.streamlit.app` URL.
6. For SEC annual revenue, add your genuine contact email as a GitHub Actions repository secret called `SEC_CONTACT_EMAIL` and also as a Streamlit secret. This respects SEC automated access identification. Leave blank if you don't need SEC data.
7. For the optional agent, add `OPENAI_API_KEY` to Streamlit **App settings → Secrets**. Never commit API keys to GitHub. Optional `OPENAI_MODEL = "gpt-4.1-mini"`.
8. In GitHub, open **Actions** → **Refresh verified AI industry research** → **Run workflow** to populate `data/latest.json`, then check the report for source coverage and warnings. If GitHub Actions asks you to enable workflows, do so. Repository workflow permissions must permit writing contents for snapshots to be committed.

Example Streamlit secrets (paste into Streamlit's secure secrets settings, not a GitHub file):

```toml
SEC_CONTACT_EMAIL = "your-real-email@example.com"
# Optional, uses paid OpenAI API credits:
# OPENAI_API_KEY = "sk-..."
# OPENAI_MODEL = "gpt-4.1-mini"
```

## Architecture

```text
Official RSS / HTML newsrooms ----> verify official URL + full page ----> article corpus
SEC Company Facts API ----------> 10-K annual revenue ----> historical chart
article corpus ------------------> TF-IDF retrieval -----> cited extracts (free)
article corpus + revenue -------> OpenAI function tools --> sourced answer + trace (paid optional)
GitHub Actions (6h cron) -------> data/latest.json ------> deployed Streamlit app
```

## Trust and known limitations

- Source provenance is verified through an allowlisted official host, **not independent fact-checking**. Corporate statements can reflect marketing bias.
- Official newsrooms may change their RSS endpoints/HTML structure. Collector warnings report missing data instead of inserting fake results.
- Article dates may be absent; the app labels them as undisclosed and omits them from time series.
- News coverage is not exhaustive. Announcement counts are not product adoption, revenue, market share, or return on investment.
- Only Google/Alphabet, Apple and NVIDIA have SEC filings here. The report deliberately excludes unsupported private-company financial estimates and stock prices.
- Historical SEC 10-K annual revenues reflect non-identical fiscal calendars and sometimes restated disclosures.
- Auto-refresh while the Streamlit page is open differs from unattended server-side cron. The included GitHub Actions job provides the latter, subject to GitHub service restrictions.
- OpenAI API usage is separate from any ChatGPT subscription. Leave the key unset to keep the app's AI retrieval free.

## Presentation suggested sequence (5 minutes)

- 0:00–0:45: stakeholder question and five-company scope.
- 0:45–1:45: show official-source comparisons and a linked announcement.
- 1:45–2:40: compare fiscal revenue across the public firms and explain private-data gaps.
- 2:40–4:10: run an RAG query, show retrieved sources, optionally run the agent and open the tool trace.
- 4:10–5:00: show source verification/auto-refresh limitations and business implications.

## Tests

`python -m pytest -q` (install `pytest` if not installed).
