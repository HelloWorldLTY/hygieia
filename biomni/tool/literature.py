import re
import os
import time
from typing import Optional, Tuple, Dict, Any
from io import BytesIO
from urllib.parse import urljoin

import PyPDF2
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import os
from openai import AzureOpenAI
import pandas as pd
from openai import OpenAI
from anthropic import AnthropicFoundry

#run_diagnosis
#run_generank

import pandas as pd
from openai import OpenAI
from anthropic import AnthropicFoundry
from tqdm import tqdm

# ─────────────────────────────────────────────
# Azure AI Foundry credentials for Claude
# ─────────────────────────────────────────────
AZURE_BASE_URL = ""
AZURE_API_KEY = ""


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def _extract_tag(text: str, tag: str) -> str:
    """Extract content inside XML-like tags, e.g. <solution>...</solution>."""
    if not text:
        return ""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_yes_no(text: str) -> str:
    answer = _extract_tag(text, "solution") or _extract_tag(text, "answer") or text
    answer = answer.upper()
    if "YES" in answer:
        return "YES"
    return "NO"


# ─────────────────────────────────────────────
# Prompt builders  (improved over GPT version)
# ─────────────────────────────────────────────

def _make_diagnosis_prompt(
    input_patient_information: str,
    prior_knowledge: str,
    disease_num: str,
) -> str:
    context_section = (
        f"\n\n## Supporting Literature / Context\n{prior_knowledge}"
        if prior_knowledge.strip()
        else ""
    )
    return f"""You are an expert clinical geneticist and rare disease specialist with deep knowledge of OMIM, Orphanet, and CCRD disease classifications.

## Patient Phenotypes
{input_patient_information}
{context_section}

## Task
Systematically analyze these phenotypes and identify the top {disease_num} most likely rare disease diagnosis(es).

Follow this structured approach:
1. **Phenotype grouping** – Cluster phenotypes by organ system; flag the most distinctive or pathognomonic features.
2. **Pattern recognition** – Infer likely inheritance mode, age of onset, and disease trajectory.
3. **Differential diagnosis** – List candidate diseases ranked by how well they explain *all* phenotypes; briefly note key supporting and contradicting evidence for each.
4. **Final selection** – Choose the most parsimonious diagnosis that covers the broadest phenotype set.

Formatting requirements:
- Disease names must follow the official OMIM / Orphanet / CCRD nomenclature (not synonyms or abbreviations).
- Prefer specific disease names over broad category labels.

Document your full reasoning within <thinking></thinking>.
List the final top {disease_num} disease name(s) within <solution></solution>, one per line, no numbers or bullets."""


def _make_verifier_prompt(
    prior_knowledge: str,
    input_patient_information: str,
    diagnosis_text: str,
) -> str:
    context_section = (
        f"\n## Supporting Context\n{prior_knowledge}"
        if prior_knowledge.strip()
        else ""
    )
    return f"""You are an expert clinical geneticist critically reviewing a rare disease diagnosis.

## Patient Phenotypes
{input_patient_information}
{context_section}

## Diagnosis Under Review
{diagnosis_text}

## Your Evaluation Criteria
1. Does the proposed disease explain **all** listed phenotypes?
2. Are there phenotypes that contradict the proposed diagnosis?
3. Is a more specific or more likely disease possible given this phenotype combination?
4. Is the disease name correctly formatted per OMIM/Orphanet conventions?

Document your critical reasoning within <thinking></thinking>.
- If the diagnosis is well-supported: output YES within <solution></solution>.
- If the diagnosis needs improvement: output NO within <solution></solution> AND provide 1-3 specific alternative diseases to consider within <feedback></feedback>, with a brief rationale for each."""


def _make_retry_prompt(
    input_patient_information: str,
    prior_knowledge: str,
    previous_diagnosis: str,
    verifier_feedback: str,
    disease_num: str,
) -> str:
    context_section = (
        f"\n\n## Supporting Literature / Context\n{prior_knowledge}"
        if prior_knowledge.strip()
        else ""
    )
    return f"""You are an expert clinical geneticist. Your previous diagnosis was reviewed by a specialist and found to need revision.

## Patient Phenotypes
{input_patient_information}
{context_section}

## Previous Diagnosis (rejected)
{previous_diagnosis}

## Reviewer Feedback
{verifier_feedback}

## Instructions
Incorporate the reviewer's feedback and produce a revised diagnosis.
Focus especially on:
- Phenotypes that were unexplained or contradicted by the previous answer
- The specific alternative diseases the reviewer mentioned
- Using the most precise OMIM/Orphanet disease name

Document your revised reasoning within <thinking></thinking>.
List the revised top {disease_num} disease name(s) within <solution></solution>, one per line."""


def _call_claude(
    client,
    model_name,
    prompt,
    max_tokens: int = 4096,
) -> str:
    """Call Claude via Azure AI Foundry and return the text response."""
    verify_kwargs = {
        "model": model_name,
        "input": prompt,
    }
    verify_response = client.responses.create(**verify_kwargs)
    verifier_text = verify_response.output_text
    return verifier_text

def run_diagnosis(
    input_patient_information: str,
    knowinte = True,
    knowgene = None,
    disease_num: str = "one",
    model: str = "gpt-5.4-mini",
    verifier_model: str = "gpt-5.4-mini",
    max_verify_rounds: int = 3,
) -> Tuple[str, str]:
    """
    Rare disease diagnosis via Claude on Azure AI Foundry + iterative self-verification.

    Improvements over the GPT-based version
    ----------------------------------------
    * Separate verifier call reduces shared bias (different prompt framing).
    * Verifier returns structured <feedback> with specific alternative diagnoses.
    * Retry prompt feeds that feedback back, giving targeted guidance rather than a generic retry.
    * prior_knowledge (pre-retrieved literature context) is integrated into every prompt.

    Returns
    -------
    (model_output, prior_knowledge)
    """
    if not _safe_str(input_patient_information):
        raise ValueError("input_patient_information must be a non-empty string.")

    client = OpenAI(
        base_url=AZURE_BASE_URL,
        api_key=AZURE_API_KEY
    )
    
    prior_knowledge = ""
    gene_knowledge = ""
    if knowinte:
        prior_knowledge = "Context Information: "
        prior_knowledge += query_scholar(input_patient_information)
        prior_knowledge += search_google(input_patient_information, num_results=3)
        prior_knowledge += query_pubmed(input_patient_information, max_retries=10)
    if knowgene != None:
        prior_knowledge += "Risk Gene Information: The detected risk gene is: " + knowgene +"."
#     if geneinfo!=None and geneinfo!='nan':
#         prior_knowledge += "The function of this gene is: "+geneinfo

    # ── Initial diagnosis ──────────────────────────────────────────────────────
    diagnosis_prompt = _make_diagnosis_prompt(
        input_patient_information=input_patient_information,
        prior_knowledge=prior_knowledge,
        disease_num=disease_num,
    )
    modelout = _call_claude(client, model, diagnosis_prompt)

    # ── Self-verification loop ─────────────────────────────────────────────────
    for round_idx in range(max_verify_rounds):
        print(f"  [verify {round_idx + 1}/{max_verify_rounds}]", end=" ", flush=True)

        solution_text = _extract_tag(modelout, "solution") or modelout

        verifier_prompt = _make_verifier_prompt(
            prior_knowledge=prior_knowledge,
            input_patient_information=input_patient_information,
            diagnosis_text=solution_text,
        )
        try:
            verifier_out = _call_claude(client, verifier_model, verifier_prompt, max_tokens=1024)
            verdict = _normalize_yes_no(verifier_out)
            feedback = _extract_tag(verifier_out, "feedback") or ""
        except Exception as e:
            print(f"verification error: {e}")
            verdict = "NO"
            feedback = ""

        if verdict == "YES":
            print("accepted.")
            break

        print("rejected. Retrying with feedback.")
        fallback_feedback = (
            "The diagnosis needs reconsideration. "
            "Please re-examine the phenotypes more carefully and consider alternative rare diseases."
        )
        retry_prompt = _make_retry_prompt(
            input_patient_information=input_patient_information,
            prior_knowledge=prior_knowledge,
            previous_diagnosis=solution_text,
            verifier_feedback=feedback if feedback else fallback_feedback,
            disease_num=disease_num,
        )
        modelout = _call_claude(client, model, retry_prompt)

    return modelout


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def _extract_tag(text: str, tag: str) -> str:
    """Extract content inside XML-like tags, e.g. <solution>...</solution>."""
    if not text:
        return ""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_yes_no(text: str) -> str:
    answer = _extract_tag(text, "solution") or _extract_tag(text, "answer") or text
    answer = answer.upper()
    if "YES" in answer:
        return "YES"
    return "NO"


def _extract_gene_list(text: str) -> str:
    """
    Try to extract the gene list from <solution> or <answer>.
    Falls back to raw text.
    """
    out = _extract_tag(text, "solution") or _extract_tag(text, "answer") or text
    return out.strip()


def _make_gene_prioritization_prompt(
    input_patient_information: str,
    prior_knowledge: str = "",
    gene_num: str = "10",
    knowdisease: Optional[str] = None,
) -> str:
    context_section = (
        f"\n\n## Supporting Literature / Context\n{prior_knowledge}"
        if _safe_str(prior_knowledge)
        else ""
    )

    if _safe_str(knowdisease):
        task_line = (
            f"Identify the top {gene_num} most likely causal / high-priority genes "
            f"for the disease context: {knowdisease}."
        )
    else:
        task_line = (
            f"Identify the top {gene_num} most likely causal / high-priority genes "
            f"to test based on the phenotype profile."
        )

    return f"""You are an expert clinical geneticist, molecular biologist, and rare disease gene prioritization specialist.

## Patient Phenotypes / Clinical Information
{input_patient_information}
{context_section}

## Task
{task_line}

Use this structured approach:
1. **Phenotype synthesis** – summarize the most informative phenotypes and organ systems.
2. **Disease mechanism inference** – infer likely inheritance pattern, pathway, tissue relevance, age of onset, and syndrome class.
3. **Gene prioritization** – rank candidate genes by how well they explain the phenotype constellation.
4. **Evidence review** – for each top gene, briefly note supporting evidence and key phenotype matches.
5. **Final ranking** – provide the most plausible gene list in ranked order.

Requirements:
- Use official HGNC gene symbols only.
- Prefer specific causal genes over broad pathway descriptions.
- Rank genes from most likely to least likely.
- Do not output disease names in the final answer unless needed briefly in reasoning.

Return:
- Full reasoning inside <thinking></thinking>
- Final ranked gene list inside <solution></solution>, one gene symbol per line, no numbering, no bullets
"""


def _make_gene_verifier_prompt(
    input_patient_information: str,
    gene_output_text: str,
    prior_knowledge: str = "",
    knowdisease: Optional[str] = None,
) -> str:
    context_section = (
        f"\n## Supporting Literature / Context\n{prior_knowledge}"
        if _safe_str(prior_knowledge)
        else ""
    )

    disease_section = (
        f"\n## Known Disease Context\n{knowdisease}" if _safe_str(knowdisease) else ""
    )

    return f"""You are an expert clinical geneticist critically reviewing a ranked gene prioritization result.

## Patient Phenotypes / Clinical Information
{input_patient_information}
{disease_section}
{context_section}

## Gene Ranking Under Review
{gene_output_text}

## Evaluation Criteria
1. Do the proposed genes explain the phenotype profile well?
2. Are the genes biologically and clinically plausible?
3. Are there more likely genes missing from the ranking?
4. Are the gene symbols valid HGNC-style symbols?
5. Is the ranking reasonably ordered from strongest to weaker candidates?

Return:
- Critical reasoning inside <thinking></thinking>
- If the ranking is acceptable: output YES inside <solution></solution>
- If the ranking needs improvement: output NO inside <solution></solution>
- If NO, also provide 1-10 better replacement or missing genes inside <feedback></feedback>, one gene per line, with short rationale if useful
"""


def _make_gene_retry_prompt(
    input_patient_information: str,
    previous_output: str,
    verifier_feedback: str,
    prior_knowledge: str = "",
    gene_num: str = "10",
    knowdisease: Optional[str] = None,
) -> str:
    context_section = (
        f"\n\n## Supporting Literature / Context\n{prior_knowledge}"
        if _safe_str(prior_knowledge)
        else ""
    )

    disease_section = (
        f"\n## Known Disease Context\n{knowdisease}" if _safe_str(knowdisease) else ""
    )

    return f"""You are an expert clinical geneticist and gene prioritization specialist. Your previous ranked gene list was reviewed and needs revision.

## Patient Phenotypes / Clinical Information
{input_patient_information}
{disease_section}
{context_section}

## Previous Gene Ranking (rejected)
{previous_output}

## Reviewer Feedback
{verifier_feedback}

## Instructions
Revise the gene prioritization using the reviewer feedback.
Focus especially on:
- phenotype features not well explained by the previous ranking
- missing genes suggested by the reviewer
- better biological fit, inheritance consistency, and syndrome specificity
- correct HGNC gene symbols

Return:
- Revised reasoning inside <thinking></thinking>
- Revised top {gene_num} ranked genes inside <solution></solution>, one gene symbol per line, no numbering, no bullets
"""


# ─────────────────────────────────────────────
# Core caller
# ─────────────────────────────────────────────

def _call_model(
    client,
    model_name: str,
    prompt: str,
    max_output_tokens: int = 4096,
) -> str:
    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    return response.output_text



def run_generank(
    input_patient_information: str,
    knowinte: bool = True,
    knowdisease: Optional[str] = None,
    gene_num: str = "10",
    model: str = "gpt-5.4-mini",
    verifier_model: str = "gpt-5.4-mini",
    max_verify_rounds: int = 5,
    sleep_sec: float = 0.0,
) -> Tuple[str, str]:
    """
    Gene prioritization with iterative self-verification.

    Parameters
    ----------
    input_patient_information : str
        Phenotype / patient description.
    knowinte : bool
        Whether to retrieve supporting context from external sources.
    knowdisease : Optional[str]
        Known disease label, if any.
    gene_num : str
        Number of genes to rank, e.g. "10".
    model : str
        Main model name.
    verifier_model : str
        Verifier model name.
    max_verify_rounds : int
        Number of self-verification rounds.
    sleep_sec : float
        Optional sleep between retries.

    Returns
    -------
    (model_output, prior_knowledge)
    """
    if not _safe_str(input_patient_information):
        raise ValueError("input_patient_information must be a non-empty string.")

    client = OpenAI(
        base_url=AZURE_BASE_URL,
        api_key=AZURE_API_KEY
    )

    # ── Retrieve prior knowledge ──────────────────────────────────────────────
    prior_knowledge = ""
    if knowinte:
        query_text = _safe_str(input_patient_information)
        if _safe_str(knowdisease):
            query_text = f"{knowdisease}\n{query_text}"

        knowledge_parts = []
        try:
            scholar_text = query_scholar(query_text)
            if _safe_str(scholar_text):
                knowledge_parts.append("Scholar:\n" + scholar_text)
        except Exception as e:
            print(f"query_scholar error: {e}")

        try:
            google_text = search_google(query_text, num_results=3)
            if _safe_str(google_text):
                knowledge_parts.append("Google:\n" + google_text)
        except Exception as e:
            print(f"search_google error: {e}")

        try:
            pubmed_text = query_pubmed(query_text, max_retries=10)
            if _safe_str(pubmed_text):
                knowledge_parts.append("PubMed:\n" + pubmed_text)
        except Exception as e:
            print(f"query_pubmed error: {e}")

        prior_knowledge = "\n\n".join(knowledge_parts).strip()

    # ── Initial gene prioritization ───────────────────────────────────────────
    gene_prompt = _make_gene_prioritization_prompt(
        input_patient_information=input_patient_information,
        prior_knowledge=prior_knowledge,
        gene_num=gene_num,
        knowdisease=knowdisease,
    )
    modelout = _call_model(client, model, gene_prompt)

    # ── Self-verification loop ────────────────────────────────────────────────
    for round_idx in range(max_verify_rounds):
        print(f"  [verify {round_idx + 1}/{max_verify_rounds}]", end=" ", flush=True)

        solution_text = _extract_gene_list(modelout)

        verifier_prompt = _make_gene_verifier_prompt(
            input_patient_information=input_patient_information,
            gene_output_text=solution_text,
            prior_knowledge=prior_knowledge,
            knowdisease=knowdisease,
        )

        try:
            verifier_out = _call_model(client, verifier_model, verifier_prompt, max_output_tokens=1024)
            verdict = _normalize_yes_no(verifier_out)
            feedback = _extract_tag(verifier_out, "feedback") or ""
        except Exception as e:
            print(f"verification error: {e}")
            verdict = "NO"
            feedback = ""

        if verdict == "YES":
            print("accepted.")
            break

        print("rejected. Retrying with feedback.")

        fallback_feedback = (
            "The previous gene ranking is not sufficiently supported. "
            "Please reconsider phenotype fit, inheritance, pathway relevance, and missing high-priority genes."
        )

        retry_prompt = _make_gene_retry_prompt(
            input_patient_information=input_patient_information,
            previous_output=solution_text,
            verifier_feedback=feedback if feedback else fallback_feedback,
            prior_knowledge=prior_knowledge,
            gene_num=gene_num,
            knowdisease=knowdisease,
        )
        modelout = _call_model(client, model, retry_prompt)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return modelout

def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info"):
    """Fetches supplementary information for a paper given its DOI and returns a research log.

    Args:
        doi: The paper DOI.
        output_dir: Directory to save supplementary files.

    Returns:
        dict: A dictionary containing a research log and the downloaded file paths.

    """
    research_log = []
    research_log.append(f"Starting process for DOI: {doi}")

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(crossref_url, headers=headers)

    if response.status_code != 200:
        log_message = f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    publisher_url = response.url
    research_log.append(f"Resolved DOI to publisher page: {publisher_url}")

    # Fetch publisher page
    response = requests.get(publisher_url, headers=headers)
    if response.status_code != 200:
        log_message = f"Failed to access publisher page for DOI {doi}."
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    # Parse page content
    soup = BeautifulSoup(response.content, "html.parser")
    supplementary_links = []

    # Look for supplementary materials by keywords or links
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        text = link.get_text().lower()
        if "supplementary" in text or "supplemental" in text or "appendix" in text:
            full_url = urljoin(publisher_url, href)
            supplementary_links.append(full_url)
            research_log.append(f"Found supplementary material link: {full_url}")

    if not supplementary_links:
        log_message = f"No supplementary materials found for DOI {doi}."
        research_log.append(log_message)
        return research_log

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    research_log.append(f"Created output directory: {output_dir}")

    # Download supplementary materials
    downloaded_files = []
    for link in supplementary_links:
        file_name = os.path.join(output_dir, link.split("/")[-1])
        file_response = requests.get(link, headers=headers)
        if file_response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(file_response.content)
            downloaded_files.append(file_name)
            research_log.append(f"Downloaded file: {file_name}")
        else:
            research_log.append(f"Failed to download file from {link}")

    if downloaded_files:
        research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
    else:
        research_log.append(f"No files could be downloaded for DOI {doi}.")

    return "\n".join(research_log)


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        results = "\n\n".join([f"Title: {paper.title}\nSummary: {paper.summary}" for paper in client.results(search)])
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def query_scholar(query: str) -> str:
    """Query Google Scholar for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.

    Returns
    -------
    - str: The first search result formatted or an error message.

    """
    from scholarly import scholarly
    from scholarly import ProxyGenerator

    # Set up a ProxyGenerator object to use free proxies
    # This needs to be done only once per session
    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)

    try:
        search_query = scholarly.search_pubs(query)
        result = next(search_query, None)
        if result:
            return f"Title: {result['bib']['title']}\nYear: {result['bib']['pub_year']}\nVenue: {result['bib']['venue']}\nAbstract: {result['bib']['abstract']}"
        else:
            return "No results found on Google Scholar."
    except Exception as e:
        return f"Error querying Google Scholar: {e}"


def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> str:
    """Query PubMed for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).
    - max_retries (int): Maximum number of retry attempts with modified queries (default: 3).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    from pymed import PubMed

    try:
        pubmed = PubMed(tool="MyTool", email="tianyu.liu@yale.edu")  # Update with a valid email address

        # Initial attempt
        papers = list(pubmed.query(query, max_results=max_papers))

        # Retry with modified queries if no results
        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            # Simplify query with each retry by removing the last word
            simplified_query = " ".join(query.split()[:-retries]) if len(query.split()) > retries else query
            time.sleep(1)  # Add delay between requests
            papers = list(pubmed.query(simplified_query, max_results=max_papers))

        if papers:
            results = "\n\n".join(
                [f"Title: {paper.title}\nAbstract: {paper.abstract}" for paper in papers]
            )
            return results
        else:
            return "No papers found on PubMed after multiple query attempts."
    except Exception as e:
        return f"Error querying PubMed: {e}"


def search_google(query: str, num_results: int = 3, language: str = "en") -> list[dict]:
    """Search using Google search.

    Args:
        query (str): The search query (e.g., "protocol text or seach question")
        num_results (int): Number of results to return (default: 10)
        language (str): Language code for search results (default: 'en')
        pause (float): Pause between searches to avoid rate limiting (default: 2.0 seconds)

    Returns:
        List[dict]: List of dictionaries containing search results with title and URL

    """
    try:
        results_string = ""
        search_query = f"{query}"

        print(f"Searching for {search_query} with {num_results} results and {language} language")

        for res in search(search_query, num_results=num_results, lang=language, advanced=True):
            print(f"Found result: {res.title}")
            title = res.title
            url = res.url
            description = res.description

            results_string += f"Title: {title}\nURL: {url}\nDescription: {description}\n\n"

    except Exception as e:
        print(f"Error performing search: {str(e)}")
    return results_string


def advanced_web_search_claude(
    query: str,
    max_searches: int = 1,
    max_retries: int = 3,
) -> tuple[str, list[dict[str, str]], list]:
    """
    Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.
    Craft the query carefully for the search agent to find the most relevant information.

    Parameters
    ----------
    query : str
        The search phrase you want Claude to look up.
    max_searches : int, optional
        Upper-bound on searches Claude may issue inside this request.
    max_retries : int, optional
        Maximum number of retry attempts with exponential backoff.

    Returns
    -------
    full_text : str
        A formatted string containing the full text response from Claude and the citations.
    """
    import random

    import anthropic

    try:
        from biomni.config import default_config

        model = default_config.llm
        api_key = default_config.api_key
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
    except ImportError:
        model = "claude-4-sonnet-latest"
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if "claude" not in model:
        raise ValueError("Model must be a Claude model.")

    if not api_key:
        raise ValueError("Set your api_key explicitly.")

    client = anthropic.Anthropic(api_key=api_key)
    tool_def = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_searches,
    }

    delay = random.randint(1, 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": query}],
                tools=[tool_def],
            )

            paragraphs, citations = [], []
            response.content = response.content
            formatted_response = ""
            for blk in response.content:
                if blk.type == "text":
                    paragraphs.append(blk.text)
                    formatted_response += blk.text

                    if blk.citations:
                        for cite in blk.citations:
                            citations.append({"url": cite.url, "title": cite.title, "cited_text": cite.cited_text})
                            formatted_response += f"(Citation: {cite.title} - {cite.url})"
            return formatted_response

        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"Error performing web search after {max_retries} attempts: {str(e)}")
            return f"Error performing web search after {max_retries} attempts: {str(e)}"


def extract_url_content(url: str) -> str:
    """Extract the text content of a webpage using requests and BeautifulSoup.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage

    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # Check if the response is in text format
    if "text/plain" in response.headers.get("Content-Type", "") or "application/json" in response.headers.get(
        "Content-Type", ""
    ):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content first, fallback to body
    content = soup.find("main") or soup.find("article") or soup.body

    # Remove unwanted elements
    for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        element.decompose()

    # Extract text with better formatting
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_text = []

    for p in paragraphs:
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)


def extract_pdf_content(url: str) -> str:
    """Extract the text content of a PDF file given its URL.

    Args:
        url: URL of the PDF file to extract text from

    Returns:
        The extracted text content from the PDF

    """
    try:
        # Check if the URL ends with .pdf
        if not url.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found
                    if not pdf_links[0].startswith("http"):
                        # Handle relative URLs
                        base_url = "/".join(url.split("/")[:3])
                        url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

        pdf_file = BytesIO(response.content)

        # Try with PyPDF2 first
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        return text

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

