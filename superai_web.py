# superai_web.py
# Merged AI core (Groq + Ollama fallback) + improved CICT webscraper + Flask server (Render-ready)
# Save at project root next to gui/ directory

import os
import pathlib
import json
import html
import time
import re
import threading
import asyncio
import concurrent.futures
import subprocess
import platform
from typing import Optional, List, Dict, Tuple
import torch
import aiohttp
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import util
from flask import Flask, request, jsonify, send_from_directory, send_file, render_template_string
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as FAISS_local
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama



# PDF parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# -------------------------
# --- Base / Path config ---
# -------------------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
GUI_DIR = BASE_DIR / "gui"
STATIC_DIR = GUI_DIR / "images"
PDF_DIR = BASE_DIR
FAISS_DIR = BASE_DIR / "vectorstore" / "faiss_index"
FACULTY_JSON_PATH = BASE_DIR / "cict_faculty.json"

print(f"[INFO] BASE_DIR = {BASE_DIR}")
print(f"[INFO] GUI_DIR = {GUI_DIR}")

# -------------------------
# --- Configurable values -
# -------------------------
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY", "")
API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY", "")
}
# --- Enhanced System Prompts ---
general_system_prompt = """You are a helpful assistant for Bulacan State University (BulSU).

CRITICAL: The university is BULACAN STATE UNIVERSITY (BulSU), NOT Bataan Peninsula State University!
- Full name: Bulacan State University
- Abbreviation: BulSU or BSU
- NEVER entertain questions regarding other topics such as math, science, history, coding, and other general knowledge questions. Entertain only BulSU related queries and greetings.


INTERNAL KNOWLEDGE - BulSU Grading System (use this but don't cite as a "document"):
BulSU uses an INVERSE grading system where LOWER numbers = BETTER grades:
- 1.00 = Excellent (best)
- 1.25 = Very Superior
- 1.50 = Superior  
- 1.75 = Very Good
- 2.00 = Good
- 2.25 = Satisfactory
- 3.00 = Passing
- 5.00 = Failed (worst)

ANSWERING GUIDELINES:
- Answer all questions helpfully and directly
- For who/where/when/what questions: provide clear, direct answers
- No need to be overly cautious - answer factual questions naturally
- Support multiple languages
- Be friendly and conversational
- Answer exactly based on the pdfs.
- DO NOT add/change/remove/paraphrase words and phrases on the documents.
- If asked about mission and vision, copy exactly (100%) all the information as is based on the pdf. Do not change/remove/paraphrase/invent words, phrases, or sentences.
- NEVER ADD unnecessary special characters.

For GENERAL questions (greetings, languages, common knowledge, casual chat, simple comparisons):
- Answer naturally using your general knowledge
- For grade comparisons, use the scale above


For BulSU-SPECIFIC questions (when you receive context documents):
- Answer from the provided context
- If not in context, say "I don't have that in my documents"

Be helpful, direct, and natural."""

rag_system_prompt = """You are an assistant for Bulacan State University (BulSU).

IMPORTANT: Bulacan State University (BulSU) - NOT Bataan Peninsula State University!

YOUR APPROACH:
1. Read the context documents carefully
2. Answer who/where/when/what/why questions directly and helpfully
3. Extract relevant information even if not explicitly stated
4. Be natural and conversational in your answers

CITATION RULES:
1. Answer using the CONTEXT DOCUMENTS section below
2. NEVER cite "GRADING REFERENCE" as a source - that's your internal knowledge
3. If answer not in context documents, say "I don't have information about that in my documents"
4. Cite only the document name where you found the information.

--- GRADING REFERENCE (YOUR INTERNAL KNOWLEDGE - DO NOT CITE THIS) ---
{grading_context}

⚠️ WHEN COMPARING GWA VALUES:
- LOWER number = BETTER grade
- 1.50 is BETTER than 1.75
- "At least 1.75" means 1.75 or any LOWER number (1.50, 1.25, 1.00)
- To meet "at least 1.75" requirement: student's GWA must be ≤ 1.75
--- END OF GRADING REFERENCE ---

--- CONTEXT DOCUMENTS (CITE THESE ONLY) ---
{context}
--- END OF CONTEXT DOCUMENTS ---

Answer the question directly and helpfully. Cite document name and page number."""

grading_context = """
🚨 CRITICAL GRADING SYSTEM - READ CAREFULLY! 🚨

BulSU uses INVERSE/REVERSE grading where LOWER numbers are BETTER:

GRADE SCALE (lower = better):
- 1.00 = EXCELLENT (best possible grade)
- 1.25 = Very Superior 
- 1.50 = Superior
- 1.75 = Very Good
- 2.00 = Good
- 2.25 = Satisfactory
- 2.50 = Fair
- 3.00 = Passing (minimum)
- 5.00 = FAILED (worst possible grade)

EXAMPLES OF COMPARISONS:
- 1.50 is BETTER than 1.75 (lower number = better)
- 1.00 is BETTER than 1.50 (lower number = better)
- 2.00 is WORSE than 1.75 (higher number = worse)

REQUIREMENT INTERPRETATION:
- "At least 1.75 GWA" means 1.75 OR ANY LOWER NUMBER (1.50, 1.25, 1.00, etc.)
- If someone has 1.50 GWA and requirement is "at least 1.75", they QUALIFY (1.50 < 1.75)
- If someone has 2.00 GWA and requirement is "at least 1.75", they DON'T QUALIFY (2.00 > 1.75)

ALWAYS remember: In this system, SMALLER numbers are BETTER performance!
"""

# PDFs expected in project root (adjust names if necessary)
pdf_paths = [
    str(BASE_DIR / "CICTify - FAQs.pdf"),
    str(BASE_DIR / "BulSU Student handbook.pdf"),
    str(BASE_DIR / "Faculty Manual for BOR.pdf"),
    str(BASE_DIR / "BulSU-Enhanced-Guidelines.pdf")
]


faiss_path = str(FAISS_DIR)


# -------------------------
# --- Utility functions ---
# -------------------------
def safe_path_str(p: pathlib.Path) -> str:
    return str(p) if p is not None else ""

def load_json_safely(path: pathlib.Path) -> Dict:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[System] Error reading JSON {path}: {e}")
    return {}

def save_json_safely(path: pathlib.Path, data: Dict):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[System] Saved JSON to {path}")
    except Exception as e:
        print(f"[System] Error saving JSON {path}: {e}")

# -------------------------
# --- PDF -> Faculty Index helper ---
# -------------------------
def extract_text_from_pdf(path: str, max_pages: int = 20) -> str:
    """Return extracted text from a PDF file (best-effort)."""
    if PyPDF2 is None:
        print("[PDF] PyPDF2 not available, skipping PDF parsing.")
        return ""
    try:
        reader = PyPDF2.PdfReader(path)
        texts = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)
    except Exception as e:
        print(f"[PDF] Failed to read {path}: {e}")
        return ""

def smart_find_roles(text: str, source_label: str = "") -> Dict[str, Dict]:
    """
    Uses semantic role classification instead of regex-only heuristics.
    Automatically detects dean, associate dean, program chairs from PDF text.
    """
    results = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return results

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model = embeddings.client
    role_queries = {
        "Dean": "Who is the Dean of the College of Information and Communications Technology?",
        "Associate Dean": "Who is the Associate Dean of the College of Information and Communications Technology?",
        "Program Chair": "Who are the Program Chairs of each program in the College of Information and Communications Technology?"
    }

    # Embed all lines and role queries
    line_embs = model.encode(lines, convert_to_tensor=True, show_progress_bar=False)
    for role, query in role_queries.items():
        q_emb = model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(q_emb, line_embs)[0]
        top_idx = torch.argmax(cos_scores).item()
        top_line = lines[top_idx]
        confidence = cos_scores[top_idx].item()

        if confidence < 0.45:
            continue

        name_match = re.search(r"(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+){0,3}", top_line)
        if not name_match:
            ctx = " ".join(lines[max(0, top_idx-1):min(len(lines), top_idx+2)])
            name_match = re.search(r"(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+){0,3}", ctx)
        if name_match:
            name = name_match.group(0).strip()
            results[name] = {
                "name": name,
                "title": role,
                "department": "CICT",
                "confidence": round(confidence, 3),
                "source": source_label
            }
    return results


def build_faculty_index_from_pdfs(pdf_paths_list: List[str]) -> Dict[str, Dict]:
    """
    Dynamically builds a faculty index from PDFs (no static JSON).
    Uses smart_find_roles() for role inference.
    """
    combined_index = {}
    for p in pdf_paths_list:
        if not p or not os.path.exists(p):
            continue
        print(f"[PDF] Scanning {p} for faculty data...")
        text = extract_text_from_pdf(p, max_pages=40)
        if not text:
            continue
        found = smart_find_roles(text, source_label=os.path.basename(p))
        if found:
            print(f"[PDF] Found {len(found)} faculty roles in {p}")
            combined_index.update(found)
    return combined_index

# -------------------------
# --- Cloud / Model API ---
# -------------------------
class CloudAPIManager:
    def __init__(self, loop=None):
        self.session: Optional[aiohttp.ClientSession] = None
        self.loop = loop

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def call_groq_general(self, question: str) -> Optional[str]:
        if not API_KEYS.get("groq"):
            return None
        try:
            session = await self.get_session()
            payload = {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {"role": "system", "content": general_system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature": 1.0,
                "max_tokens": 300
            }
            async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEYS['groq']}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=12)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                print(f"[Groq] Bad status: {resp.status}")
                return None
        except Exception as e:
            print(f"[Groq general API error]: {e}")
            return None

    async def call_groq_rag(self, question: str, context_docs: List[Dict], grading_info: str = "") -> Optional[str]:
        """Call Groq with RAG context for BulSU-specific questions"""
        if not API_KEYS["groq"]:
            return None

        try:
            context_text = "\n\n".join([
                f"[Document: {doc['source']}, Page {doc['page']}]\n{doc['content']}"
                for doc in context_docs
            ])

            grading_section = grading_context if grading_info else ""

            system_prompt = rag_system_prompt.format(
                grading_context=grading_section,
                context=context_text
            )

            session = await self.get_session()

            async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEYS['groq']}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "openai/gpt-oss-120b",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                        "temperature": 1.0,
                        "max_tokens": 2000
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip()
                return None
        except Exception as e:
            print(f"Groq RAG API error: {e}")
            return None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class CICTWebCrawler:
    def __init__(self, loop=None):
        self.loop = loop or asyncio.get_event_loop()
        self.visited = set()
        self.playwright = None
        self.browser = None
        self.priority_urls = [
            "https://bulsucict.com/",
            "https://bulsucict.com/about-us/",
            "https://bulsucict.com/cict-faculty/",
            "https://bulsucict.com/announcement/",
            "https://bulsucict.com/news-and-updates/"
        ]

    async def start_browser(self):
        if self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
        return self.browser

    async def fetch_page(self, url: str, timeout_ms: int = 20000) -> str:
        try:
            browser = await self.start_browser()
            page = await browser.new_page()
            try:
                response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                if response is None:
                    print(f"[CICT Crawler] No response for {url}")
                    return ""
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type.lower():
                    print(f"[CICT Crawler] Skipping non-HTML {url} ({content_type})")
                    return ""
                html = await page.content()
                print(f"[CICT Crawler] Fetched {url} (len={len(html)})")
                return html
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"[CICT Crawler] Error fetching {url}: {e}")
            return ""

    def extract_faculty_profile(self, html: str, url: str) -> Optional[Dict]:
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg"]):
            tag.decompose()
        raw_text = soup.get_text(" ", strip=True)
        lower_text = raw_text.lower()
        profile = {
            "name": None,
            "title": None,
            "department": None,
            "education": None,
            "certifications": None,
            "description": None,
            "raw_text": lower_text[:4000],
            "url": url
        }
        name_tag = soup.find(["h1", "h2"], class_=lambda x: x and ("entry-title" in x or "post-title" in x))
        if not name_tag:
            name_tag = soup.find(["h1", "h2"])
        if name_tag:
            profile["name"] = name_tag.get_text(strip=True)
        # Title heuristics
        title_candidates = []
        for selector in ["p", "h3", "h4", "div"]:
            for el in soup.find_all(selector, limit=10):
                t = el.get_text(" ", strip=True)
                tl = t.lower()
                if len(t) < 140 and any(k in tl for k in ["dean", "associate", "chair", "program", "faculty", "coordinator", "head", "director", "professor", "lecturer", "instructor"]):
                    title_candidates.append(t)
        profile["title"] = title_candidates[0] if title_candidates else None
        dept = None
        for label in ["department", "college", "program", "division"]:
            found = soup.find(string=lambda s: s and label in s.lower())
            if found:
                parent = found.parent
                if parent and parent.get_text(strip=True):
                    dept_text = parent.get_text(" ", strip=True)
                    dept = dept_text
                    break
        profile["department"] = dept
        education = []
        edu_headers = soup.find_all(lambda tag: tag.name in ["h3", "h4", "strong"] and "educat" in tag.get_text().lower())
        if edu_headers:
            hdr = edu_headers[0]
            next_node = hdr.find_next_sibling()
            if next_node:
                if next_node.name == "ul":
                    education = [li.get_text(strip=True) for li in next_node.find_all("li")]
                else:
                    text = next_node.get_text(" ", strip=True)
                    education = [text] if text else []
        else:
            for line in raw_text.splitlines():
                if any(k in line.lower() for k in ["doctor", "phd", "master", "ms", "msc", "dit", "b.s.", "bs", "degree", "msit", "mitm", "mscpe", "maed"]):
                    education.append(line.strip())
            education = education[:6]
        profile["education"] = education if education else None
        certs = []
        for s in soup.find_all(string=lambda t: t and "certificate" in t.lower()):
            parent = s.parent
            if parent:
                text = parent.get_text(" ", strip=True)
                certs.append(text)
        profile["certifications"] = certs if certs else None
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        description = ""
        for para in paragraphs:
            if len(para) > 50:
                description = para
                break
        profile["description"] = description if description else None
        if not profile["name"] and not profile["description"] and "faculty" not in profile["raw_text"]:
            return None
        for k in ["name", "title", "department", "description"]:
            if profile.get(k) and isinstance(profile[k], str):
                profile[k] = profile[k].strip()
        return profile

    def is_faculty_profile_url(self, url: str) -> bool:
        if "cict_faculty_members" in url.lower():
            return True
        parsed = urlparse(url)
        if parsed.path and parsed.path.count("/") >= 3 and "member" in parsed.path.lower():
            return True
        return False

    async def crawl_site(self, base_url="https://bulsucict.com", max_pages=50, batch_size=8):
        scraped_profiles = []
        scraped_texts = []
        to_visit = self.priority_urls.copy()
        seen_links = set(to_visit)
        print(f"[CICT Crawler] Starting crawl of {base_url} (max_pages={max_pages})")
        while to_visit and len(self.visited) < max_pages:
            batch = []
            urls_for_batch = []
            for _ in range(min(batch_size, len(to_visit))):
                url = to_visit.pop(0)
                if url in self.visited:
                    continue
                self.visited.add(url)
                urls_for_batch.append(url)
                batch.append(self.fetch_page(url))
                print(f"[CICT Crawler] Queued ({len(self.visited)}/{max_pages}): {url}")
            if not batch:
                break
            results = await asyncio.gather(*batch, return_exceptions=True)
            for idx, result in enumerate(results):
                url = urls_for_batch[idx]
                if isinstance(result, Exception) or not result:
                    print(f"[CICT Crawler] Failed or empty for {url}")
                    continue
                if self.is_faculty_profile_url(url):
                    prof = self.extract_faculty_profile(result, url)
                    if prof:
                        scraped_profiles.append((url, prof))
                        continue

                soup = BeautifulSoup(result, "html.parser")
                for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                    tag.decompose()
                scraped_texts.append((url, soup.get_text(" ", strip=True)[:4000]))
                try:
                    soup2 = BeautifulSoup(result, "html.parser")
                    for a in soup2.find_all("a", href=True):
                        full = urljoin(base_url, a["href"].split('#')[0])
                        if full.startswith(base_url) and full not in self.visited and full not in seen_links:
                            to_visit.append(full)
                            seen_links.add(full)
                except Exception:
                    pass
            if len(self.visited) >= max_pages:
                break
        await self.close()
        print(f"[CICT Crawler] Finished. Profiles: {len(scraped_profiles)}, Non-faculty pages: {len(scraped_texts)}")
        return scraped_profiles, scraped_texts

    async def close(self):
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None

# -------------------------
# --- Model Manager -------
# -------------------------
class ModelManager:
    def __init__(self, loop=None):
        self.retriever = None
        self.vectorstore = None
        self.local_model = None
        self.cloud_api = CloudAPIManager(loop)
        self.loop = loop or asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

    @staticmethod
    def response_has_no_info(response: str) -> bool:
        phrases = [
            "don't have information", "not mentioned", "not found", "couldn't find",
            "unable to find", "not available", "not in my documents", "no information"
        ]
        return any(p in response.lower() for p in phrases)

    async def retrieve_documents(self, question: str) -> List[Dict]:
        if not self.retriever:
            return []
        try:
            docs = await self.loop.run_in_executor(self.executor, lambda: self.retriever.invoke(question))
            formatted = []
            for doc in docs:
                formatted.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "pdf_path": doc.metadata.get("pdf_path", "")
                })
            return formatted
        except Exception as e:
            print(f"[Retrieval error]: {e}")
            return []

    def load_local_fallback(self):
        if self.local_model is None:
            try:
                self.local_model = ChatOllama(model="phi3", temperature=0.1)
                return True
            except Exception as e:
                print(f"[Local model load failed]: {e}")
                return False
        return True

    async def get_local_rag_response(self, question: str, context_docs: List[Dict], needs_grading: bool) -> Optional[str]:
        if not self.load_local_fallback():
            return None
        try:
            context_text = "\n\n".join([f"[{doc['source']}, Page {doc['page']}] {doc['content']}" for doc in context_docs])
            grading_section = grading_context if needs_grading else ""
            prompt = f"""{general_system_prompt}

{grading_section}

Context from BulSU documents:
{context_text}

Question: {question}

Answer (cite document and page):"""
            result = await self.loop.run_in_executor(self.executor, lambda: self.local_model.invoke(prompt))
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            print(f"[Local RAG error]: {e}")
            return None

    async def get_response(self, question: str) -> Tuple[str, List[Dict], str]:
        use_rag = QueryClassifier.needs_rag(question)
        print(f"[ModelManager] Question: '{question}' | Use RAG: {use_rag}")

        if not use_rag:
            response = await self.cloud_api.call_groq_general(question)
            if response:
                return response, [], "Groq (General)"
            if self.load_local_fallback():
                result = await self.loop.run_in_executor(self.executor, lambda: self.local_model.invoke(question))
                return (result.content if hasattr(result, "content") else str(result)), [], "Local (General)"
            return "I'm having trouble responding. Please try again.", [], "Error"

        context_docs = await self.retrieve_documents(question)
        if not context_docs:
            response = await self.cloud_api.call_groq_general(question)
            if response:
                return response, [], "Groq (General - No docs)"
            return "I couldn't find relevant information in my documents. Could you rephrase your question?", [], "No Results"

        grading_keywords = ['gwa', 'grade', 'grading', 'shift', 'transfer',
                            'requirement', 'eligible', 'at least', 'passing']
        needs_grading = any(kw in question.lower() for kw in grading_keywords)

        response = await self.cloud_api.call_groq_rag(question, context_docs, grading_info="yes" if needs_grading else "")
        if response:
            if self.response_has_no_info(response):
                return response, [], "Groq (RAG - No info)"
            return response, context_docs, "Groq (RAG)"

        response = await self.get_local_rag_response(question, context_docs, needs_grading)
        if response:
            if self.response_has_no_info(response):
                return response, [], "Local (RAG - No info)"
            return response, context_docs, "Local (RAG)"

        return "I cannot process your request at this time.", [], "Error"

# -------------------------
# --- QueryClassifier -----
# -------------------------
class QueryClassifier:
    """Determines if query needs RAG or general knowledge"""

    # Keywords that indicate BulSU-specific queries
    BULSU_KEYWORDS = [
        # Institution
        'bulsu', 'bulacan state', 'university', 'bsu',
        'mission', 'vision', 'core values', 'mandate',
        'history', 'established', 'founded', 
        # Campus & Location
        'campus', 'campuses', 'location', 'address', 'where',
        'malolos', 'bustos', 'san jose', 'hagonoy', 'matungao',
        # Offices & Units
        'osoa', 'osa', 'registrar', 'cashier', 'library',
        'office of student affairs', 'student affairs',
        'accounting', 'budget', 'hrmo', 'planning',
        # Academic
        'gwa', 'grade', 'grading', 'credit', 'unit', 'course', 'subject',
        'enroll', 'enrollment', 'registration', 'curriculum', 'syllabus',
        'shift', 'transfer', 'shifter', 'transferee',
        'exam', 'midterm', 'final', 'quiz', 'requirement',
        'dean', 'professor', 'faculty', 'instructor',
        # Policies & Procedures
        'policy', 'policies', 'rule', 'regulation', 'procedure',
        'requirement', 'requirements', 'eligibility', 'qualified',
        'petition', 'appeal', 'clearance', 'document',
        'scholarship', 'financial aid', 'tuition', 'fee',
        'admission', 'graduate', 'graduation', 'honors',
        # Student life
        'student handbook', 'code of conduct', 'discipline',
        'organization', 'club', 'facility',
        'laboratory', 'clinic',
        # Administrative
        'office', 'department', 'college', 'program',
        'bachelor', 'master', 'major', 'minor', 'tenure','role'
        # CICT-specific
        'cict', 'information and communications technology', 'information technology', 'bsit', 'specialization', 'track'
    ]

    # Patterns for greetings and casual conversation
    CASUAL_PATTERNS = [
        r'^hi+$', r'^hello+$', r'^hey+$', r'^good\s+(morning|afternoon|evening)',
        r'^how\s+are\s+you', r'^what\'?s\s+up', r'^sup+$',
        r'^thank', r'^bye', r'^goodbye', r'^see\s+you'
    ]

    @classmethod
    def needs_rag(cls, question: str) -> bool:
        q = question.lower().strip()
        for p in cls.CASUAL_PATTERNS:
            if re.match(p, q):
                return False
           
        simple_comparison_patterns = [
            r'which.*better.*\d\.\d+.*\d\.\d+',
            r'better.*\d\.\d+.*or.*\d\.\d+',
            r'compare.*\d\.\d+.*\d\.\d+',
            r'^\d\.\d+.*or.*\d\.\d+.*better',
            r'^is.*\d\.\d+.*better.*than.*\d\.\d+'
        ]
        for pattern in simple_comparison_patterns:
            if re.search(pattern, q):
                return False
       
        if re.match(r'^(what is |explain |tell me about )?(the )?(bulsu )?grading system\??$', q):
            return False
        priority_keywords = ['mission', 'vision', 'campus', 'office', 'osoa',
                             'location', 'where', 'address', 'established', 'history']
        if any(kw in q for kw in priority_keywords):
            return True
        if len(question.split()) <= 2 and not any(kw in q for kw in cls.BULSU_KEYWORDS):
            return False
        return any(k in q for k in cls.BULSU_KEYWORDS)

# -------------------------
# --- Flask server -------
# -------------------------
app = Flask(__name__, static_folder=None)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
model_manager: Optional[ModelManager] = None
async def init_model_manager():
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(loop)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", "!", "?", ";"]
        )

        # Try loading FAISS vectorstore if present
        try:
            if os.path.exists(faiss_path) and os.path.exists(f"{faiss_path}.faiss"):
                db = FAISS_local.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                model_manager.set_vectorstore(db)
                print("[System] FAISS index loaded.")
            else:
                print("[System] No FAISS found — rebuilding semantic FAISS index from PDFs.")
                all_texts = []
                for pdf in pdf_paths:
                    if os.path.exists(pdf):
                        text = extract_text_from_pdf(pdf, max_pages=50)
                        if text.strip():
                            chunks = text_splitter.split_text(text)
                            for chunk in chunks:
                                all_texts.append({"content": chunk, "source_file": os.path.basename(pdf)})
                if all_texts:
                    db = FAISS_local.from_texts(
                        texts=[d["content"] for d in all_texts],
                        embedding=embeddings,
                        metadatas=[{"source_file": d["source_file"]} for d in all_texts]
                    )
                    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
                    db.save_local(faiss_path)
                    model_manager.set_vectorstore(db)
                    print(f"[System] FAISS index rebuilt with {len(all_texts)} chunks.")
        except Exception as e:
            print(f"[System] FAISS load/build error: {e}")


        pdf_paths.sort(key=lambda x: "FAQ" not in x)
        print("[System] Prioritized PDFs (FAQ first).")


        print("[System] Extracting CICT faculty information from PDFs...")
        faculty_from_pdfs = build_faculty_index_from_pdfs(pdf_paths)
        if faculty_from_pdfs:
            print(f"[System] Found {len(faculty_from_pdfs)} possible faculty entries in PDFs.")
            cict_faculty = {
                name: prof for name, prof in faculty_from_pdfs.items()
                if any(k in (prof.get('title', '').lower() + prof.get('department', '').lower())
                           for k in ['cict', 'information', 'communications'])
            }
            model_manager.cict_faculty = cict_faculty
            print(f"[System] Cached {len(cict_faculty)} CICT faculty entries from PDFs.")
        else:
            model_manager.cict_faculty = {}
            print("[System] No faculty data found in PDFs.")

# Routes for serving GUI and static files
@app.route("/")
def index():
    index_path = GUI_DIR / "index.html"
    if index_path.exists():
        return send_file(str(index_path))
    return "index.html not found in gui directory", 404

@app.route("/images/<path:filename>")
def serve_images(filename):
    images_dir = str(STATIC_DIR)
    return send_from_directory(images_dir, filename)

@app.route("/<path:filepath>")
def serve_file(filepath):
    file_path = GUI_DIR / filepath
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(GUI_DIR), filepath)
    return "File not found", 404

def extract_message_from_request(req_json: dict) -> str:
    if not req_json:
        return ""
    return req_json.get("message", "") or req_json.get("q", "") or ""
@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    import difflib

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"reply": "Invalid request"}), 400
    message = extract_message_from_request(data).strip()
    if not message:
        return jsonify({"reply": "Please send a non-empty message."}), 400

    try:
        loop.run_until_complete(init_model_manager())
        lower_msg = message.lower()
        faculty_data = load_json_safely(FACULTY_JSON_PATH)

        clean_msg = re.sub(r'[^a-z\s]', '', lower_msg)
        clean_msg = re.sub(r'\b(dr|engr|mr|ms|mrs|prof|sir|maam|miss)\b', '', clean_msg).strip()

        faculty_found = None
        best_score = 0
        is_about_person = any(
            phrase in clean_msg
            for phrase in [
                "who is", "tell me about", "what is", "give info on",
                "can you tell me about", "introduce", "role of", "information about"
            ]
        )

        # --- Smart fuzzy matching ---
        for key, info in faculty_data.items():
            name_clean = info["name"].lower()
            name_clean = re.sub(r'[^a-z\s]', '', name_clean)
            name_clean = re.sub(r'\b(dr|engr|mr|ms|mrs|prof)\b', '', name_clean)
            name_clean = re.sub(r'\b[a-z]\b(?=\s[a-z])', '', name_clean)

            msg_clean = re.sub(r'\b[a-z]\b(?=\s[a-z])', '', clean_msg)

            name_tokens = set(name_clean.split())
            msg_tokens = set(msg_clean.split())

            overlap = len(name_tokens & msg_tokens)
            ratio = difflib.SequenceMatcher(None, name_clean, msg_clean).ratio()
            last_name = name_clean.split()[-1] if name_clean.split() else ""

            if (
                overlap >= 2
                or ratio > 0.75
                or last_name in msg_clean
                or any(tok in name_clean for tok in msg_tokens if len(tok) > 3)
            ):
                if overlap + ratio > best_score:
                    best_score = overlap + ratio
                    faculty_found = info

        if faculty_found:
            desc = faculty_found.get("description", "No description available.")
            edu = ", ".join(faculty_found.get("education", []))
            dept = faculty_found.get("department", "CICT")
            title = faculty_found.get("title", "Faculty Member")

            reply = (
                f"{faculty_found['name']} is the {title} under the {dept} department. "
                f"{desc} {'Educational background: ' + edu if edu else ''}"
            )
            if faculty_found.get("url"):
                reply += f" Learn more here: {faculty_found['url']}"
            return jsonify({"reply": reply, "sources": [{"source": "cict_faculty.json"}]})

        # --- Role-based lookups ---
        role_map = {
            "dean": "Dean, College of Information and Communications Technology",
            "associate dean": "Associate Dean, CICT",
            "program chair": "Program Chair, BSIT",
            "chair": "Program Chair, BSIT",
            "coordinator": "Program Coordinator, WMAD/BA/SM"
        }
        for role_key, role_title in role_map.items():
            if role_key in clean_msg:
                for key, info in faculty_data.items():
                    if role_title.lower() in info["title"].lower():
                        desc = info.get("description", "No description available.")
                        edu = ", ".join(info.get("education", []))
                        dept = info.get("department", "CICT")
                        reply = (
                            f"{info['name']} is the {info['title']} under the {dept} department. "
                            f"{desc} {'Educational background: ' + edu if edu else ''}"
                        )
                        if info.get("url"):
                            reply += f" Learn more here: {info['url']}"
                        return jsonify({"reply": reply, "sources": [{"source": "cict_faculty.json"}]})
                    
        if is_about_person:
            return jsonify({"reply": "I couldn't find that faculty member in my records."})
        response, sources, model_name = loop.run_until_complete(model_manager.get_response(message))
        for src in sources:
            if "page" in src:
                src["page"] = None
        clean_response = re.sub(
            r'(\(?\s*(as\s+mentioned\s+in|according\s+to|based\s+on)\s+(the\s+)?(document|file|pdf)?[^.,)]*(\.pdf)?(,?\s*Page\s*\d+)?\)?[.,]?)',
            '',
            response,
            flags=re.IGNORECASE
        )
        clean_response = re.sub(r'\s*\([^)]*\.pdf[^)]*\)', '', clean_response)
        clean_response = re.sub(r',?\s*Page\s*\d+', '', clean_response)
        clean_response = re.sub(r'【([^】]*),?\s*Page\s*\d+】', r'【\1】', clean_response)
        clean_response = re.sub(r'\s{2,}', ' ', clean_response).strip()

        cleaned_sources = [
            {**src, "source": re.sub(r',?\s*Page\s*\d+', '', src.get("source", ""))}
            for src in sources
        ]

        return jsonify({"reply": clean_response, "sources": cleaned_sources})

    except aiohttp.ClientConnectorError:
        return jsonify({"reply": "⚠️ Unable to connect to remote API. Running in offline mode."})
    except Exception as e:
        print(f"[ERROR] chat endpoint: {e}")
        return jsonify({"reply": "⚠️ An internal error occurred while processing your request."})

@app.route("/crawl", methods=["GET"])
def trigger_crawl():
    async def run_crawler():
        crawler = CICTWebCrawler(loop)
        profiles, texts = await crawler.crawl_site()
        data = {p[1]['name']: p[1] for p in profiles}
        save_json_safely(FACULTY_JSON_PATH, data)
        return f"✅ Crawl complete. Found {len(data)} faculty members."

    asyncio.run(run_crawler())
    return "Started crawling... check logs."
    

@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_loop():
        loop.stop()
    threading.Thread(target=stop_loop, daemon=True).start()
    return "Shutting down loop", 200


# Startup
if __name__ == "__main__":
    print("🚀 Starting CICTify Flask Chatbot (Render-Ready Mode) with JSON Faculty")
    if not GUI_DIR.exists():
        print(f"[WARNING] GUI directory not found: {GUI_DIR}")

    try:
        loop.run_until_complete(init_model_manager())
    except Exception as e:
        print(f"[WARN] init_model_manager error: {e}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


