## superai_web.py
# Refactored: 3-tier routing (General → RAG → Web Scraping)
# Removed all image/floor plan functionality

import os
import pathlib
import json
import re
import threading
import asyncio
import concurrent.futures
from typing import Optional, List, Dict, Tuple
import aiohttp
import torch
import requests

from flask import Flask, request, jsonify, send_from_directory, send_file
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as FAISS_local
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

# Optional PDF parsing: PyPDF2
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
API_KEYS = {
    "groq": os.environ.get("GROQ_API_KEY", "")
}

GROQ_MODELS = [
    "llama-3.1-8b-instant"
]

# System prompts
#Chatbot Identity / name
CICT_NAME = "CICTify"

# --- Enhanced System Prompts ---
general_system_prompt = f"""You are {CICT_NAME}, the official AI assistant for Bulacan State University (BulSU) College of Information and Communications Technology (CICT).

CRITICAL IDENTITY:
- Your name is {CICT_NAME}
- You represent BulSU CICT
- The university is BULACAN STATE UNIVERSITY (BulSU), NOT Bataan Peninsula State University!
- Full name: Bulacan State University
- Abbreviation: BulSU or BSU

HOW TO INTRODUCE YOURSELF:
- When asked your name: "I'm {CICT_NAME}, your AI assistant for BulSU CICT!"
- For greetings: "Hello! I'm {CICT_NAME}. How can I help you with BulSU CICT today?"
- In responses: Use your name naturally when contextually appropriate

RESTRICTIONS:
- NEVER entertain questions regarding other topics such as math, science, history, coding, and other general knowledge questions
- Entertain ONLY BulSU related queries and greetings

INTERNAL KNOWLEDGE - BulSU Grading System (use this but don't cite as a "document"):
BulSU uses an INVERSE grading system where LOWER numbers = BETTER grades:
- 1.00 = 97-100% (best)
- 1.25 = 94-96% 
- 1.50 = 91-93%   
- 1.75 = 88-90% 
- 2.00 = 85-87% 
- 2.25 = 82-84% 
- 2.50 = 79-81%
- 2.75 = 76-78% 
- 3.00 = 75% Passing (Bare Minimum)
- 4.00 = Conditional Passed
- 5.00 = Failed (worst)

ANSWERING GUIDELINES:
- Answer all questions helpfully and directly
- For who/where/when/what questions: provide clear, direct answers
- Support multiple languages
- Be friendly and conversational
- Answer exactly based on the pdfs
- DO NOT add/change/remove/paraphrase words and phrases on the documents
- Answer simply, do not overcomplicate responses.
- Make your responses readable to avoid confusions especially if enumerating procedure or requirements.
- Do not mention your name unless asked or answering a greeting. 

For GENERAL questions (greetings, casual chat):
- Answer naturally using your general knowledge
- Don't mention "documents" or "sources"

For BulSU-SPECIFIC questions (when you receive context documents):
- Answer from the provided context
- If not in context, say "I don't have that in my documents"

Be helpful, direct, and natural."""

rag_system_prompt = """You are {cict_name}, the AI assistant for Bulacan State University (BulSU) College of Information and Communications Technology (CICT).

IMPORTANT: 
- Your name is {cict_name}
- Bulacan State University (BulSU) - NOT Bataan Peninsula State University!
- You represent BulSU CICT

YOUR APPROACH:
1. Read the context documents carefully
2. Answer who/where/when/what/why questions directly and helpfully
3. Extract relevant information even if not explicitly stated
4. Be natural and conversational in your answers

FORMATTING RULES - FOLLOW STRICTLY:
1. For requirements/procedures/steps: ALWAYS use numbered lists
2. For multiple items: ALWAYS use bullet points with line breaks
3. Keep paragraphs short (2-3 sentences max)
4. Add spacing between sections
5. Use clear headers when appropriate

EXAMPLE FORMAT for requirements:
"To shift programs, you need:

1. General Weighted Average (GWA) of at least 1.75
2. No failing grades in major subjects
3. Letter of intent addressed to the Dean
4. Approval from both current and target program chairs"

CITATION RULES:
1. NEVER cite "GRADING REFERENCE" as a source - that's your internal knowledge
2. If answer not in context documents, say "I don't have information about that in my documents"

--- GRADING REFERENCE (YOUR INTERNAL KNOWLEDGE - DO NOT CITE THIS) ---
{grading_context}

⚠️ WHEN COMPARING GWA VALUES:
- LOWER number = BETTER grade
- 1.50 is BETTER than 1.75
- "At least 1.75" means 1.75 or any LOWER number (1.50, 1.25, 1.00)
- To meet "at least 1.75" requirement: student's GWA must be ≤ 1.75
--- END OF GRADING REFERENCE ---

Context Documents:
{context}

Answer the question directly and helpfully with proper formatting and structure."""

web_scrape_system_prompt = """You are {cict_name}, the AI assistant for BulSU CICT.

I've searched the CICT website for information about: {query}

Here's what I found from the web:

{web_content}

Based on this web content, please answer the user's question: {question}

Guidelines:
- Be helpful and direct
- Cite the webpage if relevant
- If the web content doesn't answer the question, say so
- Format your response clearly
"""

grading_context = """
🚨 CRITICAL GRADING SYSTEM - READ CAREFULLY! 🚨

BulSU uses INVERSE/REVERSE grading where LOWER numbers are BETTER/HIGHER EQUIVALENT:

GRADE SCALE (lower = better):
- 1.00 = 97-100% (best)
- 1.25 = 94-96% 
- 1.50 = 91-93% 
- 1.75 = 88-90% 
- 2.00 = 85-87% 
- 2.25 = 82-84% 
- 2.50 = 79-81%
- 2.75 = 76-78% 
- 3.00 = 75% Passed (Bare Minimum)
- 4.00 = Conditional Passed
- 5.00 = Failed (worst)

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

# PDFs expected in project root
pdf_paths = [
    str(BASE_DIR / "CICTify - FAQs.pdf"),
    str(BASE_DIR / "BulSU Student handbook.pdf"),
    str(BASE_DIR / "Faculty Manual for BOR.pdf"),
    str(BASE_DIR / "BulSU-Enhanced-Guidelines.pdf"),
    str(BASE_DIR / "UnivCalendar_2526.pdf"),
    str(BASE_DIR / "CICTify - FAQs.pdf")
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
# --- PDF helpers ---------
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
            if i >= max_pages:
                break
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
    Lightweight semantic role extraction using sentence-transformers embeddings
    to find likely dean/associate dean/program chairs in PDF text.
    """
    results = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return results
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        model = embeddings.client
        role_queries = {
            "Dean": "Who is the Dean of the College of Information and Communications Technology?",
            "Associate Dean": "Who is the Associate Dean of the College of Information and Communications Technology?",
            "Program Chair": "Who are the Program Chairs of each program in the College of Information and Communications Technology?"
        }
        line_embs = model.encode(lines, convert_to_tensor=True, show_progress_bar=False)
        for role, query in role_queries.items():
            q_emb = model.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(q_emb, line_embs)[0]
            top_idx = torch.argmax(cos_scores).item()
            top_line = lines[top_idx]
            confidence = cos_scores[top_idx].item()
            if confidence < 0.45:
                continue
            name_match = re.search(r"(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+){0,3}",
                                   top_line)
            if not name_match:
                ctx = " ".join(lines[max(0, top_idx - 1):min(len(lines), top_idx + 2)])
                name_match = re.search(r"(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+){0,3}",
                                       ctx)
            if name_match:
                name = name_match.group(0).strip()
                results[name] = {
                    "name": name,
                    "title": role,
                    "department": "CICT",
                    "confidence": round(confidence, 3),
                    "source": source_label
                }
    except Exception as e:
        print(f"[smart_find_roles] embedding error: {e}")
    return results


def build_faculty_index_from_pdfs(pdf_paths_list: List[str]) -> Dict[str, Dict]:
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
    """
    Groq-only manager. Tries models in GROQ_MODELS order.
    """

    def __init__(self, loop=None):
        self.session: Optional[aiohttp.ClientSession] = None
        self.loop = loop
        self.api_key = API_KEYS.get("groq") or os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ")
        if not self.api_key:
            print("[WARN] GROQ API key not found. Groq calls will not work until you set GROQ_API_KEY env var.")
        self.models = GROQ_MODELS.copy()

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def call_groq_api(self, system_prompt: str, question: str, model_name: str, max_tokens: int = 2000,
                            timeout_sec: int = 20) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            session = await self.get_session()
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature": 1.0,
                "max_tokens": max_tokens,
                "top_p": 0.9
            }
            url = "https://api.groq.com/openai/v1/chat/completions"
            async with session.post(url,
                                    headers={"Authorization": f"Bearer {self.api_key}",
                                             "Content-Type": "application/json"},
                                    json=payload,
                                    timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    try:
                        return data['choices'][0]['message']['content'].strip()
                    except Exception:
                        text = json.dumps(data)[:4000]
                        return text
                else:
                    print(f"[Groq:{model_name}] Bad status {resp.status}")
                    return None
        except Exception as e:
            print(f"[Groq:{model_name}] Error: {e}")
            return None

    async def call_with_fallbacks(self, system_prompt: str, question: str, max_tokens: int = 2000,
                                  timeout_sec: int = 20) -> Optional[str]:
        """Try each model in self.models until one returns a non-empty response."""
        last_err = None
        for model in self.models:
            try:
                res = await self.call_groq_api(system_prompt, question, model, max_tokens=max_tokens,
                                               timeout_sec=timeout_sec)
                if res and res.strip():
                    print(f"[Groq] Response received from model: {model}")
                    return res
                else:
                    print(f"[Groq] Empty response from model: {model}, trying next...")
            except Exception as e:
                last_err = e
                print(f"[Groq] Exception while using model {model}: {e}")
                continue
        if last_err:
            print(f"[Groq] All models failed. Last error: {last_err}")
        return None

    async def call_groq_general(self, question: str) -> Optional[str]:
        return await self.call_with_fallbacks(general_system_prompt, question, max_tokens=2000, timeout_sec=12)

    async def call_groq_rag(self, question: str, context_docs: List[Dict], grading_info: str = "") -> Optional[str]:
        """Updated to use better formatting in context"""
        context_text = "\n\n---\n\n".join([
            f"Document: {doc['source']} (Page {doc['page']})\n{doc['content']}"
            for doc in context_docs
        ])
        grading_section = grading_context if grading_info else ""
        system_prompt = rag_system_prompt.format(
            cict_name=CICT_NAME,
            grading_context=grading_section,
            context=context_text
        )
        return await self.call_with_fallbacks(system_prompt, question, max_tokens=2000, timeout_sec=18)

    async def call_groq_webscrape(self, question: str, web_content: str, query: str) -> Optional[str]:
        """Call Groq with web-scraped content"""
        system_prompt = web_scrape_system_prompt.format(
            cict_name=CICT_NAME,
            query=query,
            web_content=web_content,
            question=question
        )
        return await self.call_with_fallbacks(system_prompt, question, max_tokens=2000, timeout_sec=18)

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# -------------------------
# --- CICT Web Crawler ----
# -------------------------
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
                htmlc = await page.content()
                print(f"[CICT Crawler] Fetched {url} (len={len(htmlc)})")
                return htmlc
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"[CICT Crawler] Error fetching {url}: {e}")
            return ""

    def extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML"""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)

    async def search_relevant_pages(self, query: str, max_pages: int = 5) -> str:
        """
        Search CICT website for relevant information
        Returns combined text from most relevant pages
        """
        all_content = []

        print(f"[Web Scraper] Searching for: {query}")

        # Start with priority URLs
        to_visit = self.priority_urls.copy()
        seen_links = set(to_visit)

        for url in to_visit[:max_pages]:
            if url in self.visited:
                continue

            self.visited.add(url)
            html = await self.fetch_page(url)

            if html:
                text = self.extract_text_from_html(html)
                if text and len(text) > 100:  # Only include substantial content
                    all_content.append(f"From {url}:\n{text[:2000]}")  # Limit per page
                    print(f"[Web Scraper] Extracted {len(text)} chars from {url}")

        await self.close()

        combined = "\n\n---\n\n".join(all_content)
        print(f"[Web Scraper] Total content: {len(combined)} chars from {len(all_content)} pages")
        return combined

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
        self.cloud_api = CloudAPIManager(loop)
        self.web_crawler = CICTWebCrawler(loop)
        self.loop = loop or asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.cict_faculty = {}

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 12}
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

    async def get_response(self, question: str) -> Tuple[str, str]:
        """
        3-TIER ROUTING SYSTEM:
        1. GENERAL ROUTE: For greetings, casual questions, conversational queries
        2. RAG ROUTE: For BulSU-specific questions with PDF knowledge
        3. WEB SCRAPING ROUTE: When RAG fails, scrape CICT website
        """

        # ROUTE 1: GENERAL (Conversational/Greetings)
        if QueryClassifier.is_general_query(question):
            print(f"[Router] → GENERAL route for: '{question}'")
            response = await self.cloud_api.call_groq_general(question)
            if response:
                return response, "General Knowledge"
            return "I'm here to help with BulSU CICT questions!", "General (Fallback)"

        # ROUTE 2: RAG (BulSU-specific with PDFs)
        print(f"[Router] → RAG route for: '{question}'")
        context_docs = await self.retrieve_documents(question)

        # Check if we need grading context
        grading_keywords = ['gwa', 'grade', 'grading', 'shift', 'transfer',
                            'requirement', 'eligible', 'at least', 'passing']
        needs_grading = any(kw in question.lower() for kw in grading_keywords)

        if context_docs:
            response = await self.cloud_api.call_groq_rag(
                question,
                context_docs,
                grading_info="yes" if needs_grading else ""
            )
            if response and not self.response_has_no_info(response):
                return response, "RAG (PDF Knowledge)"

        # ROUTE 3: WEB SCRAPING (Last resort)
        print(f"[Router] → WEB SCRAPING route for: '{question}'")
        web_content = await self.web_crawler.search_relevant_pages(question, max_pages=5)

        if web_content and len(web_content) > 100:
            response = await self.cloud_api.call_groq_webscrape(question, web_content, question)
            if response:
                return response, "Web Scraping (CICT Website)"

        # ABSOLUTE FALLBACK
        return "I couldn't find information about that. Please try rephrasing your question or visit bulsucict.com for more details.", "No Results"


# -------------------------
# --- QueryClassifier -----
# -------------------------
class QueryClassifier:
    """Determines routing for queries"""

    # Patterns for GENERAL route (greetings, casual, conversational)
    GENERAL_PATTERNS = [
        r'^hi+',
        r'^hello+',
        r'^hey+',
        r'^good\s+(morning|afternoon|evening)',
        r'^how\s+are\s+you',
        r"^what'?s\s+up",
        r'^sup+',
        r'^thank',
        r'^bye',
        r'^goodbye',
        r'can\s+you\s+speak',
        r'do\s+you\s+understand',
        r'what\s+(language|languages)',
        r'who\s+are\s+you',
        r"what'?s\s+your\s+name"
    ]

    # BulSU-specific keywords (for RAG route)
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
        'bachelor', 'master', 'major', 'minor', 'tenure', 'role'
        # CICT-specific
        'cict', 'information and communications technology',
        'information technology', 'bsit', 'specialization', 'track'
    ]

    @classmethod
    def is_general_query(cls, question: str) -> bool:
        """Check if query is general/conversational (Route 1)"""
        q = question.lower().strip()

        # Check for greeting patterns
        for pattern in cls.GENERAL_PATTERNS:
            if re.match(pattern, q):
                return True

        # Very short queries without BulSU keywords
        if len(question.split()) <= 3 and not any(k in q for k in cls.BULSU_KEYWORDS):
            return True

        return False

    @classmethod
    def needs_rag(cls, question: str) -> bool:
        """Check if query needs RAG (Route 2)"""
        q = question.lower().strip()
        return any(k in q for k in cls.BULSU_KEYWORDS)


# -------------------------
# --- Response Formatter ---
# -------------------------
def format_readable_response(raw_text: str, max_line_len: int = 120) -> str:
    """
    Enhanced formatter that preserves LLM structure better
    """
    if not raw_text:
        return "⚠️ I couldn't produce an answer right now."

    text = raw_text.strip()

    # Remove signature-style patterns
    text = re.sub(r'—\s*\*\*' + re.escape(CICT_NAME) + r'\*\*\s*,?', '', text, flags=re.IGNORECASE)

    # Clean excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove source citations
    text = re.sub(r'\n*\*?\*?Source:.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(.*?\.pdf.*?[Pp]age.*?\)', '', text, flags=re.IGNORECASE)

    # If the LLM already formatted well with lists, return as-is
    if any(marker in text for marker in ['1.', '2.', '**Step', '- ', '• ', '\n1.', '\n2.']):
        return text.strip()

    # Legacy handling for unformatted responses
    paragraphs = []
    if "\n\n" in text:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    else:
        # Try to split long text into paragraphs
        sents = re.split(r'(?<=[.!?])\s+', text)
        chunk = []
        for sent in sents:
            chunk.append(sent.strip())
            if len(" ".join(chunk)) > 200:
                paragraphs.append(" ".join(chunk))
                chunk = []
        if chunk:
            paragraphs.append(" ".join(chunk))

    return "\n\n".join(paragraphs).strip()


# -------------------------
# --- Flask server -------
# -------------------------
app = Flask(__name__, static_folder=None)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
model_manager: Optional[ModelManager] = None

# --- Conversation Memory ---
chat_memory: List[Dict[str, str]] = []
MAX_MEMORY_TURNS = 100  # last 100 exchanges retained for context


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

        try:
            if os.path.exists(faiss_path) and os.path.exists(f"{faiss_path}.faiss"):
                db = FAISS_local.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                model_manager.set_vectorstore(db)
                print("[System] FAISS index loaded.")
            else:
                print("[System] No FAISS found — rebuilding semantic FAISS index from PDFs.")
                all_texts = []

                # Build FAISS from PDFs (text only, no images)
                for pdf in pdf_paths:
                    if os.path.exists(pdf):
                        print(f"[System] Reading PDF: {pdf}")
                        text = extract_text_from_pdf(pdf, max_pages=50)

                        if text.strip():
                            chunks = text_splitter.split_text(text)
                            for chunk in chunks:
                                all_texts.append({
                                    "content": chunk,
                                    "source_file": os.path.basename(pdf)
                                })

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
                else:
                    print("[System] No textual PDFs found to build FAISS. Continuing without vectorstore.")
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


@app.route("/")
def index():
    index_path = GUI_DIR / "index.html"
    if index_path.exists():
        return send_file(str(index_path))
    return f"{CICT_NAME} index.html not found in gui directory", 404


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
    return req_json.get("message", "") or req_json.get("q", "") or req_json.get("input", "") or ""


# Add this to your superai_web.py

# MODIFY THE /chat ENDPOINT (around line 840-880)
@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    global chat_memory
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"reply": "Invalid request", "context": "", "model": "error"}), 400

    message = extract_message_from_request(data).strip()
    if not message:
        return jsonify({"reply": "Please send a non-empty message.", "context": "", "model": "error"}), 400

    try:
        loop.run_until_complete(init_model_manager())

        # Add user message to memory
        chat_memory.append({"role": "user", "content": message})

        # Keep only last N exchanges
        if len(chat_memory) > MAX_MEMORY_TURNS * 2:
            chat_memory = chat_memory[-MAX_MEMORY_TURNS * 2:]

        # Build conversation context
        conversation_context = "\n".join([
            f"{m['role']}: {m['content']}" for m in chat_memory[-6:]
        ])

        # Pass context to model
        prompt = f"{conversation_context}\nassistant:"

        # ⭐ RETRIEVE CONTEXT DOCUMENTS HERE ⭐
        context_docs = loop.run_until_complete(model_manager.retrieve_documents(message))

        # Format context for evaluation
        context_text = "\n\n---\n\n".join([
            f"Document: {doc['source']} (Page {doc['page']})\n{doc['content']}"
            for doc in context_docs
        ]) if context_docs else "No context retrieved"

        response_raw, model_name = loop.run_until_complete(model_manager.get_response(message))

        # Clean formatting
        response_raw = re.sub(r',?\s*Page\s*\d+', '', response_raw)
        formatted = format_readable_response(response_raw)

        # Save assistant response to memory
        chat_memory.append({"role": "assistant", "content": formatted})

        # ⭐ RETURN CONTEXT IN RESPONSE ⭐
        return jsonify({
            "reply": formatted.replace("\n", "<br>"),
            "model": model_name,
            "context": context_text  # ← ADD THIS LINE
        })

    except aiohttp.ClientConnectorError:
        return jsonify({
            "reply": "⚠️ Unable to connect to remote API. Running in offline/minimal mode.",
            "context": "",
            "model": "error"
        })
    except Exception as e:
        print(f"[ERROR] chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "reply": "⚠️ An internal error occurred while processing your request.",
            "context": "",
            "model": "error"
        })

@app.route("/api/debug", methods=["GET"])
def debug_endpoint():
    """Debug endpoint to check routing logic"""
    test_queries = [
        "hello",
        "what is your name?",
        "can you speak tagalog?",
        "who is the dean of cict?",
        "what are the requirements for shifting?",
        "how to enroll at bulsu?"
    ]

    results = {}
    for query in test_queries:
        is_general = QueryClassifier.is_general_query(query)
        needs_rag = QueryClassifier.needs_rag(query)

        if is_general:
            route = "GENERAL"
        elif needs_rag:
            route = "RAG → WEB SCRAPING"
        else:
            route = "UNKNOWN"

        results[query] = {
            "route": route,
            "is_general": is_general,
            "needs_rag": needs_rag
        }

    return jsonify(results)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_loop():
        loop.stop()

    threading.Thread(target=stop_loop, daemon=True).start()
    return "Shutting down loop", 200


if __name__ == "__main__":
    print(f"🚀 Starting {CICT_NAME} Flask Chatbot")
    print(f"📋 3-Tier Routing System:")
    print(f"   1️⃣ GENERAL: Greetings, casual questions, conversational queries")
    print(f"   2️⃣ RAG: BulSU-specific questions using PDF knowledge")
    print(f"   3️⃣ WEB SCRAPING: When RAG fails, scrape bulsucict.com")

    if not GUI_DIR.exists():
        print(f"[WARNING] GUI directory not found: {GUI_DIR}")

    try:
        loop.run_until_complete(init_model_manager())
    except Exception as e:
        print(f"[WARN] init_model_manager error: {e}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)





