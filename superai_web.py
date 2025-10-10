# superai_web.py
# Standalone file (AI core + improved CICT webscraper + Flask web server that serves your web GUI)
# Save at D:\BulsuAssistant\superai_web.py
#test
import html
import os
import threading
import asyncio
import aiohttp
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import concurrent.futures
import subprocess
import platform
import re
from typing import Optional, List, Dict, Tuple
import json
from pathlib import Path
import time
import pathlib

# -------------------------
# --- Configuration ------
# -------------------------

BASE_DIR = pathlib.Path(__file__).parent.resolve()
GUI_DIR = os.path.join(BASE_DIR, "gui")
STATIC_DIR = os.path.join(GUI_DIR, "images")
PDF_DIR = BASE_DIR 

app = Flask(__name__, static_folder=STATIC_DIR)
# PDF paths
pdf_paths = [
    r"D:\BulsuAssistant\guide.pdf",
    r"D:\BulsuAssistant\BulSU Student handbook.pdf",
    r"D:\BulsuAssistant\FacultyManual.pdf",
]

faiss_path = "vectorstore/faiss_index"

# API keys
API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY", "")
}

# CICT faculty JSON file
FACULTY_JSON_PATH = Path(r"D:\BulsuAssistant\cict_faculty.json")
FACULTY_CACHE_PATH = Path("cict_faculty_cache.json")
FACULTY_CACHE_TTL_SECONDS = 24 * 60 * 60  # 1 day

# Load CICT faculty profiles from JSON
def load_cict_faculty_profiles() -> Dict[str, Dict]:
    try:
        if FACULTY_JSON_PATH.exists():
            with FACULTY_JSON_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        print(f"[System] Faculty JSON not found at {FACULTY_JSON_PATH}")
        return {}
    except Exception as e:
        print(f"[System] Error loading faculty JSON: {e}")
        return {}

# -------------------------
# --- Prompts & Context ---
# -------------------------

general_system_prompt = """You are a helpful assistant for Bulacan State University (BulSU).

CRITICAL: The university is BULACAN STATE UNIVERSITY (BulSU), NOT Bataan Peninsula State University!
- Full name: Bulacan State University
- Abbreviation: BulSU
- Location: Bulacan province, Philippines
- Main campus: Malolos City, Bulacan

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

For GENERAL questions (greetings, languages, common knowledge, casual chat, simple comparisons):
- Answer naturally using your general knowledge
- For grade comparisons, use the scale above
- Don't mention "documents" or "sources"

For BulSU-SPECIFIC questions (when you receive context documents):
- Answer from the provided context
- Cite PDF document and page
- If not in context, say "I don't have that in my documents"

Be helpful, direct, and natural."""

rag_system_prompt = """You are an assistant for Bulacan State University (BulSU).

IMPORTANT: Bulacan State University (BulSU) - NOT Bataan Peninsula State University!

YOUR APPROACH:
1. Read the context documents carefully
2. Answer who/where/when/what/why questions directly and helpfully
3. Extract relevant information even if not explicitly stated
4. Be natural and conversational in your answers
5. Always cite your sources (PDF name and page)

CITATION RULES:
1. Answer using the CONTEXT DOCUMENTS section below
2. ONLY cite actual PDF documents (guide.pdf, BulSU Student handbook.pdf, FacultyManual.pdf)
3. NEVER cite "GRADING REFERENCE" as a source - that's your internal knowledge
4. Always mention which PDF document (filename) and page number
5. If answer not in context documents, say "I don't have information about that in my documents"

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

Answer the question directly and helpfully. Cite document name and page number.
"""

grading_context = """
🚨 CRITICAL GRADING SYSTEM - READ CAREFULLY! 🚨

BulSU uses INVERSE/REVERSE grading where LOWER numbers = BETTER:

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

# ------------------------- 
# --- Query Classifier ----
# -------------------------
class QueryClassifier:
    """Determines if query needs RAG or general knowledge"""

    BULSU_KEYWORDS = [
        'bulsu', 'bulacan state', 'university', 'bsu',
        'mission', 'vision', 'core values', 'mandate',
        'history', 'established', 'founded',
        'campus', 'campuses', 'location', 'address', 'where',
        'malolos', 'bustos', 'san jose', 'hagonoy', 'matungao',
        'osoa', 'osa', 'registrar', 'cashier', 'library',
        'office of student affairs', 'student affairs',
        'gwa', 'grade', 'grading', 'credit', 'unit', 'course', 'subject',
        'enroll', 'enrollment', 'registration', 'curriculum', 'syllabus',
        'shift', 'transfer', 'shifter', 'transferee',
        'exam', 'midterm', 'final', 'quiz', 'requirement',
        'dean', 'professor', 'faculty', 'instructor',
        'policy', 'policies', 'rule', 'regulation', 'procedure',
        'requirement', 'requirements', 'eligibility', 'qualified',
        'petition', 'appeal', 'clearance', 'document',
        'scholarship', 'financial aid', 'tuition', 'fee',
        'admission', 'graduate', 'graduation', 'honors',
        'student handbook', 'code of conduct', 'discipline',
        'organization', 'club', 'facility', 'laboratory', 'clinic',
        'office', 'department', 'college', 'program',
        'bachelor', 'master', 'major', 'minor'
    ]

    CASUAL_PATTERNS = [
        r'^hi+$', r'^hello+$', r'^hey+$', r'^good\s+(morning|afternoon|evening)',
        r'^how\s+are\s+you', r'^what\'?s\s+up', r'^sup+$',
        r'^thank', r'^bye', r'^goodbye', r'^see\s+you'
    ]

    @classmethod
    def needs_rag(cls, question: str) -> bool:
        question_lower = question.lower().strip()

        # Check for casual greetings
        for pattern in cls.CASUAL_PATTERNS:
            if re.match(pattern, question_lower):
                return False

        # Simple grade comparisons don't need RAG
        simple_comparison_patterns = [
            r'which.*better.*\d\.\d+.*\d\.\d+',
            r'better.*\d\.\d+.*or.*\d\.\d+',
            r'compare.*\d\.\d+.*\d\.\d+',
            r'^\d\.\d+.*or.*\d\.\d+.*better',
            r'^is.*\d\.\d+.*better.*than.*\d\.\d+'
        ]
        for pattern in simple_comparison_patterns:
            if re.search(pattern, question_lower):
                return False

        # "grading system" alone doesn't need RAG if it's just asking what it is
        if re.match(r'^(what is |explain |tell me about )?(the )?(bulsu )?grading system\??$', question_lower):
            return False

        # Questions about mission/vision/campuses/offices always need RAG
        priority_keywords = ['mission', 'vision', 'campus', 'office', 'osoa', 'osa',
                             'location', 'where', 'address', 'established', 'history']
        if any(kw in question_lower for kw in priority_keywords):
            return True

        # Very short questions are usually casual
        if len(question.split()) <= 2 and not any(kw in question_lower for kw in cls.BULSU_KEYWORDS):
            return False

        # Check for BulSU-specific keywords
        return any(keyword in question_lower for keyword in cls.BULSU_KEYWORDS)

# ------------------------- 
# --- PDF Page Opener -----
# -------------------------
class PDFPageOpener:
    @staticmethod
    def open_pdf_page(pdf_path: str, page_number: int):
        """Open PDF at specific page"""
        try:
            system = platform.system()

            if system == "Windows":
                sumatra_paths = [
                    r"C:\Program Files\SumatraPDF\SumatraPDF.exe",
                    r"C:\Program Files (x86)\SumatraPDF\SumatraPDF.exe"
                ]

                sumatra_found = False
                for sumatra_path in sumatra_paths:
                    if os.path.exists(sumatra_path):
                        subprocess.Popen([sumatra_path, "-page", str(page_number), pdf_path])
                        sumatra_found = True
                        break

                if not sumatra_found:
                    os.startfile(pdf_path)

            elif system == "Darwin":  # macOS
                subprocess.run(["open", "-a", "Preview", pdf_path])

            else:  # Linux
                subprocess.run(["xdg-open", pdf_path])

        except Exception as e:
            print(f"Error opening PDF: {e}")
# ------------------------- 
# --- Cloud API Manager ---
# -------------------------
class CloudAPIManager:
    def __init__(self, loop=None):
        self.session = None
        self.loop = loop

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def call_groq_general(self, question: str) -> Optional[str]:
        """Call Groq for general knowledge questions (no RAG)"""
        if not API_KEYS["groq"]:
            return None

        try:
            session = await self.get_session()

            async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEYS['groq']}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": general_system_prompt},
                            {"role": "user", "content": question}
                        ],
                        "temperature": 1.0,
                        "max_tokens": 300
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip()
                return None
        except Exception as e:
            print(f"Groq general API error: {e}")
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
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                        "temperature": 1.0,
                        "max_tokens": 500
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

# ------------------------- 
# --- Improved CICT Web Crawler ----
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
                html = await page.content()
                print(f"[CICT Crawler] Successfully fetched {url} (len={len(html)})")
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

        title_candidates = []
        for selector in ["p", "h3", "h4", "div"]:
            for el in soup.find_all(selector, limit=10):
                t = el.get_text(" ", strip=True)
                tl = t.lower()
                if len(t) < 100 and any(k in tl for k in ["dean", "associate", "chair", "program", "faculty", "coordinator", "head", "director", "professor", "lecturer", "instructor"]):
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

        print(f"[CICT Crawler] Starting crawl of {base_url} (max_pages={max_pages}, non-faculty only)")

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
                    print(f"[CICT Crawler] Failed or empty result for {url}")
                    continue

                if self.is_faculty_profile_url(url):
                    continue

                soup = BeautifulSoup(result, "html.parser")
                for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(" ", strip=True)
                scraped_texts.append((url, text[:4000]))

                try:
                    soup2 = BeautifulSoup(result, "html.parser")
                    for a in soup2.find_all("a", href=True):
                        full = urljoin(base_url, a["href"].split('#')[0])
                        if full.startswith(base_url) and full not in self.visited:
                            to_visit.append(full)
                            seen_links.add(full)
                except Exception:
                    pass

            if len(self.visited) >= max_pages:
                break

        await self.close()
        print(f"[CICT Crawler] Finished. Non-faculty pages: {len(scraped_texts)}")
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
        self.loop = loop
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

    @staticmethod
    def response_has_no_info(response: str) -> bool:
        no_info_phrases = [
            "don't have information",
            "don't have that information",
            "don't have any information",
            "no information",
            "not mentioned",
            "not found",
            "couldn't find",
            "unable to find",
            "not available",
            "cannot find",
            "i don't know",
            "not in my documents",
            "not in the documents"
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in no_info_phrases)

    async def retrieve_documents(self, question: str) -> List[Dict]:
        if not self.retriever:
            return []

        try:
            docs = await self.loop.run_in_executor(
                self.executor,
                lambda: self.retriever.invoke(question)
            )

            formatted_docs = []
            for doc in docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                pdf_path = doc.metadata.get("pdf_path", "")
                page_num = doc.metadata.get("page", 0)

                formatted_docs.append({
                    "content": doc.page_content,
                    "source": source_file,
                    "page": page_num + 1,
                    "pdf_path": pdf_path
                })

            return formatted_docs

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    async def get_response(self, question: str) -> Tuple[str, List[Dict], str]:
        # Load faculty profiles
        faculty_index = load_cict_faculty_profiles()

        # Normalize message
        lower_msg = question.lower().strip()
        clean_msg = re.sub(r"[^a-z\s]", "", lower_msg)

        # Detect if the question relates to BulSU CICT or faculty
        academic_keywords = [
            'grading', 'gwa', 'honor', 'honors', 'grade', 'requirement', 'requirements',
            'passing', 'scholarship', 'curriculum', 'student handbook'
        ]
        cict_keywords = [
            'cict', 'bulsucict', 'college of information', 'faculty', 'professor',
            'dean', 'associate dean', 'chair', 'program chair', 'coordinator',
            'instructor', 'lecturer', 'teacher', 'faculty member'
        ]

        q_lower = question.lower().strip()
        clean_msg = re.sub(r"[^a-z\s]", "", q_lower)

        is_academic_query = any(word in q_lower for word in academic_keywords)
        is_cict_query = any(word in q_lower for word in cict_keywords)

        # Faculty JSON loaded earlier
        faculty_index = load_cict_faculty_profiles()

        # Smart routing: distinguish academic vs faculty
        if is_cict_query and not is_academic_query:
            print("[System] Detected CICT faculty/website query - using JSON or scraper.")
            # (Existing faculty detection + scraping logic follows this)
        elif is_academic_query:
            print("[System] Academic query detected - skip CICT scraping, use RAG or Groq General.")
            use_rag = QueryClassifier.needs_rag(question)
            if use_rag:
                context_docs = await self.retrieve_documents(question)
                if context_docs:
                    response = await self.cloud_api.call_groq_rag(question, context_docs)
                    return response, context_docs, "Groq (RAG)"
            response = await self.cloud_api.call_groq_general(question)
            return response or "I'm having trouble answering that right now.", [], "Groq (General)"

        # Detect if it directly mentions a faculty member’s name
        name_detected = False
        if faculty_index:
            for name_key in faculty_index.keys():
                name_clean = re.sub(r"[^a-z\s]", "", name_key.lower())
                key_tokens = [t for t in name_clean.split() if len(t) > 2]
                if sum(1 for t in key_tokens if t in clean_msg) >= 2:
                    name_detected = True
                    print(f"[DEBUG] Name detected in query → {name_key}")
                    break

        # This will now classify any faculty-related or name-based query as CICT
        is_cict_query = (
            any(keyword in lower_msg for keyword in cict_keywords)
            or name_detected
        )

        if is_cict_query and faculty_index:
            print("[DEBUG] Classified as CICT-related → Using JSON faculty data.")

        
        if is_cict_query and faculty_index:
            print("[System] Detected CICT query - checking JSON faculty data...")
            
            lower_msg = question.lower()

            # 1) Specific name or role-based faculty match
            print("[DEBUG] Checking for specific name or role match...")
            # --- A) Final Robust NAME match (handles "who is", "tell me about", or just the name) ---
            name_hit = None

            # Clean and normalize input
            clean_msg = re.sub(r"[^a-z\s]", "", lower_msg).strip()

            for name_key, profile in faculty_index.items():
                # Normalize faculty name
                key_clean = re.sub(r"[^a-z\s]", "", name_key).strip()

                # Tokenize and remove short tokens (like middle initials)
                msg_tokens = [t for t in clean_msg.split() if len(t) > 2]
                key_tokens = [t for t in key_clean.split() if len(t) > 2]

                # Check how many overlapping tokens exist
                overlap = sum(1 for t in key_tokens if t in msg_tokens)

                # Match if at least 2 name tokens overlap (e.g. first + last)
                if overlap >= 2:
                    name_hit = profile
                    print(f"[DEBUG] ✅ Matched name → {profile['name']} ({overlap} overlapping tokens)")
                    break

            # Fallback: handle "who is"/"tell me about" phrasing for single-token names
            if not name_hit:
                match = re.search(r"(who\s+(is|s)|tell\s+me\s+about)\s+([a-z\s]+)", clean_msg)
                if match:
                    candidate = match.group(3).strip()
                    for name_key, profile in faculty_index.items():
                        key_clean = re.sub(r"[^a-z\s]", "", name_key)
                        if candidate in key_clean:
                            name_hit = profile
                            print(f"[DEBUG] ✅ Secondary match by phrase → {profile['name']}")
                            break

            # If a name match is found → build profile summary
            if name_hit:
                details = []
                details.append(f"{name_hit['name']} is the {name_hit['title']}.")
                if name_hit.get('department') and name_hit['department'] not in ["", "N/A"]:
                    details.append(f"Department: {name_hit['department']}")
                if name_hit.get('education') and name_hit['education'] not in ["", "N/A"]:
                    details.append(f"Education: {'; '.join(name_hit['education'])}")
                if name_hit.get('certifications') and name_hit['certifications'] not in ["", "N/A"]:
                    details.append(f"Certifications: {'; '.join(name_hit['certifications'])}")
                if name_hit.get('description') and name_hit['description'] not in ["", "N/A"]:
                    details.append(f"About: {name_hit['description']}")
                if name_hit.get('url') and name_hit['url'] not in ["", "N/A"]:
                    details.append(f"Profile: {name_hit['url']}")
                reply = "\n".join(details)
                return reply, [], "JSON CICT"

            # --- B) Role match (dean, associate dean, program chair, coordinator, etc.) ---
            print("[DEBUG] Starting role matching block")

            roles = {
                "associate dean": [
                    p for p in faculty_index.values()
                    if re.search(r"\bassociate\b.*\bdean\b", p['title'].lower())
                ],
                "dean": [
                    p for p in faculty_index.values()
                    if "dean" in p['title'].lower() and "associate" not in p['title'].lower()
                ],
                "program chair": [
                    p for p in faculty_index.values()
                    if "program chair" in p['title'].lower()
                ],
                "coordinator": [
                    p for p in faculty_index.values()
                    if "coordinator" in p['title'].lower()
                ],
                "regular faculty": [
                    p for p in faculty_index.values()
                    if "regular faculty" in p['title'].lower()
                ],
                "part-time": [
                    p for p in faculty_index.values()
                    if "part-time" in p['title'].lower()
                ],
                "internship coordinator": [
                    p for p in faculty_index.values()
                    if "internship" in p['title'].lower() and "coordinator" in p['title'].lower()
                ]
            }

            # Debug summary
            print("[DEBUG] Role matches summary:")
            for role, matches in roles.items():
                print(f"  {role}: {len(matches)} match(es)")

            for role, matches in roles.items():
                plural_role = role + "s"
                plural_role_alt = role.rstrip('y') + 'ies' if role.endswith('y') else plural_role

                # Match “who is”, “who’s”, “dean of cict”, “cict dean”, etc.
                role_patterns = [
                    rf"\b{role}\b",
                    rf"\b{plural_role}\b",
                    rf"\b{plural_role_alt}\b",
                    rf"who(['’]s| is| is the)?\s+(the\s+)?{role}",
                    rf"{role}\s+of\s+(the\s+)?cict",
                    rf"cict\s+{role}"
                ]

                if any(re.search(pat, lower_msg) for pat in role_patterns):
                    print(f"[DEBUG] Detected role '{role}' in message.")
                    if matches:
                        if len(matches) == 1:
                            top = matches[0]
                            reply = f"{top['name']} is the {top['title']}."
                            if top.get('url'):
                                reply += f" Profile: {top['url']}"
                        else:
                            reply_lines = [f"{role.title()}s:"]
                            reply_lines += [f"- {m['name']}" for m in matches]
                            reply = "\n".join(reply_lines)
                        print(f"[DEBUG] Reply formed for role '{role}' → {reply}")
                        return reply, [], "JSON CICT"

            # 3) List all if general "faculty"
            if "faculty" in lower_msg and any(k in lower_msg for k in ["list", "all", "who are"]):
                names = [p['name'] for p in faculty_index.values()]
                reply = "CICT Faculty:\n" + "\n".join(f"- {n}" for n in names)
                reply += f"\n({len(names)} total, ask for details)."
                return reply, [], "JSON CICT"

            # Fallback to FAISS or scrape for non-faculty (announcements, etc.)
            context_docs = await self.retrieve_documents(question)
            if not context_docs:
                print("[System] No cached CICT data - scraping non-faculty...")
                crawler = CICTWebCrawler(self.loop)
                _, scraped_pages = await crawler.crawl_site("https://bulsucict.com", max_pages=20)
                if scraped_pages:
                    context_docs = [{"content": text, "source": url, "page": 1} for url, text in scraped_pages]
                    try:
                        from langchain.schema import Document
                        from langchain_huggingface import HuggingFaceEmbeddings
                        from langchain_community.vectorstores import FAISS
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        docs = [Document(page_content=doc["content"], metadata={"source_file": doc["source"], "page": 0}) for doc in context_docs]
                        if self.vectorstore:
                            self.vectorstore.add_documents(docs)
                            self.vectorstore.save_local(faiss_path)
                    except Exception as e:
                        print(f"Error saving CICT to FAISS: {e}")
            
            if context_docs:
                response = await self.cloud_api.call_groq_rag(question, context_docs[:6])
                if response:
                    return response, [], "Groq (CICT Non-Faculty)"
            # No faculty or CICT-specific data found — fall back to general RAG
            print("[System] No faculty data matched — falling back to general Groq RAG...")

        # Regular query classification for PDFs
        use_rag = QueryClassifier.needs_rag(question)
        print(f"DEBUG: Question: '{question}' | Use RAG: {use_rag}")
        # --- FIX: If FAISS is missing or RAG returns no context, use Groq general instead ---
        if use_rag:
            context_docs = await self.retrieve_documents(question)
            if not context_docs:
                print("[Fallback] No RAG context found → using Groq general.")
                response = await self.cloud_api.call_groq_general(question)
                if response:
                    return response, [], "Groq (General Fallback)"

        if not use_rag:
            response = await self.cloud_api.call_groq_general(question)
            if response:
                return response, [], "Groq (General)"
            
            if self.load_local_fallback():
                try:
                    result = await self.loop.run_in_executor(
                        self.executor,
                        lambda: self.local_model.invoke(question)
                    )
                    answer = result.content if hasattr(result, 'content') else str(result)
                    return answer, [], "Local (General)"
                except Exception as e:
                    print(f"Local general error: {e}")
            
            return "I'm having trouble responding. Please try again.", [], "Error"
        
        else:
            context_docs = await self.retrieve_documents(question)
            
            if not context_docs:
                if any(word in question.lower() for word in ['where', 'when', 'who', 'what']):
                    response = await self.cloud_api.call_groq_general(question)
                    if response:
                        return response, [], "Groq (General - No docs found)"
                
                return "I couldn't find relevant information in my documents. Could you rephrase your question?", [], "No Results"
            
            grading_keywords = ['gwa', 'grade', 'grading', 'shift', 'transfer',
                                'requirement', 'requirements', 'qualify', 'eligible',
                                'at least', '1.', '2.', '3.', 'passing']
            needs_grading = any(kw in question.lower() for kw in grading_keywords)
            
            response = await self.cloud_api.call_groq_rag(
                question,
                context_docs,
                grading_info="yes" if needs_grading else ""
            )
            
            if response:
                if self.response_has_no_info(response):
                    return response, [], "Groq (RAG - No relevant info)"
                else:
                    return response, context_docs, "Groq (RAG)"
            
            response = await self.get_local_rag_response(question, context_docs, needs_grading)
            if response:
                if self.response_has_no_info(response):
                    return response, [], "Local (RAG - No relevant info)"
                else:
                    return response, context_docs, "Local (RAG)"
            
            return "I cannot process your request at this time.", [], "Error"

    def load_local_fallback(self):
        if self.local_model is None:
            try:
                from langchain_ollama import ChatOllama
                self.local_model = ChatOllama(model="phi3", temperature=0.1)
                return True
            except Exception as e:
                print(f"Local model load failed: {e}")
                return False
        return True

    async def get_local_rag_response(self, question: str, context_docs: List[Dict], needs_grading: bool) -> Optional[str]:
        if not self.load_local_fallback():
            return None

        try:
            context_text = "\n\n".join([
                f"[{doc['source']}, Page {doc['page']}]: {doc['content']}"
                for doc in context_docs
            ])

            grading_section = grading_context if needs_grading else ""

            prompt = f"""{general_system_prompt}

{grading_section}

Context from BulSU documents:
{context_text}

Question: {question}

Answer (cite document and page):"""

            result = await self.loop.run_in_executor(
                self.executor,
                lambda: self.local_model.invoke(prompt)
            )

            return result.content if hasattr(result, 'content') else str(result)

        except Exception as e:
            print(f"Local RAG error: {e}")
            return None

    async def cleanup(self):
        await self.cloud_api.close()
        self.executor.shutdown(wait=False)

# ------------------------- 
# --- Flask Web Server ---
# -------------------------

GUI_DIR = os.path.join("D:" + os.sep, "BulsuAssistant", "gui")

app = Flask(__name__, static_folder=None)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

model_manager: Optional[ModelManager] = None

async def init_model_manager():
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(loop)
        try:
            if os.path.exists(faiss_path) and os.path.exists(f"{faiss_path}.faiss"):
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_community.vectorstores import FAISS as FAISS_local
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS_local.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                model_manager.set_vectorstore(db)
                print("[System] FAISS index loaded into ModelManager")
        except Exception as e:
            print(f"[System] FAISS load error (continuing anyway): {e}")
@app.route("/")
def index():
    index_path = os.path.join(GUI_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(GUI_DIR, "index.html")
    return "index.html not found in gui directory", 404
    
@app.route("/<path:filepath>")
def serve_file(filepath):
    file_path = os.path.join(GUI_DIR, filepath)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(GUI_DIR, filepath)
    return "File not found", 404

@app.route("/images/<path:filename>")
def serve_images(filename):
    images_dir = os.path.join(GUI_DIR, "images")
    return send_from_directory(images_dir, filename)
    
@app.route("/static/<path:filename>")
def serve_static(filename):
    static_dir = os.path.join(GUI_DIR, "static")
    return send_from_directory(static_dir, filename)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"reply": "Invalid request"}), 400
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please send a non-empty message."})

    try:
        loop.run_until_complete(init_model_manager())
        response, sources, model_name = loop.run_until_complete(model_manager.get_response(message))
        return jsonify({"reply": response})
    except aiohttp.ClientConnectorError:
        return jsonify({"reply": "⚠️ Unable to connect to remote API. Running in offline mode."})
    except Exception as e:
        print(f"[ERROR] while handling chat: {e}")
        return jsonify({"reply": "⚠️ An internal error occurred while processing your request."})



@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_loop():
        loop.stop()
    threading.Thread(target=stop_loop, daemon=True).start()
    return "Shutting down loop", 200

if __name__ == "__main__":
    print("🚀 Starting CICTify Flask Chatbot (Render-Ready Mode) with JSON Faculty")

    if not os.path.isdir(GUI_DIR):
        print(f"[WARNING] GUI directory not found: {GUI_DIR}")

    try:
        loop.run_until_complete(init_model_manager())
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"[ERROR] Failed to start model manager: {e}")

