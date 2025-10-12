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

import aiohttp
from flask import Flask, request, jsonify, send_from_directory, send_file, render_template_string
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright

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
FACULTY_CACHE_PATH = BASE_DIR / "cict_faculty_cache.json"

print(f"[INFO] BASE_DIR = {BASE_DIR}")
print(f"[INFO] GUI_DIR = {GUI_DIR}")

# -------------------------
# --- Configurable values -
# -------------------------
API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY", "")  # set via env
}

# PDFs expected in project root (adjust names if necessary)
pdf_paths = [
    str(BASE_DIR / "CICTify - FAQs.pdf"),
    str(BASE_DIR / "BulSU Student handbook.pdf"),
    str(BASE_DIR / "Faculty Manual for BOR.pdf"),
    str(BASE_DIR / "BulSU-Enhanced-Guidelines.pdf")
]


faiss_path = str(FAISS_DIR)

# -------------------------
# --- Prompts & Context ---
# -------------------------
general_system_prompt = """You are a helpful assistant for Bulacan State University (BulSU).

CRITICAL: The university is BULACAN STATE UNIVERSITY (BulSU).

Be friendly, direct and helpful.""".strip()

rag_system_prompt = """You are an assistant for Bulacan State University (BulSU).
Use the provided context documents to answer BulSU-specific questions. Cite document name and page if possible.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Answer directly and helpfully.""".strip()

grading_context = """
BulSU grading system (internal reference): Lower numbers are better (1.00 best, 5.00 failed).
""".strip()

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

def heuristically_find_faculty_from_text(text: str, source_label: str = "") -> Dict[str, Dict]:
    """
    Heuristic scanner to find 'Dean' and 'Associate Dean' mentions and possible names.
    Returns dict keyed by guessed name with minimal metadata.
    """
    results = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # join short neighbor lines to increase chance of "Name — Title" patterns
    joined = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and len(lines[i]) < 30:
            joined.append(lines[i] + " " + lines[i+1])
            i += 2
        else:
            joined.append(lines[i])
            i += 1

    patt_title = re.compile(r"(associate\s+dean|associate dean|deputy dean|dean)\b", re.I)
    # Candidate name pattern: sequences of Title Case words, allow middle initials
    name_pattern = re.compile(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.-]+){0,4})")

    for line in joined:
        if patt_title.search(line):
            lower = line.lower()
            # try to extract name from line
            # patterns: "Dr. John Doe — Associate Dean" or "Associate Dean: John Doe"
            # Try "Associate Dean: NAME" first
            m = re.search(r"(associate\s+dean[:\-\–\—\s]+)(.+)$", line, re.I)
            if m:
                name_candidate = m.group(2).strip()
            else:
                # try other direction: NAME - Associate Dean
                m2 = re.search(r"^(.{2,120}?)\s+[-–—:]\s+(associate\s+dean|dean)", line, re.I)
                if m2:
                    name_candidate = m2.group(1).strip()
                else:
                    # fallback: take first TitleCase sequence
                    nm = name_pattern.search(line)
                    name_candidate = nm.group(1).strip() if nm else None

            if not name_candidate:
                continue

            # clean name candidate: remove trailing commas, titles like PhD
            name_candidate = re.sub(r",?\s*(Ph\.?D|PhD|DIT|MSIT|MS|MAED|MITM|MSCPE|MSc|Dr\.?)\b.*", "", name_candidate, flags=re.I).strip()
            # take only first 4 words max
            name_candidate = " ".join(name_candidate.split()[:4]).strip()
            if len(name_candidate) < 3:
                continue

            # normalize title presence
            title_match = "Associate Dean" if "associate" in lower else "Dean" if "dean" in lower else "Faculty"
            profile = {
                "name": name_candidate,
                "title": title_match,
                "department": "CICT" if "cict" in lower or "college" in lower else None,
                "education": None,
                "certifications": None,
                "description": None,
                "url": source_label
            }
            results[name_candidate] = profile
    return results

def build_faculty_index_from_pdfs(pdf_paths_list: List[str]) -> Dict[str, Dict]:
    """
    Try to build a minimal faculty index from available PDFs.
    Returns dict mapping 'Name' -> profile dict.
    """
    combined_index = {}
    for p in pdf_paths_list:
        if not p or not os.path.exists(p):
            continue
        print(f"[PDF] Scanning {p} for faculty entries...")
        text = extract_text_from_pdf(p, max_pages=40)
        if not text:
            continue
        found = heuristically_find_faculty_from_text(text, source_label=os.path.basename(p))
        if found:
            print(f"[PDF] Heuristics found {len(found)} entries in {p}")
            for name, prof in found.items():
                # do not overwrite existing more detailed profiles
                if name not in combined_index:
                    combined_index[name] = prof
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
                "model": "llama-3.3-70b-versatile",
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
        if not API_KEYS.get("groq"):
            return None
        try:
            context_text = "\n\n".join([f"[{doc.get('source','unknown')}] {doc.get('content','')}" for doc in context_docs])
            system_prompt = rag_system_prompt.format(context=context_text)
            session = await self.get_session()
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature": 1.0,
                "max_tokens": 500
            }
            async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEYS['groq']}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                print(f"[Groq RAG] Bad status: {resp.status}")
                return None
        except Exception as e:
            print(f"[Groq RAG API error]: {e}")
            return None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

# -------------------------
# --- CICT Web Crawler ----
# (kept as-is - omitted HERE in this listing for brevity, but unchanged)
# -------------------------
# For brevity I will reuse your existing CICTWebCrawler class below (unchanged)
# — the full class is the same as in your code above (fetch_page, extract_faculty_profile, crawl_site, etc.)
# Inserted exactly as in your previous file (kept behavior identical).

# (To keep the response compact I'm re-adding it below exactly)
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
        # department heuristics
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
        # Education heuristics
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
        # certifications
        certs = []
        for s in soup.find_all(string=lambda t: t and "certificate" in t.lower()):
            parent = s.parent
            if parent:
                text = parent.get_text(" ", strip=True)
                certs.append(text)
        profile["certifications"] = certs if certs else None
        # description
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
                # If faculty profile url, try to extract into profiles list
                if self.is_faculty_profile_url(url):
                    prof = self.extract_faculty_profile(result, url)
                    if prof:
                        scraped_profiles.append((url, prof))
                        continue
                # otherwise save page text
                soup = BeautifulSoup(result, "html.parser")
                for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                    tag.decompose()
                scraped_texts.append((url, soup.get_text(" ", strip=True)[:4000]))
                # find links to expand
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

    @staticmethod
    def response_has_no_info(response: str) -> bool:
        if not response:
            return True
        no_info_phrases = [
            "don't have information", "i couldn't find", "not in my documents",
            "no information", "couldn't find", "unable to find", "not found"
        ]
        rl = response.lower()
        return any(p in rl for p in no_info_phrases)

    async def retrieve_documents(self, question: str) -> List[Dict]:
        if not self.retriever:
            return []
        try:
            docs = await self.loop.run_in_executor(
                self.executor,
                lambda: self.retriever.invoke(question)
            )
            formatted = []
            for d in docs:
                formatted.append({
                    "content": d.page_content,
                    "source": d.metadata.get("source_file", "local"),
                    "page": d.metadata.get("page", 0) + 1,
                    "pdf_path": d.metadata.get("pdf_path", "")
                })
            return formatted
        except Exception as e:
            print(f"[ModelManager] retrieve_documents error: {e}")
            return []

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore
        try:
            self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        except Exception as e:
            print(f"[ModelManager] set_vectorstore error: {e}")

    def load_local_fallback(self):
        if self.local_model is None:
            try:
                from langchain_ollama import ChatOllama
                self.local_model = ChatOllama(model="phi3", temperature=0.1)
                return True
            except Exception as e:
                print(f"[ModelManager] local model load failed: {e}")
                return False
        return True

    async def get_local_rag_response(self, question: str, context_docs: List[Dict], needs_grading: bool) -> Optional[str]:
        if not self.load_local_fallback():
            return None
        try:
            context_text = "\n\n".join([f"[{d['source']}] {d['content']}" for d in context_docs])
            grading_section = grading_context if needs_grading else ""
            prompt = f"""{general_system_prompt}

{grading_section}

Context:
{context_text}

Question: {question}

Answer (cite doc and page):"""
            result = await self.loop.run_in_executor(self.executor, lambda: self.local_model.invoke(prompt))
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            print(f"[Local RAG error]: {e}")
            return None

    async def get_response(self, question: str) -> Tuple[str, List[Dict], str]:
        """
        High-level routing:
         - Use faculty JSON (quick) for faculty queries
         - Otherwise try FAISS retriever -> Groq RAG -> Groq general -> local fallback
        """
        # load faculty JSON (if available)
        faculty_index = load_json_safely(FACULTY_JSON_PATH)

        lower_msg = question.lower().strip()
        clean_msg = re.sub(r"[^a-z\s]", "", lower_msg)

        # categorize queries: academic vs cict/faculty
        academic_keywords = ['grading', 'gwa', 'honor', 'honors', 'grade', 'requirement', 'requirements',
                             'passing', 'scholarship', 'curriculum', 'student handbook']
        cict_keywords = ['cict', 'bulsucict', 'college of information', 'faculty', 'professor', 'dean',
                         'associate dean', 'chair', 'program chair', 'coordinator', 'instructor', 'lecturer']

        is_academic_query = any(word in lower_msg for word in academic_keywords)
        # name detection against faculty JSON
        name_detected = False
        matched_profile = None
        if faculty_index:
            for name_key in faculty_index.keys():
                name_clean = re.sub(r"[^a-z\s]", "", name_key.lower())
                key_tokens = [t for t in name_clean.split() if len(t) > 2]
                if sum(1 for t in key_tokens if t in clean_msg) >= 2:
                    name_detected = True
                    matched_profile = faculty_index[name_key]
                    print(f"[ModelManager] Name detected: {name_key}")
                    break
        is_cict_query = any(word in lower_msg for word in cict_keywords) or name_detected

        # PRIORITY: faculty queries handled by JSON (fast)
        if is_cict_query and faculty_index:
            print("[System] CICT/faculty query detected - using faculty JSON.")
            lower_msg = lower_msg  # already
            # Try name match first
            if name_detected and matched_profile:
                p = matched_profile
                details = []
                details.append(f"{p.get('name','Unknown')} — {p.get('title','')}")
                if p.get('department'):
                    details.append(f"Department: {p.get('department')}")
                if p.get('education'):
                    details.append(f"Education: {'; '.join(p.get('education')[:6])}")
                if p.get('certifications'):
                    details.append(f"Certifications: {'; '.join(p.get('certifications')[:6])}")
                if p.get('description'):
                    details.append(f"About: {p.get('description')}")
                if p.get('url'):
                    details.append(f"Profile: {p.get('url')}")
                return "\n".join(details), [], "JSON CICT"
            # role matching
            roles = {
                "dean": [p for p in faculty_index.values() if p.get('title') and 'dean' in p['title'].lower() and 'associate' not in p['title'].lower()],
                "associate dean": [p for p in faculty_index.values() if p.get('title') and 'associate' in p['title'].lower() and 'dean' in p['title'].lower()],
            }
            for role, matches in roles.items():
                if re.search(rf"\b{role}\b", lower_msg):
                    if matches:
                        if len(matches) == 1:
                            top = matches[0]
                            reply = f"{top.get('name','Unknown')} — {top.get('title','')}"
                            if top.get('url'):
                                reply += f" Profile: {top.get('url')}"
                            return reply, [], "JSON CICT"
                        else:
                            return "Matches:\n" + "\n".join([m.get("name","") for m in matches]), [], "JSON CICT"
            # list faculty
            if "list" in lower_msg or "who are" in lower_msg or "all faculty" in lower_msg:
                names = [p.get('name','') for p in faculty_index.values()]
                return "CICT Faculty:\n" + "\n".join(f"- {n}" for n in names), [], "JSON CICT"
            # otherwise fallback to scraping non-faculty or FAISS
            # try FAISS/RAG
            context_docs = await self.retrieve_documents(question)
            if not context_docs:
                print("[System] No cached CICT data - scraping non-faculty pages (fallback).")
                crawler = CICTWebCrawler(self.loop)
                _, scraped_texts = await crawler.crawl_site("https://bulsucict.com", max_pages=20)
                if scraped_texts:
                    context_docs = [{"content": t, "source": u, "page": 1} for (u, t) in scraped_texts]
            if context_docs:
                resp = await self.cloud_api.call_groq_rag(question, context_docs[:6])
                if resp:
                    return resp, [], "Groq (CICT Non-Faculty)"
            # fallback to general
            print("[System] No faculty data matched — falling back to general RAG/General.")
            response = await self.cloud_api.call_groq_general(question)
            if response:
                return response, [], "Groq (General - fallback)"
        # Non-faculty queries: use QueryClassifier rules to decide RAG vs general
        use_rag = QueryClassifier.needs_rag(question)
        print(f"[ModelManager] Question: '{question}' | Use RAG: {use_rag}")
        if not use_rag:
            # Groq general
            response = await self.cloud_api.call_groq_general(question)
            if response:
                return response, [], "Groq (General)"
            # local fallback
            if self.load_local_fallback():
                try:
                    result = await self.loop.run_in_executor(self.executor, lambda: self.local_model.invoke(question))
                    return (result.content if hasattr(result, "content") else str(result)), [], "Local (General)"
                except Exception as e:
                    print(f"[ModelManager] local general error: {e}")
            return "I'm having trouble responding. Please try again.", [], "Error"
        else:
            # RAG flow: retrieve docs -> call Groq RAG -> local RAG fallback
            context_docs = await self.retrieve_documents(question)
            if not context_docs:
                # attempt Groq general as fallback
                response = await self.cloud_api.call_groq_general(question)
                if response:
                    return response, [], "Groq (General fallback)"
                return "I couldn't find relevant information in my documents. Could you rephrase your question?", [], "No Results"
            needs_grading = any(w in lower_msg for w in ['gwa', 'grade', 'grading', 'honor', 'honours'])
            response = await self.cloud_api.call_groq_rag(question, context_docs, grading_info="yes" if needs_grading else "")
            if response:
                if self.response_has_no_info(response):
                    return response, [], "Groq (RAG - No info)"
                return response, context_docs, "Groq (RAG)"
            # local RAG fallback
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
    BULSU_KEYWORDS = [
        'bulsu', 'bulacan state', 'university', 'bsu', 'mission', 'vision', 'history', 'campus',
        'gwa', 'grade', 'grading', 'dean', 'faculty', 'student handbook', 'honor', 'honours'
    ]
    CASUAL_PATTERNS = [
        r'^hi+$', r'^hello+$', r'^hey+$', r'^good\s+(morning|afternoon|evening)',
        r'^how\s+are\s+you', r'^what\'?s\s+up', r'^thanks?', r'^bye', r'^goodbye'
    ]

    @classmethod
    def needs_rag(cls, question: str) -> bool:
        q = question.lower().strip()
        for p in cls.CASUAL_PATTERNS:
            if re.match(p, q):
                return False
        # "grading system" explicitly should not always trigger RAG (we prefer general)
        if re.match(r'^(what is |explain )?(the )?(bulsu )?grading system\??$', q):
            return False
        # otherwise if contains BulSU keywords -> RAG
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
        # try loading FAISS vectorstore if present
        try:
            if os.path.exists(faiss_path) and os.path.exists(f"{faiss_path}.faiss"):
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    from langchain_community.vectorstores import FAISS as FAISS_local
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    db = FAISS_local.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                    model_manager.set_vectorstore(db)
                    print("[System] FAISS index loaded.")
                except Exception as e:
                    print(f"[System] FAISS load failed: {e}")
        except Exception as e:
            print(f"[System] FAISS check error: {e}")
        # Ensure new PDFs are prioritized in retrieval
        print("[System] Prioritizing new BulSU and CICT PDFs for retrieval order.")
        pdf_paths.sort(key=lambda x: "FAQ" not in x)  # make FAQs first

        # If faculty JSON doesn't exist, try to build it from PDFs (CICTify - FAQs, etc.)
        if not FACULTY_JSON_PATH.exists():
            print("[System] cict_faculty.json not found - attempting to build from PDFs...")
            built = build_faculty_index_from_pdfs(pdf_paths)
            if built:
                try:
                    save_json_safely(FACULTY_JSON_PATH, built)
                    print(f"[System] Built faculty JSON with {len(built)} entries.")
                except Exception as e:
                    print(f"[System] Could not save built faculty JSON: {e}")
            else:
                print("[System] Could not heuristically build faculty index from PDFs.")

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
    # serve any file inside gui directory (css/js/widget)
    file_path = GUI_DIR / filepath
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(GUI_DIR), filepath)
    return "File not found", 404

# Chat endpoints (legacy and api)
def extract_message_from_request(req_json: dict) -> str:
    if not req_json:
        return ""
    return req_json.get("message", "") or req_json.get("q", "") or ""

@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"reply": "Invalid request"}), 400
    message = extract_message_from_request(data).strip()
    if not message:
        return jsonify({"reply": "Please send a non-empty message."}), 400

    try:
        # ensure model manager created
        loop.run_until_complete(init_model_manager())
        response, sources, model_name = loop.run_until_complete(model_manager.get_response(message))
        return jsonify({"reply": response, "sources": sources, "model": model_name})
    except aiohttp.ClientConnectorError:
        return jsonify({"reply": "⚠️ Unable to connect to remote API. Running in offline mode."})
    except Exception as e:
        print(f"[ERROR] chat endpoint: {e}")
        return jsonify({"reply": "⚠️ An internal error occurred while processing your request."})

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
    # Use 0.0.0.0 so Render can bind
    if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

