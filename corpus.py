"""Build the two text corpora: reference texts, title-conditioned generations, one rewrite.

Reference abstracts come from the public arXiv Atom API and reference introductions
from the public Wikipedia REST API (no key needed). Generation and rewriting call an
OpenAI-compatible endpoint ($GEN_EP, $LLM_KEY; temperature 0.8, max 320 tokens).
Generation is not seeded, so a rebuilt corpus will differ from the one evaluated in
the paper; the evaluated corpora are distributed with the supplementary material.

Output format (one file per corpus):
    {"human": [...], "ai": [...], "humanized_ai": [...]}   # aligned by index
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from common import chat, save_json

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_CATEGORIES = ["cs.CL", "cs.LG", "cs.IR", "cs.CY"]
USER_AGENT = {"User-Agent": "research-corpus/0.1"}

ABSTRACT_PROMPT = ("Write the abstract (120-200 words) of a research paper titled "
                   "\"{title}\". Output only the abstract.")
ABSTRACT_REWRITE = ("Rewrite the following abstract so it reads like it was written by a "
                    "human researcher (vary phrasing, remove any AI-typical cadence); keep the "
                    "content and length. Output only the rewritten abstract:\n\n{text}")
WIKI_PROMPT = ("Write a concise encyclopedic introduction (100-150 words) for the "
               "Wikipedia topic \"{title}\". Output only the introduction.")
WIKI_REWRITE = ("Rewrite the following so it reads like a human Wikipedia editor wrote "
                "it (vary phrasing, keep the content and length). Output only the text:\n\n{text}")

# Fixed topic list (140 topics); entries without a standard intro > 200 chars are dropped.
WIKI_TOPICS = [
    "Photosynthesis", "Quantum mechanics", "Black hole", "General relativity",
    "Thermodynamics", "Electromagnetism", "Periodic table", "Standard Model",
    "Big Bang", "Galaxy", "Solar System", "Photon", "Entropy", "Plate tectonics",
    "Volcano", "Earthquake", "Nuclear fission", "Superconductivity", "Higgs boson",
    "Dark matter", "DNA", "Evolution", "Natural selection", "Genome", "Cell (biology)",
    "Immune system", "Neuron", "Mitochondrion", "Protein", "Enzyme", "Virus",
    "Bacteria", "Vaccine", "Antibiotic", "Cancer", "Diabetes", "CRISPR", "Stem cell",
    "Human brain", "Heart", "Penicillin", "Hormone", "Water", "Carbon", "Oxygen",
    "Chemical bond", "Acid", "Catalysis", "Polymer", "Graphene", "Crystal",
    "Combustion", "Climate change", "Atmosphere of Earth", "Ocean", "Amazon rainforest",
    "Mount Everest", "Sahara", "Antarctica", "Moon", "Mars", "Sun", "Jupiter",
    "Comet", "Aurora", "Internet", "World Wide Web", "Machine learning",
    "Artificial intelligence", "Cryptography", "Operating system", "Blockchain",
    "Transistor", "Integrated circuit", "Database", "Algorithm", "Compiler",
    "Quantum computing", "Computer network", "Encryption", "Roman Empire",
    "French Revolution", "World War II", "World War I", "Cold War", "Renaissance",
    "Industrial Revolution", "Ancient Egypt", "Ottoman Empire", "Byzantine Empire",
    "American Civil War", "Silk Road", "Mongol Empire", "Cuban Missile Crisis",
    "Age of Discovery", "Black Death", "Roman Republic", "Han dynasty",
    "Pacific Ocean", "Himalayas", "Nile", "Great Barrier Reef", "Mediterranean Sea",
    "Amazon River", "Mount Kilimanjaro", "Grand Canyon", "Jazz", "Impressionism",
    "Cubism", "Baroque", "Opera", "William Shakespeare", "Ludwig van Beethoven",
    "Wolfgang Amadeus Mozart", "Leonardo da Vinci", "Vincent van Gogh", "Symphony",
    "Gothic architecture", "Romanticism", "Photography", "Democracy", "Capitalism",
    "Inflation", "Game theory", "Supply and demand", "Utilitarianism",
    "Existentialism", "Stoicism", "Globalization", "Gross domestic product",
    "Federalism", "Constitution", "Coffee", "Tea", "Chess", "Olympic Games",
    "Chocolate", "Wine", "Volcanology", "Telescope",
]


def fetch_arxiv(n_per_category=60):
    """Most recent abstracts per category, in category order: [(title, abstract)]."""
    out = []
    for cat in ARXIV_CATEGORIES:
        url = (f"http://export.arxiv.org/api/query?search_query=cat:{cat}&start=0"
               f"&max_results={n_per_category}&sortBy=submittedDate&sortOrder=descending")
        root = ET.fromstring(urllib.request.urlopen(
            urllib.request.Request(url, headers=USER_AGENT), timeout=30).read())
        for e in root.findall(f"{ATOM}entry"):
            title = " ".join((e.findtext(f"{ATOM}title") or "").split())
            abstract = " ".join((e.findtext(f"{ATOM}summary") or "").split())
            if len(abstract) > 200 and len(title) > 10:
                out.append((title, abstract))
        time.sleep(3)  # arXiv API rate limit
    return out


def fetch_wikipedia(topics=WIKI_TOPICS):
    """Unique standard-page summaries longer than 200 characters: [(title, intro)]."""
    base = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    out, seen = [], set()
    for topic in topics:
        try:
            url = base + urllib.parse.quote(topic.replace(" ", "_"))
            d = json.loads(urllib.request.urlopen(
                urllib.request.Request(url, headers=USER_AGENT), timeout=20).read())
        except Exception:
            continue
        title, intro = d.get("title", ""), d.get("extract", "")
        if title and title not in seen and len(intro) > 200 and d.get("type") == "standard":
            seen.add(title)
            out.append((title, " ".join(intro.split())))
        time.sleep(0.1)
    return out


def generate_and_rewrite(references, prompt, rewrite_prompt):
    human = [text for _, text in references]
    ai = [chat(prompt.format(title=title), 0.8, 320) or "" for title, _ in references]
    rewritten = [chat(rewrite_prompt.format(text=a), 0.8, 320) or a for a in ai]
    return {"human": human, "ai": ai, "humanized_ai": rewritten}


def build(out_dir, n_abstracts=120):
    abstracts = generate_and_rewrite(fetch_arxiv()[:n_abstracts], ABSTRACT_PROMPT, ABSTRACT_REWRITE)
    save_json(abstracts, out_dir / "abstracts_texts.json")
    wiki = generate_and_rewrite(fetch_wikipedia(), WIKI_PROMPT, WIKI_REWRITE)
    save_json(wiki, out_dir / "wiki_texts.json")
    print(f"wrote {len(abstracts['human'])} abstract and {len(wiki['human'])} Wikipedia triples to {out_dir}")
