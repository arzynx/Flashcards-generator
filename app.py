import csv
import html
import io
import json
import os
import re
from collections import Counter
from datetime import datetime

import streamlit as st


st.set_page_config(page_title="Recallly — AI Flashcards", page_icon="✦", layout="wide")

DEFAULT_NOTES = ""

SYSTEM_PROMPT = """You are an expert learning designer. Turn the source material into exactly {count} high-quality revision flashcards.

Return ONLY a JSON array. Every item must have exactly these keys: question, answer, difficulty.
Difficulty must be one of: Easy, Medium, Hard.

Rules:
- Test understanding, not sentence copying.
- Make a balanced mix of definition, why/how, comparison/cause-effect, and application questions.
- Avoid repetitive wording and avoid more than two questions beginning with 'What is'.
- Answers must be self-contained, concise, and accurate (1–3 sentences).
- Use only ideas supported by the source material. Do not invent facts.

SOURCE MATERIAL:
{notes}
"""

VISUAL_PROMPT = """You are an expert learning designer and visual study-note editor. Convert the source material into exactly {count} compact, exam-ready VISUAL REVISION CARDS. Each card must teach ONE atomic concept, like an excellent handwritten revision page: a title, organised facts, a small visual explanation, a memory hook, and a quick self-test.

Return ONLY a JSON array. Every item must have exactly these keys:
- title: short topic title
- sections: an array of exactly 4 objects. Every object has heading and points containing exactly TWO items: first a concise main point, then one lighter supporting subpoint. This creates 8 facts per page. Each item must fit in 2–3 visual lines (maximum 150 characters). In every main point, wrap the 1–2 most important terms in double brackets, for example: "[[Centripetal force]] acts towards the centre."
- formula: a formula, relationship, example, or empty string if none is supported by the source
- remember: one concise memory hook, exam trap, or summary line
- visual: an object with exactly: kind, title, items. kind must be one of flow, relationship, comparison, equation, table. title is a short label. items is an array of 2–4 objects, each with label and value.
- challenge: an object with exactly question and answer. It must test application or understanding—not just repeat a heading.
- difficulty: Easy, Medium, or Hard

Rules:
- Each card must be self-contained, factual, and visually scannable. Do not put more than one major concept on a card.
- Across the two-page draft, use a mix of definitions, mechanism/why, examples, limitations, comparisons, and applications where relevant.
- Do not invent facts. Do not repeat the same idea between sheets.
- Summarise; never dump the source text. Keep every point short enough to fit in 2–3 lines on a revision card. The first point in each section is the major fact and the second is its lighter supporting detail. Include formulas only when the source supports them. Write headings in 1–4 words.
- Use only plain text and Unicode symbols; no markdown.

SOURCE MATERIAL:
{notes}
"""

SUMMARY_PROMPT = """You are an expert study-note summariser. Reduce the source into a factual study blueprint before any visual design is attempted.

Return ONLY one JSON object with these exact keys:
- topic: a concise topic title
- overview: a clear 1–2 sentence explanation of the topic
- key_points: 5–8 non-repetitive, exam-relevant facts in logical order
- mechanism: 2–4 short steps explaining how it works, or an empty list if not applicable
- examples: 0–3 examples supported by the source
- formula: one relevant formula/relationship, or an empty string
- remember: a single short memory hook or exam trap

Do not invent facts. Preserve relationships, cause-effect, definitions, quantities and conditions. Write concise plain text, no markdown.

SOURCE MATERIAL:
{notes}
"""


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def local_api_key() -> str:
    """Prefer an environment variable, then this machine's Streamlit secrets."""
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def validate_cards(payload, count: int):
    if not isinstance(payload, list):
        raise ValueError("The response was not a list of cards.")
    cards = []
    seen = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = clean_text(str(item.get("question", "")))
        answer = clean_text(str(item.get("answer", "")))
        difficulty = str(item.get("difficulty", "Medium")).title()
        if difficulty not in {"Easy", "Medium", "Hard"}:
            difficulty = "Medium"
        key = question.lower()
        if len(question) > 12 and len(answer) > 12 and key not in seen:
            cards.append({"question": question, "answer": answer, "difficulty": difficulty})
            seen.add(key)
    if len(cards) < min(count, 5):
        raise ValueError("Not enough usable cards came back. Try again.")
    return cards[:count]


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    match = re.search(r"\[.*\]|\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object or array found in the response.")
    return json.loads(match.group(0))


def sentence_list(notes: str):
    return [clean_text(sentence) for sentence in re.split(r"(?<=[.!?])\s+", notes) if len(clean_text(sentence)) > 25]


def extractive_summary(notes: str, deck_title: str = ""):
    """A local, fact-preserving summariser used when the online model is unavailable."""
    sentences = sentence_list(notes) or [clean_text(notes)]
    stop_words = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are", "was", "were", "by", "with", "that", "this", "it", "as", "at", "be", "from", "into", "which", "their", "its", "can", "may", "also"}
    words = re.findall(r"[a-zA-Z]{3,}", notes.lower())
    frequencies = Counter(word for word in words if word not in stop_words)
    scored = []
    for position, sentence in enumerate(sentences):
        sentence_words = re.findall(r"[a-zA-Z]{3,}", sentence.lower())
        score = sum(frequencies[word] for word in sentence_words) / max(len(sentence_words), 1)
        scored.append((score, position, sentence))
    key_sentences = [item[2] for item in sorted(sorted(scored, reverse=True)[:8], key=lambda item: item[1])]
    if not key_sentences:
        key_sentences = sentences
    derived_topic = re.split(r"\s+(?:is|are|was|were)\s+", sentences[0], maxsplit=1)[0].strip().title()
    formulas = [sentence for sentence in sentences if "=" in sentence or re.search(r"\b(proportional|formula|equation|ratio)\b", sentence, re.I)]
    mechanism = [sentence for sentence in sentences if re.search(r"\b(first|then|next|finally|process|cycle|reaction|causes|leads|converts|uses)\b", sentence, re.I)]
    return {
        "topic": deck_title.strip() or derived_topic,
        "overview": " ".join(key_sentences[:2]),
        "key_points": key_sentences[:8],
        "mechanism": mechanism[:4],
        "examples": [sentence for sentence in sentences if re.search(r"\b(example|such as|e\.g\.|including)\b", sentence, re.I)][:3],
        "formula": formulas[0] if formulas else "",
        "remember": key_sentences[0],
    }


def validate_summary(payload, deck_title: str = ""):
    if not isinstance(payload, dict):
        raise ValueError("The summary response was not an object.")
    def clean_list(value, limit):
        return [clean_text(str(item)) for item in value[:limit] if clean_text(str(item))] if isinstance(value, list) else []
    key_points = clean_list(payload.get("key_points", []), 8)
    if len(key_points) < 3:
        raise ValueError("The summary did not contain enough key points.")
    return {
        "topic": deck_title.strip() or clean_text(str(payload.get("topic", "Study topic"))),
        "overview": clean_text(str(payload.get("overview", ""))),
        "key_points": key_points,
        "mechanism": clean_list(payload.get("mechanism", []), 4),
        "examples": clean_list(payload.get("examples", []), 3),
        "formula": clean_text(str(payload.get("formula", ""))),
        "remember": clean_text(str(payload.get("remember", ""))),
    }


def gemini_cards(notes: str, count: int, api_key: str):
    from google import genai

    # Ignore a broken local proxy setting so the app can reach Gemini directly.
    client = genai.Client(api_key=api_key, http_options={"client_args": {"trust_env": False}, "timeout": 20000})
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=SYSTEM_PROMPT.format(notes=notes, count=count),
    )
    return validate_cards(parse_json(response.text), count)


def validate_sheets(payload, count: int):
    if not isinstance(payload, list):
        raise ValueError("The response was not a list of revision sheets.")
    sheets = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = clean_text(str(item.get("title", "")))
        raw_sections = item.get("sections", [])
        sections = []
        for section in raw_sections if isinstance(raw_sections, list) else []:
            heading = clean_text(str(section.get("heading", ""))) if isinstance(section, dict) else ""
            points = [clean_text(str(point))[:150] for point in section.get("points", [])] if isinstance(section, dict) else []
            points = [point for point in points if point][:2]
            if heading and points:
                sections.append({"heading": heading, "points": points[:4]})
        difficulty = str(item.get("difficulty", "Medium")).title()
        if difficulty not in {"Easy", "Medium", "Hard"}:
            difficulty = "Medium"
        raw_visual = item.get("visual", {}) if isinstance(item.get("visual", {}), dict) else {}
        kind = str(raw_visual.get("kind", "relationship")).lower()
        if kind not in {"flow", "relationship", "comparison", "equation", "table"}:
            kind = "relationship"
        visual_items = []
        for visual_item in raw_visual.get("items", []) if isinstance(raw_visual.get("items", []), list) else []:
            if isinstance(visual_item, dict):
                label = clean_text(str(visual_item.get("label", "")))
                value = clean_text(str(visual_item.get("value", "")))
                if label or value:
                    visual_items.append({"label": label, "value": value})
        raw_challenge = item.get("challenge", {}) if isinstance(item.get("challenge", {}), dict) else {}
        challenge = {"question": clean_text(str(raw_challenge.get("question", ""))), "answer": clean_text(str(raw_challenge.get("answer", "")))}
        if title and len(sections) >= 2:
            sheets.append({"title": title, "sections": sections[:4], "formula": clean_text(str(item.get("formula", ""))), "remember": clean_text(str(item.get("remember", ""))), "visual": {"kind": kind, "title": clean_text(str(raw_visual.get("title", ""))), "items": visual_items[:4]}, "challenge": challenge, "difficulty": difficulty})
    if len(sheets) < min(count, 2):
        raise ValueError("Not enough usable revision sheets came back. Try again.")
    return sheets[:count]


def gemini_sheets(notes: str, count: int, api_key: str, deck_title: str = ""):
    from google import genai

    # Ignore a broken local proxy setting so the app can reach Gemini directly.
    client = genai.Client(api_key=api_key, http_options={"client_args": {"trust_env": False}, "timeout": 20000})
    summary_response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=SUMMARY_PROMPT.format(notes=notes),
    )
    blueprint = validate_summary(parse_json(summary_response.text), deck_title)
    sheet_response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=VISUAL_PROMPT.format(notes=json.dumps(blueprint, ensure_ascii=False), count=count),
    )
    return validate_sheets(parse_json(sheet_response.text), count), blueprint


def fallback_cards(notes: str, count: int):
    """A dependable offline demo mode; Gemini is used for production-quality cards."""
    sentences = [clean_text(x) for x in re.split(r"(?<=[.!?])\s+", notes) if len(clean_text(x)) > 35]
    if not sentences:
        sentences = [clean_text(notes)]
    cards = []
    starters = [
        ("What is the main idea in this statement?", "Easy"),
        ("Why is this idea important in the overall topic?", "Medium"),
        ("How would you explain this point to a classmate?", "Medium"),
        ("What could happen if this process or idea did not occur?", "Hard"),
        ("Which detail from the notes best supports this claim?", "Hard"),
    ]
    for index in range(count):
        sentence = sentences[index % len(sentences)]
        question, difficulty = starters[index % len(starters)]
        if index % 3 == 0:
            subject = re.split(r"\s+(?:is|are|was|were|can|will|uses|use)\s+", sentence, maxsplit=1)[0]
            question = f"Explain the role of: {subject.strip()}."
        cards.append({"question": question, "answer": sentence, "difficulty": difficulty})
    return cards


def fallback_sheets(notes: str, count: int, deck_title: str = ""):
    """Create two coherent visual cards from a local fact-preserving study blueprint."""
    blueprint = extractive_summary(notes, deck_title)
    topic = blueprint["topic"]
    points = blueprint["key_points"]
    while len(points) < 6:
        points.append(points[-1])
    def highlighted(sentence):
        words = sentence[:170].rsplit(" ", 1)[0].split() if len(sentence) > 170 else sentence.split()
        key = " ".join(words[: min(4, len(words))])
        return f"[[{key}]]" + (f" {' '.join(words[4:])}" if len(words) > 4 else "")
    mechanism = blueprint["mechanism"] or points[2:5]
    examples = blueprint["examples"] or [points[-1]]
    first_sheet = {
        "title": topic,
        "sections": [
            {"heading": "Meaning", "points": [highlighted(blueprint["overview"] or points[0]), points[0][:150]]},
            {"heading": "Key fact", "points": [highlighted(points[1]), points[2][:150]]},
            {"heading": "Key link", "points": [highlighted(points[3]), points[4][:150]]},
            {"heading": "See it", "points": [highlighted(examples[0]), points[5][:150]]},
        ],
        "formula": blueprint["formula"],
        "remember": blueprint["remember"],
        "visual": {"kind": "relationship", "title": "Core connections", "items": [{"label": "Idea", "value": topic}, {"label": "Key", "value": points[1][:36]}, {"label": "Link", "value": points[2][:36]}]},
        "challenge": {"question": f"What is the central idea behind {topic}?", "answer": blueprint["overview"] or points[0]},
        "difficulty": "Easy",
    }
    second_sheet = {
        "title": "How it works",
        "sections": [
            {"heading": "Process", "points": [highlighted(mechanism[0]), mechanism[1][:150] if len(mechanism) > 1 else points[3][:150]]},
            {"heading": "Then", "points": [highlighted(mechanism[2] if len(mechanism) > 2 else points[3]), points[4][:150]]},
            {"heading": "Why it matters", "points": [highlighted(points[4]), points[5][:150]]},
            {"heading": "Exam link", "points": [highlighted(points[5]), examples[0][:150]]},
        ],
        "formula": blueprint["formula"],
        "remember": blueprint["remember"],
        "visual": {"kind": "flow", "title": "Cause → process → result", "items": [{"label": "Cause", "value": mechanism[0][:28]}, {"label": "Process", "value": mechanism[1][:28] if len(mechanism) > 1 else points[3][:28]}, {"label": "Result", "value": points[4][:28]}]},
        "challenge": {"question": f"How does {topic} connect to the result described in the notes?", "answer": " ".join(mechanism[:2])},
        "difficulty": "Medium",
    }
    return [first_sheet, second_sheet][:count], blueprint


def csv_download(cards):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["question", "answer", "difficulty"])
    writer.writeheader()
    writer.writerows(cards)
    return output.getvalue().encode("utf-8")


def sheets_csv_download(sheets):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["title", "section", "points", "formula", "remember", "difficulty"])
    writer.writeheader()
    for sheet in sheets:
        for section in sheet["sections"]:
            writer.writerow({"title": sheet["title"], "section": section["heading"], "points": " • ".join(section["points"]), "formula": sheet["formula"], "remember": sheet["remember"], "difficulty": sheet["difficulty"]})
    return output.getvalue().encode("utf-8")


def render_visual(visual):
    kind = visual.get("kind", "relationship")
    title = html.escape(visual.get("title", "Visual connection"))
    items = visual.get("items", [])
    if not items:
        return ""
    cells = "".join(
        f"<div class='visual-cell'><b>{html.escape(item.get('label', 'Key point'))}</b><span>{html.escape(item.get('value', ''))}</span></div>"
        for item in items
    )
    if kind == "flow":
        labels = [html.escape(item.get("label", "")) for item in items[:3]]
        while len(labels) < 3:
            labels.append("Idea")
        return f"<div class='visual-panel flow-panel'><div class='visual-label'>↝ {title}</div><svg class='sketch-diagram flow-sketch' viewBox='0 0 420 120' aria-hidden='true'><defs><marker id='tiny-arrow' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0,0 L0,6 L7,3 z' fill='#4d91bc'/></marker></defs><path d='M78 58 C125 4 170 107 210 58 S300 8 347 58' marker-end='url(#tiny-arrow)'/><circle cx='70' cy='58' r='25'/><circle cx='210' cy='58' r='25'/><circle cx='350' cy='58' r='25'/><text x='70' y='62'>{labels[0][:12]}</text><text x='210' y='62'>{labels[1][:12]}</text><text x='350' y='62'>{labels[2][:12]}</text></svg></div>"
    if kind == "comparison":
        return f"<div class='visual-panel compare-panel'><div class='visual-label'>⇄ {title}</div><div class='visual-compare'>{cells}</div></div>"
    if kind == "table":
        return f"<div class='visual-panel table-panel'><div class='visual-label'>▦ {title}</div><div class='visual-table'>{cells}</div></div>"
    if kind == "equation":
        return f"<div class='visual-panel equation-panel'><div class='visual-label'>∑ {title}</div><div class='visual-equation'>{cells}</div></div>"
    labels = [html.escape(item.get("label", "")) for item in items[:3]]
    while len(labels) < 3:
        labels.append("Link")
    return f"<div class='visual-panel relationship-panel'><div class='visual-label'>✦ {title}</div><svg class='sketch-diagram relationship-sketch' viewBox='0 0 420 120' aria-hidden='true'><path d='M210 58 Q125 0 65 30 M210 58 Q300 0 355 30 M210 58 Q210 112 210 103'/><circle cx='210' cy='58' r='29' class='centre'/><circle cx='60' cy='30' r='20'/><circle cx='360' cy='30' r='20'/><circle cx='210' cy='105' r='18'/><text x='210' y='62'>IDEA</text><text x='60' y='34'>{labels[0][:8]}</text><text x='360' y='34'>{labels[1][:8]}</text><text x='210' y='109'>{labels[2][:8]}</text></svg></div>"


def paint_text(value: str) -> str:
    """Safely turn the model's [[important phrase]] marks into colourful ink."""
    safe = html.escape(value)
    return re.sub(r"\[\[(.+?)\]\]", r"<mark>\1</mark>", safe)


def reset_index():
    st.session_state.card_index = 0
    st.session_state.revealed = False
    st.session_state.show_challenge = False


def move_card(step):
    deck_length = len(st.session_state.sheets) or len(st.session_state.cards)
    st.session_state.card_index = max(0, min(deck_length - 1, st.session_state.card_index + step))
    st.session_state.revealed = False
    st.session_state.show_challenge = False


def toggle_challenge():
    st.session_state.show_challenge = not st.session_state.show_challenge


def toggle_mastered(index):
    mastered = set(st.session_state.mastered)
    if index in mastered:
        mastered.remove(index)
    else:
        mastered.add(index)
    st.session_state.mastered = mastered


st.markdown(
    """<style>
    .stApp { background: #f7f1e7; color: #172142; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif !important; color: #f6f8ff !important; }
    .hero { padding: 1.5rem 0 1.6rem; }
    .eyebrow { color: #72e9ff; font: 700 .72rem/1.2 monospace; letter-spacing: .14em; }
    .hero h1 { font-size: clamp(3rem, 6vw, 5.4rem); line-height: .87; margin: .55rem 0 .8rem; letter-spacing: -.055em; }
    .hero h1 em { color: #9de9d6; font-style: italic; }
    .hero p { max-width: 620px; color: #b7c7dd; font-size: 1.05rem; }
    .constellation { height: 240px; max-width: 830px; margin: 1.3rem auto .3rem; position: relative; }
    .constellation svg { width: 100%; height: 100%; overflow: visible; }
    .constellation .link { fill: none; stroke: url(#signal); stroke-width: 1.6; stroke-dasharray: 4 8; animation: flow 9s linear infinite; }
    .constellation .link:nth-child(2n) { animation-direction: reverse; animation-duration: 12s; }
    .constellation .node { fill: #edfaff; stroke: #8be8ff; stroke-width: 2; filter: url(#glow); animation: breathe 3s ease-in-out infinite; }
    .constellation .node.violet { stroke: #c9a1ff; fill: #eee5ff; animation-delay: -1.2s; }
    .constellation .node.mint { stroke: #93f3ca; fill: #e3fff4; animation-delay: -2s; }
    .constellation .core { fill: url(#core); stroke: #c8fbff; stroke-width: 1.5; filter: url(#glow); }
    .constellation text { fill: #ddecff; font: 700 10px monospace; letter-spacing: 2px; }
    .constellation .caption { color: #a8c7e9; font: 700 .68rem monospace; letter-spacing: .15em; text-align:center; margin-top:-.8rem; }
    .transformation { position:relative; max-width:860px; height:260px; margin:1.15rem auto .15rem; overflow:visible; }
    .raw-note { position:absolute; left:5%; top:45px; z-index:2; width:245px; min-height:142px; padding:1rem; background:#30364a; border:1px solid #5d6376; border-radius:7px; color:#aeb5c6; transform:rotate(-5deg); box-shadow:0 14px 28px rgba(0,0,0,.28); }
    .raw-note small { display:block; color:#aabfff; font:700 .62rem monospace; letter-spacing:.13em; margin-bottom:.55rem; }.raw-note i { display:block; height:7px; margin:.45rem 0; border-radius:99px; background:linear-gradient(90deg,#9ba1b2 0 79%,transparent 79%); opacity:.72; }.raw-note i:nth-child(3){width:82%}.raw-note i:nth-child(4){width:91%}.raw-note i:nth-child(5){width:65%}.raw-note i:nth-child(6){width:86%}
    .transform-arrow { position:absolute; left:30%; top:34px; z-index:1; width:43%; height:190px; overflow:visible; }.transform-arrow path { fill:none; stroke:#8eeedd; stroke-width:2.2; stroke-dasharray:4 8; stroke-linecap:round; filter:drop-shadow(0 0 5px rgba(117,235,215,.55)); animation:flow 7s linear infinite; }.transform-arrow circle { fill:#9de9d6; filter:drop-shadow(0 0 8px #9de9d6); }.transform-arrow text { fill:#bdeee7; font:700 10px monospace; letter-spacing:2px; }
    .mini-sheet { position:absolute; right:4%; top:15px; z-index:3; width:286px; min-height:190px; padding:1rem 1.08rem; background:#fffdf7; color:#172142; border-radius:19px 11px 20px 13px; transform:rotate(4deg); box-shadow:0 18px 38px rgba(0,0,0,.32), 0 0 0 2px rgba(141,224,255,.34); overflow:hidden; }.mini-sheet:before { content:''; position:absolute; inset:0; opacity:.16; background:repeating-linear-gradient(0deg,transparent 0 24px,#84bfe8 25px 26px); }.mini-sheet > * { position:relative; z-index:1; }.mini-sheet .tag { display:inline-block; padding:.18rem .55rem; border-radius:999px; color:#226981; background:#d9f5f7; font:700 .62rem monospace; letter-spacing:.1em; }.mini-sheet h4 { margin:.35rem 0 .42rem; color:#15244a; font:700 2rem/.9 'Caveat',cursive; }.mini-sheet p { margin:.2rem 0; color:#33425c; font:1.05rem/1.08 'Patrick Hand',cursive; }.mini-sheet mark { color:#237a70; background:#d9f2c9; font-weight:700; }.mini-sheet .spark { position:absolute; right:18px; bottom:12px; color:#846bd4; font:700 2.7rem/1 'Caveat',cursive; }
    @keyframes flow { to { stroke-dashoffset: -150; } }
    @keyframes breathe { 50% { opacity: .48; transform: scale(1.35); transform-origin: center; } }
    .panel { background: rgba(12, 20, 41, .82); border: 1px solid rgba(170, 202, 255, .18); border-radius: 20px; padding: 1.25rem; box-shadow: 0 22px 60px rgba(0, 0, 0, .25); }
    [data-testid="stVerticalBlockBorderWrapper"] { background: rgba(23, 34, 66, .62); border-color: rgba(135, 213, 255, .24) !important; border-radius: 15px; }
    [data-testid="stSegmentedControl"] { background: rgba(5, 11, 27, .65); border-radius: 10px; padding: 3px; }
    [data-testid="stSegmentedControl"] label { color: #dceaff !important; font-weight: 700; font-size: .78rem; }
    [data-testid="stSegmentedControl"] [data-checked="true"] { background: linear-gradient(110deg, #82eadb, #8ab5ff) !important; color: #07101e !important; border-radius: 8px; }
    .metric { border-left: 1px solid rgba(170, 202, 255, .2); padding-left: 1rem; color: #aab9cf; }
    .metric b { display: block; color: #8eeedd; font: 700 2rem/1 Georgia, serif; }
    .flashcard { min-height: 330px; border-radius: 24px; padding: 2rem; background: linear-gradient(140deg, #1b2861, #121a38 58%, #20264c); border: 1px solid rgba(157, 233, 214, .34); box-shadow: 0 22px 55px rgba(0,0,0,.3); display: flex; flex-direction: column; justify-content: space-between; }
    .flashcard.answer { background: linear-gradient(140deg, #164f5d, #132746 60%, #242252); }
    .card-side { color: #8eeedd; font: 700 .7rem monospace; letter-spacing: .14em; }
    .card-text { font: 400 clamp(1.65rem, 3vw, 2.45rem)/1.05 Georgia, serif; max-width: 850px; color: #fbfdff; }
    .card-footer { color: #c1d1e7; font-size: .9rem; }
    .revision-sheet { min-height: 570px; background: #fffdf9; color: #172142; border-radius: 24px; padding: 1.55rem; box-shadow: 0 24px 70px rgba(0,0,0,.36); position: relative; overflow:hidden; }
    .revision-sheet:before { content: ''; position:absolute; inset:0; opacity:.18; background: repeating-linear-gradient(0deg, transparent 0 28px, #a9c8e8 29px 30px); pointer-events:none; }
    .sheet-top, .sheet-grid, .memory, .sheet-meta { position:relative; z-index:1; }
    .sheet-top { border-bottom: 3px solid #89b9e8; padding: .1rem .2rem .75rem; display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }
    .sheet-top h2 { color: #13234c !important; font-family: Georgia, serif !important; margin:0; font-size:clamp(2rem,4vw,3.2rem); line-height:.95; }
    .sheet-top span { background:#e5f2ff; border:1px solid #78aee0; color:#1a5590; border-radius:999px; padding:.35rem .62rem; font:700 .68rem monospace; white-space:nowrap; }
    .sheet-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:.9rem; padding-top:1rem; }
    .sheet-section { border:2px solid var(--sheet-accent, #8abce9); background:rgba(255,255,255,.88); border-radius:15px; padding:.85rem 1rem .75rem; }
    .sheet-section:nth-child(2n) { --sheet-accent:#9bcf9d; }.sheet-section:nth-child(3n) { --sheet-accent:#d4a6dc; }.sheet-section:nth-child(4n) { --sheet-accent:#f0ba83; }
    .sheet-section h3 { display:inline-block; color:#172142 !important; font:700 1.05rem/1 Georgia,serif !important; background:color-mix(in srgb, var(--sheet-accent) 30%, white); border-radius:6px; padding:.18rem .45rem; margin: -.2rem 0 .5rem; }
    .sheet-section ul { margin:.1rem 0 0; padding-left:1.1rem; }.sheet-section li { margin:.28rem 0; color:#263454; font-size:.94rem; line-height:1.35; }
    .formula { position:relative; z-index:1; margin-top:.9rem; padding:.72rem 1rem; text-align:center; background:#eff9ef; border:2px dashed #88b98a; border-radius:13px; color:#254e2a; font:700 1.1rem/1.25 Georgia, serif; }
    .memory { margin-top:.9rem; background:#fff8d8; border:2px dashed #e4ba55; border-radius:13px; padding:.76rem 1rem; color:#54400b; }.memory b { font-family:Georgia,serif; font-size:1.08rem; }.memory span { display:block; margin-top:.2rem; }
    .sheet-meta { margin-top:.7rem; color:#6a7896; font:700 .67rem monospace; letter-spacing:.09em; }
    .visual-panel { position:relative; z-index:1; margin-top:.9rem; padding:.75rem .85rem; border-radius:14px; background:#f5fbff; border:2px solid #a9cce9; }.visual-label { color:#285880; font:800 .72rem monospace; letter-spacing:.08em; margin-bottom:.55rem; }
    .visual-flow,.visual-relationship { display:flex; gap:.45rem; align-items:stretch; }.visual-flow .visual-cell { flex:1; position:relative; }.visual-flow .visual-cell:not(:last-child):after { content:'→'; position:absolute; right:-.52rem; top:42%; color:#5797bf; font-weight:900; z-index:3; }.visual-cell { border-radius:9px; padding:.55rem .6rem; background:#fff; border:1px solid rgba(69,126,165,.3); }.visual-cell b,.visual-cell span { display:block; }.visual-cell b { color:#214b75; font:700 .78rem Georgia,serif; }.visual-cell span { color:#3e5570; font-size:.74rem; line-height:1.25; margin-top:.15rem; }.visual-relationship .visual-cell { flex:1; text-align:center; }.visual-relationship .visual-cell:first-child { background:#e8f6ff; }.visual-relationship .visual-cell:nth-child(2) { background:#f2edff; }.visual-relationship .visual-cell:nth-child(3) { background:#edfff3; }
    .visual-compare,.visual-table { display:grid; grid-template-columns:repeat(2,1fr); gap:.45rem; }.visual-compare .visual-cell:first-child { background:#edf7ff; border-color:#a9cae3; }.visual-compare .visual-cell:nth-child(2) { background:#effaf0; border-color:#aad5ae; }.visual-table { gap:0; border:1px solid #aac8df; border-radius:9px; overflow:hidden; }.visual-table .visual-cell { border-radius:0; border:0; border-right:1px solid #c5d6e4; border-bottom:1px solid #c5d6e4; }.visual-equation { display:flex; gap:.4rem; flex-wrap:wrap; justify-content:center; }.visual-equation .visual-cell { background:#effaf0; border-color:#9bc79f; text-align:center; min-width:100px; }
    .study-actions { margin-top:.9rem; } .study-actions button { min-height:42px !important; } .progress-line { color:#a8bed7; font-size:.8rem; margin:.25rem 0 .7rem; }
    /* Hand-drawn visual-card language: intentionally organic, never dashboard-like. */
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;600;700&family=Pacifico&family=Patrick+Hand&display=swap');
    .revision-sheet { min-height: 930px; background:linear-gradient(112deg,rgba(255,255,255,.035),transparent 20%,rgba(0,0,0,.20) 51%,transparent 78%), repeating-linear-gradient(4deg,#201712 0 3px,#281b15 4px 8px,#1b130f 9px 12px); border:1px solid rgba(222,174,115,.22); border-radius:12px 20px 15px 24px; padding:1.1rem; box-shadow:inset 0 0 55px rgba(0,0,0,.45), 0 25px 70px rgba(0,0,0,.38); transform:rotate(-.15deg); }
    .revision-sheet:before { opacity:.38; background:repeating-linear-gradient(93deg,transparent 0 29px,rgba(239,197,133,.055) 30px 31px,transparent 32px 65px); }
    .sheet-top { display:none; }
    .concept-core { position:absolute; z-index:3; left:50%; top:42%; transform:translate(-50%,-50%) rotate(-2deg); width:min(49%,340px); min-height:132px; padding:1.05rem 1rem; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; isolation:isolate; background:linear-gradient(145deg,#183962,#0c203d); border:3px solid #87e8e0; border-radius:48% 52% 45% 55% / 54% 45% 55% 46%; box-shadow:0 0 0 6px rgba(180,246,235,.16), 0 12px 0 rgba(7,29,50,.3), 0 0 42px rgba(72,213,202,.22); color:white; }
    .concept-core:before,.concept-core:after { content:''; position:absolute; z-index:0; display:block; background:linear-gradient(145deg,#183962,#0c203d); border:3px solid #87e8e0; border-bottom:0; border-radius:50%; box-shadow:0 -1px 12px rgba(72,213,202,.18); }.concept-core:before { width:65px; height:57px; left:20%; top:-31px; }.concept-core:after { width:48px; height:42px; right:21%; top:-23px; }
    .concept-core small,.concept-core h2 { position:relative; z-index:1; }.concept-core small { color:#9defff; font:700 .66rem monospace; letter-spacing:.16em; }.concept-core h2 { margin:.35rem 0 0 !important; color:#fff !important; font:700 clamp(1.8rem,3.4vw,2.85rem)/.9 'Caveat',cursive !important; letter-spacing:-.04em; }
    .orbit-notes { position:relative; z-index:2; min-height:880px; }.orbit-note { position:absolute; z-index:3; width:37%; min-height:132px; border:2px solid var(--ink,#76b6e3); background:rgba(255,255,255,.94); border-radius:16px 11px 18px 9px; padding:.66rem .78rem .58rem; box-shadow:3px 4px 0 rgba(75,97,148,.11); }.orbit-note:before { content:''; position:absolute; width:19px; height:19px; border:2px solid var(--ink,#76b6e3); border-radius:50%; background:#fff; }.orbit-note h3 { display:inline-block; color:#172142 !important; background:color-mix(in srgb,var(--ink,#76b6e3) 34%,white); border-radius:7px; padding:.06rem .4rem; margin:-1.1rem 0 .3rem !important; font:700 1.35rem/1 'Caveat',cursive !important; transform:rotate(-1deg); }.orbit-note ul { padding-left:1rem; margin:.24rem 0 0; }.orbit-note li { color:#27344f; font:.91rem/1.48 'Pacifico',cursive; margin:.24rem 0; }.orbit-note li::marker { color:var(--ink,#76b6e3); font-size:1.1em; }.orbit-note mark { background:transparent; color:var(--ink-dark,#155c99); font-weight:700; padding:0 .05rem; text-decoration:underline 2px var(--ink,#76b6e3); text-underline-offset:2px; }
    .orbit-note.note-0 { --ink:#69aee2;--ink-dark:#1d609d; left:10%; top:7%; transform:rotate(-2.4deg); }.orbit-note.note-0:before { right:-10px; bottom:-10px; }
    .orbit-note.note-1 { --ink:#8abf82;--ink-dark:#397c36; right:9%; top:9%; transform:rotate(2.1deg); }.orbit-note.note-1:before { left:-10px; top:-11px; }
    .orbit-note.note-2 { --ink:#c09bdf;--ink-dark:#77509e; width:31%; left:14%; top:58%; transform:rotate(1.4deg); }.orbit-note.note-2:before { right:12px; top:-11px; }
    .orbit-note.note-3 { --ink:#e9af71;--ink-dark:#ad6828; width:31%; right:14%; top:59%; transform:rotate(-2deg); }.orbit-note.note-3:before { left:-10px; bottom:12px; }
    .curly-links { position:absolute; z-index:1; inset:0; width:100%; height:100%; pointer-events:none; }.curly-links > path { fill:none; stroke:#74aee0; stroke-width:2.3; stroke-linecap:round; stroke-dasharray:5 7; filter:drop-shadow(0 0 2px rgba(95,161,220,.36)); }.curly-links > path:nth-of-type(2){stroke:#91be86}.curly-links > path:nth-of-type(3){stroke:#b491d2}.curly-links > path:nth-of-type(4){stroke:#dfae76}.curly-links marker path { fill:#74aee0; stroke:none; }
    .visual-panel { position:absolute; z-index:4; left:50%; top:60%; transform:translateX(-50%) rotate(.9deg); width:min(34%,315px); margin:0; padding:.5rem .65rem; background:#f2fbff; border:2px dashed #6aabd6; box-shadow:2px 3px 0 rgba(79,107,151,.13); }.visual-label { font:700 1rem 'Caveat',cursive; letter-spacing:.02em; color:#215d87; margin:0 0 .2rem; }.visual-cell b { font:700 1rem 'Caveat',cursive; }.visual-cell span { font:.78rem/1.1 'Patrick Hand',cursive; }.visual-flow .visual-cell:not(:last-child):after { color:#5e99c8; }.visual-relationship .visual-cell { padding:.3rem; }
    .sketch-diagram { width:100%; height:100px; overflow:visible; }.sketch-diagram path { fill:none; stroke:#4d91bc; stroke-width:2.3; stroke-linecap:round; stroke-dasharray:4 5; }.sketch-diagram circle { fill:#fff; stroke:#4d91bc; stroke-width:2; }.sketch-diagram .centre { fill:#dff4ff; stroke:#586ea4; }.sketch-diagram text { text-anchor:middle; font:700 10px 'Patrick Hand',cursive; fill:#244d71; }
    .formula { position:absolute; z-index:4; left:50%; bottom:11%; transform:translateX(-50%) rotate(-1deg); width:min(42%,300px); margin:0; padding:.44rem .7rem; font:700 1.2rem/1.1 'Caveat',cursive; }.memory { position:absolute; z-index:5; left:50%; bottom:1.5%; transform:translateX(-50%) rotate(.8deg); width:min(62%,440px); margin:0; padding:.45rem .7rem; border-radius:20px 9px 20px 11px; background:#fff6ca; }.memory b { font:700 1.22rem 'Caveat',cursive; }.memory span { font:1rem/1.1 'Patrick Hand',cursive; }.sheet-meta { position:absolute; z-index:4; left:15px; bottom:7px; font-size:.54rem; opacity:.62; }
    @media(max-width:700px) { .revision-sheet { min-height: auto; padding:.85rem; transform:none; }.orbit-notes { min-height:0; display:grid; grid-template-columns:1fr; gap:1rem; padding-top:10px; }.concept-core { position:relative; left:auto; top:auto; transform:rotate(-1.3deg); width:74%; margin:1rem auto 1.7rem; min-height:120px; }.orbit-note { position:relative; inset:auto !important; width:100%; min-height:0; }.orbit-note.note-0{transform:rotate(-1.4deg)}.orbit-note.note-1{transform:rotate(1deg)}.orbit-note.note-2{transform:rotate(-.6deg)}.orbit-note.note-3{transform:rotate(1.2deg)}.visual-panel,.formula,.memory { position:relative; left:auto; top:auto; bottom:auto; transform:rotate(.4deg); width:92%; margin:1rem auto 0; }.sheet-meta { position:relative; left:auto; bottom:auto; margin-top:.8rem; }.orbit-note li { font-size:1rem; } }
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { background: #090f22 !important; color: #eef5ff !important; border-color: #31436d !important; border-radius: 15px !important; box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 8px 24px rgba(0,0,0,.12); }
    div[data-testid="stTextArea"] textarea { border-radius: 18px !important; }
    div[data-testid="stTextInput"] label, div[data-testid="stTextArea"] label { font-weight: 800 !important; letter-spacing: -.015em; color: #e6f1ff !important; }
    .stButton > button, .stDownloadButton > button { border-radius: 999px; border: 1px solid rgba(150, 236, 220, .5); background: linear-gradient(105deg, #85eddc, #8eb9ff); color: #07101e; font-weight: 800; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: white; color: #07101e; }
    .stAlert { border-radius: 14px; }
    @media(max-width:700px) { .constellation { height: 175px; } .constellation text { font-size: 7px; letter-spacing: 1px; } .transformation{height:430px;max-width:420px}.raw-note{left:3%;top:20px;width:205px}.mini-sheet{right:3%;top:214px;width:235px}.transform-arrow{left:28%;top:116px;width:55%;height:150px;transform:rotate(42deg)} }
    @media(max-width:650px) { .sheet-grid { grid-template-columns:1fr; } .revision-sheet { padding:1rem; } .sheet-section li { font-size:.88rem; } }

    /* The app stays cinematic and dark; only the generated revision paper is off-white. */
    .stApp { background: radial-gradient(circle at 84% 8%, #253b84 0, transparent 28%), radial-gradient(circle at 5% 60%, #174e69 0, transparent 25%), #090d1b; color:#edf5ff; }
    .stApp > h1, .stApp > h2, .stApp > h3, .hero h1, .panel h3 { color:#f6f8ff !important; }.hero p { color:#b7c7dd; }.eyebrow { color:#72e9ff; }.panel, [data-testid="stVerticalBlockBorderWrapper"] { background:rgba(12,20,41,.82) !important; border-color:rgba(170,202,255,.18) !important; box-shadow:0 22px 60px rgba(0,0,0,.25) !important; }.metric { color:#aab9cf; border-color:rgba(170,202,255,.2); }.metric b { color:#8eeedd; }
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { background:#090f22 !important; color:#eef5ff !important; border-color:#31436d !important; box-shadow:none !important; } div[data-testid="stTextInput"] label, div[data-testid="stTextArea"] label { color:#e6f1ff !important; }
    [data-testid="stSegmentedControl"] { background:rgba(5,11,27,.65) !important; }
    [data-testid="stSegmentedControl"] label { color:#dceaff !important; }
    [data-testid="stSegmentedControl"] [data-checked="true"],
    [data-testid="stSegmentedControl"] button[aria-checked="true"],
    [data-testid="stSegmentedControl"] button[aria-selected="true"],
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] { background:rgba(7,16,30,.72) !important; border:1px solid #75f1da !important; color:#9ff8e7 !important; box-shadow:0 0 0 1px rgba(117,241,218,.22), 0 0 14px rgba(89,231,218,.34), inset 0 0 15px rgba(75,197,231,.08) !important; }
    [data-testid="stSegmentedControl"] [data-checked="true"] *,
    [data-testid="stSegmentedControl"] button[aria-checked="true"] *,
    [data-testid="stSegmentedControl"] button[aria-selected="true"] *,
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] * { color:#9ff8e7 !important; }
    /* Streamlit renders this control as BaseWeb radio labels in some builds. */
    [data-testid="stSegmentedControl"] label:has(input:checked),
    [data-testid="stSegmentedControl"] [data-baseweb="button"]:has(input:checked),
    [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] { background:rgba(7,16,30,.72) !important; border:1px solid #75f1da !important; color:#9ff8e7 !important; box-shadow:0 0 0 1px rgba(117,241,218,.25), 0 0 16px rgba(89,231,218,.35), inset 0 0 13px rgba(75,197,231,.08) !important; }
    [data-testid="stSegmentedControl"] label:has(input:checked) *,
    [data-testid="stSegmentedControl"] [data-baseweb="button"]:has(input:checked) *,
    [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] * { color:#9ff8e7 !important; }
    .stButton > button, .stDownloadButton > button { background:linear-gradient(105deg,#85eddc,#8eb9ff); color:#07101e; border-color:rgba(150,236,220,.5); }.stButton > button:hover, .stDownloadButton > button:hover { color:#07101e; background:#a6f3e5; }

    /* Collision-proof concept map: uneven placement, but every element owns a grid zone. */
    .revision-sheet { min-height:0; transform:none; padding:1.4rem; }
    .orbit-notes { min-height:0; display:grid; grid-template-columns:minmax(180px,1fr) minmax(200px,.78fr) minmax(180px,1fr); grid-template-areas:"note0 . note1" ". core ." "note2 visual note3" ". formula ." "memory memory memory"; gap:36px 25px; align-items:start; }
    .curly-links { display:block; position:absolute; z-index:1; inset:0; width:100%; height:100%; pointer-events:none; }.curly-links > path { fill:none; stroke:#79b9e6; stroke-width:2.4; stroke-linecap:round; stroke-dasharray:5 7; }.curly-links > path:nth-of-type(2){stroke:#8cbb86}.curly-links > path:nth-of-type(3){stroke:#bd94df}.curly-links > path:nth-of-type(4){stroke:#e0ad76}.curly-links circle { fill:#fffefa; stroke:#79b9e6; stroke-width:2; }.curly-links marker path { fill:#79b9e6; stroke:none; }
    .concept-core { position:relative; grid-area:core; left:auto; top:auto; transform:rotate(-2deg); width:min(540px,94vw); min-height:138px; margin:0; justify-self:center; }
    .concept-core h2 { white-space:normal; overflow-wrap:normal; word-break:normal; max-width:94%; font-size:clamp(1.35rem,2.45vw,2.35rem) !important; line-height:.94 !important; }
    .orbit-note { position:relative; inset:auto !important; width:auto !important; min-height:0; margin:0; }.orbit-note.note-0{grid-area:note0;transform:rotate(-2deg)}.orbit-note.note-1{grid-area:note1;transform:rotate(1.5deg)}.orbit-note.note-2{grid-area:note2;transform:rotate(1.2deg)}.orbit-note.note-3{grid-area:note3;transform:rotate(-1.6deg)}
    .orbit-note:after { position:absolute; z-index:5; color:var(--ink-dark,#155c99); font:700 2.1rem/1 'Caveat',cursive; text-shadow:1px 1px #fff; }.orbit-note.note-0:after{content:'↘';right:-29px;bottom:-31px;transform:rotate(-12deg)}.orbit-note.note-1:after{content:'↙';left:-29px;bottom:-31px;transform:rotate(12deg)}.orbit-note.note-2:after{content:'↗';right:-29px;top:-31px;transform:rotate(12deg)}.orbit-note.note-3:after{content:'↖';left:-29px;top:-31px;transform:rotate(-12deg)}
    /* Compact thought-cloud summaries: organic, but still stable enough for text. */
    .orbit-note { padding:.9rem 1.05rem .78rem; border-radius:46% 43% 48% 42% / 43% 51% 42% 49%; background:color-mix(in srgb,var(--ink,#76b6e3) 7%,white); box-shadow:3px 4px 0 rgba(75,97,148,.11), inset 0 0 0 7px rgba(255,255,255,.48); }
    .orbit-note:before { z-index:0; width:35px; height:35px; background:color-mix(in srgb,var(--ink,#76b6e3) 7%,white); box-shadow:25px -10px 0 2px color-mix(in srgb,var(--ink,#76b6e3) 7%,white), 52px -1px 0 -2px color-mix(in srgb,var(--ink,#76b6e3) 7%,white); }
    .orbit-note.note-0:before { left:17%; top:-18px; right:auto; bottom:auto; }.orbit-note.note-1:before { left:18%; top:-18px; right:auto; bottom:auto; }.orbit-note.note-2:before { left:18%; top:-18px; right:auto; bottom:auto; }.orbit-note.note-3:before { left:18%; top:-18px; right:auto; bottom:auto; }
    .orbit-note h3 { position:relative; z-index:1; margin:-.35rem 0 .35rem !important; }
    .orbit-note ul { position:relative; z-index:1; margin:.18rem 0 0; }
    .orbit-note li { margin:.28rem 0; }
    .visual-panel { position:relative; grid-area:visual; left:auto; top:auto; transform:rotate(.8deg); width:auto; margin:0; align-self:center; }.formula { position:relative; grid-area:formula; left:auto; bottom:auto; transform:rotate(-.7deg); width:auto; margin:0; }.memory { position:relative; grid-area:memory; left:auto; bottom:auto; transform:rotate(.5deg); width:min(74%,520px); margin:0 auto; }.sheet-meta { position:relative; left:auto; bottom:auto; margin-top:.65rem; }
    @media(max-width:700px) { .revision-sheet{padding:1rem}.orbit-notes { display:grid; grid-template-columns:1fr; grid-template-areas:"core" "note0" "note1" "note2" "note3" "visual" "formula" "memory"; gap:20px; }.curly-links{display:none}.orbit-note:after{display:none}.concept-core,.visual-panel,.formula,.memory{width:auto;margin:0;transform:rotate(0)}.concept-core h2{white-space:normal}.memory{width:auto} }
    </style>""",
    unsafe_allow_html=True,
)

if "cards" not in st.session_state:
    st.session_state.cards = []
if "sheets" not in st.session_state:
    st.session_state.sheets = []
if "card_index" not in st.session_state:
    st.session_state.card_index = 0
if "revealed" not in st.session_state:
    st.session_state.revealed = False
if "generation_notice" not in st.session_state:
    st.session_state.generation_notice = ""
if "study_summary" not in st.session_state:
    st.session_state.study_summary = None
if "mastered" not in st.session_state:
    st.session_state.mastered = set()
if "show_challenge" not in st.session_state:
    st.session_state.show_challenge = False

st.markdown("""<div class='hero'>
<div class='eyebrow'>STUDY SMARTER · REMEMBER LONGER</div>
<h1>Turn notes into<br><em>better recall.</em></h1>
<p>Generate revision cards that test definitions, reasoning, and application—not just memory.</p>
<div class='constellation'>
  <svg viewBox='0 0 900 255' aria-label='A living knowledge network'>
    <defs>
      <linearGradient id='signal' x1='0' x2='1'><stop stop-color='#70e9ff'/><stop offset='.5' stop-color='#bba1ff'/><stop offset='1' stop-color='#8ff0cd'/></linearGradient>
      <radialGradient id='core'><stop stop-color='#e8ffff'/><stop offset='.25' stop-color='#75d9ff'/><stop offset='1' stop-color='#263987'/></radialGradient>
      <filter id='glow'><feGaussianBlur stdDeviation='3' result='b'/><feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter>
    </defs>
    <path class='link' d='M88 143 C198 34 289 94 444 123 S621 40 800 82'/>
    <path class='link' d='M121 53 C254 154 332 222 450 128 S650 208 825 176'/>
    <path class='link' d='M88 143 C229 226 341 182 444 123 S665 83 800 82'/>
    <path class='link' d='M121 53 C266 65 331 82 444 123 S636 158 825 176'/>
    <path class='link' d='M88 143 C220 130 300 117 444 123 S670 141 825 176'/>
    <circle class='node' cx='88' cy='143' r='7'/><circle class='node violet' cx='121' cy='53' r='6'/><circle class='node mint' cx='800' cy='82' r='7'/><circle class='node violet' cx='825' cy='176' r='6'/><circle class='node mint' cx='235' cy='89' r='5'/><circle class='node' cx='651' cy='68' r='5'/><circle class='node violet' cx='289' cy='194' r='5'/><circle class='node' cx='650' cy='170' r='5'/>
    <circle class='core' cx='444' cy='123' r='29'/><circle fill='none' stroke='#9ceeff' stroke-width='1' stroke-dasharray='3 4' cx='444' cy='123' r='41'><animateTransform attributeName='transform' type='rotate' from='0 444 123' to='360 444 123' dur='12s' repeatCount='indefinite'/></circle>
    <text x='50' y='166'>NOTES</text><text x='94' y='34'>CONCEPTS</text><text x='402' y='128'>RECALL</text><text x='765' y='65'>APPLY</text><text x='780' y='200'>MASTER</text>
  </svg>
  <div class='caption'>YOUR NOTES → CONNECTED UNDERSTANDING → LASTING MEMORY</div>
</div></div>""", unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")
with left:
    study_title = st.text_input("Give this study set a title", placeholder="e.g. Electric potential energy · Class 12 Physics")
    notes = st.text_area("Paste your notes or article", value=DEFAULT_NOTES, height=235, placeholder="START HERE… Paste a chapter, lecture notes, or an article.")
    count = 2
    st.caption("TWO-PAGE VISUAL STUDY DRAFT · ONE IDEA, THEN ITS CONNECTIONS")
    controls_a, controls_b = st.columns(2)
    with controls_a:
        with st.container(border=True):
            st.caption("01 · STUDY FORMAT")
            deck_type = st.segmented_control("Output", ["Visual sheets", "Q&A cards"], default="Visual sheets", label_visibility="collapsed")
    with controls_b:
        with st.container(border=True):
            st.caption("02 · GENERATION MODE")
            mode = st.segmented_control("Engine", ["Smart AI", "Offline demo"], default="Smart AI", label_visibility="collapsed")
    with st.expander("Advanced: connect your own Gemini key (optional)"):
        st.caption("Most people can ignore this. Offline demo always works; this is only for someone who already has a Gemini API key.")
        api_key = st.text_input("Gemini API key", value=local_api_key(), type="password", placeholder="Optional for demo mode")
    generate = st.button("✦ Create visual revision sheets" if deck_type == "Visual sheets" else "✦ Generate Q&A flashcards", type="primary", use_container_width=True)

with right:
    st.markdown("<div class='panel'><div class='eyebrow'>QUALITY STANDARD</div><h3>Not another copy-paste quiz.</h3><p>Every generation is prompted to balance definition, why/how, cause-effect, and application questions.</p></div>", unsafe_allow_html=True)
    metric_one, metric_two = st.columns(2)
    with metric_one:
        st.markdown("<div class='metric'><b>3×</b>question styles</div>", unsafe_allow_html=True)
    with metric_two:
        st.markdown("<div class='metric'><b>CSV</b>ready to export</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='transformation' aria-label='Plain notes transforming into a visual flashcard'>
      <div class='raw-note'>
        <small>RAW NOTES · BORING</small>
        <i></i><i></i><i></i><i></i>
      </div>
      <svg class='transform-arrow' viewBox='0 0 440 190' aria-hidden='true'>
        <defs><marker id='magic-arrow' markerWidth='9' markerHeight='9' refX='7' refY='4' orient='auto'><path d='M0,0 L0,8 L8,4 z' fill='#8eeedd'/></marker></defs>
        <path d='M22 132 C104 180 128 17 219 92 S332 159 409 54' marker-end='url(#magic-arrow)'/>
        <path d='M33 74 C114 12 159 164 255 108 S344 34 400 94'/>
        <circle cx='216' cy='92' r='5'/><text x='160' y='48'>RECALLLY MAGIC</text>
      </svg>
      <div class='mini-sheet'>
        <span class='tag'>VISUAL RECALL</span>
        <h4>Photosynthesis</h4>
        <p><mark>Light + water</mark> become stored energy.</p>
        <p>One idea. Easy to remember.</p>
        <span class='spark'>✦</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

if generate:
    if len(clean_text(notes)) < 80:
        st.error("Please paste at least a short paragraph so the cards have enough context.")
    else:
        with st.spinner("Designing questions that make you think…"):
            try:
                st.session_state.generation_notice = ""
                source_material = f"STUDY SET TITLE: {study_title}\n\n{notes}" if study_title.strip() else notes
                if mode == "Smart AI":
                    if not api_key:
                        raise ValueError("Add a Gemini API key, or choose Demo fallback for an offline demo.")
                    try:
                        if deck_type == "Visual sheets":
                            st.session_state.sheets, st.session_state.study_summary = gemini_sheets(source_material, count, api_key, study_title)
                            st.session_state.cards = []
                        else:
                            st.session_state.cards = gemini_cards(source_material, count, api_key)
                            st.session_state.sheets = []
                        st.session_state.generation_mode = "Gemini AI"
                    except Exception as gemini_error:
                        if deck_type == "Visual sheets":
                            st.session_state.sheets, st.session_state.study_summary = fallback_sheets(notes, count, study_title)
                            st.session_state.cards = []
                        else:
                            st.session_state.cards = fallback_cards(notes, count)
                            st.session_state.sheets = []
                        st.session_state.generation_mode = "Demo fallback"
                        st.session_state.generation_notice = "Gemini could not be reached, so Recallly switched to its offline demo generator. Your deck, flipping, and CSV export still work."
                else:
                    if deck_type == "Visual sheets":
                        st.session_state.sheets, st.session_state.study_summary = fallback_sheets(notes, count, study_title)
                        st.session_state.cards = []
                    else:
                        st.session_state.cards = fallback_cards(notes, count)
                        st.session_state.sheets = []
                    st.session_state.generation_mode = "Demo fallback"
                reset_index()
                st.session_state.mastered = set()
                output_count = len(st.session_state.sheets) if deck_type == "Visual sheets" else len(st.session_state.cards)
                noun = "revision sheets" if deck_type == "Visual sheets" else "flashcards"
                st.success(f"Created {output_count} {noun} with {st.session_state.generation_mode}.")
                if st.session_state.generation_notice:
                    st.warning(st.session_state.generation_notice)
            except Exception as error:
                st.error(f"Could not generate cards: {error}")

if st.session_state.sheets:
    sheets = st.session_state.sheets
    index = st.session_state.card_index
    sheet = sheets[index]
    st.markdown("<br>", unsafe_allow_html=True)
    title_col, download_col = st.columns([4, 1])
    with title_col:
        st.subheader("Your visual revision deck")
    with download_col:
        st.download_button("↓ Download CSV", data=sheets_csv_download(sheets), file_name=f"recally-revision-sheets-{datetime.now():%Y%m%d}.csv", mime="text/csv", use_container_width=True)

    if st.session_state.study_summary:
        with st.expander("Study blueprint used to build these two pages"):
            blueprint = st.session_state.study_summary
            st.write(blueprint.get("overview", ""))
            st.caption(" · ".join(blueprint.get("key_points", [])[:5]))

    rendered_sections = []
    for number, section in enumerate(sheet["sections"]):
        list_items = "".join(
            f"<li class='{'main-point' if point_number == 0 else 'sub-point'}'>{paint_text(point)}</li>"
            for point_number, point in enumerate(section["points"])
        )
        rendered_sections.append(f"<article class='orbit-note note-{number}'><h3>{html.escape(section['heading'])}</h3><ul>{list_items}</ul></article>")
    section_html = "".join(rendered_sections)
    formula_html = f"<div class='formula'>{html.escape(sheet['formula'])}</div>" if sheet["formula"] else ""
    remember_html = f"<div class='memory'><b>✦ Remember</b><span>{html.escape(sheet['remember'])}</span></div>" if sheet["remember"] else ""
    visual_html = render_visual(sheet.get("visual", {}))
    arrow_paths = [
        "<path d='M280 165 C338 208 385 253 440 303' marker-end='url(#card-arrow)'/><circle cx='280' cy='165' r='8'/>",
        "<path d='M720 173 C665 218 619 256 565 303' marker-end='url(#card-arrow)'/><circle cx='720' cy='173' r='8'/>",
        "<path d='M320 540 C360 486 403 440 452 394' marker-end='url(#card-arrow)'/><circle cx='320' cy='540' r='8'/>",
        "<path d='M680 548 C638 493 598 446 550 394' marker-end='url(#card-arrow)'/><circle cx='680' cy='548' r='8'/>",
    ][:len(sheet["sections"])]
    arrows_html = "".join(arrow_paths)
    mastered_count = len(st.session_state.mastered)
    st.markdown(f"<div class='progress-line'>LEARNING PROGRESS · {mastered_count} OF {len(sheets)} CARDS MEMORIZED</div>", unsafe_allow_html=True)
    st.markdown(
        f"<section class='revision-sheet'><div class='orbit-notes'><svg class='curly-links' viewBox='0 0 1000 880' aria-hidden='true'><defs><marker id='card-arrow' markerWidth='9' markerHeight='9' refX='7' refY='4' orient='auto'><path d='M0,0 L0,8 L8,4 z' fill='#74aee0'/></marker></defs>{arrows_html}</svg><div class='concept-core'><small>CORE CONCEPT · {sheet['difficulty'].upper()}</small><h2>{html.escape(sheet['title'])}</h2></div>{section_html}{visual_html}{formula_html}{remember_html}</div><div class='sheet-meta'>RECALLLY VISUAL NOTES · SHEET {index + 1} OF {len(sheets)}</div></section>",
        unsafe_allow_html=True,
    )
    previous, challenge, memorize, next_sheet = st.columns([1, 1.35, 1.35, 1])
    with previous:
        st.button("← Previous sheet", disabled=index == 0, on_click=move_card, args=(-1,), use_container_width=True)
    with challenge:
        st.button("🧠 Test me", on_click=toggle_challenge, use_container_width=True)
    with memorize:
        st.button("✓ Memorized" if index in st.session_state.mastered else "◌ Mark memorized", on_click=toggle_mastered, args=(index,), use_container_width=True)
    with next_sheet:
        st.button("Next sheet →", disabled=index == len(sheets) - 1, on_click=move_card, args=(1,), type="primary", use_container_width=True)
    if st.session_state.show_challenge:
        challenge_data = sheet.get("challenge", {})
        st.info(f"**Quick check:** {challenge_data.get('question', 'Explain the key idea in your own words.')}\n\n**Answer:** {challenge_data.get('answer', sheet['remember'])}")

elif st.session_state.cards:
    cards = st.session_state.cards
    index = st.session_state.card_index
    card = cards[index]
    st.markdown("<br>", unsafe_allow_html=True)
    title_col, download_col = st.columns([4, 1])
    with title_col:
        st.subheader("Your revision deck")
    with download_col:
        st.download_button("↓ Download CSV", data=csv_download(cards), file_name=f"recally-flashcards-{datetime.now():%Y%m%d}.csv", mime="text/csv", use_container_width=True)

    side = "ANSWER" if st.session_state.revealed else "QUESTION"
    text = card["answer"] if st.session_state.revealed else card["question"]
    extra_class = " answer" if st.session_state.revealed else ""
    st.markdown(f"<div class='flashcard{extra_class}'><div class='card-side'>{side} · {card['difficulty'].upper()}</div><div class='card-text'>{text}</div><div class='card-footer'>Card {index + 1} of {len(cards)} · Click reveal to test your recall first.</div></div>", unsafe_allow_html=True)
    previous, reveal, next_card = st.columns([1, 2, 1])
    with previous:
        st.button("← Previous", disabled=index == 0, on_click=move_card, args=(-1,), use_container_width=True)
    with reveal:
        st.button("Show answer" if not st.session_state.revealed else "Show question", on_click=lambda: st.session_state.__setitem__("revealed", not st.session_state.revealed), type="primary", use_container_width=True)
    with next_card:
        st.button("Next →", disabled=index == len(cards) - 1, on_click=move_card, args=(1,), use_container_width=True)

    with st.expander("See all generated cards"):
        for number, item in enumerate(cards, start=1):
            st.markdown(f"**{number}. {item['question']}**  \\n+            {item['answer']}  \\n+            *{item['difficulty']}*")
else:
    st.info("Paste notes and generate a deck. The demo fallback works without any API key; use Gemini AI for the strongest hackathon demo.")

st.caption("Recallly · AI Flashcard Generator · Built for fast, thoughtful revision")
