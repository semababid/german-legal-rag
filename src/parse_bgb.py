"""Flatten a gesetze-im-internet.de law XML file into retrieval-ready records.

The source format keeps two things separate that a retriever needs together:

  * Structural headings (Buch / Abschnitt / Titel / Untertitel) live in their own
    `<norm>` elements and carry a `gliederungskennzahl` — a nested numeric key
    using three digits per level ("010" -> "010010" -> "010010010").
  * The actual sections (`§ 573c`) carry no hierarchy of their own; they inherit
    it from whichever headings preceded them in document order.

So this walks the document once, maintaining a hierarchy stack, and attaches the
current breadcrumb to every section it emits. Output is one JSON object per
Absatz, which is the natural chunk boundary for German statutes — an Absatz is
self-contained enough to answer a question and short enough to embed well.

    python src/parse_bgb.py --law bgb
    python src/parse_bgb.py --law bgb --granularity paragraph --stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://www.gesetze-im-internet.de"

# Levels in a gliederungskennzahl are three digits wide.
KZ_LEVEL_WIDTH = 3

# "(1)", "(2a)" at the start of a paragraph opens a new Absatz.
ABSATZ_RE = re.compile(r"^\((\d+[a-z]?)\)\s*")

# Sections the legislator has emptied out rather than renumbered. These markers
# appear both as a section's title and as the body text of an individual
# Absatz, so the check runs at both levels.
REPEALED_RE = re.compile(r"^\(?(weggefallen|aufgehoben|gegenstandslos|XXXX)\)?$", re.IGNORECASE)


@dataclass
class Record:
    id: str
    law: str
    law_title: str
    paragraph: str
    paragraph_key: str
    absatz: str | None
    title: str | None
    breadcrumb: list[str]
    citation: str
    text: str
    source_url: str
    builddate: str | None
    repealed: bool = False
    context: str = field(default="", repr=False)


def flatten_text(el: ET.Element) -> str:
    """Render an element's subtree as plain text, keeping list structure legible.

    Numbered lists arrive as <DL><DT>1.</DT><DD>...</DD></DL>. Naive itertext()
    would run the marker into the item text ("1.die Grundsätze"), so definition
    terms get a leading newline and a trailing space.
    """
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag == "DT":
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            if child.tag == "BR":
                parts.append("\n")
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)
        if node.tag == "DT":
            parts.append(" ")

    walk(el)

    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_absaetze(content: ET.Element) -> list[tuple[str | None, str]]:
    """Group a section's <P> elements into (absatz_number, text) pairs.

    Roughly half of BGB sections are a single unnumbered block; those come back
    as one pair with absatz=None. A <P> that does not open with "(n)" is treated
    as a continuation of the Absatz above it.
    """
    blocks: list[tuple[str | None, list[str]]] = []

    for p in content.findall("P"):
        text = flatten_text(p)
        if not text:
            continue

        match = ABSATZ_RE.match(text)
        if match:
            blocks.append((match.group(1), [text[match.end():].strip()]))
        elif blocks:
            blocks[-1][1].append(text)
        else:
            blocks.append((None, [text]))

    return [(num, "\n".join(chunk).strip()) for num, chunk in blocks if "".join(chunk).strip()]


def paragraph_key(enbez: str) -> str:
    """'§ 573c' -> '573c'. Used for IDs and for building per-section URLs."""
    return enbez.replace("§", "").replace("§§", "").strip().replace(" ", "")


def section_url(law: str, key: str) -> str:
    """gesetze-im-internet publishes one page per section as __<num>.html."""
    return f"{BASE_URL}/{law}/__{key.lower()}.html"


def parse(xml_path: Path, law: str, granularity: str) -> tuple[list[Record], dict]:
    root = ET.parse(xml_path).getroot()
    norms = root.findall("norm")

    builddate = root.get("builddate")
    law_title = norms[0].findtext("metadaten/langue") if norms else None
    law_title = law_title or law.upper()

    hierarchy: dict[int, str] = {}
    records: list[Record] = []
    stats = {
        "norms": len(norms),
        "headings": 0,
        "sections": 0,
        "repealed": 0,
        "repealed_absaetze": 0,
        "records": 0,
        "absatz_numbered": 0,
        "single_block": 0,
    }

    for norm in norms:
        enbez = (norm.findtext("metadaten/enbez") or "").strip()
        gliederung = norm.find("metadaten/gliederungseinheit")

        # A structural heading: update the stack and drop everything below it.
        if gliederung is not None:
            kz = (gliederung.findtext("gliederungskennzahl") or "").strip()
            bez = (gliederung.findtext("gliederungsbez") or "").strip()
            titel = (gliederung.findtext("gliederungstitel") or "").strip()
            if kz:
                level = len(kz) // KZ_LEVEL_WIDTH
                label = " — ".join(part for part in (bez, titel) if part)
                hierarchy = {lvl: v for lvl, v in hierarchy.items() if lvl < level}
                hierarchy[level] = label or bez or titel
                stats["headings"] += 1
            continue

        if not enbez.startswith("§"):
            continue  # Inhaltsübersicht, preamble, vacated ranges

        stats["sections"] += 1

        key = paragraph_key(enbez)
        title = (norm.findtext("metadaten/titel") or "").strip() or None
        breadcrumb = [hierarchy[lvl] for lvl in sorted(hierarchy)]
        content = norm.find("textdaten/text/Content")

        repealed = content is None or bool(title and REPEALED_RE.match(title))
        if repealed:
            stats["repealed"] += 1
            continue

        absaetze = split_absaetze(content)

        # Drop individual repealed Absätze ("(2) (weggefallen)") — as chunks
        # they carry no information and waste retrieval slots.
        dropped = len(absaetze)
        absaetze = [(num, text) for num, text in absaetze if not REPEALED_RE.match(text)]
        stats["repealed_absaetze"] += dropped - len(absaetze)

        if not absaetze:
            stats["repealed"] += 1
            continue

        if granularity == "paragraph":
            merged = "\n\n".join(
                f"({num}) {text}" if num else text for num, text in absaetze
            )
            absaetze = [(None, merged)]

        if len(absaetze) > 1 or absaetze[0][0] is not None:
            stats["absatz_numbered"] += 1
        else:
            stats["single_block"] += 1

        for num, text in absaetze:
            citation = f"{law.upper()} {enbez}"
            if num:
                citation += f" Abs. {num}"

            # A prefix carrying the breadcrumb and section title. Embedding this
            # alongside the text is what lets a query like "Kündigungsfrist"
            # reach an Absatz whose own wording never repeats the section title.
            context_parts = breadcrumb + [f"{enbez} {title}" if title else enbez]
            context = " > ".join(context_parts)

            records.append(
                Record(
                    id=f"{law}-{key}" + (f"-{num}" if num else ""),
                    law=law.upper(),
                    law_title=law_title,
                    paragraph=enbez,
                    paragraph_key=key,
                    absatz=num,
                    title=title,
                    breadcrumb=breadcrumb,
                    citation=citation,
                    text=text,
                    source_url=section_url(law, key),
                    builddate=builddate,
                    repealed=False,
                    context=context,
                )
            )

    stats["records"] = len(records)
    return records, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--law", default="bgb", help="law abbreviation, matching data/raw/<law>/ (default: bgb)")
    ap.add_argument("--input", type=Path, help="explicit XML path, overrides --law lookup")
    ap.add_argument("--output", type=Path, help="output .jsonl (default: data/processed/<law>.jsonl)")
    ap.add_argument(
        "--granularity",
        choices=("absatz", "paragraph"),
        default="absatz",
        help="one record per Absatz (default) or per section — the chunking comparison",
    )
    ap.add_argument("--stats", action="store_true", help="print a parse summary")
    args = ap.parse_args()

    if args.input:
        xml_path = args.input
    else:
        raw_dir = REPO_ROOT / "data" / "raw" / args.law
        candidates = sorted(raw_dir.glob("*.xml"))
        if not candidates:
            print(
                f"No XML under {raw_dir}. Run: ./scripts/fetch_corpus.sh {args.law}",
                file=sys.stderr,
            )
            return 1
        xml_path = candidates[0]

    records, stats = parse(xml_path, args.law, args.granularity)

    out_path = args.output or REPO_ROOT / "data" / "processed" / f"{args.law}.{args.granularity}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    print(f"{stats['records']} records -> {out_path}")

    if args.stats:
        print()
        print(f"  source            {xml_path.name}")
        print(f"  norms             {stats['norms']}")
        print(f"  headings          {stats['headings']}")
        print(f"  sections          {stats['sections']}")
        print(f"  repealed/empty    {stats['repealed']}")
        print(f"  repealed Absätze  {stats['repealed_absaetze']}")
        print(f"  multi-Absatz      {stats['absatz_numbered']}")
        print(f"  single block      {stats['single_block']}")
        if records:
            lengths = sorted(len(r.text) for r in records)
            print(f"  chars  median     {lengths[len(lengths) // 2]}")
            print(f"         p95        {lengths[int(len(lengths) * 0.95)]}")
            print(f"         max        {lengths[-1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
