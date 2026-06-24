"""
03_build_entity_dictionary.py

Build the entity ALIGNMENT DICTIONARY used by 05_compute_metrics.py for
dictionary-based (not CLIP) grounding. For each entity that can appear in a
detection — political objects (from config) and gallery person identities — we
collect the surface forms a news caption might use to refer to it: surname alone,
title+surname, nicknames, synonyms, plurals. POLVIFIDEL then counts an entity as
"mentioned" iff one of its aliases appears in the caption, avoiding both the CLIP
saturation (everything matches) and the conflation of distinct political entities.

Aliases come from (a) deterministic rules and (b) optional LLM expansion
(GPT-4o-mini) for nicknames/synonyms — per the CLAUDE.md design. Results are
cached: re-running only generates aliases for entities not already in the file.

Usage:
    python 03_build_entity_dictionary.py                      # rules + LLM, all entities
    python 03_build_entity_dictionary.py --no-llm             # deterministic only (no API)
    python 03_build_entity_dictionary.py --from-detections    # also include labels seen in data/detections/
    python 03_build_entity_dictionary.py --model gpt-4o

Output:
    data/entity_dict.json   {canonical_label: [lowercase aliases...]}
"""

from __future__ import annotations

import argparse
import json
import pickle
import re

from config import DATA_DIR, DETECTIONS_DIR, GALLERY_DIR, POLITICAL_OBJECT_QUERIES

OUT_PATH = DATA_DIR / "entity_dict.json"

# Plural/synonym seeds for the standard political-object vocabulary.
OBJECT_SEEDS = {
    "flag": ["flag", "flags", "american flag", "national flag", "banner"],
    "banner": ["banner", "banners", "sign"],
    "sign": ["sign", "signs", "placard", "placards"],
    "poster": ["poster", "posters", "placard"],
    "podium": ["podium", "podiums", "lectern", "rostrum"],
    "microphone": ["microphone", "microphones", "mic", "mics"],
    "hat": ["hat", "hats", "cap", "caps", "maga hat", "red hat"],
    "badge": ["badge", "badges", "pin", "lapel pin"],
    "button": ["button", "buttons", "campaign button", "pin"],
    "campaign shirt": ["campaign shirt", "t-shirt", "tee shirt", "shirt"],
    "protest sign": ["protest sign", "protest signs", "placard", "sign"],
}

TITLES = ["president", "former president", "senator", "governor", "vice president",
          "representative", "secretary", "speaker", "mayor", "justice"]


def normalize(s: str) -> str:
    s = s.lower().strip().replace("_", " ")   # snake_case subcategory labels -> words
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def person_alias_seeds(name: str) -> list[str]:
    """Deterministic surface forms for a person name (handles 'Last, First' too)."""
    raw = re.sub(r"\([^)]*\)", "", name).strip()        # drop '(1972- )'
    if "," in raw:
        last, first = (p.strip() for p in raw.split(",", 1))
        full = f"{first} {last}"
    else:
        full = raw
        last = raw.split()[-1] if raw.split() else raw
    full_n, last_n = normalize(full), normalize(last)
    aliases = {full_n, last_n}
    aliases.update(f"{t} {last_n}" for t in TITLES)       # 'president trump'
    return sorted(a for a in aliases if a)


def collect_entities(use_detections: bool) -> tuple[set, set]:
    """Return (object_labels, person_labels)."""
    objects = set(POLITICAL_OBJECT_QUERIES)
    people = set()

    gallery_pkl = GALLERY_DIR / "embeddings.pkl"
    if gallery_pkl.exists():
        with open(gallery_pkl, "rb") as f:
            people.update(pickle.load(f).keys())

    if use_detections:
        for p in DETECTIONS_DIR.glob("*.json"):
            det = json.load(open(p))
            objects.update(o["label"] for o in det.get("objects", []))
            # Fine-grained political-object subcategories from the gallery match.
            objects.update(o["subcategory"] for o in det.get("objects", [])
                           if o.get("subcategory"))
            people.update(f["name"] for f in det.get("faces", [])
                          if f.get("match_status") == "matched" and f.get("name"))
    return objects, people


def llm_expand(entities: list[str], model: str) -> dict[str, list[str]]:
    """Ask the LLM for extra surface forms. Batched. Returns {entity: [aliases]}."""
    import os
    from portkey_ai import Portkey
    client = Portkey(api_key=os.getenv("AI_SANDBOX_KEY"))   # Princeton AI Sandbox gateway
    out: dict[str, list[str]] = {}
    BATCH = 20
    for i in range(0, len(entities), BATCH):
        chunk = entities[i:i + BATCH]
        prompt = (
            "For each entity below (a person or an object seen in news photographs), "
            "list the short lowercase surface forms a news caption might use to refer "
            "to it: last name alone, common nicknames, title+name, synonyms, plurals. "
            "Keep them specific — do NOT include forms that would match a different "
            "person. Return strict JSON mapping each entity exactly as given to a list "
            f"of strings.\n\nEntities:\n{json.dumps(chunk)}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content)
            for k, v in data.items():
                if isinstance(v, list):
                    out[k] = [normalize(str(x)) for x in v if str(x).strip()]
        except Exception as e:  # noqa: BLE001
            print(f"  LLM batch {i // BATCH} failed: {e}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-llm", action="store_true",
                        help="Deterministic aliases only (no API calls)")
    parser.add_argument("--from-detections", action="store_true",
                        help="Also include labels/names found in data/detections/")
    args = parser.parse_args()

    out_path = DATA_DIR / "entity_dict.json" if args.out == str(OUT_PATH) else args.out
    existing = {}
    try:
        with open(out_path) as f:
            existing = json.load(f)
    except FileNotFoundError:
        pass

    objects, people = collect_entities(args.from_detections)
    print(f"Entities: {len(objects)} objects, {len(people)} people "
          f"({len(existing)} already in dict)")

    aliases: dict[str, list[str]] = dict(existing)

    # Deterministic seeds.
    for obj in objects:
        seeds = OBJECT_SEEDS.get(obj, [normalize(obj)])
        seeds = sorted(set(seeds) | {normalize(obj)})
        aliases[obj] = sorted(set(aliases.get(obj, [])) | set(seeds))
    for person in people:
        aliases[person] = sorted(set(aliases.get(person, [])) | set(person_alias_seeds(person)))

    # LLM expansion only for entities not previously expanded.
    if not args.no_llm:
        todo = [e for e in list(objects) + list(people) if e not in existing]
        if todo:
            print(f"LLM-expanding {len(todo)} new entities ({args.model})...")
            extra = llm_expand(todo, args.model)
            for k, v in extra.items():
                if k in aliases:
                    aliases[k] = sorted(set(aliases[k]) | set(v))

    # Always include the canonical's own normalized form.
    for canon in list(aliases):
        aliases[canon] = sorted(set(aliases[canon]) | {normalize(canon)})

    with open(out_path, "w") as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)
    n_alias = sum(len(v) for v in aliases.values())
    print(f"\nWrote {len(aliases)} entities, {n_alias} aliases → {out_path}")
    for k in list(aliases)[:8]:
        print(f"  {k!r}: {aliases[k]}")


if __name__ == "__main__":
    main()
