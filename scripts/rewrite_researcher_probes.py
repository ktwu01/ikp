"""Rewrite researcher subfield probe questions to require evidence.

Old:  "In computer science, what is the research subfield of {name}?"
New:  "In computer science, what is the research subfield of {name}, and name
       one paper, system, institution, or co-author associated with their
       work? If you don't know who this person is, say so."

This shifts the burden of proof: bluffers either refuse or fabricate evidence
that the judge can verify against the enriched gold bundle.

Reads:  data/probes/final_probe_set_v8.json
Writes: data/probes/final_probe_set_v9.json (replacement; same probe IDs)
"""

import json
from pathlib import Path

ROOT = Path("/Users/boj/ikp-paper")
SRC = ROOT / "data" / "probes" / "final_probe_set_v8.json"
DST = ROOT / "data" / "probes" / "final_probe_set_v9.json"

NEW_TEMPLATE = (
    "In computer science, what is the research subfield of {name}, and name "
    "one paper, system, institution, or co-author associated with their work? "
    "If you don't know who this person is, say so."
)


def main():
    with open(SRC) as f:
        probes = json.load(f)

    out = []
    rewritten = 0
    for p in probes:
        if p.get("source_type") == "researcher":
            name = p.get("researcher_name")
            if not name:
                # Parse from old question pattern
                q = p["question"]
                # "In computer science, what is the research subfield of NAME?"
                if "research subfield of " in q:
                    name = q.split("research subfield of ", 1)[1].rstrip("?").strip()
            if name:
                pp = dict(p)
                pp["question_v1"] = p["question"]  # preserve original
                pp["question"] = NEW_TEMPLATE.format(name=name)
                pp["researcher_name"] = name
                rewritten += 1
                out.append(pp)
                continue
        out.append(p)

    with open(DST, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Rewrote {rewritten} researcher probes; wrote {len(out)} probes to {DST}")


if __name__ == "__main__":
    main()
