"""Write the control runs without resonators from the bowtie scripts.

Each script in ``simulations/bowtie`` is copied with one line changed: the
geometry keeps only the YIG film. Everything else (antennas, absorbing layers,
solver settings, run time) stays as in the original, so every control run is
paired with its bowtie run case by case.

    python scripts/make_no_resonators.py simulations/bowtie simulations/no_resonators
"""

import sys
from pathlib import Path

OLD = "geometry := yig.add("
NEW = "geometry := yig   // control: no resonators, the only change from the bowtie script"


def convert(text: str) -> str:
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip().startswith(OLD)]
    if len(hits) != 1:
        raise ValueError(f"expected one line starting with '{OLD}', found {len(hits)}")
    lines[hits[0]] = NEW + "\n"
    return "".join(lines)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.mx3")):
        (dst / f.name).write_text(convert(f.read_text()))
        print(f"{dst / f.name}")


if __name__ == "__main__":
    main()
