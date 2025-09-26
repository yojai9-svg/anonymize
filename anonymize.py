import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import spacy


CHUNK_SIZE = 5000


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> Iterable[str]:
    """Yield text chunks with a maximum size."""
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def index_to_letters(index: int) -> str:
    """Convert a zero-based index to alphabetical sequences (A, B, ..., Z, AA, AB, ...)."""
    letters: List[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        letters.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(letters))


def greek_sequence(index: int) -> str:
    """Return Greek letter style names in sequence (Alpha, Beta, ..., Omega, Alpha2, ...)."""
    greek_names = [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "Eta",
        "Theta",
        "Iota",
        "Kappa",
        "Lambda",
        "Mu",
        "Nu",
        "Xi",
        "Omicron",
        "Pi",
        "Rho",
        "Sigma",
        "Tau",
        "Upsilon",
        "Phi",
        "Chi",
        "Psi",
        "Omega",
    ]
    base_count = len(greek_names)
    cycle, offset = divmod(index, base_count)
    name = greek_names[offset]
    if cycle == 0:
        return name
    return f"{name}{cycle + 1}"


@dataclass
class PlaceholderManager:
    person_counter: int = 0
    company_counter: int = 23  # Start at "X"
    fund_counter: int = 0
    firm_counter: int = 0
    assigned_placeholders: Dict[str, str] = field(default_factory=dict)

    def assign(self, original: str, label: str) -> str:
        """Assign a placeholder for a new original string."""
        normalized = original.strip()
        if normalized in self.assigned_placeholders:
            return self.assigned_placeholders[normalized]

        placeholder_type = self._determine_type(normalized, label)
        placeholder = self._generate_placeholder(normalized, placeholder_type)
        self.assigned_placeholders[normalized] = placeholder
        return placeholder

    def _determine_type(self, text: str, label: str) -> str:
        lower_text = text.lower()
        if text == "GIC":
            return "gic"
        if "fund" in lower_text:
            return "fund"
        upper_text = text.upper()
        if "LLP" in upper_text or "LLC" in upper_text:
            return "firm"
        if label == "PERSON":
            return "person"
        if label == "ORG":
            return "company"
        return "other"

    def _generate_placeholder(self, original: str, placeholder_type: str) -> str:
        if placeholder_type == "person":
            placeholder = f"Person {index_to_letters(self.person_counter)}"
            self.person_counter += 1
            return self._inject_granite_if_needed(original, placeholder)
        if placeholder_type == "company":
            placeholder = f"Company {index_to_letters(self.company_counter)}"
            self.company_counter += 1
            return self._inject_granite_if_needed(original, placeholder)
        if placeholder_type == "fund":
            placeholder = f"Fund {greek_sequence(self.fund_counter)}"
            self.fund_counter += 1
            return self._inject_granite_if_needed(original, placeholder, prefix_only=True)
        if placeholder_type == "firm":
            placeholder = f"Firm {greek_sequence(self.firm_counter)} LLP"
            self.firm_counter += 1
            return self._inject_granite_if_needed(original, placeholder, prefix_only=True)
        if placeholder_type == "gic":
            return "Granite"
        # Default: return original text unchanged
        return original

    def _inject_granite_if_needed(
        self, original: str, placeholder: str, prefix_only: bool = False
    ) -> str:
        if "GIC" not in original:
            return placeholder
        if prefix_only:
            return f"Granite {placeholder}"
        return f"Granite {placeholder}" if not placeholder.startswith("Granite") else placeholder


def build_replacements(
    doc,
    placeholder_manager: PlaceholderManager,
    global_lookup: "OrderedDict[str, str]",
) -> List[Tuple[int, int, str]]:
    replacements: List[Tuple[int, int, str]] = []
    for ent in doc.ents:
        original = ent.text
        normalized = original.strip()
        if not normalized:
            continue
        if normalized in global_lookup:
            placeholder = global_lookup[normalized]
        else:
            placeholder = placeholder_manager.assign(normalized, ent.label_)
            global_lookup[normalized] = placeholder
        replacements.append((ent.start_char, ent.end_char, placeholder))
    return sorted(replacements, key=lambda item: item[0])


def apply_replacements(text: str, replacements: List[Tuple[int, int, str]]) -> str:
    if not replacements:
        return text
    output: List[str] = []
    last_index = 0
    for start, end, replacement in replacements:
        if start < last_index:
            continue
        output.append(text[last_index:start])
        output.append(replacement)
        last_index = end
    output.append(text[last_index:])
    return "".join(output)


def replace_remaining_gic(
    original_text: str,
    anonymized_text: str,
    global_lookup: "OrderedDict[str, str]",
) -> str:
    pattern = re.compile(r"\bGIC\b")

    if not pattern.search(original_text):
        return anonymized_text

    def substitute(match: re.Match[str]) -> str:
        return "Granite"

    if "GIC" not in global_lookup:
        global_lookup["GIC"] = "Granite"

    return pattern.sub(substitute, anonymized_text)


def anonymize_text(text: str, nlp) -> Tuple[str, "OrderedDict[str, str]"]:
    placeholder_manager = PlaceholderManager()
    lookup: "OrderedDict[str, str]" = OrderedDict()
    processed_chunks: List[str] = []

    for chunk in chunk_text(text):
        doc = nlp(chunk)
        replacements = build_replacements(doc, placeholder_manager, lookup)
        processed_chunks.append(apply_replacements(chunk, replacements))

    anonymized = "".join(processed_chunks)
    anonymized = replace_remaining_gic(text, anonymized, lookup)
    return anonymized, lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymize legal documents.")
    parser.add_argument("input_path", type=Path, help="Path to the input text file")
    parser.add_argument(
        "output_path", type=Path, help="Path to the output JSON file"
    )
    args = parser.parse_args()

    text = args.input_path.read_text(encoding="utf-8")

    nlp = spacy.load("en_core_web_sm")
    anonymized_text, lookup = anonymize_text(text, nlp)

    lookup_table = [
        {"original": original, "anonymized": placeholder}
        for original, placeholder in lookup.items()
    ]
    result = {"anonymized_text": anonymized_text, "lookup_table": lookup_table}

    args.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
