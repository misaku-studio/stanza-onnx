"""Generate Unicode 18 tables directly from checksum-pinned official UCD files.

The data does not depend on the generating interpreter's Unicode database.
Default execution writes byte-identical Rust and Python-reference assets.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

UNICODE_VERSION = "18.0.0"
UCD_ROOT = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd"
SOURCES = {
    "ReadMe.txt": "b0ea442f29dee90584aacd3665a6e1ce87ffc917a4d31f34e1916d2d6f7f9205",
    "UnicodeData.txt": "0736451de439ae7baf1425136617da495e09ee5afbe6e394374db7009ea08950",
    "SpecialCasing.txt": "8538dea57c184f1ef3783885ea79677b10f6efa06423717157e63712f14d1ad2",
    "DerivedCoreProperties.txt": "09c928886a178fcafd93c29e4bd59073a058e5a100b716d425cb563ab50f68c9",
    "extracted/DerivedNumericType.txt": "3dade4d96bd00d71b10022bf70b370090f3ca947f977ad4dc2854e06c4d074f6",
}
ROOT = Path(__file__).resolve().parents[1]


def source_files(cache):
    """Existing cache files are read only; every file must match its pinned hash."""
    files = {}
    for name, expected in SOURCES.items():
        path = cache / name
        if not path.exists():
            request = urllib.request.Request(
                f"{UCD_ROOT}/{name}", headers={"User-Agent": "stanza-onnx"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                contents = response.read()
            if hashlib.sha256(contents).hexdigest() != expected:
                raise ValueError(f"Downloaded Unicode source checksum mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(f"Cached Unicode source checksum mismatch: {name}")
        files[name] = contents.decode("utf-8")
    if f"final data files for version {UNICODE_VERSION}" not in files["ReadMe.txt"]:
        raise ValueError("Unicode source is not the pinned final release")
    return files


def codepoints(value):
    bounds = value.strip().split("..")
    start = int(bounds[0], 16)
    end = int(bounds[-1], 16)
    if not 0 <= start <= end <= 0x10FFFF:
        raise ValueError(f"Invalid Unicode range: {value}")
    return range(start, end + 1)


def ranges(points):
    result = []
    for number in sorted(points):
        if result and number == result[-1][1] + 1:
            result[-1][1] = number
        else:
            result.append([number, number])
    return result


def property_records(text):
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            fields = [field.strip() for field in line.split(";")]
            yield fields


def mapping(value):
    return "".join(chr(int(point, 16)) for point in value.split())


def set_mapping(target, point, value):
    if value == chr(point):
        target.pop(point, None)
    else:
        target[point] = value


def _read_unicode_data(text, properties, lower, upper, title):
    """Collect categories and simple casing, including First/Last ranges."""
    first = None
    for line in text.splitlines():
        fields = line.split(";")
        if len(fields) != 15:
            raise ValueError("Invalid UnicodeData record")
        point = int(fields[0], 16)
        name = fields[1]
        category = fields[2]
        bidi = fields[4]
        if name.endswith(", First>"):
            if first is not None:
                raise ValueError("Nested UnicodeData First range")
            first = (point, category, bidi, name.removesuffix(", First>"))
            continue
        if name.endswith(", Last>"):
            if first is None or first[1:] != (
                category,
                bidi,
                name.removesuffix(", Last>"),
            ):
                raise ValueError("Unmatched UnicodeData Last range")
            points = range(first[0], point + 1)
            first = None
        else:
            if first is not None:
                raise ValueError("Unterminated UnicodeData First range")
            points = (point,)
            if fields[13]:
                set_mapping(lower, point, mapping(fields[13]))
            if fields[12]:
                set_mapping(upper, point, mapping(fields[12]))
            if fields[14] or fields[12]:
                set_mapping(title, point, mapping(fields[14] or fields[12]))
        # Python-style alpha uses Letter categories, not the broader derived
        # Alphabetic property. Numeric_Type supplies alphanumeric letter numbers.
        if category in {"Lu", "Ll", "Lt", "Lm", "Lo"}:
            properties["alpha"].update(points)
        if category == "Lt":
            properties["titlecase"].update(points)
        if category == "Zs" or bidi in {"WS", "B", "S"}:
            properties["whitespace"].update(points)
    if first is not None:
        raise ValueError("Unterminated UnicodeData range")


def _read_core_properties(text, properties):
    core = {
        "Uppercase": "uppercase",
        "Lowercase": "lowercase",
        "Cased": "cased",
        "Case_Ignorable": "caseIgnorable",
    }
    for fields in property_records(text):
        if fields[1] in core:
            properties[core[fields[1]]].update(codepoints(fields[0]))


def _read_numeric_properties(text, properties):
    # This extracted file includes numeric properties originating in Unihan.
    # UnicodeData's numeric columns alone omit many CJK numeric characters.
    for fields in property_records(text):
        kind = fields[1]
        points = codepoints(fields[0])
        if kind not in {"Decimal", "Digit", "Numeric"}:
            raise ValueError(f"Unexpected Numeric_Type: {kind}")
        if kind == "Decimal":
            properties["decimal"].update(points)
        if kind in {"Decimal", "Digit"}:
            properties["digit"].update(points)
        properties["numeric"].update(points)


def _read_special_casing(text, lower, upper, title):
    """Full mappings override simple ones; only default contextual rules apply."""
    final_sigma = None
    for fields in property_records(text):
        point = int(fields[0], 16)
        condition = fields[4]
        if not condition:
            for target, value in (
                (lower, fields[1]),
                (title, fields[2]),
                (upper, fields[3]),
            ):
                set_mapping(target, point, mapping(value))
        elif condition == "Final_Sigma":
            if point != 0x03A3 or mapping(fields[1]) != "\u03c2":
                raise ValueError("Unexpected Final_Sigma mapping")
            final_sigma = {
                "codepoint": point,
                "condition": condition,
                "lower": mapping(fields[1]),
            }
        elif not any(locale in condition.split() for locale in ("lt", "tr", "az")):
            raise ValueError(
                f"Unsupported locale-independent casing condition: {condition}"
            )
        # Locale-specific mappings do not apply to default Unicode casing.
    if final_sigma is None:
        raise ValueError("Missing default conditional Final_Sigma mapping")
    return final_sigma


def build_tables(files):
    properties = {
        name: set()
        for name in (
            "uppercase",
            "lowercase",
            "titlecase",
            "cased",
            "caseIgnorable",
            "decimal",
            "digit",
            "numeric",
            "alpha",
            "alnum",
            "whitespace",
        )
    }
    lower = {}
    upper = {}
    title = {}
    _read_unicode_data(files["UnicodeData.txt"], properties, lower, upper, title)
    _read_core_properties(files["DerivedCoreProperties.txt"], properties)
    _read_numeric_properties(files["extracted/DerivedNumericType.txt"], properties)
    properties["alnum"] = properties["alpha"] | properties["numeric"]
    final_sigma = _read_special_casing(files["SpecialCasing.txt"], lower, upper, title)

    return {
        "unicodeVersion": UNICODE_VERSION,
        "sources": [
            {"path": name, "url": f"{UCD_ROOT}/{name}", "sha256": digest}
            for name, digest in SOURCES.items()
        ],
        **{name: ranges(points) for name, points in properties.items()},
        "lower": {str(point): lower[point] for point in sorted(lower)},
        "upper": {str(point): upper[point] for point in sorted(upper)},
        "title": {str(point): title[point] for point in sorted(title)},
        "conditionalLower": [final_sigma],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/unicode18")
    parser.add_argument(
        "--output",
        type=Path,
        help="Rust output; overrides the default dual-file generation",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Optional identical Python-reference output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated bytes with outputs without changing them",
    )
    args = parser.parse_args()
    output = args.output or ROOT / "crates/stanza-onnx/src/data/unicode18.json"
    reference = args.reference_output
    if reference is None and args.output is None:
        reference = ROOT / "exporter/src/stanza_onnx/data/unicode18.json"
    data = build_tables(source_files(args.cache_dir))
    contents = (
        json.dumps(data, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    for destination in [output] + ([reference] if reference is not None else []):
        if args.check:
            if not destination.is_file() or destination.read_bytes() != contents:
                raise ValueError(f"Unicode table regeneration differs: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
    print(
        json.dumps(
            {
                "unicodeVersion": UNICODE_VERSION,
                "output": str(output),
                "referenceOutput": str(reference) if reference else None,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "checked": args.check,
            }
        )
    )


if __name__ == "__main__":
    main()
