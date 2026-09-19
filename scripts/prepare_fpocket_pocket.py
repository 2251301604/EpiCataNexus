#!/usr/bin/env python3
"""Prepare a ligand-independent fpocket pocket PDB from a protein structure.

The implemented manuscript rule is: keep protein ATOM records; run fpocket with
default parameters; retain the cavity with the highest fpocket Score; and select
protein residues having at least one heavy atom within 6 Angstrom of one of that
cavity's alpha spheres. No ligand information is read or used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
POCKET_PATTERN = re.compile(r"^\s*Pocket\s+(\d+)\s*:?\s*$", re.IGNORECASE)
SCORE_PATTERN = re.compile(rf"^\s*Score\s*:\s*({FLOAT_PATTERN})\s*$", re.IGNORECASE)
VERSION_PATTERN = re.compile(r"fpocket\s+(?:version\s+)?(\d+(?:\.\d+){1,2})", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class ResidueKey:
    chain: str
    residue_number: str
    insertion_code: str
    residue_name: str


@dataclass(frozen=True)
class AtomRecord:
    line: str
    residue: ResidueKey
    atom_name: str
    element: str
    xyz: tuple[float, float, float]

    @property
    def is_heavy(self) -> bool:
        return self.element.upper() not in {"H", "D"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_element(line: str, atom_name: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    stripped = re.sub(r"^[0-9]+", "", atom_name.strip())
    return stripped[:1].upper()


def parse_pdb_atom(line: str) -> AtomRecord:
    if len(line) < 54:
        raise ValueError(f"PDB atom line is too short: {line!r}")
    atom_name = line[12:16].strip()
    residue = ResidueKey(
        chain=line[21].strip(),
        residue_number=line[22:26].strip(),
        insertion_code=line[26].strip(),
        residue_name=line[17:20].strip().upper(),
    )
    try:
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError as exc:
        raise ValueError(f"Invalid PDB coordinates in line: {line!r}") from exc
    return AtomRecord(
        line=line.rstrip("\r\n"),
        residue=residue,
        atom_name=atom_name,
        element=infer_element(line, atom_name),
        xyz=xyz,
    )


def clean_protein_pdb(input_path: Path, output_path: Path, chain: str | None) -> list[AtomRecord]:
    """Keep ATOM records from the first model and normalize alternate locations."""
    if chain is not None and len(chain) > 1:
        raise ValueError("--chain must be a single PDB chain identifier.")

    candidates: dict[tuple[str, str, str, str], tuple[int, int, AtomRecord]] = {}
    saw_model = False
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        for order, raw_line in enumerate(handle):
            record = raw_line[:6].strip().upper()
            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                continue
            if record == "ENDMDL" and saw_model:
                break
            if record != "ATOM":
                continue
            if chain is not None and raw_line[21].strip() != chain:
                continue

            altloc = raw_line[16].strip() if len(raw_line) > 16 else ""
            priority = 0 if not altloc else (1 if altloc == "A" else 2)
            normalized = raw_line.rstrip("\r\n")
            if len(normalized) > 16 and altloc:
                normalized = normalized[:16] + " " + normalized[17:]
            atom = parse_pdb_atom(normalized)
            key = (
                atom.residue.chain,
                atom.residue.residue_number,
                atom.residue.insertion_code,
                atom.atom_name,
            )
            previous = candidates.get(key)
            if previous is None or priority < previous[0]:
                candidates[key] = (priority, order, atom)

    atoms = [entry[2] for entry in sorted(candidates.values(), key=lambda entry: entry[1])]
    if not atoms:
        raise ValueError(f"No protein ATOM records found in {input_path} for chain={chain!r}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        previous_chain: str | None = None
        for atom in atoms:
            if previous_chain is not None and atom.residue.chain != previous_chain:
                handle.write("TER\n")
            handle.write(atom.line + "\n")
            previous_chain = atom.residue.chain
        handle.write("TER\nEND\n")
    return atoms


def resolve_executable(value: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.parent != Path(".") or expanded.is_absolute():
        if not expanded.is_file():
            raise FileNotFoundError(f"fpocket executable not found: {expanded}")
        return expanded.resolve()
    located = shutil.which(value)
    if not located:
        raise FileNotFoundError(
            f"fpocket executable {value!r} was not found on PATH; pass --fpocket-bin."
        )
    return Path(located).resolve()


def detect_fpocket_version(executable: Path) -> tuple[str | None, str]:
    conda_meta = executable.parent.parent / "conda-meta"
    if conda_meta.is_dir():
        for metadata_path in sorted(conda_meta.glob("fpocket-*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("name") == "fpocket" and metadata.get("version"):
                return str(metadata["version"]), f"conda-meta:{metadata_path}"

    outputs = []
    for flag in ("--version", "-v"):
        result = subprocess.run(
            [str(executable), flag], capture_output=True, text=True, check=False
        )
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        outputs.append(combined)
        match = VERSION_PATTERN.search(combined)
        if match:
            return match.group(1), combined
    return None, "\n".join(output for output in outputs if output)


def parse_fpocket_scores(info_path: Path) -> dict[int, float]:
    scores: dict[int, float] = {}
    current_pocket: int | None = None
    with info_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            pocket_match = POCKET_PATTERN.match(line)
            if pocket_match:
                current_pocket = int(pocket_match.group(1))
                continue
            score_match = SCORE_PATTERN.match(line)
            if score_match and current_pocket is not None:
                scores[current_pocket] = float(score_match.group(1))
    if not scores:
        raise RuntimeError(f"No default fpocket Score values could be parsed from {info_path}.")
    return scores


def choose_top_pocket(scores: dict[int, float]) -> tuple[int, float]:
    pocket_number = min(scores, key=lambda number: (-scores[number], number))
    return pocket_number, scores[pocket_number]


def parse_sphere_coordinates(path: Path) -> list[tuple[float, float, float]]:
    coordinates = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[:6].strip().upper() not in {"ATOM", "HETATM"}:
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except (ValueError, IndexError):
                numeric = [float(value) for value in re.findall(FLOAT_PATTERN, line)]
                if len(numeric) < 5:
                    raise ValueError(f"Cannot parse alpha-sphere coordinates from: {line!r}")
                xyz = tuple(numeric[-5:-2])
            coordinates.append(xyz)
    if not coordinates:
        raise RuntimeError(f"No alpha-sphere coordinates found in {path}.")
    return coordinates


def distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def select_residues(
    atoms: list[AtomRecord],
    sphere_coordinates: list[tuple[float, float, float]],
    cutoff: float,
) -> tuple[list[ResidueKey], dict[ResidueKey, float], dict[ResidueKey, int]]:
    if cutoff <= 0:
        raise ValueError("--sphere-cutoff must be positive.")
    minimum_distances: dict[ResidueKey, float] = {}
    heavy_atom_counts: dict[ResidueKey, int] = {}
    for atom in atoms:
        if not atom.is_heavy:
            continue
        heavy_atom_counts[atom.residue] = heavy_atom_counts.get(atom.residue, 0) + 1
        atom_minimum = min(distance(atom.xyz, sphere) for sphere in sphere_coordinates)
        minimum_distances[atom.residue] = min(
            minimum_distances.get(atom.residue, math.inf), atom_minimum
        )

    selected_set = {
        residue for residue, minimum_distance in minimum_distances.items() if minimum_distance <= cutoff
    }
    selected = []
    seen = set()
    for atom in atoms:
        if atom.residue in selected_set and atom.residue not in seen:
            selected.append(atom.residue)
            seen.add(atom.residue)
    if not selected:
        raise RuntimeError(
            f"No protein residue has a heavy atom within {cutoff:g} Angstrom of the "
            "selected pocket alpha spheres."
        )
    return selected, minimum_distances, heavy_atom_counts


def write_pocket_pdb(
    path: Path,
    atoms: list[AtomRecord],
    selected: list[ResidueKey],
    pocket_number: int,
    score: float,
    cutoff: float,
) -> None:
    selected_set = set(selected)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("REMARK 950 GENERATED BY prepare_fpocket_pocket.py\n")
        handle.write("REMARK 950 LIGAND COORDINATES USED: NO\n")
        handle.write(f"REMARK 950 FPOCKET NUMBER: {pocket_number}\n")
        handle.write(f"REMARK 950 FPOCKET SCORE: {score:.8g}\n")
        handle.write(f"REMARK 950 ALPHA-SPHERE HEAVY-ATOM CUTOFF: {cutoff:g} ANGSTROM\n")
        previous_chain: str | None = None
        for atom in atoms:
            if atom.residue not in selected_set:
                continue
            if previous_chain is not None and atom.residue.chain != previous_chain:
                handle.write("TER\n")
            handle.write(atom.line + "\n")
            previous_chain = atom.residue.chain
        handle.write("TER\nEND\n")


def find_single(path: Path, pattern: str, description: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} matching {pattern!r} in {path}, "
            f"found {len(matches)}."
        )
    return matches[0]


def write_residue_table(
    path: Path,
    selected: list[ResidueKey],
    minimum_distances: dict[ResidueKey, float],
    heavy_atom_counts: dict[ResidueKey, int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "chain",
                "residue_number",
                "insertion_code",
                "residue_name",
                "minimum_alpha_sphere_distance_angstrom",
                "heavy_atom_count",
            ]
        )
        for residue in selected:
            writer.writerow(
                [
                    residue.chain,
                    residue.residue_number,
                    residue.insertion_code,
                    residue.residue_name,
                    f"{minimum_distances[residue]:.6f}",
                    heavy_atom_counts[residue],
                ]
            )


def write_score_table(path: Path, scores: dict[int, float], selected: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["pocket_number", "fpocket_score", "selected"])
        for pocket_number in sorted(scores):
            writer.writerow(
                [pocket_number, f"{scores[pocket_number]:.8g}", pocket_number == selected]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="Input PDB structure.")
    parser.add_argument("--protein-id", required=True, help="Stable identifier used in outputs.")
    parser.add_argument("--chain", default=None, help="Optional single PDB chain identifier.")
    parser.add_argument("--fpocket-bin", default="fpocket", help="fpocket executable or path.")
    parser.add_argument(
        "--sphere-cutoff",
        type=float,
        default=6.0,
        help="Heavy-atom-to-alpha-sphere cutoff in Angstrom (default: 6.0).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-fpocket-version",
        default="4.2.3",
        help="Manuscript fpocket version (default: 4.2.3).",
    )
    parser.add_argument(
        "--strict-version",
        action="store_true",
        help="Fail unless detected fpocket version matches the expected version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_pdb = args.pdb.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_pdb.is_file():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    fpocket_bin = resolve_executable(args.fpocket_bin)
    detected_version, version_output = detect_fpocket_version(fpocket_bin)
    version_matches = detected_version == args.expected_fpocket_version
    if args.strict_version and not version_matches:
        raise RuntimeError(
            "fpocket version mismatch: "
            f"expected {args.expected_fpocket_version}, detected {detected_version or 'unknown'}."
        )
    if not version_matches:
        print(
            f"Warning: manuscript fpocket version {args.expected_fpocket_version}; "
            f"detected {detected_version or 'unknown'}. Run recorded as a version mismatch.",
            file=sys.stderr,
        )

    protein_only_path = output_dir / "protein_only.pdb"
    with tempfile.TemporaryDirectory(prefix="epicatanexus_fpocket_") as temp_name:
        temp_dir = Path(temp_name)
        temp_pdb = temp_dir / "protein_only.pdb"
        atoms = clean_protein_pdb(input_pdb, temp_pdb, args.chain)
        shutil.copy2(temp_pdb, protein_only_path)

        command = [str(fpocket_bin), "-f", str(temp_pdb)]
        result = subprocess.run(command, cwd=temp_dir, capture_output=True, text=True, check=False)
        (output_dir / "fpocket.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_dir / "fpocket.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"fpocket failed with exit code {result.returncode}; see fpocket log files."
            )

        fpocket_output = temp_dir / "protein_only_out"
        if not fpocket_output.is_dir():
            raise RuntimeError(f"fpocket did not create expected directory: {fpocket_output}")
        shutil.copytree(fpocket_output, output_dir / "fpocket_raw")

        info_path = find_single(fpocket_output, "*_info.txt", "fpocket info file")
        scores = parse_fpocket_scores(info_path)
        pocket_number, top_score = choose_top_pocket(scores)
        sphere_path = fpocket_output / "pockets" / f"pocket{pocket_number}_vert.pqr"
        if not sphere_path.is_file():
            raise FileNotFoundError(f"Selected alpha-sphere file not found: {sphere_path}")
        sphere_coordinates = parse_sphere_coordinates(sphere_path)
        shutil.copy2(sphere_path, output_dir / "top_pocket_alpha_spheres.pqr")

    selected, minimum_distances, heavy_atom_counts = select_residues(
        atoms, sphere_coordinates, args.sphere_cutoff
    )
    pocket_pdb = output_dir / f"{args.protein_id}_fpocket_pocket.pdb"
    residue_table = output_dir / "pocket_residues.tsv"
    score_table = output_dir / "pocket_scores.tsv"
    write_pocket_pdb(
        pocket_pdb, atoms, selected, pocket_number, top_score, args.sphere_cutoff
    )
    write_residue_table(residue_table, selected, minimum_distances, heavy_atom_counts)
    write_score_table(score_table, scores, pocket_number)

    metadata = {
        "method": "fpocket_highest_score_alpha_sphere_proximity",
        "ligand_coordinates_used": False,
        "input_pdb": str(input_pdb),
        "input_pdb_sha256": sha256_file(input_pdb),
        "protein_id": args.protein_id,
        "chain": args.chain,
        "first_model_only": True,
        "protein_cleaning": "ATOM_records_only",
        "fpocket_executable": str(fpocket_bin),
        "fpocket_version_detected": detected_version,
        "fpocket_version_expected": args.expected_fpocket_version,
        "fpocket_version_matches_manuscript": version_matches,
        "fpocket_version_command_output": version_output,
        "fpocket_parameters": "default",
        "selected_pocket_number": pocket_number,
        "selected_pocket_score": top_score,
        "selection_rule": "highest_default_fpocket_Score",
        "sphere_cutoff_angstrom": args.sphere_cutoff,
        "alpha_sphere_count": len(sphere_coordinates),
        "input_protein_atom_count": len(atoms),
        "selected_residue_count": len(selected),
        "protein_only_pdb": protein_only_path.name,
        "protein_only_pdb_sha256": sha256_file(protein_only_path),
        "pocket_pdb": pocket_pdb.name,
        "pocket_pdb_sha256": sha256_file(pocket_pdb),
        "pocket_residue_table": residue_table.name,
        "pocket_score_table": score_table.name,
        "raw_fpocket_output": "fpocket_raw",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Selected fpocket cavity: pocket {pocket_number} (Score={top_score:.8g})")
    print(f"Selected protein residues: {len(selected)}")
    print(f"Pocket PDB: {pocket_pdb}")
    print(f"Residue table: {residue_table}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
