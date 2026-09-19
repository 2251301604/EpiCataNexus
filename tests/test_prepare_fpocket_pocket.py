from pathlib import Path

import pytest

from scripts.prepare_fpocket_pocket import (
    AtomRecord,
    ResidueKey,
    choose_top_pocket,
    clean_protein_pdb,
    detect_fpocket_version,
    parse_fpocket_scores,
    parse_sphere_coordinates,
    select_residues,
)


def pdb_line(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    residue_number: int,
    xyz: tuple[float, float, float],
    record: str = "ATOM",
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{serial:>5} {atom:^4} {residue:>3} {chain:1}{residue_number:>4}    "
        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{20.0:>6.2f}"
        f"          {element:>2}\n"
    )


def test_selects_highest_default_fpocket_score(tmp_path: Path):
    info = tmp_path / "protein_info.txt"
    info.write_text(
        "Pocket 1 :\n\tScore : 12.5\n\tDruggability Score : 0.99\n"
        "Pocket 2 :\n\tScore : 22.75\n\tDruggability Score : 0.10\n",
        encoding="utf-8",
    )
    scores = parse_fpocket_scores(info)
    assert scores == {1: 12.5, 2: 22.75}
    assert choose_top_pocket(scores) == (2, 22.75)


def test_detects_fpocket_version_from_conda_metadata(tmp_path: Path):
    executable = tmp_path / "env" / "bin" / "fpocket"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    metadata_dir = tmp_path / "env" / "conda-meta"
    metadata_dir.mkdir()
    metadata_path = metadata_dir / "fpocket-4.2.3-test_0.json"
    metadata_path.write_text(
        '{"name": "fpocket", "version": "4.2.3"}',
        encoding="utf-8",
    )
    version, source = detect_fpocket_version(executable)
    assert version == "4.2.3"
    assert source == f"conda-meta:{metadata_path}"


def test_cleaning_removes_ligand_and_unrequested_chain(tmp_path: Path):
    source = tmp_path / "source.pdb"
    source.write_text(
        pdb_line(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0))
        + pdb_line(2, "CA", "GLY", "B", 2, (1.0, 0.0, 0.0))
        + pdb_line(3, "C1", "LIG", "A", 900, (2.0, 0.0, 0.0), record="HETATM"),
        encoding="utf-8",
    )
    output = tmp_path / "protein_only.pdb"
    atoms = clean_protein_pdb(source, output, chain="A")
    text = output.read_text(encoding="utf-8")
    assert len(atoms) == 1
    assert "ALA A   1" in text
    assert "GLY" not in text
    assert "LIG" not in text
    assert "HETATM" not in text


def test_heavy_atom_sphere_cutoff_selects_complete_residue():
    near = ResidueKey("A", "10", "", "ALA")
    far = ResidueKey("A", "20", "", "GLY")
    atoms = [
        AtomRecord("ATOM near N", near, "N", "N", (0.0, 0.0, 0.0)),
        AtomRecord("ATOM near CA", near, "CA", "C", (2.0, 0.0, 0.0)),
        AtomRecord("ATOM far CA", far, "CA", "C", (8.0, 0.0, 0.0)),
        AtomRecord("ATOM far H", far, "H", "H", (0.1, 0.0, 0.0)),
    ]
    selected, distances, counts = select_residues(atoms, [(0.0, 0.0, 0.0)], cutoff=6.0)
    assert selected == [near]
    assert distances[near] == pytest.approx(0.0)
    assert distances[far] == pytest.approx(8.0)
    assert counts == {near: 2, far: 1}


def test_parse_alpha_sphere_pqr_coordinates(tmp_path: Path):
    sphere_file = tmp_path / "pocket2_vert.pqr"
    sphere_file.write_text(
        "HETATM    1  APOL STP     1      10.000  11.500  -2.250  0.00  3.40\n",
        encoding="utf-8",
    )
    assert parse_sphere_coordinates(sphere_file) == [(10.0, 11.5, -2.25)]
