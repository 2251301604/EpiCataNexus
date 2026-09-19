# Ligand-independent fpocket pocket preparation

`scripts/prepare_fpocket_pocket.py` implements the manuscript pocket-selection rule
for a local PDB structure:

1. retain protein `ATOM` records from the requested chain and first model;
2. remove ligands, waters, ions, and other `HETATM` records;
3. run fpocket with default parameters;
4. select the cavity with the highest default fpocket `Score`;
5. retain every protein residue with at least one heavy atom within 6 Angstrom
   of an alpha sphere in the selected cavity.

No ligand identity, ligand coordinate, catalytic annotation, or kinetic label is used.

## Usage

```bash
python scripts/prepare_fpocket_pocket.py \
  --pdb inputs/3D92.pdb \
  --protein-id 3D92_A \
  --chain A \
  --fpocket-bin /home/changda/miniforge3/envs/fpocket-tools/bin/fpocket \
  --sphere-cutoff 6.0 \
  --expected-fpocket-version 4.2.3 \
  --output-dir outputs/3D92_A_fpocket
```

Add `--strict-version` to reject any fpocket version other than 4.2.3.

The pocket PDB produced by this example is:

```text
outputs/3D92_A_fpocket/3D92_A_fpocket_pocket.pdb
```

Use it for inference with:

```bash
--pocket-pdb outputs/3D92_A_fpocket/3D92_A_fpocket_pocket.pdb
```

The output directory also contains:

- `protein_only.pdb`: exact ligand-independent input sent to fpocket;
- `fpocket_raw/`: complete raw fpocket output;
- `pocket_scores.tsv`: all default fpocket scores and the selected cavity;
- `top_pocket_alpha_spheres.pqr`: alpha spheres for the selected cavity;
- `pocket_residues.tsv`: selected residue identifiers and minimum distances;
- `metadata.json`: versions, parameters, hashes, and provenance;
- `fpocket.stdout.txt` and `fpocket.stderr.txt`: execution logs.

An output directory must be empty before a run, preventing files from separate runs
from being mixed.
