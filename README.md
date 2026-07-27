# PFS LAM1D QA

Quality-assurance tool for PFS/Subaru LAM1D (1D redshift pipeline) outputs.
For a set of LAM1D output directories it reads the per-object FITS files,
cross-matches results against a reference spectroscopic-redshift catalog, and
produces summary statistics and diagnostic plots.

## What it does

For each object the script reads from the LAM1D FITS file:

- **TARGET** — coordinates, object IDs
- **CLASSIFICATION** — best-fit class (`GALAXY`, `QSO`, `STAR`)
- **GALAXY / QSO / STAR CANDIDATES** — redshifts, probabilities, emission-line fits

It then:

1. Merges all catid / obj_group directories into a single combined table.
2. Cross-matches with a reference spec-z catalog (via sky-coordinate matching).
3. Computes redshift match rates (`|Δz| ≤ z_limit_gal` for galaxies, `|Δz| ≤ z_limit_qso` for QSOs) and classification match rates.
4. Selects representative *unmatched* objects using KMeans clustering.
5. Plots spectra for those objects with a main panel and four emission-line zoom panels.

## Outputs

| File | Contents |
| --- | --- |
| `{run_tag}_all_lam1d.csv` | All objects from all catid/obj_group directories |
| `{run_tag}_all_lam1d_catalog_matched.csv` | Subset cross-matched with the reference catalog |
| `{run_tag}_summary_stats.txt` | Object counts and redshift/classification match rates |
| `{run_tag}_galaxy_lam1d_vs_catalog_specz_comparison.png` | LAM1D vs catalog spec-z scatter (GALAXY) |
| `{run_tag}_qso_lam1d_vs_catalog_specz_comparison.png` | LAM1D vs catalog spec-z scatter (QSO) |
| `{run_tag}_unmatched_spectra.pdf` | Spectrum + zoom panels for representative unmatched objects |

## Requirements

- The LSST / PFS 2D DRP stack (`lsst.daf.butler`, `pfs.datamodel`)
- `numpy`, `pandas`, `matplotlib`, `astropy`, `scipy`, `scikit-learn`

## Usage

Edit the user-defined parameters at the top of [`1dQA.py`](1dQA.py):

```python
lam1d_dir  = "/path/to/lam1d"          # root directory for LAM1D outputs
run_name   = "run26"                    # run name (e.g. S25A, run26)
rerun      = "20260430"                 # rerun date tag
lam1d_output_dir = "bmn_400_930_output" # LAM1D output directory name
specz_catalog_path = "/path/to/catalog_specz_merged.csv"
matching_threshold_arcsec = 1.0         # sky cross-match radius (arcsec)
z_limit_gal = 0.01                      # max |Δz| for a GALAXY redshift match
z_limit_qso = 0.02                      # max |Δz| for a QSO redshift match
drp2d_dir  = "/path/to/datastore"      # Butler repository for the 2D DRP
```

Then run:

```bash
python 1dQA.py
```

The script validates the Butler repository and collections before processing and
will exit with a clear error message if they are not accessible.

It can also be run interactively in a Jupyter notebook on the **Hilo**, **Mitaka**,
or **Science Platform** servers where the PFS DRP stack and data store are available.
