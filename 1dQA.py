import pandas as pd
from astropy.io import fits
import os
import glob
import re
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from lsst.daf.butler import Butler
from pfs.datamodel import PfsCoZCandidates

# --- Step 1: User-defined parameters ---
lam1d_dir  = "/lfs_pfs/Subaru/PFS/1d/lam1d"          # root directory for LAM1D outputs
run_name   = "run26"                                   # run name (e.g. S25A, run26)
rerun      = "20260430"                                # rerun date tag
lam1d_output_dir = "brn_400_960_output"                      # exact LAM1D output directory name, such that: {lam1d_dir}/{run_name}/{rerun}/{lam1d_output_dir}
specz_catalog_path = "/data_pde01/work/lizm/packages/catalog_specz_merged.csv"  # reference spec-z catalog
matching_threshold_arcsec = 1.0  # max separation for catalog cross-matching (arcsec)
z_limit_gal = 0.01               # max |z_lam1d - z_catalog| for GALAXY redshift match
z_limit_qso = 0.02               # max |z_lam1d - z_catalog| for QSO redshift match
drp2d_dir   = "/lfs_pfs/Subaru/PFS/data/datastore_20260226"  # Butler repository location for corrsponding 2D data (2D DRP datastore)

# --- Step 2: Derived parameters ---
run_dir     = os.path.join(lam1d_dir, run_name, rerun, lam1d_output_dir)
run_tag     = f"{run_name}_{rerun}_{lam1d_output_dir}"
collections = f"PFS/science/{run_name}/coadd.{rerun}"
combination = f"{lam1d_output_dir.split('_')[0]}_{run_name}"
repo        = drp2d_dir
pdf_filename = f"{run_tag}_unmatched_spectra.pdf"
N_UNMATCHED_SPECTRA = 10
MEDIAN_FILTER_SIZE  = 1

class_config = [
    ("GALAXY", "redshift_gal", z_limit_gal),
    ("QSO",    "redshift_qso", z_limit_qso),
]

# --- Step 3: Validate Butler repo and collections ---
print("Validating Butler repo and collections...")
try:
    _butler = Butler(repo, collections=collections)
    _test = list(_butler.registry.queryCollections(collections))
    if not _test:
        raise ValueError(f"Collections '{collections}' not found in repo '{repo}'")
    print(f"Butler validation passed: repo and collections are accessible.\n")
    del _butler, _test
except Exception as e:
    raise SystemExit(f"\nButler validation FAILED — stopping execution.\nReason: {e}")


# --- Step 4: Endian-safe FITS HDU to DataFrame conversion ---
def fits_to_df(hdu_data):
    def fix_endian(arr):
        if arr.dtype.byteorder == '>':
            return arr.byteswap().view(arr.dtype.newbyteorder('<'))
        return arr
    return pd.DataFrame({name: fix_endian(hdu_data[name]) for name in hdu_data.names})


# --- Step 5: Get all subdirectories ---
all_dirs = sorted(glob.glob(os.path.join(run_dir, "*_*")))
catids = sorted(set(
    m.group(1) for d in all_dirs
    for m in [re.search(r"^(\d+)_\d+$", os.path.basename(d))] if m
))
print(f"Found {len(catids)} catids: {catids}")


# --- Step 6: Process all catids ---
all_final_dfs = []

for catid in catids:
    base_dirs = sorted(
        glob.glob(os.path.join(run_dir, f"{catid}_*")),
        key=lambda x: int(re.search(rf"{catid}_(\d+)", os.path.basename(x)).group(1))
    )
    print(f"\nProcessing catid {catid}, found {len(base_dirs)} directories (in ascending order)")

    for base_dir in base_dirs:
        fits_files = sorted(glob.glob(os.path.join(base_dir, '**', '*.fits'), recursive=True))
        if not fits_files:
            print(f"No FITS files found in {base_dir}")
            continue

        file_path = fits_files[0]
        print(f"  Reading {file_path}")

        with fits.open(file_path) as hdul:
            target_df         = fits_to_df(hdul['TARGET'].data)
            classification_df = fits_to_df(hdul['CLASSIFICATION'].data)
            warnings_df       = fits_to_df(hdul['WARNINGS'].data)
            errors_df         = fits_to_df(hdul['ERRORS'].data)
            galaxy_df         = fits_to_df(hdul['GALAXY_CANDIDATES'].data)
            qso_df            = fits_to_df(hdul['QSO_CANDIDATES'].data)
            star_df           = fits_to_df(hdul['STAR_CANDIDATES'].data)

        galaxy_df = galaxy_df.rename(columns={col: f"{col}_gal" for col in galaxy_df.columns if col != 'targetId'})
        qso_df    = qso_df.rename(columns={col: f"{col}_qso" for col in qso_df.columns if col != 'targetId'})
        star_df   = star_df.rename(columns={col: f"{col}_star" for col in star_df.columns if col != 'targetId'})

        target_cls_df = pd.merge(target_df, classification_df, on='targetId', how='inner')
        target_cls_df = pd.merge(target_cls_df, warnings_df,   on='targetId', how='inner')
        target_cls_df = pd.merge(target_cls_df, errors_df,     on='targetId', how='inner')

        best_galaxy_df = galaxy_df.sort_values('redshiftProba_gal', ascending=False).drop_duplicates('targetId')
        best_qso_df    = qso_df.sort_values('redshiftProba_qso',    ascending=False).drop_duplicates('targetId')
        best_star_df   = star_df.drop_duplicates('targetId')

        final_df = target_cls_df
        final_df = pd.merge(final_df, best_galaxy_df, on='targetId', how='inner')
        final_df = pd.merge(final_df, best_qso_df,    on='targetId', how='inner')
        final_df = pd.merge(final_df, best_star_df,   on='targetId', how='inner')

        coadd_file = os.path.basename(base_dir.rstrip('/'))
        final_df['coadd_file']  = coadd_file
        final_df['catid']       = catid
        final_df['obj_group']   = int(re.search(rf"{catid}_(\d+)", coadd_file).group(1))
        final_df['combination'] = combination

        all_final_dfs.append(final_df)


# --- Step 7: Combine and export ---
if all_final_dfs:
    combined_df = pd.concat(all_final_dfs, ignore_index=True)
    combined_df['obj_count'] = combined_df.groupby('catid').cumcount() + 1
    combined_df.to_csv(f'{run_tag}_all_lam1d.csv', index=False)

    print(f"\nCombined file containing all objects from {run_tag} saved as {run_tag}_all_lam1d.csv")
    total_objects = len(combined_df)
    total_galaxy  = (combined_df['class'].str.upper() == 'GALAXY').sum()
    total_qso     = (combined_df['class'].str.upper() == 'QSO').sum()
    total_star    = (combined_df['class'].str.upper() == 'STAR').sum()
    print("Counts:")
    print("GALAXY:", total_galaxy)
    print("QSO:",    total_qso)
    print("STAR:",   total_star)

    with open(f"{run_tag}_summary_stats.txt", "w") as f:
        f.write(f"{run_tag} statistics summary\n")
        f.write(f"Total number of objects: {total_objects}\n")
        f.write(f"Total number of GALAXY: {total_galaxy}\n")
        f.write(f"Total number of QSO: {total_qso}\n")
        f.write(f"Total number of STAR: {total_star}\n\n")
else:
    print("No data combined, all directories empty or missing FITS files.")


# --- Step 8: Catalog matching, summary, and plotting ---
if all_final_dfs:
    df_lam1d = pd.read_csv(f"{run_tag}_all_lam1d.csv", low_memory=False)
    df_specz = pd.read_csv(specz_catalog_path, low_memory=False)

    coords_lam1d = SkyCoord(ra=df_lam1d["ra"].values * u.deg, dec=df_lam1d["dec"].values * u.deg)
    coords_specz = SkyCoord(ra=df_specz["ra"].values * u.deg, dec=df_specz["dec"].values * u.deg)

    idx, sep2d, _ = coords_lam1d.match_to_catalog_sky(coords_specz)
    mask = sep2d < matching_threshold_arcsec * u.arcsec

    df_lam1d_match = df_lam1d[mask].reset_index(drop=True)
    df_specz_match = df_specz.iloc[idx[mask]].reset_index(drop=True)

    df_matched = pd.concat([df_lam1d_match, df_specz_match], axis=1)
    df_matched.to_csv(f"{run_tag}_all_lam1d_catalog_matched.csv", index=False)
    print(f"Matched {len(df_matched)} objects with redshift catalog, saved to {run_tag}_all_lam1d_catalog_matched.csv")

    summary_lines = []
    for class_name, redshift_col, z_limit in class_config:
        df_class = df_matched[df_matched['class'].str.upper() == class_name]
        n_total  = len(df_class)
        if n_total == 0:
            continue
        df_match   = df_class[(df_class[redshift_col] - df_class["z"]).abs() <= z_limit]
        n_match    = len(df_match)
        pct_match  = (n_match / n_total) * 100
        n_type     = len(df_class[df_class['spectype'].str.upper() == class_name])
        pct_type   = (n_type / n_total) * 100

        summary_lines.append(f"Total {class_name} objects matched with catalog (within {matching_threshold_arcsec} arcsec): {n_total}")
        summary_lines.append(f"{class_name} objects with catalog redshift match (±{z_limit} difference): {n_match} out of {n_total} ({pct_match:.2f}%)")
        summary_lines.append(f"{class_name} objects with catalog classification match: {n_type} of {n_total} ({pct_type:.1f}%)")

    with open(f"{run_tag}_summary_stats.txt", "a") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print(f"Summary of {run_tag} statistics saved to {run_tag}_summary_stats.txt")

    def plot_comparison(df, class_name, redshift_col, z_limit, plot_filename):
        diff            = df[redshift_col] - df["z"]
        matched         = abs(diff) <= z_limit
        df_matched_plot = df[matched]

        n_total   = len(df)
        n_match   = len(df_matched_plot)
        pct_match = (n_match / n_total) * 100 if n_total > 0 else 0
        n_type    = len(df[df['spectype'].str.upper() == class_name])
        pct_type  = (n_type / n_total) * 100 if n_total > 0 else 0

        plt.scatter(df["z"], df[redshift_col], s=2, color="red", alpha=0.6,
                    label=f"{class_name} Total No.={n_total}")
        plt.scatter(df_matched_plot["z"], df_matched_plot[redshift_col], s=2, color="blue", alpha=0.6,
                    label=(f"{class_name} Redshift Match\nNo.={n_match} ({pct_match:.1f}%)\n|\u0394z| ≤ {z_limit}"))

        zmin = 0
        zmax = df["z"].max()
        plt.plot([zmin, zmax], [zmin, zmax], "k--", lw=1)
        plt.plot([zmin, zmax], [zmin + z_limit, zmax + z_limit], "k:", lw=1)
        plt.plot([zmin, zmax], [zmin - z_limit, zmax - z_limit], "k:", lw=1)
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.text(0.05, 0.95,
                 f'{class_name} Classification Match = {pct_type:.1f}%',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 horizontalalignment='left', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel("Catalog spec-z")
        plt.ylabel("LAM1D spec-z")
        plt.title(f"{class_name}: LAM1D vs Catalog Spectroscopic Redshift\n{run_tag}")
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=200)
        plt.clf()

    plot_comparison(df_matched[df_matched['class'].str.upper() == 'GALAXY'], "GALAXY", "redshift_gal", z_limit_gal,
                    f"{run_tag}_galaxy_lam1d_vs_catalog_specz_comparison.png")
    plot_comparison(df_matched[df_matched['class'].str.upper() == 'QSO'], "QSO", "redshift_qso", z_limit_qso,
                    f"{run_tag}_qso_lam1d_vs_catalog_specz_comparison.png")


# --- Step 9: Spectrum plots for unmatched objects ---
if all_final_dfs:

    def plot_summary(df_all, df_selected, class_name, redshift_col, z_limit, pdf):
        df_class = df_all[df_all['class'].str.upper() == class_name]
        if df_class.empty:
            return

        diff           = df_class[redshift_col] - df_class["z"]
        matched        = abs(diff) <= z_limit
        df_matched_cls = df_class[matched]

        n_total   = len(df_class)
        n_match   = len(df_matched_cls)
        pct_match = (n_match / n_total) * 100 if n_total > 0 else 0
        n_type    = len(df_class[df_class['spectype'].str.upper() == class_name])
        pct_type  = (n_type / n_total) * 100 if n_total > 0 else 0

        df_highlight = df_class[df_class['objId'].isin(df_selected['objId'])]

        plt.figure(figsize=(8, 7))
        plt.scatter(df_class["z"], df_class[redshift_col], s=2, color="red", alpha=0.6,
                    label=f"{class_name} Total No.={n_total}")
        plt.scatter(df_matched_cls["z"], df_matched_cls[redshift_col], s=2, color="blue", alpha=0.6,
                    label=f"Redshift Match No.={n_match} ({pct_match:.1f}%)\n|\u0394z| ≤ {z_limit}")
        plt.scatter(df_highlight["z"], df_highlight[redshift_col], s=30, color="lime",
                    edgecolors="black", linewidths=0.5, zorder=5,
                    label=f"Selected for spectra (N={len(df_highlight)})")

        zmin = 0
        zmax = df_class["z"].max()
        plt.plot([zmin, zmax], [zmin, zmax], "k--", lw=1)
        plt.plot([zmin, zmax], [zmin + z_limit, zmax + z_limit], "k:", lw=1)
        plt.plot([zmin, zmax], [zmin - z_limit, zmax - z_limit], "k:", lw=1)
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.text(0.05, 0.95,
                 f'{class_name} Classification Match = {pct_type:.1f}%',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 horizontalalignment='left', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel("Catalog spec-z")
        plt.ylabel("LAM1D spec-z")
        plt.title(f"{class_name}: LAM1D vs Catalog Spectroscopic Redshift\n{run_tag}")
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.tight_layout()
        pdf.savefig(dpi=200, bbox_inches='tight')
        plt.close()

    def plot_spectrum(row, butler, class_name, redshift_col, pdf):
        cat_id        = int(row['catid'])
        obj_group     = int(row['obj_group'])
        objId         = int(row['objId'])
        z_catalog     = row['z']
        z_catalog_err = row['zerr']
        z_lam1d       = row[redshift_col]
        z_lam1d_err   = row[redshift_col.replace("redshift", "redshiftError")]
        proba_col     = redshift_col.replace("redshift", "redshiftProba")
        z_lam1d_proba = row[proba_col]
        diff          = z_lam1d - z_catalog

        print(f"Plotting cat_id={cat_id}, obj_group={obj_group}, objId={objId}")

        pfscoadd  = butler.get("pfsCoadd", combination=combination, cat_id=cat_id,
                               instrument='PFS', obj_group=obj_group)
        pfsobject = pfscoadd[objId]

        base_dir   = os.path.join(run_dir, f"{cat_id}_{obj_group}")
        fits_paths = glob.glob(os.path.join(base_dir, "**", "*.fits"), recursive=True)
        if not fits_paths:
            print(f"  No FITS file found in {base_dir}, skipping.")
            return
        coZCands = PfsCoZCandidates.readFits(fits_paths[0])
        zCand    = coZCands[objId]
        model    = np.array(zCand.get_classified_model(), dtype=np.float64)

        lines = zCand.qso.lines if class_name.upper() == "QSO" else zCand.galaxy.lines
        if lines is not None:
            line_names = lines['lineName'].tolist()
            line_waves = lines['lineWave'].tolist()
            lines_df   = lines.to_pandas() if hasattr(lines, 'to_pandas') else pd.DataFrame({c: lines[c] for c in lines.colnames})
            lines_df['SNR'] = lines_df['lineFlux'] / lines_df['lineFluxError'].replace(0, np.nan)
            top4 = lines_df.nlargest(4, 'SNR')
        else:
            line_names = []
            line_waves = []
            top4 = pd.DataFrame()

        ARM_RANGES = {'b': (380, 650), 'r': (630, 970), 'n': (940, 1260), 'm': (710, 885)}
        arm_str  = lam1d_output_dir.split('_')[0]
        arm_list = list(arm_str)

        pixels_bad  = (pfsobject.mask & pfsobject.flags.get('BAD', 'CR', 'SAT', 'NO_DATA')) != 0
        pixels_good = ~pixels_bad

        x_mask        = (pfsobject.wavelength[pixels_good] >= 400) & (pfsobject.wavelength[pixels_good] <= 1300)
        flux_in_range = pfsobject.flux[pixels_good][x_mask].astype(np.float64)
        p2, p98       = np.nanpercentile(flux_in_range, [2, 98])
        span          = p98 - p2
        ylim_lower    = p2  - 0.05 * span
        ylim_upper    = p98 + 0.15 * span

        fig = plt.figure(figsize=(15, 7))
        gs  = fig.add_gridspec(2, 4, height_ratios=[2.5, 1],
                               top=0.88, bottom=0.08, left=0.05, right=0.98,
                               hspace=0.35, wspace=0.12)
        ax_main = fig.add_subplot(gs[0, :])

        ax_main.plot(pfsobject.wavelength[pixels_good],
                     ndimage.median_filter(pfsobject.flux[pixels_good].astype(np.float64), size=MEDIAN_FILTER_SIZE),
                     linewidth=0.5, color='black', label=f'PFSCoadd (MF={MEDIAN_FILTER_SIZE})', alpha=1.0)
        ax_main.plot(pfsobject.wavelength,
                     ndimage.median_filter(model, size=MEDIAN_FILTER_SIZE),
                     linewidth=0.5, color='red', label=f'LAM1D model (\u0394z={diff:.4f})', alpha=0.7)

        ax_main.set_xlim([400, 1300])
        ax_main.set_ylim([ylim_lower, ylim_upper])
        ax_main.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax_main.minorticks_on()

        x_lo, x_hi = ax_main.get_xlim()
        y_top       = ax_main.get_ylim()[1]
        for lw, ln in zip(line_waves, line_names):
            if not (x_lo <= lw <= x_hi):
                continue
            ax_main.axvline(x=lw, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
            ax_main.text(lw, y_top * 0.95, ln, rotation=90,
                         verticalalignment='top', horizontalalignment='right',
                         fontsize=7, color='red',
                         bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

        ax_main.legend(loc='upper left', fancybox=True, framealpha=0.5)
        ax_main.set_xlabel('Wavelength [nm]')
        ax_main.set_ylabel('Flux [nJy]')
        ax_main.set_title(
            f'{class_name} | objId={objId}, obj_group={obj_group}, cat_id={cat_id}\n'
            f'{run_tag}\n'
            f'z_catalog={z_catalog:.4f} \u00b1 {z_catalog_err:.4f}, '
            f'z_lam1d={z_lam1d:.4f} \u00b1 {z_lam1d_err:.4f}, '
            f'z_proba_{class_name.lower()}={z_lam1d_proba:.4f}  \u0394z={diff:+.4f}')

        for zi in range(4):
            ax = fig.add_subplot(gs[1, zi])
            if zi < len(top4):
                r      = top4.iloc[zi]
                lw_ctr = r['lineWave']
                hw     = np.clip(round(r['lineSigma'] * 6), 3, 40)
                w_lo   = lw_ctr - hw
                w_hi   = lw_ctr + hw
                zmask  = pixels_good & (pfsobject.wavelength >= w_lo) & (pfsobject.wavelength <= w_hi)
                zarm   = (pfsobject.wavelength >= w_lo) & (pfsobject.wavelength <= w_hi)
                if zmask.sum() > 0:
                    ax.plot(pfsobject.wavelength[zmask],
                            ndimage.median_filter(pfsobject.flux[zmask].astype(np.float64), size=MEDIAN_FILTER_SIZE),
                            linewidth=0.8, color='black')
                    ax.plot(pfsobject.wavelength[zarm],
                            ndimage.median_filter(model[zarm], size=MEDIAN_FILTER_SIZE),
                            linewidth=0.8, color='red', alpha=0.7)
                    zp2, _  = np.nanpercentile(pfsobject.flux[zmask].astype(np.float64), [2, 98])
                    z_top   = max(np.nanpercentile(pfsobject.flux[zmask].astype(np.float64), 99.5),
                                  np.nanmax(model[zarm]))
                    zspan   = z_top - zp2
                    ax.set_ylim([zp2 - 0.1 * zspan, z_top + 0.2 * zspan])
                ax.axvline(x=lw_ctr, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.set_xlim([w_lo, w_hi])
                ax.set_title(f"{r['lineName']}  SNR={r['SNR']:.1f}\nEW={r['lineEW']:.1f}\u00b1{r['lineEWError']:.1f}", fontsize=7)
                ax.set_xlabel('Wavelength [nm]', fontsize=7)
                ax.tick_params(labelsize=7)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax.minorticks_on()
                if zi == 0:
                    ax.set_ylabel('Flux [nJy]', fontsize=7)
            else:
                ax.set_visible(False)

        pdf.savefig(dpi=200, bbox_inches='tight')
        plt.close()

    butler = Butler(repo, collections=collections)

    with PdfPages(pdf_filename) as pdf:
        for class_name, redshift_col, z_limit in class_config:
            df_class_all = df_matched[df_matched['class'].str.upper() == class_name]
            df_unmatched = df_class_all[(df_class_all[redshift_col] - df_class_all['z']).abs() > z_limit].copy()

            print(f"\nFound {len(df_unmatched)} unmatched {class_name} objects")

            if df_unmatched.empty:
                print(f"  No unmatched {class_name} objects, skipping.")
                continue

            if len(df_unmatched) <= N_UNMATCHED_SPECTRA:
                df_selected = df_unmatched
            else:
                coords = df_unmatched[['z', redshift_col]].to_numpy()
                kmeans = KMeans(n_clusters=N_UNMATCHED_SPECTRA, random_state=42, n_init=10).fit(coords)
                selected_indices = []
                for centroid in kmeans.cluster_centers_:
                    distances = np.linalg.norm(coords - centroid, axis=1)
                    selected_indices.append(df_unmatched.index[np.argmin(distances)])
                df_selected = df_unmatched.loc[selected_indices]

            plot_summary(df_matched, df_selected, class_name, redshift_col, z_limit, pdf)

            for _, row in df_selected.iterrows():
                plot_spectrum(row, butler, class_name, redshift_col, pdf)

    print(f"Saved spectrum plots to {pdf_filename}")
