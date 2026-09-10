import numpy as np
import pandas as pd
from astroquery.gaia import Gaia


def classify_gaia_sources(df, source_id_col='gaia_source_id', chunk_size=5000):
    """
    Classify already-matched Gaia DR3 sources as star / galaxy / quasar / white
    dwarf, using the Discrete Source Classifier (DSC) results in
    gaiadr3.astrophysical_parameters, plus the boolean crossmatch flags in
    gaia_source as a cheap cross-check / fallback.
 
    This assumes you've already run something like query_gaia_for_stars() and
    have a 'gaia_source_id' column in df.
 
    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column of Gaia DR3 source_id values.
    source_id_col : str
        Name of the column holding the source_id.
    chunk_size : int
        Number of source_ids per batch query. ADQL IN(...) clauses get slow /
        hit length limits above a few thousand ids, so we chunk.
 
    Returns
    -------
    pd.DataFrame
        df with new columns:
            classlabel_dsc          - 'star' / 'galaxy' / 'quasar' / 'white_dwarf' / NaN
            classprob_star
            classprob_galaxy
            classprob_quasar
            classprob_white_dwarf
            in_galaxy_candidates    - bool crossmatch flag from gaia_source
            in_qso_candidates       - bool crossmatch flag from gaia_source
            parallax_over_error     - astrometric fallback signal
            pm_over_error           - astrometric fallback signal (combined pmra/pmdec)
            likely_star_astrometric - True if parallax or pm is significant (>3 sigma)
    """
    out = df.copy()
    ids = out[source_id_col].dropna().astype('int64').unique().tolist()
 
    if len(ids) == 0:
        print("No valid gaia_source_id values found, nothing to classify.")
        return out
 
    results = []
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        id_list = ",".join(str(s) for s in chunk)
 
        query = f"""
        SELECT
            g.source_id,
            g.in_galaxy_candidates,
            g.in_qso_candidates,
            g.parallax,
            g.parallax_error,
            g.pmra,
            g.pmra_error,
            g.pmdec,
            g.pmdec_error,
            ap.classprob_dsc_combmod_star,
            ap.classprob_dsc_combmod_galaxy,
            ap.classprob_dsc_combmod_quasar,
            ap.classprob_dsc_combmod_whitedwarf
        FROM gaiadr3.gaia_source AS g
        LEFT JOIN gaiadr3.astrophysical_parameters AS ap
            ON g.source_id = ap.source_id
        WHERE g.source_id IN ({id_list})
        """
        job = Gaia.launch_job_async(query)
        res = job.get_results().to_pandas()
        results.append(res)
        print(f"Classified {min(i + chunk_size, len(ids))}/{len(ids)} sources")
 
    class_df = pd.concat(results, ignore_index=True)
    class_df = class_df.rename(columns={
        'classprob_dsc_combmod_star': 'classprob_star',
        'classprob_dsc_combmod_galaxy': 'classprob_galaxy',
        'classprob_dsc_combmod_quasar': 'classprob_quasar',
        'classprob_dsc_combmod_whitedwarf': 'classprob_white_dwarf',
    })
 
    # astrophysical_parameters doesn't provide a ready-made "best label" column
    # (that only exists in the separate qso_candidates/galaxy_candidates
    # tables), so derive it ourselves: whichever of the 4 DSC-Combmod classes
    # has the highest probability, provided that probability exceeds 0.5
    # (Gaia's own convention for classlabel_dsc). Rows with no DSC coverage
    # (all four probabilities NaN) get classlabel_dsc = None.
    prob_cols = ['classprob_star', 'classprob_galaxy', 'classprob_quasar', 'classprob_white_dwarf']
    label_map = {
        'classprob_star': 'star',
        'classprob_galaxy': 'galaxy',
        'classprob_quasar': 'quasar',
        'classprob_white_dwarf': 'white_dwarf',
    }
    probs = class_df[prob_cols]
    has_any = probs.notna().any(axis=1)
    best_col = probs.idxmax(axis=1)
    best_val = probs.max(axis=1)
    class_df['classlabel_dsc'] = np.where(
        has_any & (best_val > 0.5), best_col.map(label_map), None
    )
 
    # Astrometric fallback signal, for sources DSC doesn't cover (typically
    # faint sources without processed BP/RP spectra).
    class_df['parallax_over_error'] = (
        class_df['parallax'] / class_df['parallax_error']
    )
    pm_snr = np.sqrt(
        (class_df['pmra'] / class_df['pmra_error']) ** 2
        + (class_df['pmdec'] / class_df['pmdec_error']) ** 2
    )
    class_df['pm_over_error'] = pm_snr
    class_df['likely_star_astrometric'] = (
        (class_df['parallax_over_error'].abs() > 3) | (class_df['pm_over_error'] > 3)
    )
 
    out = out.merge(
        class_df,
        left_on=source_id_col,
        right_on='source_id',
        how='left',
        suffixes=('', '_dsc'),
    )
    return out
 
merged_lcs_gaia = pd.read_csv('/lustre/lrspec/ZTF_data/merged_lcs_gaia')
df = classify_gaia_sources(merged_lcs_gaia[~merged_lcs_gaia['lc_filename'].isna()], source_id_col='gaia_source_id', chunk_size=5000)
df.to_csv('/lustre/lrspec/ZTF_data/gaia_class_initial')