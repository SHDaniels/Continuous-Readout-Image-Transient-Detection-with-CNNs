'''
figure out who's a quasar etc

'''

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import precision_recall as pr
import gc  # garbage collector
import review_findings_functions as rff
import os
import importlib
import implantation_functions as imp
import corner
from astropy.wcs import WCS
import shutil
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table


import time
import requests
import http.client
import socket

TRANSIENT_EXCEPTIONS = (
    requests.exceptions.HTTPError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionResetError,
    socket.timeout,
)

def _launch_job_with_retry(query, upload_resource=None, upload_table_name=None,
                            max_retries=5, base_delay=15):
    for attempt in range(1, max_retries + 1):
        try:
            if upload_resource is not None:
                job = Gaia.launch_job_async(
                    query=query,
                    upload_resource=upload_resource,
                    upload_table_name=upload_table_name,
                )
            else:
                job = Gaia.launch_job_async(query=query)
            return job.get_results().to_pandas()
        except TRANSIENT_EXCEPTIONS as e:
            if attempt == max_retries:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            print(f"    Gaia error (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)
            

def query_and_classify_gaia_sources_batched(
    olddf,
    radius=9,
    chunk_size=2000,
    checkpoint_path=None,
    resume=True,
    classification_id_chunk=500,
):
    """
    Batched Gaia DR3 crossmatch + classification.

    Pipeline
    --------
    1. Upload input coordinates to Gaia.
    2. Spatially crossmatch against gaiadr3.gaia_source only.
    3. Select the best Gaia match locally in pandas:
         - Band == 'g'  -> smallest phot_bp_mean_mag
         - otherwise    -> smallest phot_rp_mean_mag
    4. Query astrophysical_parameters for the selected source_ids.
    5. Query vari_summary separately for the selected source_ids.
    6. Merge everything locally.
    7. Save checkpoint after every input chunk.

    The output columns are exactly OUTPUT_COLUMNS.
    Original row order and original DataFrame index are preserved.
    """

    # ==============================================================
    # Validation
    # ==============================================================

    required_columns = {"star_ra", "star_dec", "Band"}
    missing = required_columns - set(olddf.columns)

    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: "
            f"{sorted(missing)}"
        )

    if radius <= 0:
        raise ValueError("radius must be > 0")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if classification_id_chunk <= 0:
        raise ValueError("classification_id_chunk must be > 0")

    # ==============================================================
    # Preserve original index and create guaranteed unique ID
    # ==============================================================

    original_index = olddf.index.copy()

    df = olddf.copy()

    df["_upload_id"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    # ==============================================================
    # Helper: empty result
    # ==============================================================

    def _empty_result(upload_ids):

        return pd.DataFrame(
            {
                "_upload_id": np.asarray(
                    upload_ids,
                    dtype=np.int64,
                ),
                **{
                    col: [None] * len(upload_ids)
                    for col in OUTPUT_COLUMNS
                },
            }
        )

    # ==============================================================
    # Helper: make ADQL IN clause
    # ==============================================================

    def _source_id_in_clause(source_ids):

        source_ids = np.asarray(
            source_ids,
            dtype=np.int64,
        )

        return ", ".join(
            str(int(x))
            for x in source_ids
        )

    # ==============================================================
    # Helper: query astrophysical_parameters
    # ==============================================================

    def _query_astrophysical_parameters(source_ids):

        source_ids = np.asarray(
            source_ids,
            dtype=np.int64,
        )

        if len(source_ids) == 0:
            return pd.DataFrame(
                columns=[
                    "gaia_source_id",
                    "classprob_star",
                    "classprob_galaxy",
                    "classprob_quasar",
                    "classprob_white_dwarf",
                ]
            )

        pieces = []

        for j in range(
            0,
            len(source_ids),
            classification_id_chunk,
        ):

            ids = source_ids[
                j:j + classification_id_chunk
            ]

            id_list = _source_id_in_clause(ids)

            query = f"""
            SELECT
                source_id,
                classprob_dsc_combmod_star
                    AS classprob_star,
                classprob_dsc_combmod_galaxy
                    AS classprob_galaxy,
                classprob_dsc_combmod_quasar
                    AS classprob_quasar,
                classprob_dsc_combmod_whitedwarf
                    AS classprob_white_dwarf
            FROM gaiadr3.astrophysical_parameters
            WHERE source_id IN ({id_list})
            """

            print(
                f"    astrophysical_parameters: "
                f"{j + 1}-{min(j + classification_id_chunk, len(source_ids))}"
                f"/{len(source_ids)}"
            )

            # job = Gaia.launch_job_async(
            #     query=query
            # )
            try:
                piece = _launch_job_with_retry(query=query)
                if not piece.empty:
                    pieces.append(piece)
            except requests.exceptions.HTTPError as e:
                print(f"  Chunk failed after retries, skipping for now: {e}")
                failed_chunks.append(chunk_ids)
                continue

            # piece = job.get_results().to_pandas()

            # if not piece.empty:
            #     pieces.append(piece)

        if not pieces:
            return pd.DataFrame(
                columns=[
                    "gaia_source_id",
                    "classprob_star",
                    "classprob_galaxy",
                    "classprob_quasar",
                    "classprob_white_dwarf",
                ]
            )

        result = pd.concat(
            pieces,
            ignore_index=True,
        )

        result = result.rename(
            columns={
                "source_id": "gaia_source_id",
            }
        )

        return result[
            [
                "gaia_source_id",
                "classprob_star",
                "classprob_galaxy",
                "classprob_quasar",
                "classprob_white_dwarf",
            ]
        ]

    # ==============================================================
    # Helper: query vari_summary separately
    # ==============================================================

    def _query_vari_summary(source_ids):

        source_ids = np.asarray(
            source_ids,
            dtype=np.int64,
        )

        if len(source_ids) == 0:
            return pd.DataFrame(
                columns=[
                    "gaia_source_id",
                    "in_vari_summary",
                ]
            )

        pieces = []

        for j in range(
            0,
            len(source_ids),
            classification_id_chunk,
        ):

            ids = source_ids[
                j:j + classification_id_chunk
            ]

            id_list = _source_id_in_clause(ids)

            query = f"""
            SELECT DISTINCT
                source_id
            FROM gaiadr3.vari_summary
            WHERE source_id IN ({id_list})
            """

            print(
                f"    vari_summary: "
                f"{j + 1}-{min(j + classification_id_chunk, len(source_ids))}"
                f"/{len(source_ids)}"
            )

            # piece = _launch_job_with_retry(
            #     query=query
            # )

            # piece = job.get_results().to_pandas()

            # if not piece.empty:
            #     pieces.append(piece)
            try:
                pieces = _launch_job_with_retry(query=query)
                if not piece.empty:
                    pieces.append(piece)
            except requests.exceptions.HTTPError as e:
                print(f"  Chunk failed after retries, skipping for now: {e}")
                failed_chunks.append(chunk_ids)
                continue

        # No sources in vari_summary.
        if not pieces:

            return pd.DataFrame(
                {
                    "gaia_source_id": source_ids,
                    "in_vari_summary": False,
                }
            )

        found = pd.concat(
            pieces,
            ignore_index=True,
        )

        found = found.rename(
            columns={
                "source_id": "gaia_source_id",
            }
        )

        found_ids = set(
            found["gaia_source_id"]
            .astype(np.int64)
        )

        return pd.DataFrame(
            {
                "gaia_source_id": source_ids,
                "in_vari_summary": [
                    int(x) in found_ids
                    for x in source_ids
                ],
            }
        )

    # ==============================================================
    # 1. Load checkpoint
    # ==============================================================

    all_results = []
    done_ids = set()

    if (
        checkpoint_path is not None
        and resume
        and os.path.exists(checkpoint_path)
    ):

        print(
            f"Loading checkpoint: {checkpoint_path}"
        )

        prior = pd.read_pickle(
            checkpoint_path
        )

        expected = {
            "_upload_id",
            *OUTPUT_COLUMNS,
        }

        missing_checkpoint = (
            expected - set(prior.columns)
        )

        if missing_checkpoint:
            raise ValueError(
                "Checkpoint is missing columns: "
                f"{sorted(missing_checkpoint)}"
            )

        all_results.append(prior)

        done_ids = set(
            prior["_upload_id"]
            .astype(np.int64)
        )

        print(
            f"Resuming: {len(done_ids)} rows already completed."
        )

    # ==============================================================
    # 2. Valid coordinates
    # ==============================================================

    valid_mask = (
        df["star_ra"].notna()
        & df["star_dec"].notna()
    )

    valid = df.loc[
        valid_mask
    ].copy()

    invalid_ids = (
        df.loc[
            ~valid_mask,
            "_upload_id"
        ]
        .astype(np.int64)
        .tolist()
    )

    todo_ids = [
        int(x)
        for x in valid["_upload_id"]
        if int(x) not in done_ids
    ]

    print(
        f"Total input rows:       {len(df)}"
    )
    print(
        f"Rows with coordinates:  {len(valid)}"
    )
    print(
        f"Rows without coords:    {len(invalid_ids)}"
    )
    print(
        f"Rows already completed: {len(done_ids)}"
    )
    print(
        f"Rows remaining:         {len(todo_ids)}"
    )

    # ==============================================================
    # 3. Process input chunks
    # ==============================================================

    for start in range(
        0,
        len(todo_ids),
        chunk_size,
    ):

        chunk_ids = todo_ids[
            start:start + chunk_size
        ]

        chunk = valid[
            valid["_upload_id"].isin(chunk_ids)
        ].copy()

        print()
        print("=" * 70)
        print(
            f"Processing rows "
            f"{start + 1}-"
            f"{min(start + len(chunk), len(todo_ids))} "
            f"of {len(todo_ids)}"
        )

        # ==========================================================
        # STAGE 1
        # Spatial crossmatch
        # ==========================================================

        upload_table = Table(
            {
                "upload_id":
                    chunk["_upload_id"]
                    .astype("int64")
                    .to_numpy(),

                "ra":
                    chunk["star_ra"]
                    .astype("float64")
                    .to_numpy(),

                "dec":
                    chunk["star_dec"]
                    .astype("float64")
                    .to_numpy(),

                "band":
                    chunk["Band"]
                    .astype(str)
                    .to_numpy(),
            }
        )

        spatial_query = f"""
        SELECT
            u.upload_id,
            u.band,

            g.source_id,
            g.ra,
            g.dec,

            g.phot_g_mean_mag,
            g.bp_rp,
            g.phot_rp_mean_mag,
            g.phot_bp_mean_mag,
            g.phot_variable_flag,

            g.in_galaxy_candidates,
            g.in_qso_candidates

        FROM tap_upload.srctable AS u

        JOIN gaiadr3.gaia_source AS g
            ON 1 = CONTAINS(
                POINT('ICRS', g.ra, g.dec),
                CIRCLE(
                    'ICRS',
                    u.ra,
                    u.dec,
                    {float(radius)}/3600.0
                )
            )
        """

        print(
            "Stage 1: spatial crossmatch..."
        )

        # job = Gaia.launch_job_async(
        #     query=spatial_query,
        #     upload_resource=upload_table,
        #     upload_table_name="srctable",
        # )
        # job = _launch_job_with_retry(
        #         query=spatial_query,
        #     upload_resource=upload_table,
        #     upload_table_name="srctable",
        #     )
        try:
            candidates = _launch_job_with_retry(query=spatial_query,
            upload_resource=upload_table,
            upload_table_name="srctable")
        except requests.exceptions.HTTPError as e:
            print(f"  Chunk failed after retries, skipping for now: {e}")
            failed_chunks.append(chunk_ids)
            continue

        # candidates = (
        #     job.get_results()
        #     .to_pandas()
        # )

        print(
            f"  Gaia returned "
            f"{len(candidates)} candidate matches."
        )

        # ==========================================================
        # No Gaia candidates
        # ==========================================================

        if candidates.empty:

            print(
                "  No Gaia matches for this chunk."
            )

            res = _empty_result(
                chunk_ids
            )

            all_results.append(
                res
            )

            if checkpoint_path is not None:

                checkpoint = (
                    pd.concat(
                        all_results,
                        ignore_index=True,
                    )
                    .sort_values("_upload_id")
                    .drop_duplicates(
                        "_upload_id",
                        keep="last",
                    )
                )

                checkpoint.to_pickle(
                    checkpoint_path
                )

            continue

        # ==========================================================
        # Normalize column names
        # ==========================================================

        candidates.columns = [
            str(c).lower()
            for c in candidates.columns
        ]

        # ==========================================================
        # STAGE 2
        # Select best Gaia match locally
        # ==========================================================

        print(
            "Stage 2: selecting best Gaia match locally..."
        )

        candidates[
            "phot_bp_mean_mag"
        ] = pd.to_numeric(
            candidates[
                "phot_bp_mean_mag"
            ],
            errors="coerce",
        )

        candidates[
            "phot_rp_mean_mag"
        ] = pd.to_numeric(
            candidates[
                "phot_rp_mean_mag"
            ],
            errors="coerce",
        )

        candidates[
            "match_mag"
        ] = np.where(
            candidates["band"]
            .astype(str)
            .str.lower()
            .eq("g"),

            candidates[
                "phot_bp_mean_mag"
            ],

            candidates[
                "phot_rp_mean_mag"
            ],
        )

        # Sort by requested magnitude.
        #
        # source_id provides a deterministic tie breaker.
        candidates = candidates.sort_values(
            [
                "upload_id",
                "match_mag",
                "source_id",
            ],
            ascending=True,
            na_position="last",
        )

        best = (
            candidates
            .drop_duplicates(
                "upload_id",
                keep="first",
            )
            .copy()
        )

        best = best.rename(
            columns={
                "source_id": "gaia_source_id",
            }
        )

        print(
            f"  Selected "
            f"{len(best)} best Gaia matches."
        )

        # ==========================================================
        # STAGE 3A
        # astrophysical_parameters
        # ==========================================================

        selected_source_ids = (
            best["gaia_source_id"]
            .dropna()
            .astype(np.int64)
            .unique()
        )

        print(
            "Stage 3a: retrieving "
            "astrophysical parameters..."
        )

        astrophysical = (
            _query_astrophysical_parameters(
                selected_source_ids
            )
        )

        print(
            f"  Returned classification data "
            f"for {len(astrophysical)} sources."
        )

        # ==========================================================
        # STAGE 3B
        # vari_summary
        # ==========================================================

        print(
            "Stage 3b: retrieving variability summary..."
        )

        variability = (
            _query_vari_summary(
                selected_source_ids
            )
        )

        print(
            f"  Checked variability status "
            f"for {len(variability)} sources."
        )

        # ==========================================================
        # Merge auxiliary information locally
        # ==========================================================

        best = best.merge(
            astrophysical,
            on="gaia_source_id",
            how="left",
        )

        best = best.merge(
            variability,
            on="gaia_source_id",
            how="left",
        )

        # Any source that somehow did not appear in vari_summary
        # is explicitly False.
        best["in_vari_summary"] = (
            best["in_vari_summary"]
            .fillna(False)
            .astype(bool)
        )

        # ==========================================================
        # Construct exact output columns
        # ==========================================================

        res = pd.DataFrame(
            {
                "_upload_id":
                    best["upload_id"],

                "gaia_source_id":
                    best["gaia_source_id"],

                "ztf_g_gaia_bp_rp_mag":
                    best["phot_bp_mean_mag"],

                "ztf_ri_gaia_rp_mag":
                    best["phot_rp_mean_mag"],

                "variability":
                    best["phot_variable_flag"],

                "in_vari_summary":
                    best["in_vari_summary"],

                "in_galaxy_candidates":
                    best["in_galaxy_candidates"],

                "in_qso_candidates":
                    best["in_qso_candidates"],

                "classprob_star":
                    best["classprob_star"],

                "classprob_galaxy":
                    best["classprob_galaxy"],

                "classprob_quasar":
                    best["classprob_quasar"],

                "classprob_white_dwarf":
                    best["classprob_white_dwarf"],
            }
        )

        # ==========================================================
        # Best DSC label
        # ==========================================================

        res["classlabel_dsc"] = res.apply(
            _best_label,
            axis=1,
        )

        # Exact column order.
        res = res[
            [
                "_upload_id",
                *OUTPUT_COLUMNS,
            ]
        ]

        # ==========================================================
        # Add unmatched sources
        # ==========================================================

        matched_ids = set(
            res["_upload_id"]
            .astype(np.int64)
        )

        missing_ids = [
            int(x)
            for x in chunk_ids
            if int(x) not in matched_ids
        ]

        if missing_ids:

            print(
                f"  {len(missing_ids)} sources "
                f"had no Gaia match within {radius}\"."
            )

            res = pd.concat(
                [
                    res,
                    _empty_result(
                        missing_ids
                    ),
                ],
                ignore_index=True,
            )

        # ==========================================================
        # Store chunk
        # ==========================================================

        res = (
            res
            .sort_values("_upload_id")
            .drop_duplicates(
                "_upload_id",
                keep="last",
            )
        )

        all_results.append(
            res
        )

        # ==========================================================
        # Checkpoint
        # ==========================================================

        if checkpoint_path is not None:

            checkpoint = (
                pd.concat(
                    all_results,
                    ignore_index=True,
                )
                .sort_values("_upload_id")
                .drop_duplicates(
                    "_upload_id",
                    keep="last",
                )
            )

            checkpoint.to_pickle(
                checkpoint_path
            )

            print(
                f"  Checkpoint saved to "
                f"{checkpoint_path}"
            )

    # ==============================================================
    # 4. Invalid-coordinate rows
    # ==============================================================

    if invalid_ids:

        all_results.append(
            _empty_result(
                invalid_ids
            )
        )

    # ==============================================================
    # 5. Combine results
    # ==============================================================

    if all_results:

        matches = (
            pd.concat(
                all_results,
                ignore_index=True,
            )
            .sort_values("_upload_id")
            .drop_duplicates(
                "_upload_id",
                keep="last",
            )
        )

    else:

        matches = pd.DataFrame(
            columns=[
                "_upload_id",
                *OUTPUT_COLUMNS,
            ]
        )

    # ==============================================================
    # 6. Merge back onto original DataFrame
    # ==============================================================

    result = df.merge(
        matches,
        on="_upload_id",
        how="left",
        sort=False,
    )

    # Guaranteed original order because _upload_id
    # was generated as np.arange(len(df)).
    result = (
        result
        .sort_values("_upload_id")
        .drop(columns="_upload_id")
    )

    # Restore original index.
    result.index = original_index

    return result

def _best_label(row):
    probs = {
        "star": row.get("classprob_star"),
        "galaxy": row.get("classprob_galaxy"),
        "quasar": row.get("classprob_quasar"),
        "white_dwarf": row.get("classprob_white_dwarf"),
    }

    clean = {}

    for label, value in probs.items():
        if pd.notna(value):
            clean[label] = float(value)

    if not clean:
        return None

    best_label = max(clean, key=clean.get)

    return (
        best_label
        if clean[best_label] > 0.5
        else None
    )


lcs = pd.read_csv('/lustre/lrspec/ZTF_data/all_lcs_no_rf_no_29')
lcs = rff.add_ra_dec(lcs)
lcs = query_and_classify_gaia_sources_batched(lcs, resume=False, chunk_size=500, checkpoint_path='/lustre/lrspec/ZTF_data/gaia_all_partial')
lcs.to_csv('/home/3155/all_lcs_no_rf_no_29_crossmatched')

