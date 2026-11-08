'''splicing in the data to test pca on implantations'''

import os, re
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

CLEAN_DIR  = "/lustre/lrspec/ZTF_data/lightcurves/1020/"
CONCAT_DIR = "/lustre/lrspec/ZTF_data/lightcurves/1020_time_concat/"
EFF_BASE   = "/lustre/lrspec/ZTF_data/efficiency_testing/oneperstreak_eff_data/"
IMPL_DIR   = {"pos": EFF_BASE + "full_image_implantations_pos/lightcurves_pos/",
              "neg": EFF_BASE + "full_image_implantations_neg/lightcurves_neg/"}
SPLICE_DIR = {k: EFF_BASE + f"spliced_concat_{k}/" for k in IMPL_DIR}

CLEAN_RE = re.compile(r"^lightcurve_(?P<pre>ztf_\d+_\d+_z[a-z]_c\d+_x)_(?P<chunk>\d{4})_(?P<quad>\d+)_(?P<col>\d+)_(?P<y>\d+)\.csv$")
IMPL_RE  = re.compile(r"^lightcurve_(?P<pre>ztf_\d+_\d+_z[a-z]_c\d+_x)_(?P<chunk>\d{4})\._(?P<q1>\d+)\.npy_(?P<quad>\d+)_(?P<col>\d+)_(?P<y>\d+)\.csv$")

def build_clean_index():
    """(pre, quad) -> [(col, y, col_str, y_str)] using the chunk-0000 files, as in your concat script"""
    idx = defaultdict(list)
    with os.scandir(CLEAN_DIR) as it:
        for e in it:
            m = CLEAN_RE.match(e.name)
            if m and m["chunk"] == "0000":
                idx[(m["pre"], int(m["quad"]))].append((int(m["col"]), int(m["y"]), m["col"], m["y"]))
    return idx

def match_clean(idx, pre, quad, col, y, col_tol=5, y_tol=5):
    cands = [c for c in idx.get((pre, quad), []) if abs(c[0]-col) <= col_tol and abs(c[1]-y) <= y_tol]
    if not cands:
        return None, "no clean match"
    cands.sort(key=lambda c: abs(c[0]-col) + abs(c[1]-y))
    return cands[0], ("ok" if len(cands) == 1 else f"{len(cands)} candidates, used nearest")

def load_chunk(path):
    return pd.read_csv(path)["flux_hap_2"].to_numpy()

def splice_star(pre, quad, col_s, y_s, implants, total_len=3080*28):
    """implants: {chunk_int: implanted partial LC path}. Same as make_big_array, with chunks swapped."""
    parts, changed = [], {}
    for i in range(29):
        lc = load_chunk(f"{CLEAN_DIR}lightcurve_{pre}_{i:04d}_{quad}_{col_s}_{y_s}.csv")
        assert len(lc) == 3080, f"clean chunk {i} has length {len(lc)}"
        if i in implants:
            imp = load_chunk(implants[i])
            assert len(imp) == 3080, f"implanted chunk {i} has length {len(imp)}"
            changed[i] = int(np.sum(np.abs(imp - lc) > 1e-9))   # sanity: how many samples differ from clean
            lc = imp
        parts.append(lc)
    y = int(y_s)
    return np.hstack(parts)[y:total_len + y], changed

def make_spliced_set(kind, col_tol=5, y_tol=5):
    clean_idx = build_clean_index()
    out_dir = SPLICE_DIR[kind]
    os.makedirs(out_dir, exist_ok=True)

    stars, rows = defaultdict(dict), []
    with os.scandir(IMPL_DIR[kind]) as it:
        for e in it:
            m = IMPL_RE.match(e.name)
            if not m:
                continue
            cand, status = match_clean(clean_idx, m["pre"], int(m["quad"]), int(m["col"]), int(m["y"]), col_tol, y_tol)
            row = dict(impl_name=e.name, pre=m["pre"], chunk=int(m["chunk"]), quad=int(m["quad"]),
                       impl_col=int(m["col"]), impl_y=int(m["y"]), status=status,
                       spliced_name=None, clean_y=np.nan)
            if cand is not None:
                key = (m["pre"], m["quad"], cand[2], cand[3])
                if int(m["chunk"]) in stars[key]:
                    row["status"] = "duplicate star/chunk, skipped"
                else:
                    stars[key][int(m["chunk"])] = e.path
                    row["spliced_name"] = f"lightcurve_{m['pre']}_0000_{m['quad']}_{cand[2]}_{cand[3]}_time_adjusted_concat.npy"
                    row["clean_y"] = int(cand[3])
            rows.append(row)
    df_map = pd.DataFrame(rows)

    saved, changed_all = set(), {}
    for (pre, quad, col_s, y_s), impls in tqdm(stars.items()):
        name = f"lightcurve_{pre}_0000_{quad}_{col_s}_{y_s}_time_adjusted_concat.npy"
        try:
            arr, changed = splice_star(pre, quad, col_s, y_s, impls)
            np.save(os.path.join(out_dir, name), arr)
            saved.add(name)
            changed_all.update({(name, c): n for c, n in changed.items()})
        except Exception as ex:
            print("had an issue:", name, repr(ex))
    failed = df_map["spliced_name"].notna() & ~df_map["spliced_name"].isin(saved)
    df_map.loc[failed, ["status", "spliced_name"]] = ["splice failed", None]
    df_map["n_changed"] = [changed_all.get((r.spliced_name, r.chunk), np.nan) for r in df_map.itertuples()]

    # symlink every other star so the PCA group is identical to production
    for fn in os.listdir(CONCAT_DIR):
        dst = os.path.join(out_dir, fn)
        if fn not in saved and not os.path.lexists(dst):
            os.symlink(os.path.join(CONCAT_DIR, fn), dst)

    df_map.to_csv(os.path.join(EFF_BASE, f"spliced_mapping_{kind}.csv"), index=False)
    print(df_map["status"].value_counts())
    return df_map

df_map_pos = make_spliced_set("pos")
df_map_neg = make_spliced_set("neg")