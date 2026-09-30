"""Compact, human-readable tables for the Exp4 notebook.

Every table shows values as ``mean ± standard error`` in a single cell (e.g. the ORS gap
and its SEM), uses readable column names, and keeps only the columns needed for the
feasibility narrative. All inputs are the cached Exp4 results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hardware_specs import INDICATIVE_RATES, PLATFORMS, assumptions_table, facts_table
from .tasks import CHEMICAL_ACCURACY_HA

SYMBOL = {"iqm": "L", "aqt": "ρ", "aqt_powerlaw": "α", "quandela": "D", "haar_qubit": ""}
NAME = {"iqm": "Euro-Q-Exa (IQM)", "aqt": "Piast-Q (AQT)", "aqt_powerlaw": "Piast-Q (power-law)",
        "quandela": "Lucy (Quandela)", "haar_qubit": "Haar reference (not native)"}
MAIN = ("iqm", "aqt", "quandela", "haar_qubit")


def pm(m, s=None, fmt="{:.2f}") -> str:
    """'mean ± sem' (or just 'mean' when the error is missing / zero-width)."""
    if m is None or not np.isfinite(m):
        return "—"
    if s is None or not np.isfinite(s):
        return fmt.format(m)
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def _level(p, lv):
    return "Haar U(12)" if (p == "quandela" and lv == "haar") else ("Haar U(2⁶)" if p == "haar_qubit" else f"{SYMBOL[p]} = {lv}")


def _rows(S, platforms=MAIN):
    return S[S.platform.isin(platforms)].sort_values(["platform"], key=lambda c: c.map({p: i for i, p in enumerate(platforms)}), kind="stable")


# ── Diagnostics ───────────────────────────────────────────────────────────────
def ors_table(S) -> pd.DataFrame:
    """ORS gap (gate) / ORS_LO gap (photonic) per native complexity level, with its standard error."""
    out = []
    for _, r in _rows(S).iterrows():
        out.append({"platform": NAME[r.platform], "complexity": _level(r.platform, r.level),
                    "metric": "ORS_LO gap" if r.platform == "quandela" else "ORS gap",
                    "gap ± error": pm(r.ors, r.ors_sem, "{:.2f}" if abs(r.ors) < 100 else "{:.0f}")})
    return pd.DataFrame(out)


# ── LiH ───────────────────────────────────────────────────────────────────────
def lih_table(S) -> pd.DataFrame:
    out = []
    for _, r in _rows(S).iterrows():
        out.append({"platform": NAME[r.platform], "complexity": _level(r.platform, r.level),
                    "RMSE (mHa)": pm(1e3 * r.lih_rmse, 1e3 * r.lih_rmse_sem),
                    "test points within 1.6 mHa": f"{100 * r.lih_frac_chem:.0f}%",
                    "worst point (mHa)": pm(1e3 * r.lih_maxerr, None),
                    "ORS / ORS_LO gap": pm(r.ors, r.ors_sem, "{:.2f}" if abs(r.ors) < 100 else "{:.0f}")})
    return pd.DataFrame(out)


def lih_baselines_table(BL, exp3: dict) -> pd.DataFrame:
    rows = [{"model": m, "RMSE (mHa)": f"{1e3 * v:.2f}"} for m, v in zip(BL.model, BL.lih_rmse)]
    rows += [{"model": f"Exp3 {k} (full 8-qubit state)", "RMSE (mHa)": f"{v:.2f}"} for k, v in exp3.items()]
    return pd.DataFrame(rows)


def learning_curve_table(R, BL, representative: dict, sizes, show=(10, 20, 30, 40, 60, 201)) -> pd.DataFrame:
    """Median test RMSE (mHa) vs number of training geometries + minimum n for chemical accuracy."""
    rows = []
    for p, lv in representative.items():
        r = R[(R.platform == p) & (R.level == str(lv))]
        med = {n: float(np.median(r[f"lih_lc_rmse_{n}"])) for n in sizes}
        rows.append({"model": f"{NAME[p]} {'' if p == 'haar_qubit' else _level(p, lv)}".strip(), **med})
    for _, b in BL.iterrows():
        rows.append({"model": b.model, **{n: b[f"lih_lc_rmse_{n}"] for n in sizes}})
    out = []
    for r in rows:
        n_min = next((n for n in sizes if r[n] < CHEMICAL_ACCURACY_HA), None)
        out.append({"model": r["model"], **{f"n = {n}": f"{1e3 * r[n]:.2f}" for n in show},
                    "min n for < 1.6 mHa": n_min if n_min else "not reached"})
    return pd.DataFrame(out)


# ── Energy QRC ────────────────────────────────────────────────────────────────
def qrc_table(S) -> pd.DataFrame:
    out = []
    for _, r in _rows(S).iterrows():
        out.append({"platform": NAME[r.platform], "complexity": _level(r.platform, r.level),
                    "window 2, no reset (executable)": pm(r.qrc_test_r2, r.qrc_test_r2_sem, "{:.3f}"),
                    "window 8, reset": pm(r.get("qrcR_test_r2"), r.get("qrcR_test_r2_sem"), "{:.3f}"),
                    "continuous (idealised)": pm(r.get("qrcC_test_r2"), r.get("qrcC_test_r2_sem"), "{:.3f}")})
    return pd.DataFrame(out)


def energy_baselines_table(BE) -> pd.DataFrame:
    keep = BE[BE.model.str.startswith("persistence") | BE.window.isin([2, 8])]
    return pd.DataFrame({"classical model (same readout)": keep.model, "test R²": keep.test_r2.map("{:.3f}".format)})


# ── Hardware cost ─────────────────────────────────────────────────────────────
def recommended_table(S) -> pd.DataFrame:
    """Cheapest native level within 10% of the best LiH RMSE / within 0.005 of the best energy R²."""
    rows = []
    for p in ("iqm", "aqt", "quandela"):
        c = S[(S.platform == p) & (S.level != "haar")].copy()
        c["cost"] = c["res_mzi_reservoir"] if p == "quandela" else c["res_ops_2q"]
        lih = c[c.lih_rmse <= 1.1 * c.lih_rmse.min()].sort_values("cost").iloc[0]
        qrc = c[c.qrc_test_r2 >= c.qrc_test_r2.max() - 0.005].sort_values("cost").iloc[0]
        if p == "quandela":
            lc = f"{int(lih.res_mzi_reservoir)} MZIs in 2 meshes + 6 data phases, 3 photons"
            qc = f"{int(qrc.resqrc_beam_splitters)} BS, {int(qrc.resqrc_phase_shifters_data)} data phases, 3 photons"
        else:
            lc = f"{lih.res_ops_2q:.0f} two-qubit gates, depth {lih.res_depth:.0f}, f ≈ {lih.res_f_est:.2f}"
            qc = f"{qrc.resqrc_ops_2q:.0f} two-qubit gates, depth {qrc.resqrc_depth:.0f}, f ≈ {qrc.resqrc_f_est:.2f}"
        rows.append({"platform": NAME[p],
                     "LiH: level": _level(p, lih.level), "LiH: circuit": lc, "LiH: RMSE (mHa)": f"{1e3 * lih.lih_rmse:.2f}",
                     "energy: level": _level(p, qrc.level), "energy: circuit per time step": qc,
                     "energy: R²": f"{qrc.qrc_test_r2:.3f}"})
    return pd.DataFrame(rows)


def campaign_table(SB, lc_table, S, cfg, representative, rates=(100, 1e3, 1e4), qrc_shots=1000, n_test=30) -> pd.DataFrame:
    """Total shots for a hardware campaign (shot rates are illustrative, not verified)."""
    rows = []
    settings = int(S.loc[S.platform == "iqm", "resqrc_settings"].iloc[0])
    for p in ("iqm", "aqt", "quandela"):
        d = SB[(SB.platform == p) & np.isfinite(SB.shots)].groupby("shots_per_input_total")["lih_rmse_cv"].mean()
        ok = d[d < CHEMICAL_ACCURACY_HA]
        s_geom = float(ok.index.min()) if len(ok) else np.nan
        row = lc_table[lc_table.model.str.startswith(NAME[p])]
        n_tr = row["min n for < 1.6 mHa"].iloc[0] if len(row) else 201
        n_tr = n_tr if isinstance(n_tr, (int, np.integer)) else 201
        lih_total = (n_tr + n_test) * s_geom
        qrc_total = cfg.energy_N * (1 if p == "quandela" else settings) * qrc_shots
        rows.append({"platform": NAME[p],
                     "LiH: geometries": n_tr + n_test, "LiH: shots / geometry": f"{s_geom:.0e}",
                     "energy: steps × settings × shots": f"{cfg.energy_N} × {1 if p == 'quandela' else settings} × {qrc_shots}",
                     "total shots": f"{lih_total + qrc_total:.1e}",
                     **{f"hours @ {r:g}/s": f"{(lih_total + qrc_total) / r / 3600:.0f}" for r in rates}})
    return pd.DataFrame(rows)


# ── Appendix helpers ──────────────────────────────────────────────────────────
def facts_compact() -> pd.DataFrame:
    f = facts_table()
    return f[["platform", "fact", "kind"]]


def assumptions_compact() -> pd.DataFrame:
    a = assumptions_table()
    return a[["platform", "assumption", "resolved_by"]].rename(columns={"resolved_by": "resolved with QPU access by"})


def vendor_summary(val: dict) -> pd.DataFrame:
    V = pd.DataFrame(val["circuits"])
    g = V.groupby(["platform", "kind"]).agg(circuits=("ok", "size"), passed=("ok", "sum"), swaps=("swaps", "sum")).reset_index()
    g["platform"] = g.platform.map(NAME)
    g["kind"] = g.kind.map({"reservoir": "reservoir circuits", "qrc_window": "full QRC time step (window 2)"})
    return g


# ── Hardware plan (inference only) ────────────────────────────────────────────
def _interp_log(curve: pd.Series, S: float):
    """Interpolate a metric vs shots in log(shots); clip to the simulated range."""
    c = curve[np.isfinite(curve.index)].sort_index()
    x = np.log(c.index.values.astype(float))
    return float(np.interp(np.log(max(S, 1.0)), x, c.values)), S < c.index.min(), S > c.index.max()


def _median_curve(df, col):
    """Metric vs shots: mean over repetitions, then median over reservoir instances."""
    if "instance" not in df:
        return df.groupby("shots")[col].mean()
    return df.groupby(["shots", "instance"])[col].mean().groupby("shots").median()


def inference_table(LI, var) -> pd.DataFrame:
    """LiH inference-only: RMSE (mHa) and R² vs shots per setting (median over reservoir instances)."""
    g = pd.concat({p: pd.DataFrame({"rmse": _median_curve(d, "rmse"), "r2": _median_curve(d, "r2")})
                   for p, d in LI.groupby("platform")}, names=["platform"]).reset_index()
    g["cell"] = [f"{1e3 * a:.1f} mHa (R² {b:.3f})" for a, b in zip(g.rmse, g.r2)]
    t = g.pivot(index="shots", columns="platform", values="cell")
    t.columns = [NAME[c] for c in t.columns]
    t.index = [("exact" if not np.isfinite(i) else f"{int(i):,}") for i in t.index]
    return t.rename_axis("shots per setting / event count").reset_index()


def hardware_plan(LI, QI, minutes: float = 15, n_test: int = 20, T_test: int = 200) -> pd.DataFrame:
    """What fits into ``minutes`` of QPU time per provider, if the whole window goes to one task.

    Budget = indicative rate × time × parallel reservoirs (INDICATIVE_RATES; NOT verified).
    Expected LiH RMSE / energy R² are interpolated from the inference-only simulations.
    """
    rows = []
    for p in ("iqm", "aqt", "quandela"):
        r = INDICATIVE_RATES[p]
        budget = r["per_s"] * 60 * minutes * r["parallel"]
        settings = 1 if p == "quandela" else 3
        S_lih = budget / (n_test * settings)
        lih = LI[LI.platform == p]
        rmse, lo, hi = _interp_log(_median_curve(lih, "rmse"), S_lih)
        r2, _, _ = _interp_log(_median_curve(lih, "r2"), S_lih)
        q = _median_curve(QI[QI.platform == p], "test_r2")
        S_q = budget / (T_test * settings)
        qr2, qlo, _ = _interp_log(q, S_q)
        rows.append({
            "platform": NAME[p],
            "assumed rate": f"{r['per_s']:.3g} {r['unit']}/s" + (f" × {r['parallel']} in parallel" if r["parallel"] > 1 else ""),
            f"budget in {minutes:g} min": f"{budget:.1e}",
            f"LiH ({n_test} test geometries): shots/setting": f"{S_lih:.1e}",
            "LiH: expected RMSE": (">" if lo else "≈") + f" {1e3 * rmse:.1f} mHa (R² {'<' if lo else '≈'} {r2:.2f})",
            f"energy ({T_test} test steps): shots/setting": f"{S_q:.1e}",
            "energy: expected R²": ("< " if qlo else "≈ ") + f"{qr2:.2f}",
            "ORS: circuits at 10⁴ shots": f"{budget / 1e4:.0f}",
        })
    return pd.DataFrame(rows)


# ── Task performance under realistic noise (§6) ──────────────────────────────
def noisy_task_table(NP) -> pd.DataFrame:
    """LiH RMSE / energy R², from noise-free to the noisy emulators (mean ± spread over shot-noise draws)."""
    rows = []
    for _, r in NP.iterrows():
        lih = r.task == "lih"
        fmt = "{:.2f}" if lih else "{:.3f}"
        spread = lambda k: r.get(f"{k}_std") if (r.get(f"{k}_std") or 0) > 1e-9 else None     # deterministic → no ±
        cell = lambda k: pm(r[k], spread(k), fmt) + (" mHa" if lih and np.isfinite(r[k]) else "")
        conf = _level(r.platform, f"{float(r.level):g}") + (f", {int(r.n_layers)} layers" if pd.notna(r.n_layers) and r.n_layers else "")
        rows.append({"platform": NAME[r.platform], "task": "LiH RMSE" if lih else "energy R²", "configuration": conf,
                     "noise-free": cell("noise_free"), "twin prediction": cell("twin"),
                     "emulator: twin readout": cell("emulator_twin"),
                     "emulator: calibrated twin": cell("emulator_calibrated"),
                     "emulator: trained on emulator": cell("emulator")})
    return pd.DataFrame(rows)
