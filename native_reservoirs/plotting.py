"""Figures and tables for the Exp4 EuroHPC feasibility study.

Colour follows the platform (fixed categorical order, reference palette slots 1–3:
IQM blue, AQT orange, Quandela aqua); non-native Haar references are neutral grey and
dashed. Static figures (PNG + PDF) are written to ``figures/exp4_native_*``; tables to
``results/native_hardware/tables`` as CSV, Markdown and LaTeX.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .hardware_specs import PLATFORMS, REPO_ROOT

FIG_DIR = REPO_ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "iqm": dict(color="#2a78d6", marker="o", label="IQM Euro-Q-Exa"),
    "aqt": dict(color="#eb6834", marker="s", label="AQT Piast-Q"),
    "aqt_powerlaw": dict(color="#eb6834", marker="D", label="AQT Piast-Q (power-law)"),
    "quandela": dict(color="#1baf7a", marker="^", label="Quandela Lucy"),
    "haar_qubit": dict(color="#8a8984", marker="*", label="Haar U(2⁶) (not native)"),
}
COMPLEXITY_LABEL = {"iqm": "PRX+CZ cycles L", "aqt": "entangling density ρ",
                    "quandela": "mesh depth D (MZI columns)", "aqt_powerlaw": "power-law exponent α"}
METRICS = {
    "lih_rmse_mha": ("LiH test RMSE (mHa)", True),
    "qrc_test_r2": ("Energy test R² (window protocol)", False),
    "lih_reff": ("R_eff (LiH features)", False),
    "qrc_reff": ("R_eff (QRC features)", False),
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 6, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"exp4_native_{name}.{ext}", bbox_inches="tight")


# ── Aggregation ───────────────────────────────────────────────────────────────
def _level_key(v):
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, 0.0)


def _derived(records: pd.DataFrame) -> pd.DataFrame:
    """Add the LiH RMSE in mHa (the unit used in all tables and text)."""
    return records.assign(lih_rmse_mha=1e3 * records["lih_rmse"]) if "lih_rmse" in records else records


def cell_summary(records: pd.DataFrame, ors: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SEM per (platform, level) of task metrics, R_eff and resources, merged with ORS."""
    records = _derived(records)
    num = records.select_dtypes("number").columns.drop(["instance", "seed"], errors="ignore")
    g = records.groupby(["platform", "level"])
    mean = g[num].mean()
    sem = g[num].sem().add_suffix("_sem")
    out = pd.concat([mean, sem, g.size().rename("n")], axis=1).reset_index()
    out = out.merge(ors[["platform", "level", "metric", "gap", "gap_sem"]].rename(
        columns={"gap": "ors", "gap_sem": "ors_sem", "metric": "ors_metric"}), on=["platform", "level"], how="left")
    out["_k"] = out["level"].map(_level_key)
    return out.sort_values(["platform", "_k"]).drop(columns="_k").reset_index(drop=True)


# ── Figure 1: ORS vs performance ──────────────────────────────────────────────
def fig_ors_vs_performance(summary: pd.DataFrame, records: pd.DataFrame):
    set_style()
    records = _derived(records)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))
    for row, metric in enumerate(("lih_rmse_mha", "qrc_test_r2")):
        ylabel, logy = METRICS[metric]
        for col, (plats, title, xlab) in enumerate((
                (("iqm", "aqt", "haar_qubit"), "Gate-based: original ORS gap", "ORS gap  Λ − Λ_Haar  (K=4, B=5)"),
                (("quandela",), "Photonic: ORS_LO gap (separate metric)", "ORS_LO gap  (Haar-interferometer ref.)"))):
            ax = axes[row, col]
            for p in plats:
                st = STYLE[p]
                r = records[records.platform == p]
                ax.scatter(r["ors_gap"], r[metric], s=10, color=st["color"], alpha=0.25, lw=0)
                s = summary[summary.platform == p]
                ax.errorbar(s["ors"], s[metric], xerr=s["ors_sem"], yerr=s[f"{metric}_sem"],
                            fmt=st["marker"], color=st["color"], ms=6, lw=1, capsize=0,
                            mec="white", mew=0.8, label=st["label"])
                for _, c in s.iterrows():
                    ax.annotate(str(c["level"]), (c["ors"], c[metric]), xytext=(4, 3),
                                textcoords="offset points", fontsize=6.5, color=INK2)
            if col == 1:
                ax.set_xscale("symlog", linthresh=10)
            if logy:
                ax.set_yscale("log")
            if metric == "lih_rmse_mha":
                ax.axhline(1.594, color=INK, lw=0.9, ls=":", label="chemical accuracy" if col == 0 else None)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(title, loc="left", color=INK)
            if len(plats) > 1:
                ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "fig1_ors_vs_performance")
    return fig


# ── Figure 2: R_eff vs performance ────────────────────────────────────────────
def fig_reff_vs_performance(summary: pd.DataFrame, records: pd.DataFrame):
    set_style()
    records = _derived(records)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, (metric, reff) in zip(axes, (("lih_rmse_mha", "lih_reff"), ("qrc_test_r2", "qrc_reff"))):
        for p in ("iqm", "aqt", "quandela", "haar_qubit"):
            st = STYLE[p]
            r = records[records.platform == p]
            ax.scatter(r[reff], r[metric], s=10, color=st["color"], alpha=0.25, lw=0)
            s = summary[summary.platform == p]
            ax.errorbar(s[reff], s[metric], xerr=s[f"{reff}_sem"], yerr=s[f"{metric}_sem"],
                        fmt=st["marker"], color=st["color"], ms=6, lw=1, mec="white", mew=0.8,
                        label=st["label"])
        ax.set_xlabel(METRICS[reff][0])
        ax.set_ylabel(METRICS[metric][0])
        if METRICS[metric][1]:
            ax.set_yscale("log")
            ax.axhline(1.594, color=INK, lw=0.9, ls=":", label="chemical accuracy")
        ax.legend(loc="best")
    axes[0].set_title("LiH QELM", loc="left")
    axes[1].set_title("Energy QRC", loc="left")
    fig.tight_layout()
    save(fig, "fig2_reff_vs_performance")
    return fig


# ── Figure 3: diagnostics / performance vs native complexity ──────────────────
def fig_vs_complexity(summary: pd.DataFrame, task: str = "lih"):
    set_style()
    rows = [("ors", "ORS / ORS_LO gap", False)]
    if task == "lih":
        rows += [("lih_reff", METRICS["lih_reff"][0], False), ("lih_rmse_mha", METRICS["lih_rmse_mha"][0], True)]
    else:
        rows += [("qrc_reff", METRICS["qrc_reff"][0], False), ("qrc_test_r2", METRICS["qrc_test_r2"][0], False)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.6, 6.4), sharex="col")
    haar_gate = summary[summary.platform == "haar_qubit"]
    for c, p in enumerate(("iqm", "aqt", "quandela")):
        s = summary[(summary.platform == p) & (summary.level != "haar")].copy()
        s["x"] = s["level"].astype(float)
        ref = summary[(summary.platform == "quandela") & (summary.level == "haar")] if p == "quandela" else haar_gate
        for r, (col, lab, logy) in enumerate(rows):
            ax = axes[r, c]
            st = STYLE[p]
            ax.errorbar(s["x"], s[col], yerr=s.get(f"{col}_sem"), fmt="-" + st["marker"], color=st["color"],
                        ms=5, lw=1.6, mec="white", mew=0.8, label=st["label"])
            if len(ref) and np.isfinite(ref[col].iloc[0]):
                ax.axhline(ref[col].iloc[0], color="#8a8984", ls="--", lw=1.2,
                           label="Haar U(12) (photonic)" if p == "quandela" else "Haar U(2⁶) (not native)")
            if logy:
                ax.set_yscale("log")
            if col == "lih_rmse_mha":
                ax.axhline(1.594, color=INK, lw=0.9, ls=":", label="chemical accuracy")
                if r == len(rows) - 1 and c == 0:
                    ax.legend(loc="best", fontsize=7)
            if col == "ors" and p == "quandela":
                ax.set_yscale("symlog", linthresh=10)
            if c == 0:
                ax.set_ylabel(lab)
            if r == 0:
                ax.set_title(PLATFORMS[p]["machine"] + f" — {st['label'].split()[0]}", loc="left")
                ax.legend(loc="best", fontsize=7)
        axes[-1, c].set_xlabel(COMPLEXITY_LABEL[p])
        if p in ("iqm", "quandela"):
            axes[-1, c].set_xscale("log", base=2)
    fig.suptitle(("LiH QELM" if task == "lih" else "Energy QRC") + ": diagnostics and performance vs platform-native complexity",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"fig3_{task}_vs_complexity")
    return fig


# ── Hardware / architecture figures ───────────────────────────────────────────
def fig_architectures(iqm_res, aqt_res, ph_res):
    """Crystal-54 map with the chosen patch, AQT entangling graph, photonic mesh schematic."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.2, 1, 1.4]})
    ax = axes[0]
    G = iqm_res.G_hw
    pos = nx.kamada_kawai_layout(G)
    patch = set(iqm_res.layout)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRID, width=1.2)
    pe = [e for e in G.edges() if e[0] in patch and e[1] in patch]
    nx.draw_networkx_edges(G, pos, edgelist=pe, ax=ax, edge_color=STYLE["iqm"]["color"], width=2.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=18, node_color="#c3c2b7")
    nx.draw_networkx_nodes(G, pos, nodelist=list(patch), ax=ax, node_size=50, node_color=STYLE["iqm"]["color"])
    ax.set_title("IQM Crystal-54 map (SDK) + 6-qubit patch", loc="left")
    ax.axis("off")

    ax = axes[1]
    n = aqt_res.n_qubits
    K = nx.complete_graph(n)
    pos = nx.circular_layout(K)
    nx.draw_networkx_edges(K, pos, ax=ax, edge_color=GRID, width=1.0, style="dotted")
    nx.draw_networkx_edges(K, pos, edgelist=aqt_res.graph, ax=ax, edge_color=STYLE["aqt"]["color"], width=2.2)
    nx.draw_networkx_nodes(K, pos, ax=ax, node_size=160, node_color=STYLE["aqt"]["color"])
    nx.draw_networkx_labels(K, pos, ax=ax, font_size=7, font_color="white")
    ax.set_title(f"AQT all-to-all, RXX graph ρ={aqt_res.graph_density:.2f}", loc="left")
    ax.axis("off")

    ax = axes[2]
    m, D = ph_res.n_modes, ph_res.mesh_depth
    x_data = 0.6 + D                                      # U_pre | data phases | U_res
    end = x_data + 1.0 + D
    for k in range(m):
        ax.plot([0, end + 0.4], [k, k], color=INK2, lw=0.8)
    for k in ph_res.input_modes:
        ax.plot(0, k, "o", color=STYLE["quandela"]["color"], ms=5)

    def mesh(x0):
        for d in range(D):
            for k in range(d % 2, m - 1, 2):
                ax.add_patch(plt.Rectangle((x0 + d, k - 0.15), 0.6, 1.3, color="#cfeee2",
                                           ec=STYLE["quandela"]["color"], lw=0.8))
    mesh(0.6)
    for k in range(ph_res.n_data):
        ax.add_patch(plt.Rectangle((x_data + 0.15, k - 0.3), 0.5, 0.6, color="#fbe3d3", ec=STYLE["aqt"]["color"], lw=0.8))
    mesh(x_data + 1.0)
    ax.text(0.6 + D / 2 - 0.2, m - 0.1, "random mesh", ha="center", fontsize=7, color=INK2)
    ax.text(x_data + 0.4, m - 0.1, "data\nphases", ha="center", va="top", fontsize=7, color=INK2)
    ax.text(x_data + 1.0 + D / 2 - 0.2, m - 0.1, "random mesh", ha="center", fontsize=7, color=INK2)
    ax.text(end + 0.5, m / 2, "PNR\ndetectors", fontsize=7, color=INK2, va="center")
    ax.set_xlim(-0.4, end + 1.9)
    ax.set_ylim(-0.8, m + 1.2)
    ax.invert_yaxis()
    ax.set_title(f"Quandela: {ph_res.n_photons} photons in {m} modes (D = {D})", loc="left")
    ax.axis("off")
    fig.tight_layout()
    save(fig, "architectures")
    return fig


def fig_lih_symmetry(lih: dict, W: dict):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    gs = lih["ground_states"]
    amp = np.abs(gs).max(axis=0)
    ax = axes[0]
    hw2 = np.array([bin(i).count("1") == 2 for i in range(gs.shape[1])])
    ax.bar(np.arange(gs.shape[1])[hw2], amp[hw2] + 1e-16, width=2.5, color="#c3c2b7", label="allowed (2 electrons)")
    sup = amp > 1e-6
    ax.bar(np.arange(gs.shape[1])[sup], amp[sup], width=2.5, color=STYLE["iqm"]["color"], label="populated (10)")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel("computational basis index (8 qubits)")
    ax.set_ylabel("max |amplitude| over R")
    ax.set_title("Particle-number + spin symmetry", loc="left")
    ax.legend(loc="lower right")
    ax = axes[1]
    for j, name in enumerate(W["names"]):
        ax.plot(lih["bond_lengths"], W["w6"][:, j], lw=1.6, label=name)
    ax.set_xlabel("bond length R (Å)")
    ax.set_ylabel("configuration weight w_j")
    ax.set_title("6 encoded weights (θ_j = 2 arcsin √w_j)", loc="left")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, "lih_symmetry")
    return fig


def fig_shot_budget(df: pd.DataFrame):
    """LiH test RMSE (mHa) vs measured shots per geometry; dashed: exact expectation values."""
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for p in ("iqm", "aqt", "quandela"):
        st = STYLE[p]
        d = df[(df.platform == p) & np.isfinite(df.shots)].groupby("shots_per_input_total")
        m = d["lih_rmse_cv"].mean() * 1e3
        ax.plot(m.index, m.values, "-" + st["marker"], color=st["color"], ms=5, lw=1.6, label=st["label"])
        ex = df[(df.platform == p) & ~np.isfinite(df.shots)]["lih_rmse"].mean() * 1e3
        ax.axhline(ex, color=st["color"], lw=0.8, ls="--")
    ax.axhline(1.594, color=INK, lw=1.0, ls=":", label="chemical accuracy (1.6 mHa)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured shots per geometry (gate: all 3 settings; photonic: post-selected events)")
    ax.set_ylabel("LiH test RMSE (mHa)   [MSE = RMSE²]")
    ax.set_title("Sampling budget (dashed: exact expectation values)", loc="left")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "shot_budget")
    return fig


def fig_learning_curves(records: pd.DataFrame, baselines: pd.DataFrame, representative: dict, sizes):
    """LiH test RMSE vs number of training geometries (grid, interpolation only)."""
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for p, lv in representative.items():
        r = records[(records.platform == p) & (records.level == str(lv))]
        if r.empty:
            continue
        st = STYLE[p]
        med = [np.median(r[f"lih_lc_rmse_{n}"]) * 1e3 for n in sizes]
        lab = st["label"] + ("" if p == "haar_qubit" else f" ({ {'iqm': 'L', 'aqt': 'ρ', 'quandela': 'D'}[p] }={lv})")
        ax.plot(sizes, med, "-" + st["marker"], color=st["color"], ms=5, lw=1.6, label=lab)
    b = baselines[baselines.model.str.contains("quadratic")]
    if len(b):
        ax.plot(sizes, [b.iloc[0][f"lih_lc_rmse_{n}"] * 1e3 for n in sizes], "--", color=INK2, lw=1.4,
                label="classical ridge, quadratic in w")
    ax.axhline(1.594, color=INK, lw=1.0, ls=":", label="chemical accuracy (1.6 mHa)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training geometries (bond-length grid; test inside the scanned range)")
    ax.set_ylabel("LiH test RMSE (mHa)")
    ax.set_title("How many geometries are needed? (median over instances and grids)", loc="left")
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "lih_learning_curves")
    return fig


def fig_qrc_protocols(summary: pd.DataFrame, baselines: pd.DataFrame):
    """Energy test R² vs native complexity for the three QRC protocols + classical baselines."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), sharey=True)
    protos = (("qrc_test_r2", "-", "window W=2, no reset (executable)"),
              ("qrcR_test_r2", "--", "window W=8, reset between steps"),
              ("qrcC_test_r2", ":", "Exp3 continuous (idealised)"))
    best_ar = baselines[baselines.model.str.startswith("classical")]["test_r2"].max()
    pers = baselines[baselines.model.str.startswith("persistence")]["test_r2"].iloc[0]
    for ax, p in zip(axes, ("iqm", "aqt", "quandela")):
        st = STYLE[p]
        c = summary[(summary.platform == p) & (summary.level != "haar")].copy()
        c["x"] = c["level"].astype(float)
        for col, ls, lab in protos:
            if col in c and c[col].notna().any():
                ax.errorbar(c["x"], c[col], yerr=c.get(f"{col}_sem"), fmt=ls + st["marker"], color=st["color"],
                            ms=5, lw=1.5, mec="white", mew=0.8, label=lab if p != "quandela" else "window W=2 (photonic)")
        h = summary[summary.platform == ("quandela" if p == "quandela" else "haar_qubit")]
        h = h[h.level == "haar"]
        if len(h):
            ax.axhline(h["qrc_test_r2"].iloc[0], color="#8a8984", lw=1.1, ls="-.",
                       label="Haar U(12), W=2" if p == "quandela" else "Haar U(2⁶), W=2 (not native)")
        ax.axhline(best_ar, color=INK2, lw=1.0, ls="--", label=f"best classical AR ({best_ar:.2f})")
        ax.axhline(pers, color=INK2, lw=0.8, ls=":", label=f"persistence ({pers:.2f})")
        ax.set_xlabel(COMPLEXITY_LABEL[p])
        if p in ("iqm", "quandela"):
            ax.set_xscale("log", base=2)
        ax.set_title(PLATFORMS[p]["machine"], loc="left")
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("Energy test R² (ems_3, 15 min)")
    fig.tight_layout()
    save(fig, "qrc_protocols")
    return fig


def fig_qrc_shot_budget(df: pd.DataFrame):
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ls_map = {"reupload": "-", "reset": "--", "photonic": "-"}
    for (p, proto, W), d in df.groupby(["platform", "protocol", "window"]):
        st = STYLE[p]
        g = d[np.isfinite(d.shots)].groupby("shots")["test_r2"].mean()
        lab = f"{st['label']}, " + ("W=%d no reset" % W if proto == "reupload" else "W=%d reset" % W if proto == "reset" else "W=%d" % W)
        ax.plot(g.index, g.values, ls_map[proto] + st["marker"], color=st["color"], ms=5, lw=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("shots per measurement setting per time step (photonic: post-selected events)")
    ax.set_ylabel("Energy test R²")
    ax.set_title("QRC sampling budget", loc="left")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "qrc_shot_budget")
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────
def export_table(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(df.to_markdown(index=False))
    (out_dir / f"{name}.tex").write_text(df.to_latex(index=False, escape=True))


def table_architectures(cfg) -> pd.DataFrame:
    """Table A — native reservoir architectures."""
    rows = [
        {"platform": "Euro-Q-Exa (IQM)", "modality": "superconducting, gate-based",
         "native ops": "PRX(θ,φ), CZ", "topology": "Crystal square lattice (SDK map, 54 q)",
         "reservoir": "L cycles: PRX on all qubits + CZ on one edge-colour class of the patch",
         "encoding": "LiH: PRX(θ_j, π/2); QRC: PRX(π/2, π/2−πu) per step",
         "measurement / features": "Z/X/Y settings; 1-local (LiH) and 1+2-local (QRC) Paulis",
         "complexity": "L ∈ " + str(tuple(cfg.iqm_layers)), "simulated": f"{cfg.n_sites}-qubit patch",
         "hardware capacity": "54 qubits (150 planned)"},
        {"platform": "Piast-Q (AQT)", "modality": "trapped ions, gate-based",
         "native ops": "R(θ,φ), RZ (virtual), RXX(θ≤π/2)", "topology": "all-to-all (20 ions)",
         "reservoir": f"{cfg.aqt_n_layers} layers: R+RZ on all ions + RXX on a random graph of density ρ",
         "encoding": "LiH: R(θ_j, π/2); QRC: R(π/2, π/2−πu) per step",
         "measurement / features": "end-of-circuit Z/X/Y settings; same Pauli features",
         "complexity": "ρ ∈ " + str(tuple(cfg.aqt_densities)), "simulated": f"{cfg.n_sites} ions",
         "hardware capacity": "20 qubits"},
        {"platform": "Lucy (Quandela)", "modality": "photonic, linear optics",
         "native ops": "BS, PS (MZI mesh), single photons, photon counting",
         "topology": "all-to-all universal interferometer (24 modes)",
         "reservoir": "two Haar-dialled MZI meshes of depth D (U_pre, U_res) around the data phases; QRC: U_res after each of W=2 injections",
         "encoding": "6 data phase shifters θ_j ∈ [0, π] between the meshes",
         "measurement / features": "photon counting, post-selected on 3 photons; ⟨n_i⟩, ⟨n_i n_j⟩ (78)",
         "complexity": "D ∈ " + str(tuple(cfg.photonic_depths)) + " + Haar",
         "simulated": f"{cfg.photonic_photons} photons / {cfg.photonic_modes} modes",
         "hardware capacity": "24 modes, 12 photons"},
    ]
    return pd.DataFrame(rows)


def table_resources_results(summary: pd.DataFrame, representative: dict) -> pd.DataFrame:
    """Table B — representative resource counts and task results (one level per platform)."""
    rows = []
    fmt = lambda m, s, f="{:.2e}": (f + " ± " + f).format(m, s) if np.isfinite(s) else f.format(m)
    for p, lv in representative.items():
        c = summary[(summary.platform == p) & (summary.level == str(lv))]
        if c.empty:
            continue
        c = c.iloc[0]
        sym = {"iqm": "L", "aqt": "ρ", "aqt_powerlaw": "α", "quandela": "D"}.get(p)
        base = {"platform": PLATFORMS.get(p, {}).get("machine", "Haar reference"),
                "level": f"{sym}={lv}" if sym else "Haar U(2⁶)"}
        if p == "quandela":
            qrc_c = f"{int(c['resqrc_beam_splitters'])} BS, {int(c['resqrc_phase_shifters_data'])} data phases (W={int(c['resqrc_window'])})"
            res = {"sites": f"{int(c['res_photons'])} photons / {int(c['res_modes'])} modes",
                   "depth (modality-specific)": f"{int(c['res_optical_layers'])} optical layers",
                   "1Q | BS": f"{int(c['res_beam_splitters'])} BS",
                   "2Q | PS": f"{int(c['res_phase_shifters_fixed'] + c['res_phase_shifters_data'])} PS ({int(c['res_phase_shifters_data'])} data)",
                   "routing / density / patterns": f"{int(c['res_patterns_pnr'])} photon patterns"}
        elif p == "haar_qubit":
            qrc_c = "—"
            res = {"sites": "6 qubits", "depth (modality-specific)": "—", "1Q | BS": "—", "2Q | PS": "—",
                   "routing / density / patterns": "not native"}
        else:
            extra = (f"{int(c['res_routing_swaps'])} SWAPs" if p == "iqm" else f"ρ={c['res_graph_density']:.2f}")
            qrc_c = (f"W={int(c['resqrc_window'])}: depth {c['resqrc_depth']:.0f}, {c['resqrc_ops_2q']:.0f} 2Q, "
                     f"{int(c['resqrc_settings'])} settings, f≈{c['resqrc_f_est']:.2f}")
            res = {"sites": f"{int(c['res_qubits'])} qubits", "depth (modality-specific)": f"{c['res_depth']:.0f} (2Q depth {c['res_depth_2q']:.0f})",
                   "1Q | BS": f"{c['res_ops_1q']:.0f}" + (f" ({c['res_ops_1q_virtual']:.0f} virtual)" if c['res_ops_1q_virtual'] else ""),
                   "2Q | PS": f"{c['res_ops_2q']:.0f}", "routing / density / patterns": extra}
        rows.append(base | res | {
            "f_est": f"{c['res_f_est']:.2f}" if "res_f_est" in c and np.isfinite(c.get("res_f_est", np.nan)) else "—",
            "ORS gap (Lucy: ORS_LO)": fmt(c["ors"], c["ors_sem"], "{:.2f}"),
            "R_eff LiH": fmt(c["lih_reff"], c["lih_reff_sem"], "{:.2f}"),
            "LiH RMSE (mHa)": fmt(1e3 * c["lih_rmse"], 1e3 * c["lih_rmse_sem"], "{:.2f}"),
            "R_eff QRC": fmt(c["qrc_reff"], c["qrc_reff_sem"], "{:.1f}"),
            "energy R² (W=2)": fmt(c["qrc_test_r2"], c["qrc_test_r2_sem"], "{:.3f}"),
            "QRC circuit per time step": qrc_c,
        })
    return pd.DataFrame(rows)


# ── Circuit drawings ──────────────────────────────────────────────────────────
def lih_gate_circuit(res, w, angle_map, w_range, measure: bool = True):
    """Full native LiH circuit: weight-encoding PRX/R gates | reservoir | measurement."""
    from qiskit import QuantumCircuit
    enc = res.as_qelm(angle_map=angle_map, w_range=w_range)._encode(w)
    qc = QuantumCircuit(res.n_qubits, name="LiH")
    qc.compose(enc, inplace=True)
    qc.barrier()
    qc.compose(res.circuit, inplace=True)
    if measure:
        qc.measure_all()
    return qc


def draw_gate_circuit(qc, name: str, fold: int = 36, scale: float = 0.6):
    """qiskit ``draw('mpl')``; saved to figures/exp4_native_circuit_<name>.png|pdf."""
    fig = qc.draw("mpl", fold=fold, scale=scale)
    save(fig, f"circuit_{name}")
    return fig


def draw_photonic_circuit(circuit, name: str):
    """Perceval drawing (BS / PS level); saved to figures/exp4_native_circuit_<name>.png."""
    import perceval as pcvl
    FIG_DIR.mkdir(exist_ok=True)
    pcvl.pdisplay_to_file(circuit, str(FIG_DIR / f"exp4_native_circuit_{name}.png"), output_format=pcvl.Format.MPLOT)
    return circuit



def fig_inference_budget(LI, var):
    """LiH inference-only: test RMSE (mHa) vs shots per setting, with precision targets."""
    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for p in ("iqm", "aqt", "quandela"):
        st = STYLE[p]
        g = LI[(LI.platform == p) & np.isfinite(LI.shots)].groupby(["shots", "instance"]).rmse.mean().groupby("shots")
        med, q1, q3 = 1e3 * g.median(), 1e3 * g.quantile(0.25), 1e3 * g.quantile(0.75)
        ax.fill_between(med.index, q1.values, q3.values, color=st["color"], alpha=0.15, lw=0)
        ax.plot(med.index, med.values, "-" + st["marker"], color=st["color"], ms=5, lw=1.6,
                label=st["label"] + (" (3-photon events)" if p == "quandela" else ""))
    sd = 1e3 * np.sqrt(var)
    for r2, ls in ((0.9, "--"), (0.99, "-.")):
        ax.axhline(sd * np.sqrt(1 - r2), color=INK2, lw=1.0, ls=ls, label=f"R² = {r2}")
    ax.axhline(1.594, color=INK, lw=1.0, ls=":", label="chemical accuracy (1.6 mHa)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("shots per measurement setting per test geometry")
    ax.set_ylabel("LiH test RMSE (mHa)")
    ax.set_title("Inference only: median over 5 reservoir instances (band: quartiles)", loc="left")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, "lih_inference_budget")
    return fig
