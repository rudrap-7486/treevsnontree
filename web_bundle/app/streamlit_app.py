"""Campus tree mapper: Streamlit dashboard (deliverable 1).

Run locally:   streamlit run app/streamlit_app.py
Environment:   APP_CONFIG=config/experiments.yaml (default) or config/synthetic.yaml for testing
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from treeclass.io import read_cloud, thin_cloud, write_outputs   # noqa: E402
from treeclass.model import canopy_maps                 # noqa: E402
from treeclass.pipeline import classify_cloud           # noqa: E402

# Which labels taught the model. Each version has its own config, results and models folders;
# only versions that already have results are offered.
VERSIONS = ["config/experiments.yaml", "config/manual.yaml", "config/ta_123.yaml"]


def load_cfg(rel: str) -> dict:
    with open(ROOT / rel) as f:
        return yaml.safe_load(f)


def area_list(tiles: list[str]) -> str:
    return ", ".join(t[-2:].lstrip("0") for t in tiles)


def version_label(cfg: dict) -> str:
    return (f"{cfg.get('label_source', 'TA labels')}: train areas {area_list(cfg['data']['train'])}"
            f" → test areas {area_list(cfg['data']['test'])}")


st.set_page_config(page_title="IITR Campus Tree Mapper", page_icon="🌳", layout="wide")

if os.environ.get("APP_CONFIG"):
    available = [os.environ["APP_CONFIG"]]
else:
    available = [v for v in VERSIONS if (ROOT / v).exists()
                 and any((ROOT / load_cfg(v)["results_dir"]).glob("*/metrics.json"))] or [VERSIONS[0]]
st.sidebar.title("🌳 Campus Tree Mapper")
choice = st.sidebar.selectbox("Labels used to train the model", available,
                              format_func=lambda v: version_label(load_cfg(v)))
CONFIG_PATH = ROOT / choice
CFG = load_cfg(choice)
RESULTS = ROOT / CFG["results_dir"]
MODELS = ROOT / CFG["models_dir"]
SYNTHETIC = "synthetic" in CONFIG_PATH.name

ERROR_NAMES = {0: "Correct non-tree", 1: "Correct tree", 2: "False tree", 3: "Missed tree"}
ERROR_COLOURS = {"Correct non-tree": "#b0b0b0", "Correct tree": "#2e7d32",
                 "False tree": "#d32f2f", "Missed tree": "#f9a825"}
CLASS_COLOURS = {"Non-tree": "#b0b0b0", "Tree": "#2e7d32"}

@st.cache_data(show_spinner=False)
def load_pred_las(path: str) -> pd.DataFrame:
    import laspy
    las = laspy.read(path)
    df = pd.DataFrame({"x": np.asarray(las.x), "y": np.asarray(las.y), "z": np.asarray(las.z)})
    for name in ("label", "pred", "p_tree", "error_class", "hag"):
        if name in las.point_format.extra_dimension_names:
            df[name] = np.asarray(las[name])
    return df


def downsample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed) if len(df) > n else df


def scatter3d(df: pd.DataFrame, colour_col: str, colours: dict, title: str) -> go.Figure:
    fig = px.scatter_3d(df, x="x", y="y", z="z", color=colour_col, color_discrete_map=colours,
                        title=title, height=640)
    fig.update_traces(marker=dict(size=1.6))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                                 zaxis_visible=False),
                      legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    return fig


def experiments_with_results() -> list[str]:
    return sorted(p.parent.name for p in RESULTS.glob("*/metrics.json"))


# ---------------------------------------------------------------- sidebar
page = st.sidebar.radio("Go to", ["Overview", "3D viewer", "Accuracy", "Tree map", "TA vs team labels", "Try it"])
if (ROOT / "WEB_BUNDLE.txt").exists():
    st.sidebar.info("Online version: point clouds are random samples (up to 500,000 points) "
                    "of the full results, so maps are approximate.")
st.sidebar.caption("CEC201 term project · Problem 3: geometric classification of tree and "
                   "non-tree objects in urban LiDAR")
if SYNTHETIC:
    st.warning("Running on SYNTHETIC test data. These numbers are not project results.")

exps = experiments_with_results()
if not exps and page not in ("Overview", "TA vs team labels"):
    st.info("No results yet. Run `python scripts/run_experiment.py --exp all` first.")
    st.stop()

# ---------------------------------------------------------------- pages
if page == "Overview":
    st.title("Which points of the campus scan are trees?")
    st.markdown(
        """
This tool labels every point of an IIT Roorkee LiDAR point cloud as **tree** or **non-tree**,
using only the 3D shape of each point's neighbourhood, and reports how accurate that is on
areas the model never saw during training.

**Pipeline**
1. Ground filter (Cloth Simulation Filter) → height of every point above the ground
2. Eigenvalue shape features (linearity, planarity, sphericity, ...) at one or several radii
3. Random Forest classifier trained on hand-labelled tiles
4. Accuracy assessment on held-out tiles: precision, recall, F1, IoU
5. Canopy-cover and canopy-height maps for the whole area
"""
    )
    summary = RESULTS / "summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        best = df.loc[df["f1_tree"].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best experiment", best["experiment"])
        c2.metric("Tree F1 on unseen tiles", f"{best['f1_tree']:.3f}")
        c3.metric("Tree IoU on unseen tiles", f"{best['iou_tree']:.3f}")

elif page == "3D viewer":
    st.title("Ground truth vs prediction")
    c1, c2, c3 = st.columns(3)
    exp = c1.selectbox("Experiment", exps, index=exps.index("E4") if "E4" in exps else 0)
    files = sorted((RESULTS / exp).glob("*_pred.la[sz]"))
    if not files:
        st.info("This experiment has no saved predictions.")
        st.stop()
    f = c2.selectbox("Test tile", files, format_func=lambda p: p.stem.replace("_pred", ""))
    n = c3.slider("Points to draw", 20_000, 300_000, 100_000, step=20_000)
    df = downsample(load_pred_las(str(f)), n)
    df["Ground truth"] = np.where(df["label"] == 1, "Tree", "Non-tree")
    df["Prediction"] = np.where(df["pred"] == 1, "Tree", "Non-tree")
    df["Error"] = df["error_class"].map(ERROR_NAMES)
    view = st.radio("Colour by", ["Prediction", "Ground truth", "Error"], horizontal=True)
    colours = ERROR_COLOURS if view == "Error" else CLASS_COLOURS
    st.plotly_chart(scatter3d(df, view, colours, f"{exp} · {f.stem.replace('_pred', '')}"),
                    use_container_width=True)
    st.caption("Error view: red = called tree but is not (false positive), "
               "yellow = tree that was missed (false negative).")

elif page == "Accuracy":
    st.title("How accurate is it?")
    df = pd.read_csv(RESULTS / "summary.csv")
    main = df[~df["experiment"].str.startswith("R")]
    st.subheader("Experiments on held-out test tiles")
    st.dataframe(main.style.format({c: "{:.3f}" for c in main.columns if main[c].dtype == float}),
                 use_container_width=True, hide_index=True)
    long = main.melt(id_vars="experiment", value_vars=["f1_tree", "iou_tree", "miou"],
                     var_name="metric", value_name="score")
    fig = px.bar(long, x="experiment", y="score", color="metric", barmode="group", height=380,
                 color_discrete_sequence=["#1b5e20", "#66bb6a", "#9e9e9e"],
                 labels={"score": "Score (0-1)", "experiment": "Experiment"})
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    exp = st.selectbox("Details for", exps, index=exps.index("E4") if "E4" in exps else 0)
    c1, c2 = st.columns(2)
    c1.image(str(RESULTS / exp / "confusion_matrix.png"), caption="Confusion matrix (test tiles)")
    c2.image(str(RESULTS / exp / "feature_importance.png"), caption="Random Forest feature importance")
    if (RESULTS / "radius_sensitivity.png").exists():
        st.image(str(RESULTS / "radius_sensitivity.png"), width=520,
                 caption="Why one fixed radius is fragile: baseline score vs neighbourhood radius")
    with st.expander("Raw metrics JSON"):
        st.json(json.loads((RESULTS / exp / "metrics.json").read_text()))

elif page == "Tree map":
    st.title("Canopy cover and canopy height")
    sources = sorted(RESULTS.glob("*/*_pred.la[sz]")) + sorted(RESULTS.glob("campus/*_classified.la[sz]"))
    default = next((i for i, p in enumerate(sources) if p.parent.name in ("campus", "E4")), 0)
    f = st.selectbox("Classified cloud", sources, index=default,
                     format_func=lambda p: f"{p.parent.name}/{p.stem}")
    df = load_pred_las(str(f))
    xyz = df[["x", "y", "z"]].to_numpy()
    hag = df["hag"].to_numpy() if "hag" in df else None
    maps = canopy_maps(xyz, df["pred"].to_numpy(), hag)
    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{len(df):,}")
    c2.metric("Canopy cover (1 m cells)", f"{maps['canopy_cover_total_pct']:.1f}%")
    if hag is not None and np.isfinite(maps["height"]).any():
        c3.metric("Tallest tree point", f"{np.nanmax(maps['height']):.1f} m")
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(maps["cover_pct"], origin="lower", color_continuous_scale="Greens",
                             labels={"color": "Cover %"}, title="Tree cover per 10 m cell (%)"),
                   use_container_width=True)
    if hag is not None:
        b.plotly_chart(px.imshow(maps["height"], origin="lower", color_continuous_scale="Viridis",
                                 labels={"color": "m"}, title="Max tree height per 1 m cell (m)"),
                       use_container_width=True)
    else:
        b.info("This experiment did not compute height above ground (choose E2, E4 or E6).")

elif page == "TA vs team labels":
    st.title("Our hand labels vs the TA's labels")
    st.markdown(
        """
The team labelled areas 1–3 by hand in CloudCompare. Two questions:

1. **How close are our labels to the TA's?** Point-by-point agreement on areas 1–3.
2. **Does a model taught by our labels work as well?** Both models are trained on areas 1–3 with the
   same features and settings, and tested on the same unseen areas 5 and 6 (TA reference labels).
   The only difference is who drew the labels.
"""
    )
    agree_csv = ROOT / "results_manual" / "label_agreement.csv"
    if agree_csv.exists():
        st.subheader("1 · Label agreement on areas 1–3 (TA = reference)")
        ag = pd.read_csv(agree_csv)
        show = ag[["area", "team_tree_share_pct", "ta_tree_share_pct", "precision_tree", "recall_tree",
                   "f1_tree", "iou_tree", "kappa", "ta_points_covered_by_team_pct"]]
        fmt = {c: "{:.3f}" for c in ("precision_tree", "recall_tree", "f1_tree", "iou_tree", "kappa")}
        fmt.update({c: "{:.1f}" for c in ("team_tree_share_pct", "ta_tree_share_pct",
                                          "ta_points_covered_by_team_pct")})
        st.dataframe(show.style.format(fmt),
                     use_container_width=True, hide_index=True)
        st.caption("Precision: of the points we called tree, the share the TA also called tree. "
                   "Recall: of the TA's tree points, the share we also called tree. "
                   "Kappa above 0.8 is usually read as almost perfect agreement.")
        files = sorted((ROOT / "results_manual" / "label_agreement").glob("*_team_vs_TA.la[sz]"))
        if files:
            f = st.selectbox("See where we disagree", files, format_func=lambda p: p.stem.split("_")[0])
            df = downsample(load_pred_las(str(f)), 120_000)
            names = {0: "Both: non-tree", 1: "Both: tree", 2: "Only we said tree", 3: "Only the TA said tree"}
            colours = {"Both: non-tree": "#b0b0b0", "Both: tree": "#2e7d32",
                       "Only we said tree": "#d32f2f", "Only the TA said tree": "#f9a825"}
            df["Agreement"] = df["error_class"].map(names)
            st.plotly_chart(scatter3d(df, "Agreement", colours, f.stem), use_container_width=True)
    else:
        st.info("Run `python scripts/compare_labels.py` to compare the labels.")

    pair = [ROOT / "results_ta123" / "summary.csv", ROOT / "results_manual" / "summary.csv"]
    if all(p.exists() for p in pair):
        st.subheader("2 · Same areas, same method, different labels: tested on areas 5 and 6")
        both = pd.concat([pd.read_csv(pair[0]).assign(labels="TA labels"),
                          pd.read_csv(pair[1]).assign(labels="Team labels")])
        common = sorted(set(both[both["labels"] == "TA labels"]["experiment"])
                        & set(both[both["labels"] == "Team labels"]["experiment"]))
        both = both[both["experiment"].isin(common)]
        st.dataframe(both[["experiment", "labels", "precision_tree", "recall_tree", "f1_tree", "iou_tree",
                           "miou", "overall_accuracy"]]
                     .style.format({c: "{:.3f}" for c in ("precision_tree", "recall_tree", "f1_tree",
                                                          "iou_tree", "miou", "overall_accuracy")}),
                     use_container_width=True, hide_index=True)
        fig = px.bar(both, x="experiment", y="f1_tree", color="labels", barmode="group", height=360,
                     color_discrete_map={"TA labels": "#9e9e9e", "Team labels": "#2e7d32"},
                     labels={"f1_tree": "Tree F1 on areas 5 and 6", "experiment": "Experiment"})
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the team-label experiments (config/manual.yaml) and the control "
                "(config/ta_123.yaml) to fill this comparison.")

    cross = ROOT / "results_manual" / "cross_eval.csv"
    if cross.exists():
        st.subheader("3 · The original TA-trained model checked against our labels")
        cx = pd.read_csv(cross)
        st.dataframe(cx.style.format({c: "{:.3f}" for c in cx.columns if cx[c].dtype == float}),
                     use_container_width=True, hide_index=True)
        st.caption("Rows with seen_in_training = yes are areas the model learned from, so they show agreement, "
                   "not a fair test.")

elif page == "Try it":
    st.title("Classify your own tile")
    models = sorted(MODELS.glob("*.joblib"))
    if not models:
        st.info("No trained model is included in this version.")
        st.stop()
    m = st.selectbox("Model", models, format_func=lambda p: p.stem,
                     index=[p.stem for p in models].index("E4") if any(p.stem == "E4" for p in models) else 0)
    up = st.file_uploader("Upload a .las or .laz tile (up to 1 GB; files over ~200 MB take several minutes)",
                          type=["las", "laz"])
    thin = st.checkbox("Thin to one point per 5 cm first (same as the training data; much faster)", value=True)
    if up is not None and st.button("Classify", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / up.name
            src.write_bytes(up.getbuffer())
            with st.spinner("Ground filter, features, Random Forest..."):
                cloud = read_cloud(src)
                n_original = len(cloud)
                if thin:
                    cloud = thin_cloud(cloud, 0.05)
                pred, p_tree, hag, _ = classify_cloud(cloud, m)
                paths = write_outputs(tmp, Path(up.name).stem, cloud, pred, p_tree, hag)
                files = {key: (path.name, path.read_bytes()) for key, path in paths.items()}
        thinned = f" (thinned from {n_original:,})" if len(pred) != n_original else ""
        st.success(f"{len(pred):,} points classified{thinned} · {pred.mean() * 100:.1f}% tree")
        df = downsample(pd.DataFrame({"x": cloud.xyz[:, 0], "y": cloud.xyz[:, 1], "z": cloud.xyz[:, 2],
                                      "Prediction": np.where(pred == 1, "Tree", "Non-tree")}), 120_000)
        st.plotly_chart(scatter3d(df, "Prediction", CLASS_COLOURS, "Your tile"), use_container_width=True)
        st.markdown("**Download the result** (open in CloudCompare, QGIS or any LiDAR software)")
        labels = {"classified": "All points, labelled (class 5 = tree)",
                  "trees": "Trees only", "nontrees": "Everything except trees"}
        cols = st.columns(len(files))
        for col, (key, (name, data)) in zip(cols, files.items()):
            col.download_button(labels[key], io.BytesIO(data), file_name=name, key=f"dl-{key}")
