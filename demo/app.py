"""TheftLens: electricity theft inspection demo (Streamlit web app).

A simple tool for inspectors, built on the project's tuned Random Forest.
It has three pages:
  1. Sign in  - pick an account: one of the four inspectors or the
                supervisor (demo, no passwords)
  2. Cases    - an inspector sees only their own cases (high- and medium-risk
                cases are dealt out in turn); the supervisor sees all cases,
                team progress and the real label (theft / normal).
                Cases are shown in a mixed order (not ranked by risk), so
                high-, medium- and low-risk cases appear together; tabs and
                filters narrow the list.
  3. One case - usage chart, plain-language reasons for the flag, and a
                3-step review (cause, decision, notes) saved to demo/notes.csv

A flag is a lead for inspection, NOT proof of theft.

Run from the project folder, after `python -m src.prepare_data` and
`python -m src.tuning` (they create the data and the saved models):
    streamlit run demo/app.py

The app only scores the test set with the saved models and their saved
thresholds (chosen on training data). Nothing is trained or tuned here.
The light theme is set in .streamlit/config.toml.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit runs this file from demo/, so add the project folder to the
# import path to be able to use the team's code in src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import features, models  # noqa: E402  (import after the path fix)

MODEL_NAME = "Random Forest"                     # best tuned model on both datasets
NOTES_FILE = PROJECT_ROOT / "demo" / "notes.csv"  # inspector reviews (not on GitHub)
HOURS_PER_DAY = 24
PAGE_SIZE = 10            # cases per page in the list
NOT_REVIEWED = "New"

APP_NAME = "TheftLens"
INSPECTORS = ["Jood", "Sadeem", "Asma", "Hessa"]   # the project team
SUPERVISOR = "Dr. Heyfa"                           # course instructor
SEED = 42                                          # same mixed order every time
CACHE_VERSION = 2   # raise this when the saved case tables change, to rebuild them
DATASET_NAMES = {"sgcc": "China homes (SGCC)", "oedi": "US buildings (OEDI)"}
DOT = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

# Choices in the review form (short, so they fit on buttons)
CAUSES = ["Meter tampering", "Bypass / direct line", "Faulty meter",
          "Empty property", "Data error", "Normal use", "Unsure", "Other"]
DECISIONS = ["Site visit", "Theft confirmed", "Not theft"]

# Columns used to explain a flag (compared with normal training data)
SGCC_COLS = ["mean_use", "max_use", "min_use", "zero_days", "sudden_drops",
             "relative_std", "missing_share"]
OEDI_COLS = ["day_vs_usual", "avg_vs_24h_mean", "avg_rel_std_24h", "zero_hours"]

# Light, roomy look: bigger text, white rounded cards, no Streamlit menu
PAGE_CSS = """
<style>
#MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
html, body, [class*="css"] {font-size: 17px;}
.block-container {padding-top: 1.4rem; max-width: 1100px;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #FFFFFF; border-radius: 14px;}
[data-testid="stMetricValue"] {font-size: 2rem;}
h1 {font-size: 1.8rem !important;}
.step {color: #0F766E; font-weight: 700; font-size: 1.05rem; margin: 0.6rem 0 0.2rem;}
</style>
"""


# ------------------------------------------------------------ loading data

def model_path(dataset):
    """Path of the tuned Random Forest saved by src/tuning.py."""
    return models.MODELS_DIR / f"{dataset}_tuned_{models.slug(MODEL_NAME)}.joblib"


@st.cache_resource
def load_model(dataset):
    """Load the saved model, its threshold and feature list (once per app run)."""
    return joblib.load(model_path(dataset))


def risk_level(score, threshold):
    """High = flagged (at or above the model's cut-off), Medium = at least half of it."""
    return np.select([score >= threshold, score >= threshold / 2],
                     ["High", "Medium"], default="Low")


def make_reference(normal, theft):
    """Median and middle-80% range of normal rows, plus the theft median."""
    return pd.DataFrame({
        "normal_median": normal.median(), "normal_p10": normal.quantile(0.10),
        "normal_p90": normal.quantile(0.90), "theft_median": theft.median(),
    })


def oedi_day_table(path, saved=None):
    """Turn hourly OEDI rows into one row per building per day.

    Each day gets simple descriptive numbers (used to explain a flag) and,
    if a model is given, the average risk score of its 24 hours.
    """
    df = pd.read_csv(path)
    df["day"] = df["hour"] // HOURS_PER_DAY
    hourly = features.oedi_features(df)   # the team's hourly features
    df = pd.concat([df, hourly], axis=1)

    if saved is not None:
        # Score each hour with the same columns the model was trained on
        X, _, _ = models.load_xy("oedi", path.stem, add_features=True)
        df["score"] = saved["model"].predict_proba(X[saved["features"]])[:, 1]
        df["hour_flag"] = df["score"] >= saved["threshold"]

    # A building's "usual day" = its median daily total over the whole year
    daily_total = df.groupby(["recording_id", "day"])["elec_facility_kw"].sum()
    usual = daily_total.groupby(level="recording_id").median()

    agg = {"building_type": ("building_type", "first"),
           "avg_vs_24h_mean": ("elec_vs_mean_24h", "mean"),
           "avg_rel_std_24h": ("elec_rel_std_24h", "mean"),
           "zero_hours": ("elec_facility_kw", lambda s: int((s == 0).sum())),
           "theft": ("theft", "max"),
           "theft_type": ("theft_type", lambda s: s[s != "Normal"].mode().iat[0]
                          if (s != "Normal").any() else "Normal")}
    if saved is not None:
        agg.update(score=("score", "mean"), flagged_hours=("hour_flag", "sum"))
    days = df.groupby(["recording_id", "day"]).agg(**agg)
    days["day_vs_usual"] = daily_total / usual.reindex(days.index, level="recording_id").values
    return days.reset_index(), df


@st.cache_data(persist="disk", show_spinner="Scoring customers (first start only)...")
def load_sgcc(model_mtime, cache_version):
    """Score every SGCC test customer and prepare everything the pages need.

    model_mtime and cache_version are only there so the saved copy is rebuilt
    if the model is retrained or this code changes.
    Returns the case table, the model inputs (for charts) and the reference.
    """
    saved = load_model("sgcc")
    # The same loader and features the team used for the final test results
    X, y, _ = models.load_xy("sgcc", "test", add_features=True)
    ids = pd.read_csv(models.DATA_DIR / "sgcc" / "test.csv", usecols=["CONS_NO"])["CONS_NO"]
    score = saved["model"].predict_proba(X[saved["features"]])[:, 1]

    # Normal / theft TRAINING customers, used to explain a flag
    X_tr, y_tr, _ = models.load_xy("sgcc", "train", add_features=True)
    reference = make_reference(X_tr.loc[y_tr == 0, SGCC_COLS], X_tr.loc[y_tr == 1, SGCC_COLS])

    table = pd.DataFrame({
        "id": ids.values,
        # Long IDs are hard to read: show the first and last 4 characters
        "short_id": ["Customer " + i[:4] + "…" + i[-4:] for i in ids.values],
        "score": score,
        "risk": risk_level(score, saved["threshold"]),
        "true_label": np.where(y.values == 1, "Theft", "Normal"),
    })
    table["reasons"] = [sgcc_reasons(r, reference) for r in X[SGCC_COLS].to_dict("records")]
    return assign(table), X, reference


@st.cache_data(persist="disk", show_spinner="Scoring buildings (first start only)...")
def load_oedi(model_mtime, cache_version):
    """Score every OEDI test building-day (average risk over its 24 hours)."""
    saved = load_model("oedi")
    days, hourly = oedi_day_table(models.DATA_DIR / "oedi" / "test.csv", saved)
    train_days, _ = oedi_day_table(models.DATA_DIR / "oedi" / "train.csv")
    reference = make_reference(train_days.loc[train_days["theft"] == 0, OEDI_COLS],
                               train_days.loc[train_days["theft"] == 1, OEDI_COLS])

    # e.g. "StripMall #62 · day 196" (building type, recording number, day)
    ids = [f"{b} #{r} · day {d + 1}" for b, r, d in
           zip(days["building_type"], days["recording_id"], days["day"])]
    table = pd.DataFrame({
        "id": ids, "short_id": ids,
        "score": days["score"],
        # The hourly cut-off is applied to the day's average score
        "risk": risk_level(days["score"], saved["threshold"]),
        "true_label": np.where(days["theft"] == 1, "Theft (" + days["theft_type"] + ")", "Normal"),
    })
    table["reasons"] = [oedi_reasons(r, reference) for r in days[OEDI_COLS].to_dict("records")]
    return assign(table), (days, hourly[["recording_id", "day", "hour", "elec_facility_kw"]]), reference


def assign(table):
    """Deal high- and medium-risk cases out to the inspectors in turn.

    Highest risk first, so every inspector gets a fair share of the most
    urgent cases. Low-risk cases are not assigned (anyone can browse them).
    The split is the same every time the app starts.
    """
    # "row" = the case's position in the model inputs, kept for the chart
    table = table.assign(row=np.arange(len(table)))
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    to_assign = table["risk"] != "Low"
    order = np.arange(to_assign.sum())
    table["inspector"] = "Not assigned"
    table.loc[to_assign, "inspector"] = np.array(INSPECTORS)[order % len(INSPECTORS)]
    # A fixed random order, so the list can show a mix of all risk levels
    table["mix"] = np.random.default_rng(SEED).permutation(len(table))
    return table


# ------------------------------------------------------------ reasons in words

def compare(value, ref):
    """'high' / 'low' if outside the middle 80% of normal rows, else None."""
    if value > ref["normal_p90"]:
        return "high"
    if value < ref["normal_p10"]:
        return "low"
    return None


def sgcc_reasons(values, reference):
    """Short sentences for every SGCC pattern that is unusual for a normal customer."""
    reasons = []
    for col in SGCC_COLS:
        ref, v = reference.loc[col], float(values[col])
        med, side = ref["normal_median"], compare(v, ref)
        if side is None:
            continue
        if col == "mean_use":
            reasons.append(f"Uses about {v / med:.0f}× more electricity than a typical customer "
                           f"({v:.1f} vs {med:.1f} kWh a day)" if side == "high" and med > 0 else
                           f"Uses less than most customers ({v:.1f} vs {med:.1f} kWh a day)")
        elif col == "max_use" and side == "high":
            reasons.append(f"Very high peak days (up to {v:,.0f} kWh; typical {med:.0f})")
        elif col == "min_use" and side == "high":
            reasons.append(f"Never drops below {v:,.1f} kWh a day (most customers reach 0)")
        elif col == "zero_days" and side == "high":
            reasons.append(f"{v:.0f} days with zero use (typical: {med:.0f})")
        elif col == "sudden_drops":
            reasons.append(f"{v:.0f} sudden drops in use (typical: {med:.0f})" if side == "high"
                           else f"Fewer sudden changes than normal ({v:.0f} vs {med:.0f})")
        elif col == "relative_std":
            reasons.append("Use jumps around much more than normal" if side == "high"
                           else "Use is unusually steady from day to day")
        elif col == "missing_share" and side == "high":
            reasons.append(f"{v:.0%} of days had no meter reading (typical: {med:.0%})")
    return reasons


def oedi_reasons(values, reference):
    """Short sentences for every OEDI day pattern that is unusual for a normal day."""
    reasons = []
    for col in OEDI_COLS:
        ref, v = reference.loc[col], float(values[col])
        side = compare(v, ref)
        if side is None:
            continue
        if col == "day_vs_usual":
            reasons.append(f"Used {v:.0%} of this building's usual daily electricity")
        elif col == "avg_vs_24h_mean":
            reasons.append("Use fell below the building's recent average" if side == "low"
                           else "Use rose above the building's recent average")
        elif col == "avg_rel_std_24h":
            reasons.append("Use was unusually flat through the day" if side == "low"
                           else "Use jumped around more than usual through the day")
        elif col == "zero_hours" and side == "high":
            reasons.append(f"{v:.0f} hours with zero use")
    return reasons


# ------------------------------------------------------------ inspector notes

def load_notes():
    """Read saved reviews (one row per save, oldest first)."""
    if not NOTES_FILE.exists():
        return pd.DataFrame(columns=["dataset", "id", "cause", "decision",
                                     "observations", "inspector", "saved_at"])
    return pd.read_csv(NOTES_FILE, dtype=str).fillna("")


def save_note(row):
    """Append one review to demo/notes.csv (keeps the full history)."""
    notes = pd.concat([load_notes(), pd.DataFrame([row])], ignore_index=True)
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    notes.to_csv(NOTES_FILE, index=False)


# ------------------------------------------------------------ page switching
# The app remembers who is signed in and which page is open in
# st.session_state. Buttons call these small functions to switch pages.

state = st.session_state
state.setdefault("user", None)        # None = sign-in page
state.setdefault("page", "list")      # "list" or "case"
state.setdefault("dataset", "sgcc")
state.setdefault("tab", None)
state.setdefault("page_no", 0)        # which page of the case list
state.setdefault("current", None)     # id of the open case
state.setdefault("queue", [])         # the list the case was opened from


def sign_in(name):
    state.update(user=name, page="list", tab=None, page_no=0)


def sign_out():
    state.update(user=None, page="list")


def open_case(case_id, queue):
    state.update(current=case_id, queue=queue, page="case")


def back_to_list():
    state.page = "list"


def turn_page(step):
    state.page_no += step


# ------------------------------------------------------------ the pages

def page_sign_in():
    """Page 1: choose an account."""
    st.title(f"🔍 {APP_NAME}")
    st.caption("Spotting possible electricity theft in smart-meter data · CS316 project demo. "
               "Flags are leads to check, not proof of theft.")
    st.subheader("Who is using the app?")
    cols = st.columns(5)
    for col, name in zip(cols, INSPECTORS + [SUPERVISOR]):
        # Fixed height, so all five cards line up even if a caption wraps
        with col.container(border=True, height=250):
            st.markdown(f"### {'👤' if name != SUPERVISOR else '🧭'}")
            st.markdown(f"**{name}**")
            st.caption("Inspector · checks own cases" if name != SUPERVISOR
                       else "Supervisor · sees all cases and the team")
            st.button("Sign in", key=f"in_{name}", on_click=sign_in, args=(name,),
                      type="primary", width="stretch")


def header(table, notes):
    """Top bar: title, dataset switch, who is signed in, sign out."""
    left, mid, right = st.columns([3, 3, 1.3], vertical_alignment="center")
    left.markdown(f"### 🔍 {APP_NAME}")
    choice = mid.segmented_control("Data", list(DATASET_NAMES), format_func=DATASET_NAMES.get,
                                   default=state.dataset, label_visibility="collapsed")
    if choice and choice != state.dataset:          # switched dataset
        state.update(dataset=choice, page="list", tab=None, page_no=0)
        st.rerun()
    right.button(f"Sign out ({state.user})", on_click=sign_out, width="stretch")


def page_list(table, notes):
    """Page 2: the case list (an inspector's own cases, or all for the supervisor)."""
    supervisor = state.user == SUPERVISOR
    reviewed = table["status"] != NOT_REVIEWED
    mine = table["inspector"] == state.user

    if supervisor:
        st.markdown("#### Team progress")
        cols = st.columns(len(INSPECTORS))
        for col, name in zip(cols, INSPECTORS):
            theirs = table["inspector"] == name
            done = int((theirs & reviewed).sum())
            with col.container(border=True):
                st.markdown(f"**{name}**")
                st.progress(done / max(int(theirs.sum()), 1))
                st.caption(f"{done:,} of {int(theirs.sum()):,} reviewed · "
                           f"{int((theirs & ~reviewed & table['risk'].eq('High')).sum()):,} high risk left")
        tabs = ["All", "High risk", "Medium risk", "Low risk", "Reviewed"]
    else:
        to_check = int((mine & ~reviewed & table["risk"].eq("High")).sum())
        st.markdown(f"#### Hello, {state.user} 👋")
        st.markdown(f"You have **{to_check:,} high-risk cases** to check.")
        tabs = ["To check", "Medium risk", "Low risk", "Reviewed by me"]

    # Tabs and filters (the chosen tab is remembered between pages)
    tab = st.segmented_control("Show", tabs, default=state.tab if state.tab in tabs else tabs[0],
                               label_visibility="collapsed") or tabs[0]
    if tab != state.tab:
        state.update(tab=tab, page_no=0)
    f1, f2, f3 = st.columns([3, 2, 2]) if supervisor else (st.container(), None, None)
    search = f1.text_input("Search", placeholder="🔍 Search case ID", label_visibility="collapsed")

    # Which cases this tab shows
    risk = {"To check": "High", "High risk": "High", "Medium risk": "Medium", "Low risk": "Low"}
    if tab == "Reviewed":
        view = table[reviewed]
    elif tab == "Reviewed by me":
        view = table[table["reviewed_by"] == state.user]
    elif tab == "All":
        view = table
    else:
        view = table[table["risk"].eq(risk[tab]) & ~reviewed]
        if not supervisor and tab != "Low risk":
            view = view[view["inspector"] == state.user]   # only my cases
    if supervisor:
        label = f2.selectbox("Real label", ["Any label", "Theft", "Normal"],
                             label_visibility="collapsed")
        who = f3.selectbox("Inspector", ["Any inspector"] + INSPECTORS + ["Not assigned"],
                           label_visibility="collapsed")
        if label != "Any label":
            view = view[view["true_label"].str.startswith(label)]
        if who != "Any inspector":
            view = view[view["inspector"] == who]
    if search:
        view = view[view["id"].str.contains(search, case=False, regex=False)]
    # Always a mixed order (not ranked by risk), so different cases appear;
    # use the tabs and filters above to narrow the list
    view = view.sort_values("mix")

    # One page of cases at a time, so the screen is not overwhelming
    pages = max(1, -(-len(view) // PAGE_SIZE))
    state.page_no = min(state.page_no, pages - 1)
    start = state.page_no * PAGE_SIZE
    queue = list(view["id"])
    if view.empty:
        st.info("No cases here. Try another tab.")
    for _, case in view.iloc[start:start + PAGE_SIZE].iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.2, 5, 1.3], vertical_alignment="center")
            c1.markdown(f"#### {DOT[case['risk']]} {case['score']:.0%}")
            main = case["reasons"][0] if case["reasons"] else "No single pattern stands out"
            extra = []
            if case["status"] != NOT_REVIEWED:
                extra.append(f"✓ {case['status']}")
            if supervisor:
                extra += [case["inspector"], f"real: {case['true_label']}"]
            c2.markdown(f"**{case['short_id']}**  \n{main}"
                        + (f"  \n:gray[{' · '.join(extra)}]" if extra else ""))
            c3.button("Open →", key=f"open_{case['id']}", on_click=open_case,
                      args=(case["id"], queue), width="stretch")

    # Page buttons
    p1, p2, p3 = st.columns([1, 2, 1], vertical_alignment="center")
    p1.button("← Previous", on_click=turn_page, args=(-1,), disabled=state.page_no == 0,
              width="stretch")
    p2.markdown(f"<div style='text-align:center'>Page {state.page_no + 1} of {pages} · "
                f"{len(view):,} cases</div>", unsafe_allow_html=True)
    p3.button("Next →", on_click=turn_page, args=(1,), disabled=state.page_no >= pages - 1,
              width="stretch")


def page_case(table, notes, inputs, saved):
    """Page 3: one case, with the chart, the reasons and the review."""
    supervisor = state.user == SUPERVISOR
    matches = table.index[table["id"] == state.current]
    if len(matches) == 0:                    # e.g. after switching dataset
        back_to_list()
        st.rerun()
    case = table.loc[matches[0]]

    st.button("← Back to cases", on_click=back_to_list)
    with st.container(border=True):
        head, badge = st.columns([4, 1.2], vertical_alignment="center")
        head.markdown(f"## {case['short_id']}")
        head.caption(f"Full ID: {case['id']} · assigned to {case['inspector']}"
                     + (f" · **real label: {case['true_label']}**" if supervisor else ""))
        badge.metric("Risk score", f"{case['score']:.0%}",
                     f"{DOT[case['risk']]} {case['risk']}", delta_color="off",
                     help="How much this case looks like the theft cases the model learned "
                          "from (average of 300 trees). It is a ranking score, not the chance "
                          "of theft: many high-score cases turn out to be normal.")

        # Usage chart
        if state.dataset == "sgcc":
            day_cols = [c for c in inputs.columns if c[:4].isdigit()]   # "YYYY-MM-DD"
            usage = inputs.iloc[case["row"]][day_cols].astype(float)
            usage.index = pd.to_datetime(usage.index)
            st.markdown("**Electricity use** · weekly average, kWh a day")
            st.area_chart(usage.resample("W").mean().rename("kWh a day"),
                          height=300, color="#0F766E")
        else:
            days, hourly = inputs
            d = days.iloc[case["row"]]
            building = hourly[hourly["recording_id"] == d["recording_id"]].copy()
            building["hour_of_day"] = building["hour"] % HOURS_PER_DAY
            usual = building.groupby("hour_of_day")["elec_facility_kw"].median()
            this_day = (building[building["day"] == d["day"]]
                        .set_index("hour_of_day")["elec_facility_kw"])
            st.markdown(f"**Electricity use by hour (kW)** · {d['flagged_hours']:.0f} "
                        "of 24 hours were above the model's cut-off")
            st.line_chart(pd.DataFrame({"This day": this_day, "Usual day": usual}),
                          height=300, color=["#0F766E", "#B0BEC5"])

        # Only high-risk cases are flagged; for the others, just describe the pattern
        st.markdown("**Why this case was flagged**" if case["risk"] == "High"
                    else "**What stands out in this case**")
        reasons = case["reasons"] or ["No single pattern stands out; "
                                      "the model combined many small signals."]
        st.markdown("\n".join(f"- {r}" for r in reasons))
        st.caption("Compared with normal customers in the training data. "
                   "This describes the pattern, not the cause.")

    # The review in 3 numbered steps
    with st.container(border=True):
        st.markdown("### Your review")
        with st.form(f"review_{state.dataset}_{case['id']}", clear_on_submit=True, border=False):
            st.markdown("<div class='step'>① What is the likely cause?</div>", unsafe_allow_html=True)
            cause = st.pills("Cause", CAUSES, label_visibility="collapsed")
            st.markdown("<div class='step'>② Your decision</div>", unsafe_allow_html=True)
            decision = st.segmented_control("Decision", DECISIONS, label_visibility="collapsed")
            st.markdown("<div class='step'>③ Notes: what caused the risk, what you checked</div>",
                        unsafe_allow_html=True)
            observations = st.text_area("Notes", height=90, label_visibility="collapsed",
                                        placeholder="e.g. Meter seal broken, use dropped after March")
            b1, b2 = st.columns(2)
            save_next = b1.form_submit_button("Save & next case →", type="primary", width="stretch")
            save_back = b2.form_submit_button("Save & back to list", width="stretch")

        if save_next or save_back:
            if not cause or not decision:
                st.warning("Please choose a likely cause (①) and a decision (②).")
            else:
                save_note({"dataset": state.dataset, "id": case["id"], "cause": cause,
                           "decision": decision, "observations": observations,
                           "inspector": state.user,
                           "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.toast("Review saved ✓")
                # Next case = the one after this in the list it was opened from
                queue = state.queue
                pos = queue.index(case["id"]) if case["id"] in queue else -1
                if save_next and pos + 1 < len(queue):
                    state.current = queue[pos + 1]
                else:
                    back_to_list()
                st.rerun()

    # Earlier reviews of this case, newest first
    history = notes[notes["id"] == case["id"]]
    if not history.empty:
        with st.expander(f"Earlier reviews ({len(history)})"):
            st.dataframe(history[["saved_at", "inspector", "decision", "cause", "observations"]]
                         .iloc[::-1], hide_index=True, width="stretch")

    if supervisor:
        with st.expander("Presenter details (model cut-off, raw score)"):
            st.write(f"Model cut-off: **{saved['threshold']:.2f}** · raw score: "
                     f"**{case['score']:.3f}** · real label: **{case['true_label']}**")
            st.caption("A real inspector would not know the real label. "
                       "It is shown only to check the model during the demo.")


# ------------------------------------------------------------ run the app

st.set_page_config(page_title=APP_NAME, page_icon="🔍", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

if state.user is None:
    page_sign_in()
    st.stop()

dataset = state.dataset
if not model_path(dataset).exists():
    st.error(f"Model not found: {model_path(dataset).name}. Run `python -m src.tuning` first.")
    st.stop()
saved = load_model(dataset)
mtime = model_path(dataset).stat().st_mtime
table, inputs, reference = load_sgcc(mtime, CACHE_VERSION) if dataset == "sgcc" else load_oedi(mtime, CACHE_VERSION)

table = table.copy()   # do not change the cached table

# Latest review of each case: its decision is the status
all_notes = load_notes()
notes = all_notes[all_notes["dataset"] == dataset]
latest = notes.drop_duplicates("id", keep="last").set_index("id")
table["status"] = table["id"].map(latest["decision"]).fillna(NOT_REVIEWED)
table["reviewed_by"] = table["id"].map(latest["inspector"]).fillna("")

header(table, notes)
st.divider()
if state.page == "case":
    page_case(table, notes, inputs, saved)
else:
    page_list(table, notes)
