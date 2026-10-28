"""Interactive front end for the trained turnover-intention model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "turnover_data.csv"
RESULTS = ROOT / "turnover_results"
DEFAULT_MODEL = RESULTS / "categorical_nb_model.joblib"
TARGET = "Do you have any turnover intention?"
TARGET_LABEL = "Stated turnover intention"


def shift_category_codes(values: np.ndarray) -> np.ndarray:
    """Helper referenced by the saved preprocessing pipeline."""
    values = np.asarray(values, dtype=np.int64)
    return np.where(values < 0, 0, values + 1)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column was not found: {TARGET}")
    return frame.drop(columns=[TARGET]).drop_duplicates().reset_index(drop=True)


@st.cache_resource(show_spinner="Loading trained model…")
def load_model(path: str, modified: float):
    return joblib.load(path)


def display_name(column: str) -> str:
    """Make long survey-question headers easier to scan in the UI."""
    if "[" in column and column.endswith("]"):
        group, question = column.split("[", 1)
        return f"{group.strip()} · {question[:-1].strip()}"
    return column


def typical_value(frame: pd.DataFrame, column: str, numeric: list[str]):
    if column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else 0.0
    values = frame[column].dropna().astype(str).value_counts()
    return str(values.index[0]) if not values.empty else "Unknown"


def save_input_change(column: str) -> None:
    st.session_state["feature_values"][column] = st.session_state[f"field_{column}"]
    st.session_state["last_prediction"] = None


st.set_page_config(
    page_title="Turnover Intention | Model Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 2.4rem;}
      .eyebrow {color: #07827b; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .35rem;}
      .subtle {color: #596579; font-size: 1.03rem;}
      div.stButton > button[kind="primary"] {background: #ff4b16; color: white;
        border: 0; border-radius: 999px; width: 140px; height: 140px;
        font-size: 1.12rem; font-weight: 700; box-shadow: 0 8px 20px #14213d24;}
      div.stButton > button[kind="primary"]:hover {background: #e83b08; color: white;
        border: 0; transform: translateY(-2px);}
      [data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6eaf0;
        border-radius: 14px; padding: 12px 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Model settings")
    csv_path_text = st.text_input("Training CSV", str(DEFAULT_CSV))
    model_path_text = st.text_input("Trained model", str(DEFAULT_MODEL))

csv_path = Path(csv_path_text)
model_path = Path(model_path_text)
if not csv_path.is_file():
    st.error("Training CSV not found. Choose its location in Model settings.")
    st.stop()
if not model_path.is_file():
    st.error("Trained model not found. Run `turnover_nb_project.py` first.")
    st.stop()

try:
    X = load_data(str(csv_path))
    model = load_model(str(model_path), model_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load model or training data: {exc}")
    st.stop()

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = [column for column in X.columns if column not in numeric_columns]
all_columns = X.columns.tolist()
if "feature_values" not in st.session_state:
    st.session_state.feature_values = {
        column: typical_value(X, column, numeric_columns) for column in all_columns
    }
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

st.markdown('<div class="eyebrow">MSc AI · trained Categorical Naive Bayes</div>', unsafe_allow_html=True)
st.title("Ask the model")
st.markdown(
    '<div class="subtle">Search for a survey feature, adjust its value, then predict with the trained model.</div>',
    unsafe_allow_html=True,
)
st.info(
    "Fields you do not change use the dataset’s most common category or numeric median. "
    "The prediction is for this completed feature profile; it predicts stated intention, not actual resignation."
)

importance_path = RESULTS / "naive_bayes_permutation_importance.csv"
if importance_path.is_file():
    importance = pd.read_csv(importance_path)
    suggested_columns = [
        name for name in importance.sort_values("mean_f1_decrease", ascending=False)["feature"].tolist()
        if name in all_columns
    ][:5]
else:
    suggested_columns = all_columns[:5]

search_col, show_col, reset_col = st.columns([3.2, 1.5, 1.5], vertical_alignment="bottom")
with search_col:
    query = st.text_input("Search feature", placeholder="Search a survey question…", label_visibility="visible")
with show_col:
    show_all = st.toggle("Show all features", value=False)
with reset_col:
    reset = st.button("Reset typical values", use_container_width=True)

if reset:
    st.session_state.feature_values = {
        column: typical_value(X, column, numeric_columns) for column in all_columns
    }
    st.session_state.last_prediction = None
    st.rerun()

if query.strip():
    visible_columns = [
        column for column in all_columns
        if query.casefold() in column.casefold() or query.casefold() in display_name(column).casefold()
    ]
elif show_all:
    visible_columns = all_columns
else:
    visible_columns = suggested_columns

if not visible_columns:
    st.warning("No matching feature. Try another search term.")
else:
    if not query.strip() and not show_all:
        st.caption("Showing the five strongest overall model signals. Search or select **Show all features** to edit others.")
    else:
        st.caption(f"Showing {len(visible_columns)} matching feature(s).")

left, right = st.columns([1.35, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Feature values")
        st.caption("Unedited fields retain typical values and are still included in the prediction.")
        for column in visible_columns:
            key = f"field_{column}"
            value = st.session_state.feature_values[column]
            st.session_state[key] = value
            label = display_name(column)
            if column in numeric_columns:
                series = pd.to_numeric(X[column], errors="coerce").dropna()
                low = float(series.min()) if not series.empty else 0.0
                high = float(series.max()) if not series.empty else 100.0
                if low == high:
                    st.number_input(
                        label, value=low, min_value=low, max_value=high,
                        key=key, on_change=save_input_change, args=(column,),
                    )
                else:
                    st.number_input(
                        label, min_value=low, max_value=high, value=float(value),
                        key=key, on_change=save_input_change, args=(column,),
                    )
            else:
                options = X[column].dropna().astype(str).value_counts().index.tolist() or ["Unknown"]
                if value not in options:
                    options = [value] + options
                st.selectbox(
                    label, options, key=key,
                    on_change=save_input_change, args=(column,),
                )

with right:
    with st.container(border=True):
        st.subheader("Prediction")
        st.caption("The trained model scores all 46 fields, including values you did not change.")
        predict_space, predict_button, _ = st.columns([1, 1.1, 1])
        with predict_button:
            predict_clicked = st.button("Predict", type="primary", key="predict_button")
        if predict_clicked:
            row = pd.DataFrame(
                [{column: st.session_state.feature_values[column] for column in all_columns}],
                columns=all_columns,
            )
            probabilities = model.predict_proba(row)[0]
            probability_yes = float(probabilities[1])
            prediction = "Yes" if int(model.predict(row)[0]) == 1 else "No"
            encoded = np.asarray(model.named_steps["preprocess"].transform(row)).reshape(-1).astype(int)
            nb = model.named_steps["model"]
            feature_order = numeric_columns + categorical_columns
            evidence = []
            for index, feature in enumerate(feature_order):
                code = int(encoded[index])
                log_prob = nb.feature_log_prob_[index]
                if code < log_prob.shape[1]:
                    evidence.append(
                        {
                            "Question": display_name(feature),
                            "Selected response": str(row.iloc[0][feature]),
                            "Evidence for Yes": float(log_prob[1, code] - log_prob[0, code]),
                        }
                    )
            st.session_state.last_prediction = {
                "label": prediction,
                "probability_yes": probability_yes,
                "evidence": evidence,
            }
        result = st.session_state.last_prediction
        if result:
            probability_yes = result["probability_yes"]
            probability_text = f"{probability_yes:.1%}" if probability_yes >= 0.001 else "<0.1%"
            st.markdown("### Prediction")
            st.markdown(f"## {result['label']}")
            st.metric("Estimated probability · Stated turnover intention", probability_text)
            st.progress(probability_yes)
            with st.expander("Why this prediction?", expanded=True):
                st.caption("Evidence reflects the model’s learned associations, not causes or advice.")
                evidence_df = pd.DataFrame(result["evidence"])
                support, against = st.columns(2)
                with support:
                    st.markdown("**Supporting Yes**")
                    rows = evidence_df.nlargest(3, "Evidence for Yes")
                    st.dataframe(rows[rows["Evidence for Yes"] > 0], hide_index=True, use_container_width=True)
                with against:
                    st.markdown("**Supporting No**")
                    rows = evidence_df.nsmallest(3, "Evidence for Yes")
                    st.dataframe(rows[rows["Evidence for Yes"] < 0], hide_index=True, use_container_width=True)
        else:
            st.caption("Change a feature if needed, then press **Predict**.")

with st.expander("Batch prediction from CSV", expanded=False):
    st.markdown(
        "Upload one or more new responses with the same 46 predictor columns. "
        "The target column is ignored if present."
    )
    template = pd.DataFrame(columns=all_columns)
    st.download_button(
        "Download blank CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="turnover_intention_input_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Response CSV", type=["csv"], key="prediction_upload")
    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            prediction_input = uploaded_frame.drop(columns=[TARGET], errors="ignore")
            missing = [column for column in all_columns if column not in prediction_input.columns]
            if missing:
                st.error(f"The file is missing {len(missing)} required feature columns.")
                st.write("\n".join(f"- {column}" for column in missing))
            elif prediction_input.empty:
                st.error("The CSV has headers but no response rows.")
            else:
                prediction_input = prediction_input.loc[:, all_columns]
                pred = model.predict(prediction_input)
                prob = model.predict_proba(prediction_input)[:, 1]
                batch = pd.DataFrame(
                    {
                        "Response row": np.arange(1, len(prediction_input) + 1),
                        TARGET_LABEL: np.where(pred == 1, "Yes", "No"),
                        "Probability of Yes": prob,
                    }
                )
                st.dataframe(batch.style.format({"Probability of Yes": "{:.1%}"}), hide_index=True)
                st.download_button(
                    "Download predictions", batch.to_csv(index=False).encode("utf-8"),
                    file_name="turnover_intention_predictions.csv", mime="text/csv",
                )
                if TARGET in uploaded_frame.columns:
                    st.caption("Target values were excluded. Predictions on training rows are in-sample, not independent evaluation.")
        except Exception as exc:
            st.error(f"Could not read or score this CSV: {exc}")

st.divider()
