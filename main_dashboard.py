import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Customer Purchase Sequence Modeling using LSTM",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

DATA_PATH = os.path.join(ARTIFACT_DIR, "final_dataset.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "rnn_recommender_optimized.h5")
ITEM_TOKENIZER_PATH = os.path.join(ARTIFACT_DIR, "item_tokenizer_final.pkl")
CATEGORY_TOKENIZER_PATH = os.path.join(ARTIFACT_DIR, "category_tokenizer_final.pkl")
PERFORMANCE_PATH = os.path.join(ARTIFACT_DIR, "model_performance.csv")


@st.cache_resource
def load_data_and_model():
    required_files = {
        "Dataset": DATA_PATH,
        "Model": MODEL_PATH,
        "Item tokenizer": ITEM_TOKENIZER_PATH,
        "Category tokenizer": CATEGORY_TOKENIZER_PATH,
    }

    missing = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(ITEM_TOKENIZER_PATH, "rb") as f:
        item_tokenizer = pickle.load(f)

    with open(CATEGORY_TOKENIZER_PATH, "rb") as f:
        category_tokenizer = pickle.load(f)

    return data, model, item_tokenizer, category_tokenizer


@st.cache_data
def cache_sampled_visitors(_data):
    visitor_counts = _data.groupby("visitorid").size()
    valid_visitors = visitor_counts[
        (visitor_counts >= 3) & (visitor_counts <= 28)
    ].index.tolist()

    if not valid_visitors:
        return []

    sample_size = min(100, len(valid_visitors))
    rng = np.random.default_rng(42)
    return rng.choice(valid_visitors, size=sample_size, replace=False).tolist()


@st.cache_data
def load_performance_file():
    if not os.path.exists(PERFORMANCE_PATH):
        return None
    try:
        performance = pd.read_csv(PERFORMANCE_PATH)
        if not {"Metric", "Value"}.issubset(performance.columns):
            return None
        performance["Value"] = pd.to_numeric(performance["Value"], errors="coerce")
        return performance.dropna(subset=["Value"])
    except Exception:
        return None


try:
    data_full, model, item_tokenizer, category_tokenizer = load_data_and_model()
except Exception as e:
    st.error("The application could not load the required files.")
    st.exception(e)
    st.stop()

if "itemid" not in data_full.columns or "visitorid" not in data_full.columns:
    st.error("The dataset must contain 'itemid' and 'visitorid' columns.")
    st.stop()

if data_full["itemid"].dtype == "object":
    data_full["itemid"] = pd.to_numeric(
        data_full["itemid"], errors="coerce"
    ).fillna(0).astype(int)

if "value" not in data_full.columns:
    data_full["value"] = "0"

sampled_visitors = cache_sampled_visitors(data_full)
data = data_full[data_full["visitorid"].isin(sampled_visitors)].copy()
performance_df = load_performance_file()

st.sidebar.title("🛒 Project Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📊 Dataset Overview",
        "📈 Model Performance",
        "🎯 Next-Product Recommendation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Customer Purchase Sequence Modeling using LSTM "
    "for next-product recommendation."
)

st.title(
    "Customer Purchase Sequence Modeling using LSTM "
    "for Next-Product Recommendation"
)
st.caption(
    "Sequence-based e-commerce recommendation using customer purchase history."
)


if page == "🏠 Dashboard":

    st.subheader("📌 Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(data_full):,}")
    with col2:
        st.metric("Unique Customers", f"{data_full['visitorid'].nunique():,}")
    with col3:
        st.metric("Unique Products", f"{data_full['itemid'].nunique():,}")
    with col4:
        st.metric("Unique Categories", f"{data_full['value'].nunique():,}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("📦 Dataset Summary")
        summary = pd.DataFrame(
            {
                "Property": [
                    "Rows", "Columns", "Customers", "Products",
                    "Categories", "Missing Values"
                ],
                "Value": [
                    f"{len(data_full):,}",
                    len(data_full.columns),
                    f"{data_full['visitorid'].nunique():,}",
                    f"{data_full['itemid'].nunique():,}",
                    f"{data_full['value'].nunique():,}",
                    f"{int(data_full.isna().sum().sum()):,}",
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with right:
        st.subheader("🧠 Model Summary")
        model_info = pd.DataFrame(
            {
                "Model Property": [
                    "Model Type",
                    "Input",
                    "Sequence Length",
                    "Output",
                    "Recommendation Size",
                ],
                "Details": [
                    "LSTM-based sequence recommender",
                    "Item + Category sequence",
                    "28",
                    "Next product",
                    "Top-5 products",
                ],
            }
        )
        st.dataframe(model_info, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔄 Recommendation Workflow")
    st.markdown(
        "**Customer purchase sequence → Item & category encoding → "
        "Sequence padding → LSTM model → Next-product probability prediction "
        "→ Top-5 recommendations**"
    )
    st.info(
        "The original recommendation functionality is unchanged. "
        "Use 'Next-Product Recommendation' from the sidebar to make predictions."
    )


elif page == "📊 Dataset Overview":

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(data_full):,}")
    with col2:
        st.metric("Columns", len(data_full.columns))
    with col3:
        st.metric("Customers", f"{data_full['visitorid'].nunique():,}")
    with col4:
        st.metric("Products", f"{data_full['itemid'].nunique():,}")

    st.markdown("---")
    st.subheader("Dataset Columns")

    column_info = pd.DataFrame(
        {
            "Column": data_full.columns,
            "Data Type": [str(data_full[c].dtype) for c in data_full.columns],
            "Missing Values": [
                int(data_full[c].isna().sum()) for c in data_full.columns
            ],
            "Unique Values": [
                int(data_full[c].nunique()) for c in data_full.columns
            ],
        }
    )

    st.dataframe(column_info, use_container_width=True, hide_index=True)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("👥 Customer Activity")
        visitor_counts = (
            data_full.groupby("visitorid").size().sort_values(ascending=False).head(15)
        )
        st.bar_chart(visitor_counts)

    with right:
        st.subheader("🛍️ Most Frequent Products")
        product_counts = data_full["itemid"].value_counts().head(15)
        st.bar_chart(product_counts)

    st.markdown("---")
    st.subheader("🔎 Dataset Sample")
    st.dataframe(data_full.head(20), use_container_width=True, hide_index=True)


elif page == "📈 Model Performance":

    st.subheader("📈 Model Performance Comparison")

    st.markdown(
        "The project evaluates next-product prediction using **Top-1, Top-5, "
        "Top-10 Accuracy, Precision@5, Recall@5 and HitRate@5**. "
        "These are the metrics defined in the project evaluation notebook."
    )

    if performance_df is not None and not performance_df.empty:

        display_df = performance_df.copy()
        display_df["Value (%)"] = display_df["Value"] * 100

        left, right = st.columns(2)

        with left:
            st.subheader("Performance Table")
            table_df = display_df.copy()
            table_df["Value (%)"] = table_df["Value (%)"].round(2)
            st.dataframe(
                table_df[["Metric", "Value (%)"]],
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader("Metric Comparison")
            chart_df = display_df.set_index("Metric")[["Value (%)"]]
            st.bar_chart(chart_df)

    else:

        st.warning(
            "Actual evaluation numbers are not stored in the deployed artifacts. "
            "The app will not invent performance values."
        )

        st.info(
            "To display your real performance comparison, create "
            "artifacts/model_performance.csv with two columns: Metric, Value. "
            "Use the actual values printed by your Colab evaluation."
        )

        expected_metrics = pd.DataFrame(
            {
                "Metric": [
                    "Top-1 Accuracy",
                    "Top-5 Accuracy",
                    "Top-10 Accuracy",
                    "Precision@5",
                    "Recall@5",
                    "HitRate@5",
                ],
                "Meaning": [
                    "Correct next product ranked first",
                    "Correct next product appears in top 5",
                    "Correct next product appears in top 10",
                    "Precision among the top 5 recommendations",
                    "Recall of the actual next product in top 5",
                    "Whether the actual next product appears in top 5",
                ],
            }
        )

        st.dataframe(
            expected_metrics,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("🏗️ Model Architecture")

    architecture = pd.DataFrame(
        {
            "Stage": [
                "Item Input",
                "Item Embedding",
                "Item Bi-LSTM",
                "Category Input",
                "Category Embedding",
                "Category Bi-LSTM",
                "Concatenation",
                "Dense",
                "Dropout",
                "Softmax Output",
            ],
            "Configuration": [
                "Sequence length = 28",
                "50-dimensional embedding",
                "32 LSTM units per direction",
                "Sequence length = 28",
                "50-dimensional embedding",
                "32 LSTM units per direction",
                "Item + category representations",
                "64 units, ReLU",
                "0.30",
                "Next-product classification",
            ],
        }
    )

    st.dataframe(architecture, use_container_width=True, hide_index=True)


elif page == "🎯 Next-Product Recommendation":

    st.subheader("🎯 Get Your Recommendations")
    st.markdown(
        "Select a visitor or enter an item sequence to predict the next products."
    )

    input_type = st.radio(
        "Choose input method:",
        ("Select Visitor ID", "Search Visitor ID", "Enter Item Sequence"),
    )

    item_sequence = []
    category_sequence = []
    selected_visitor = None

    if input_type == "Select Visitor ID":

        if data.empty:
            st.error("No visitors are available in the sampled data.")
            st.stop()

        selected_visitor = st.selectbox(
            "Select a Visitor ID:",
            data["visitorid"].unique().tolist(),
        )

        user_data = data[data["visitorid"] == selected_visitor]
        item_sequence = user_data["itemid"].astype(str).tolist()
        category_sequence = user_data["value"].astype(str).tolist()

        st.write(f"**Your Sequence:** {', '.join(item_sequence)}")
        st.write(f"**Categories:** {', '.join(category_sequence)}")

    elif input_type == "Search Visitor ID":

        search_term = st.text_input("Search for a Visitor ID:")

        if search_term:
            filtered_visitors = [
                vid for vid in data["visitorid"].unique()
                if str(vid).startswith(search_term)
            ]

            if filtered_visitors:

                selected_visitor = st.selectbox(
                    "Select a matching Visitor ID:",
                    filtered_visitors,
                )

                user_data = data[data["visitorid"] == selected_visitor]
                item_sequence = user_data["itemid"].astype(str).tolist()
                category_sequence = user_data["value"].astype(str).tolist()

                st.write(f"**Your Sequence:** {', '.join(item_sequence)}")
                st.write(f"**Categories:** {', '.join(category_sequence)}")

            else:
                st.error("No matching Visitor ID found. Try another search term.")

    else:

        item_sequence_input = st.text_input(
            "Enter item IDs (space-separated, e.g., '355908 248676'):"
        )

        item_sequence = (
            item_sequence_input.strip().split()
            if item_sequence_input
            else []
        )

        for item in item_sequence:
            try:
                item_id = int(item)

                if item_id in data["itemid"].values:
                    cat = data.loc[
                        data["itemid"] == item_id, "value"
                    ].iloc[0]
                    category_sequence.append(str(cat))
                else:
                    category_sequence.append("0")

            except ValueError:
                category_sequence.append("0")

        if item_sequence:
            st.write(f"**Your Sequence:** {', '.join(item_sequence)}")
            st.write(f"**Categories:** {', '.join(category_sequence)}")

    def get_recommendations(item_seq, cat_seq, max_sequence_length=28):

        if isinstance(item_seq, pd.Series):
            item_seq = item_seq.tolist()

        if isinstance(cat_seq, pd.Series):
            cat_seq = cat_seq.tolist()

        if not item_seq or len(item_seq) != len(cat_seq):
            return None, (
                "Please provide a valid item sequence with corresponding categories."
            )

        valid_items = [
            item for item in item_seq if item in item_tokenizer.word_index
        ]

        if not valid_items:
            return None, "No valid item IDs found. Please check your input."

        item_seq_tokens = item_tokenizer.texts_to_sequences([item_seq])
        item_padded = pad_sequences(
            item_seq_tokens,
            maxlen=max_sequence_length,
            padding="pre",
        )

        cat_seq_tokens = category_tokenizer.texts_to_sequences([cat_seq])
        cat_padded = pad_sequences(
            cat_seq_tokens,
            maxlen=max_sequence_length,
            padding="pre",
        )

        pred_probs = model.predict(
            [np.array(item_padded), np.array(cat_padded)],
            verbose=0,
        )

        top_5_indices = np.argsort(pred_probs[0])[-5:][::-1]
        top_5_probs = pred_probs[0][top_5_indices]

        if top_5_probs.sum() > 0:
            top_5_probs = top_5_probs / top_5_probs.sum()

        reverse_item_map = {
            v: k for k, v in item_tokenizer.word_index.items()
        }

        recommended_items = [
            reverse_item_map.get(idx, "0")
            for idx in top_5_indices
        ]

        categories = []

        for item in recommended_items:
            try:
                item_id = int(item)

                if item_id in data_full["itemid"].values:
                    cat = data_full.loc[
                        data_full["itemid"] == item_id, "value"
                    ].iloc[0]
                    categories.append(str(cat))
                else:
                    categories.append("Unknown")

            except ValueError:
                categories.append("Unknown")

        item_names = [
            f"Item {item}" if item != "0" else "Unknown Product"
            for item in recommended_items
        ]

        recommendations = pd.DataFrame(
            {
                "Rank": range(1, 6),
                "Product": item_names,
                "Item ID": recommended_items,
                "Category ID": categories,
                "Confidence": [f"{p:.4f}" for p in top_5_probs],
            }
        )

        return recommendations, None

    if st.button("Get Recommendations 🎉", use_container_width=True):

        valid_input = (
            input_type in ["Select Visitor ID", "Search Visitor ID"]
            and selected_visitor is not None
        ) or (
            input_type == "Enter Item Sequence"
            and bool(item_sequence)
        )

        if valid_input:
            recommendations, error = get_recommendations(
                item_sequence,
                category_sequence,
            )
        else:
            recommendations, error = (
                None,
                "Please select a Visitor ID or enter an item sequence.",
            )

        if error:
            st.error(error)

        else:
            st.subheader("🎁 Top-5 Recommended Products")
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("📈 Recommendation Confidence")

            item_ids = recommendations["Item ID"].values
            confidences = recommendations["Confidence"].astype(float).values

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(item_ids, confidences)
            ax.set_xlabel("Item ID")
            ax.set_ylabel("Normalized Confidence")
            ax.set_title("Top-5 Recommendation Confidence")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.success(
                "Recommendations generated successfully using the loaded LSTM model."
            )
