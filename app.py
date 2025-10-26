import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import io

# -------------------------
# Page config & CSS (Royal Blue bright theme)
# -------------------------
st.set_page_config(page_title="ZP School Academic Dashboard",
                   page_icon="🏫",
                   layout="wide",
                   initial_sidebar_state="expanded")

# CSS styling - bright white + royal blue accents, dark navy text
st.markdown("""
    <style>
    /* ---------- Remove Streamlit Default Header Bar ---------- */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* ---------- General Page Background ---------- */
    div[data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        color: #111 !important;
    }

    /* ---------- Sidebar Styling ---------- */
   /* ---------- Remove Streamlit Default Header Bar ---------- */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Sidebar Text */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Sidebar Clean Title */
    .sidebar-clean-title {
        font-size: 24px;
        font-weight: 700;
        color: #60a5fa;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 2px;
    }

    .sidebar-subtext {
        font-size: 13px;
        text-align: center;
        color: #d1d5db;
        margin-bottom: 15px;
    }

    /* ---------- File Uploader ---------- */
    
    .stFileUploader {
        background-color: #1e1e2f !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        padding: 10px;
    }

    .stFileUploader label,
    .stFileUploader div {
        color: #f0f0f0 !important;
        font-weight: 500 !important;
    }

    .stFileUploader:hover {
        background-color: #2b2b3d !important;
    }

    /* ---------- Download Button ---------- */
    .stDownloadButton button {
        background-color: #1e1e2f !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stDownloadButton button:hover {
        background-color: #2b2b3d !important;
        color: #ffffff !important;
    }

    /* ---------- Selectboxes & Dropdowns ---------- */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e1e2f !important;
        color: #f0f0f0 !important;
        border-radius: 6px !important;
        border: 1px solid #444 !important;
    }

    .stSelectbox div[data-baseweb="select"] > div:hover {
        background-color: #2b2b3d !important;
    }

    .stSelectbox label,
    .stSelectbox span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* ---------- Headers & Titles ---------- */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8 !important;
    }

    /* ---------- Card / Container Styling ---------- */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    /* ---------- Fix Input Text Visibility ---------- */
    input, textarea {
        color: #111 !important;
        background-color: #f1f5f9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar (upload + nav)
# -------------------------
st.sidebar.markdown("""
    <div class="sidebar-clean-title">ZP School chorachiwadi</div>
    <div class="sidebar-subtext">Academic Dashboard</div>
""", unsafe_allow_html=True)

st.sidebar.write("Upload dataset (.xlsx or .csv) to start")

uploaded_file = st.sidebar.file_uploader("Upload student dataset", type=["xlsx", "csv"])

# Navigation (renamed)
menu = [
    "🏫 Dashboard Home",
    "📘 Student Data Summary",
    "🧰 Data Cleaning & Preparation",
    "📖 Subject Performance Analysis",
    "🧩 Group Students by Performance",
    "🎯 Pass/Fail Prediction",
    "📊 Custom Visualizations",
]
choice = st.sidebar.radio("Navigate", menu)

# -------------------------
# Helper: load user file
# -------------------------
def load_user_file(uploaded):
    if uploaded is None:
        return None
    try:
        if uploaded.name.lower().endswith(".csv") or uploaded.type == "text/csv":
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.sidebar.error(f"Failed to read the uploaded file: {e}")
        return None

df = load_user_file(uploaded_file)

# store cleaned DF in session
if "df_clean" not in st.session_state:
    st.session_state["df_clean"] = df.copy() if df is not None else None

# -------------------------
# Helper: subjects detection
# -------------------------
def get_subjects(dataframe):
    if dataframe is None:
        return []
    exclude = {"Student_ID", "Name", "Gender", "Class", "Age", "Attendance_Percentage", "Weight", "Height", "Health_Score"}
    subj_cols = [c for c in dataframe.columns if c not in exclude and not c.lower().endswith(("prev1", "prev2", "prev3"))]
    subj_cols = [c for c in subj_cols if pd.api.types.is_numeric_dtype(dataframe[c])]
    return subj_cols

# -------------------------
# Page: Dashboard Home
# -------------------------
if choice == "🏫 Dashboard Home":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="app-title">ZP School Academic Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Teacher tool — upload your dataset to begin analysis</div>', unsafe_allow_html=True)

    st.write("""
        This dashboard helps teachers:
        - Analyze subject-level strengths/weaknesses (automatic mean-based threshold)
        - Group students with similar performance for targeted teaching
        - Predict pass/fail using previous 3 years' marks
        - Build custom visualizations (height, weight, health score included)
    """)

    if df is None:
        st.info("No dataset uploaded yet. Please upload a `.xlsx` or `.csv` file using the uploader on the left sidebar.")
    else:
        subjects = get_subjects(df)
        cols1, cols2, cols3 = st.columns(3)
        cols1.markdown(f'<div class="metric"><h4 style="margin:0;color:#2563eb;">Total Students</h4><div style="font-size:18px;font-weight:600;color:#0f172a">{df.shape[0]}</div></div>', unsafe_allow_html=True)
        cols2.markdown(f'<div class="metric"><h4 style="margin:0;color:#2563eb;">Subjects</h4><div style="font-size:18px;font-weight:600;color:#0f172a">{len(subjects)}</div></div>', unsafe_allow_html=True)
        if "Attendance_Percentage" in df.columns:
            cols3.markdown(f'<div class="metric"><h4 style="margin:0;color:#2563eb;">Avg Attendance %</h4><div style="font-size:18px;font-weight:600;color:#0f172a">{df["Attendance_Percentage"].mean():.2f}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Ensure dataset present for data pages
# -------------------------
if choice != "🏫 Dashboard Home" and df is None:
    st.warning("Please upload your Excel/CSV dataset in the left sidebar to use this section.")
    st.stop()

if df is not None:
    if st.session_state.get("df_clean") is None or st.session_state["df_clean"].shape[0] != df.shape[0]:
        st.session_state["df_clean"] = df.copy()    

# -------------------------
# Page: Student Data Summary
# -------------------------
if choice == "📘 Student Data Summary":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Student Data Summary")
    st.write("Preview of the uploaded dataset (first 25 rows):")
    st.dataframe(st.session_state["df_clean"].head(25))
    st.write("### Data info")
    info = pd.DataFrame({
        "Data Type": st.session_state["df_clean"].dtypes.astype(str),
        "Missing Values": st.session_state["df_clean"].isnull().sum(),
        "Unique Values": st.session_state["df_clean"].nunique()
    })
    st.dataframe(info)
    st.write("### Summary statistics (numeric columns)")
    st.dataframe(st.session_state["df_clean"].describe().T)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Page: Data Cleaning & Preparation
# -------------------------
elif choice == "🧰 Data Cleaning & Preparation":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data Cleaning & Preparation")
    df_clean = st.session_state["df_clean"]

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Remove rows with missing values"):
            before = df_clean.shape[0]
            df_clean = df_clean.dropna()
            st.session_state["df_clean"] = df_clean
            st.success(f"Removed {before - df_clean.shape[0]} rows with missing values.")
    with c2:
        if st.button("Remove duplicate rows"):
            before = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            st.session_state["df_clean"] = df_clean
            st.success(f"Removed {before - df_clean.shape[0]} duplicate rows.")
    with c3:
        if st.button("Label-encode categorical columns"):
            cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
            if cat_cols:
                le = LabelEncoder()
                for col in cat_cols:
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                st.session_state["df_clean"] = df_clean
                st.success(f"Encoded columns: {cat_cols}")
            else:
                st.info("No categorical columns found to encode.")

    st.markdown("#### Preview (session copy)")
    st.dataframe(st.session_state["df_clean"].head(30))

    st.markdown("#### Download cleaned dataset")
    to_dl = st.session_state["df_clean"]
    buffer = io.BytesIO()
    try:
        to_dl.to_excel(buffer, index=False)
        st.download_button("Download cleaned (Excel)", data=buffer, file_name="students_cleaned.xlsx", mime="application/vnd.ms-excel")
    except Exception:
        st.download_button("Download cleaned (CSV)", data=to_dl.to_csv(index=False).encode('utf-8'), file_name="students_cleaned.csv")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Page: Subject Performance Analysis
# -------------------------
elif choice == "📖 Subject Performance Analysis":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Subject Performance Analysis (automatic threshold = subject mean)")

    df_used = st.session_state["df_clean"]
    subjects = get_subjects(df_used)
    if not subjects:
        st.warning("No numeric subject columns found. Ensure columns like 'Math', 'Science', etc. are present.")
    else:
        subj = st.selectbox("Select subject", subjects)
        subj_mean = df_used[subj].mean()
        st.info(f"Automatic threshold: mean = {subj_mean:.2f}. Below mean = Weak; At/above mean = Strong.")

        weak_df = df_used[df_used[subj] < subj_mean].sort_values(subj)
        strong_df = df_used[df_used[subj] >= subj_mean].sort_values(subj, ascending=False)

        left, right = st.columns([2,1])
        with left:
            st.markdown("**Weak Students (below mean)**")
            st.dataframe(weak_df[["Student_ID","Name","Class",subj,"Attendance_Percentage"]].head(200))
        with right:
            st.markdown("**Strong Students (at/above mean)**")
            st.dataframe(strong_df[["Student_ID","Name","Class",subj,"Attendance_Percentage"]].head(200))

        st.markdown("### Distribution")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df_used[subj].dropna(), bins=10, kde=True, ax=ax, color="#0b57a0")
        ax.axvline(subj_mean, color="#ff6b6b", linestyle="--", label=f"Mean = {subj_mean:.1f}")
        ax.set_xlabel("Marks")
        ax.set_title(f"{subj} distribution")
        ax.legend()
        st.pyplot(fig)

        top_n = st.slider("Top/Bottom N to show", 3, 15, 8)
        top_n_df = strong_df.head(top_n)[["Name", subj]].set_index("Name")
        bot_n_df = weak_df.head(top_n)[["Name", subj]].set_index("Name")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Top students")
            st.bar_chart(top_n_df)
        with c2:
            st.markdown("Bottom students")
            st.bar_chart(bot_n_df)

        if st.checkbox("Show subject heatmap across students (sample if large)"):
            heat_df = df_used[["Name"] + subjects].set_index("Name")
            if heat_df.shape[0] > 80:
                heat_df = heat_df.sample(80, random_state=1)
            fig, ax = plt.subplots(figsize=(10, max(4, heat_df.shape[0]*0.12)))
            sns.heatmap(heat_df, cmap="YlGnBu", cbar_kws={'label':'Marks'}, ax=ax)
            st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Page: Group Students by Performance (KMeans + meaningful labels)
# -------------------------
elif choice == "🧩 Group Students by Performance":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Group Students by Performance (KMeans) — labelled for teachers")

    df_used = st.session_state["df_clean"]
    subjects = get_subjects(df_used)
    st.write("Select 2–5 subjects (features) to create groups of similar-performing students.")
    features = st.multiselect("Performance features (subjects)", subjects, default=subjects[:3])

    if len(features) < 2:
        st.info("Please choose at least 2 subject features.")
    else:
        k = st.slider("Number of groups (k)", 2, 6, 3)

        feat_df = df_used[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feat_df.values)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        cluster_df = feat_df.copy()
        cluster_df["GroupID"] = labels

        # compute cluster mean (average across selected features) to rank clusters
        cluster_means = cluster_df.groupby("GroupID")[features].mean().mean(axis=1)
        rank_order = cluster_means.sort_values(ascending=False).index.tolist()

        # map cluster ids to human labels
        if k == 2:
            label_names = ["High Performers", "Needs Improvement"]
        elif k == 3:
            label_names = ["High Performers", "Average Learners", "Needs Improvement"]
        else:
            label_names = ["Excellent", "Above Average", "Average", "Below Average", "At Risk", "Critical"][:k]

        label_map = {}
        for rank_pos, cluster_id in enumerate(rank_order):
            label_map[cluster_id] = label_names[rank_pos]

        # attach groups back to original df
        df_grouped = df_used.copy()
        df_grouped.loc[cluster_df.index, "GroupID"] = cluster_df["GroupID"]
        df_grouped.loc[cluster_df.index, "GroupLabel"] = cluster_df["GroupID"].map(label_map)

        st.write("Group counts:")
        counts = df_grouped["GroupLabel"].value_counts().rename_axis("GroupLabel").reset_index(name="Count")
        st.dataframe(counts)

        st.write("Sample students with groups:")
        display_cols = ["Student_ID", "Name", "Class"] + features + ["GroupLabel"]
        st.dataframe(df_grouped[display_cols].sort_values("GroupLabel").head(200))

        # Scatter plot on first two features
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=df_grouped.dropna(), x=features[0], y=features[1], hue="GroupLabel", s=80, ax=ax, palette="viridis")
        ax.set_title("Student Groups (visual on two features)")
        st.pyplot(fig)

        # Show cluster centers (interpretable)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=features)
        centers_df["Assigned_Label"] = [label_map[i] for i in range(k)]
        st.markdown("Cluster centers (average marks per group):")
        st.dataframe(centers_df.round(2))

        # save grouped df in session
        st.session_state["df_grouped"] = df_grouped
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Page: Pass/Fail Prediction
# -------------------------
elif choice == "🎯 Pass/Fail Prediction":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Pass/Fail Prediction (uses previous 3 years' marks)")

    df_used = st.session_state["df_clean"]
    subjects = get_subjects(df_used)
    if not subjects:
        st.warning("No subject columns available.")
    else:
        subj = st.selectbox("Select target subject", subjects)
        pass_mark = st.slider("Pass mark threshold", 20, 60, 35)

        prev_cols = [f"{subj}_prev1", f"{subj}_prev2", f"{subj}_prev3"]
        extras = [c for c in ["Attendance_Percentage", "Health_Score", "Weight", "Height", "Class", "Age"] if c in df_used.columns]
        missing_prev = [c for c in prev_cols if c not in df_used.columns]
        if missing_prev:
            st.error(f"Missing previous-year columns for {subj}. Expected: {prev_cols}. Please ensure your file contains these columns.")
        else:
            model_df = df_used[[subj] + prev_cols + extras].dropna().copy()
            model_df["target_pass"] = (model_df[subj] >= pass_mark).astype(int)
            X = model_df[prev_cols + extras]
            # encode object columns if present
            for col in X.select_dtypes(include="object").columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            y = model_df["target_pass"]

            if y.nunique() < 2:
                st.warning("Target has no variation (all pass or all fail). Model cannot be trained.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                clf = RandomForestClassifier(n_estimators=150, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1]

                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")

                st.write("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig)

                st.write("Classification report")
                st.text(classification_report(y_test, y_pred, digits=3))

                # AUC
                try:
                    auc = roc_auc_score(y_test, y_prob)
                    st.metric("ROC AUC", f"{auc:.3f}")
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                    ax.plot([0,1],[0,1],"--", color="#999")
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
                    st.pyplot(fig)
                except Exception:
                    pass

                st.markdown("### Students most at-risk (lowest predicted probability of pass)")
                full_probs = clf.predict_proba(X)[:, 1]
                risk_df = model_df.copy()
                risk_df["prob_pass"] = full_probs
                risk_sorted = risk_df.sort_values("prob_pass").head(30)
                st.dataframe(risk_sorted[[*prev_cols, *extras, subj, "prob_pass", "target_pass"]].head(30))

                st.markdown("#### Predict single student (manual input)")
                sample_vals = {}
                for c in prev_cols + extras:
                    default = float(df_used[c].median()) if c in df_used.columns else 0.0
                    sample_vals[c] = st.number_input(c, value=float(default))
                if st.button("Predict single student"):
                    sample_X = pd.DataFrame([sample_vals])
                    for col in sample_X.select_dtypes(include="object").columns:
                        sample_X[col] = LabelEncoder().fit_transform(sample_X[col].astype(str))
                    prob = clf.predict_proba(sample_X)[:, 1][0]
                    st.info(f"Predicted probability of PASS in {subj}: {prob:.3f}")
                    if prob < 0.5:
                        st.warning("Likely to FAIL — consider intervention.")
                    else:
                        st.success("Likely to PASS.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Page: Custom Visualizations
# -------------------------
elif choice == "📊 Custom Visualizations":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Custom Visualizations")

    df_used = st.session_state["df_clean"]
    numeric_cols = df_used.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for plotting.")
    else:
        graph_type = st.selectbox("Select graph type", ["Histogram", "Boxplot", "Scatter", "Line", "Bar", "Correlation Heatmap"])
        if graph_type == "Histogram":
            col = st.selectbox("Select numeric column", numeric_cols)
            bins = st.slider("Bins", 5, 60, 12)
            fig, ax = plt.subplots()
            ax.hist(df_used[col].dropna(), bins=bins)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
        elif graph_type == "Boxplot":
            col = st.selectbox("Select column", numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(y=df_used[col], ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)
        elif graph_type == "Scatter":
            x = st.selectbox("X-axis", numeric_cols, index=0)
            y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            hue = st.selectbox("Color by (optional)", ["None"] + df_used.columns.tolist())
            fig, ax = plt.subplots()
            if hue != "None":
                sns.scatterplot(data=df_used, x=x, y=y, hue=hue, ax=ax, palette="tab10")
            else:
                ax.scatter(df_used[x], df_used[y])
            ax.set_title(f"{x} vs {y}")
            st.pyplot(fig)
        elif graph_type == "Line":
            col = st.selectbox("Column for Y (index will be X)", numeric_cols)
            fig, ax = plt.subplots()
            ax.plot(df_used.index, df_used[col])
            ax.set_title(f"Line plot of {col}")
            st.pyplot(fig)
        elif graph_type == "Bar":
            ycol = st.selectbox("Numeric column (Y)", numeric_cols)
            xcol = st.selectbox("Categorical X", df_used.select_dtypes(exclude=['number']).columns.tolist())
            agg = df_used.groupby(xcol)[ycol].mean().reset_index().sort_values(ycol, ascending=False)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.bar(agg[xcol].astype(str), agg[ycol])
            ax.set_xticklabels(agg[xcol].astype(str), rotation=45, ha="right")
            ax.set_title(f"Average {ycol} by {xcol}")
            st.pyplot(fig)
        elif graph_type == "Correlation Heatmap":
            cols = st.multiselect("Select numeric columns (2+)", numeric_cols, default=numeric_cols[:6])
            if len(cols) < 2:
                st.info("Pick at least 2 numeric columns.")
            else:
                corr = df_used[cols].corr()
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# End
# -------------------------
else:
    st.info("Choose a page from the sidebar.")