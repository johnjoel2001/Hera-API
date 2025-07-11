import streamlit as st
import json
import pandas as pd
import pickle
import tempfile
import os
from pdf_processor import extract_structured_data
import numpy as np
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from run_rag_from_json import generate_rag_explanation
import google.generativeai as genai
from report_manager import save_new_report, compare_to_last  # 👈 Import here

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Load model and feature names
with open("logistic_regression.pkl", "rb") as f:
    package = pickle.load(f)
    model = package["model"]
    feature_names = package["feature_names"]

# Load SHAP background dataset
try:
    with open("background_data.pkl", "rb") as f:
        background = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load background data. SHAP values may not be accurate.")
    background = None

# App title
st.title("🧪 Semen Analysis Fertility Prediction")

# Upload
uploaded_file = st.file_uploader("Upload your semen analysis report (PDF/Image)", type=["pdf", "jpg", "png"])

if uploaded_file:
    file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info("🔍 Extracting data from uploaded file...")

    try:
        # Extract structured data
        result = extract_structured_data(tmp_file_path)
        extracted_data = result.model_dump()

        # Save extracted JSON
        json_path = tmp_file_path.replace(".pdf", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        st.success("✅ Data extracted successfully!")

        # Extract values
        volume = extracted_data['semen_analysis']['volume']['value']
        concentration = extracted_data['semen_analysis']['concentration']['value']
        motility = extracted_data['semen_analysis']['motility']['value']

        # Create input DataFrame
        input_data = pd.DataFrame([{
            feature_names[0]: volume,
            feature_names[1]: concentration,
            feature_names[2]: motility
        }])

        # Predict fertility probability
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("🎯 Fertility Score")
        st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{probability*100:.2f}%</h1>", unsafe_allow_html=True)

        # Sidebar interpretation
        st.sidebar.title("📊 Interpretation Guide")
        st.sidebar.subheader("0-40%: Low Fertility Probability")
        st.sidebar.write("Unfavorable indicators. Recommend follow-up consultation.")
        st.sidebar.subheader("40-70%: Borderline")
        st.sidebar.write("Results are inconclusive; consider retesting.")
        st.sidebar.subheader("70-100%: High Fertility Probability")
        st.sidebar.write("Strong indicators of fertility.")

        # Table of input values
        st.table(pd.DataFrame({
            "Label Name": ["Volume", "Concentration", "Motility"],
            "Value": [volume, concentration, motility]
        }).set_index("Label Name"))

        # SHAP Analysis
        st.subheader("🔎 Feature Importance Analysis (SHAP)")
        if background is not None:
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(input_data)
            base_value = explainer.expected_value

            feature_impact = pd.DataFrame({
                "Feature": feature_names,
                "Value": input_data.iloc[0].values,
                "SHAP_Impact": shap_values[0]
            }).sort_values(by="SHAP_Impact")

            feature_impact["Abs_Impact"] = np.abs(feature_impact["SHAP_Impact"])
            colors = ['#ff4d4d' if x < 0 else '#28a745' for x in feature_impact["SHAP_Impact"]]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(feature_impact["Feature"], feature_impact["SHAP_Impact"], color=colors, edgecolor='black')

            for bar, impact in zip(bars, feature_impact["SHAP_Impact"]):
                x_offset = 0.08 if impact < 0 else 0
                ax.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height() / 2,
                        f'{impact:+.2f}', va='center', ha='left', fontsize=10, fontweight='bold')

            ax.axvline(0, color='black', linewidth=1)
            ax.set_title("Feature Contribution to Fertility Prediction")
            ax.set_xlabel("SHAP Impact")
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Feature details
            def get_impact_strength(abs_impact):
                if abs_impact > 0.15:
                    return "Very Strong"
                elif abs_impact > 0.10:
                    return "Strong"
                elif abs_impact > 0.05:
                    return "Moderate"
                else:
                    return "Mild"

            st.subheader("📋 DETAILED FEATURE ANALYSIS")
            st.markdown("---")

            positive_factors = feature_impact[feature_impact['SHAP_Impact'] >= 0]
            negative_factors = feature_impact[feature_impact['SHAP_Impact'] < 0]
            if not positive_factors.empty:
                st.markdown("✅ **POSITIVE INFLUENCES (Supporting Fertility)**")
                for _, row in positive_factors.iterrows():
                    st.markdown(f"💪 **{row['Feature']}** = {row['Value']:.2f}")
                    st.markdown(f"→ {get_impact_strength(row['Abs_Impact'])} **positive** impact (**+{row['SHAP_Impact']:.2f}**)")

            if not negative_factors.empty:
                st.markdown("⚠️ **NEGATIVE INFLUENCES (Areas for Improvement)**")
                for _, row in negative_factors.iterrows():
                    st.markdown(f"🎯 **{row['Feature']}** = {row['Value']:.2f}")
                    st.markdown(f"→ {get_impact_strength(row['Abs_Impact'])} **negative** impact (**{row['SHAP_Impact']:.2f}**)  \n💡 Consider improving this parameter if possible.")

            # Prepare JSON for RAG and Saving
            shap_json = {
                "fertility_score": round(probability * 100, 2),
                "features": {}
            }

            for feature, value in zip(feature_names, input_data.iloc[0]):
                shap_json["features"][feature] = {
                    "value": float(value),
                    "impact": float(shap_values[0][feature_names.index(feature)])
                }

            # ✅ Save report
            save_new_report(shap_json, user_id="me")

            # ✅ Compare to previous report
            comparison = compare_to_last(shap_json, user_id="me")
            st.markdown(comparison)

            # ✅ RAG Explanation
            try:
                st.subheader("📖 AI Fertility Explanation")
                with st.spinner("Generating explanation..."):
                    explanation = generate_rag_explanation()
                    st.markdown(explanation)
                    with open("rag_output.txt", "w") as f:
                        f.write(explanation)
                st.success("✅ Explanation complete and saved!")
            except Exception as e:
                st.error(f"❌ Failed to generate explanation: {e}")

        else:
            st.warning("⚠️ SHAP explanation not available due to missing background data.")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")

    os.remove(tmp_file_path)

# Footer
st.caption("🧠 Disclaimer: This tool is for educational use only. Not a substitute for professional diagnosis.")
st.markdown("[Terms of Service](https://herafertility.co/policies/terms-of-service) | [Privacy Policy](https://herafertility.co/policies/privacy-policy)")


# import streamlit as st
# import json
# import pandas as pd
# import pickle
# import tempfile
# import os
# from pdf_processor import extract_structured_data
# import numpy as np
# import shap
# import matplotlib.pyplot as plt
# from run_rag_from_json import generate_rag_explanation
# import google.generativeai as genai
# from report_manager import save_new_report, compare_to_last

# # MINIMAL cloud deployment setup
# st.set_page_config(
#     page_title="Semen Analysis Fertility Prediction",
#     page_icon="🧪",
#     layout="wide"
# )

# # Handle API keys for cloud deployment (ONLY change for cloud)
# def get_google_api_key():
#     """Get Google API key from secrets or user input"""
#     try:
#         # Try Streamlit secrets first (for cloud deployment)
#         api_key = st.secrets["GOOGLE_API_KEY"]
#         return api_key
#     except:
#         # Fall back to environment variable or user input
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             api_key = st.text_input("Enter Google API Key:", type="password")
#         return api_key

# # Configure API (keep your original logic)
# api_key = get_google_api_key()
# if api_key:
#     genai.configure(api_key=api_key)
# else:
#     st.error("❌ Google API Key required")
#     st.stop()

# # Load model and feature names (EXACTLY like your original)
# try:
#     with open("logistic_regression.pkl", "rb") as f:
#         package = pickle.load(f)
#         model = package["model"]
#         feature_names = package["feature_names"]
# except Exception as e:
#     st.error(f"❌ Failed to load model: {e}")
#     st.stop()

# # Load SHAP background dataset (EXACTLY like your original)
# try:
#     with open("background_data.pkl", "rb") as f:
#         background = pickle.load(f)
# except Exception as e:
#     st.error("❌ Failed to load background data. SHAP values may not be accurate.")
#     background = None

# # App title (EXACTLY like your original)
# st.title("🧪 Semen Analysis Fertility Prediction")

# # Upload (EXACTLY like your original)
# uploaded_file = st.file_uploader("Upload your semen analysis report (PDF/Image)", type=["pdf", "jpg", "png"])

# if uploaded_file:
#     file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name

#     st.info("🔍 Extracting data from uploaded file...")

#     try:
#         # Extract structured data (EXACTLY like your original)
#         result = extract_structured_data(tmp_file_path)
#         extracted_data = result.model_dump()

#         # Save extracted JSON (EXACTLY like your original)
#         json_path = tmp_file_path.replace(".pdf", ".json")
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(extracted_data, f, indent=2, ensure_ascii=False)

#         st.success("✅ Data extracted successfully!")

#         # Extract values (EXACTLY like your original)
#         volume = extracted_data['semen_analysis']['volume']['value']
#         concentration = extracted_data['semen_analysis']['concentration']['value']
#         motility = extracted_data['semen_analysis']['motility']['value']

#         # Create input DataFrame (EXACTLY like your original)
#         input_data = pd.DataFrame([{
#             feature_names[0]: volume,
#             feature_names[1]: concentration,
#             feature_names[2]: motility
#         }])

#         # Predict fertility probability (EXACTLY like your original)
#         probability = model.predict_proba(input_data)[0][1]

#         st.subheader("🎯 Fertility Score")
#         st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{probability*100:.2f}%</h1>", unsafe_allow_html=True)

#         # Sidebar interpretation (EXACTLY like your original)
#         st.sidebar.title("📊 Interpretation Guide")
#         st.sidebar.subheader("0-40%: Low Fertility Probability")
#         st.sidebar.write("Unfavorable indicators. Recommend follow-up consultation.")
#         st.sidebar.subheader("40-70%: Borderline")
#         st.sidebar.write("Results are inconclusive; consider retesting.")
#         st.sidebar.subheader("70-100%: High Fertility Probability")
#         st.sidebar.write("Strong indicators of fertility.")

#         # Table of input values (EXACTLY like your original)
#         st.table(pd.DataFrame({
#             "Label Name": ["Volume", "Concentration", "Motility"],
#             "Value": [volume, concentration, motility]
#         }).set_index("Label Name"))

#         # SHAP Analysis (EXACTLY like your original)
#         st.subheader("🔎 Feature Importance Analysis (SHAP)")
#         if background is not None:
#             explainer = shap.LinearExplainer(model, background)
#             shap_values = explainer.shap_values(input_data)
#             base_value = explainer.expected_value

#             feature_impact = pd.DataFrame({
#                 "Feature": feature_names,
#                 "Value": input_data.iloc[0].values,
#                 "SHAP_Impact": shap_values[0]
#             }).sort_values(by="SHAP_Impact")

#             feature_impact["Abs_Impact"] = np.abs(feature_impact["SHAP_Impact"])
#             colors = ['#ff4d4d' if x < 0 else '#28a745' for x in feature_impact["SHAP_Impact"]]

#             fig, ax = plt.subplots(figsize=(8, 5))
#             bars = ax.barh(feature_impact["Feature"], feature_impact["SHAP_Impact"], color=colors, edgecolor='black')

#             for bar, impact in zip(bars, feature_impact["SHAP_Impact"]):
#                 x_offset = 0.08 if impact < 0 else 0
#                 ax.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height() / 2,
#                         f'{impact:+.2f}', va='center', ha='left', fontsize=10, fontweight='bold')

#             ax.axvline(0, color='black', linewidth=1)
#             ax.set_title("Feature Contribution to Fertility Prediction")
#             ax.set_xlabel("SHAP Impact")
#             ax.grid(axis='x', linestyle='--', alpha=0.4)
#             plt.tight_layout()
#             st.pyplot(fig)
#             plt.close()

#             # Feature details (EXACTLY like your original)
#             def get_impact_strength(abs_impact):
#                 if abs_impact > 0.15:
#                     return "Very Strong"
#                 elif abs_impact > 0.10:
#                     return "Strong"
#                 elif abs_impact > 0.05:
#                     return "Moderate"
#                 else:
#                     return "Mild"

#             st.subheader("📋 DETAILED FEATURE ANALYSIS")
#             st.markdown("---")

#             positive_factors = feature_impact[feature_impact['SHAP_Impact'] >= 0]
#             negative_factors = feature_impact[feature_impact['SHAP_Impact'] < 0]
#             if not positive_factors.empty:
#                 st.markdown("✅ **POSITIVE INFLUENCES (Supporting Fertility)**")
#                 for _, row in positive_factors.iterrows():
#                     st.markdown(f"💪 **{row['Feature']}** = {row['Value']:.2f}")
#                     st.markdown(f"→ {get_impact_strength(row['Abs_Impact'])} **positive** impact (**+{row['SHAP_Impact']:.2f}**)")

#             if not negative_factors.empty:
#                 st.markdown("⚠️ **NEGATIVE INFLUENCES (Areas for Improvement)**")
#                 for _, row in negative_factors.iterrows():
#                     st.markdown(f"🎯 **{row['Feature']}** = {row['Value']:.2f}")
#                     st.markdown(f"→ {get_impact_strength(row['Abs_Impact'])} **negative** impact (**{row['SHAP_Impact']:.2f}**)  \n💡 Consider improving this parameter if possible.")

#             # Prepare JSON for RAG and Saving (EXACTLY like your original)
#             shap_json = {
#                 "fertility_score": round(probability * 100, 2),
#                 "features": {}
#             }

#             for feature, value in zip(feature_names, input_data.iloc[0]):
#                 shap_json["features"][feature] = {
#                     "value": float(value),
#                     "impact": float(shap_values[0][feature_names.index(feature)])
#                 }

#             # ✅ Save report (EXACTLY like your original - keep "me" as user_id)
#             save_new_report(shap_json, user_id="me")

#             # ✅ Compare to previous report (EXACTLY like your original)
#             comparison = compare_to_last(shap_json, user_id="me")
#             st.markdown(comparison)

#             # ✅ RAG Explanation (EXACTLY like your original)
#             try:
#                 st.subheader("📖 AI Fertility Explanation")
#                 with st.spinner("Generating explanation..."):
#                     explanation = generate_rag_explanation()
#                     st.markdown(explanation)
#                     with open("rag_output.txt", "w") as f:
#                         f.write(explanation)
#                 st.success("✅ Explanation complete and saved!")
#             except Exception as e:
#                 st.error(f"❌ Failed to generate explanation: {e}")

#         else:
#             st.warning("⚠️ SHAP explanation not available due to missing background data.")

#     except Exception as e:
#         st.error(f"❌ Error processing the file: {e}")

#     os.remove(tmp_file_path)

# # Footer (EXACTLY like your original)
# st.caption("🧠 Disclaimer: This tool is for educational use only. Not a substitute for professional diagnosis.")
# st.markdown("[Terms of Service](https://herafertility.co/policies/terms-of-service) | [Privacy Policy](https://herafertility.co/policies/privacy-policy)")