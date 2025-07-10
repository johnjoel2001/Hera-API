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
# from dotenv import load_dotenv
# from run_rag_from_json import generate_rag_explanation
# import google.generativeai as genai
# from report_manager import save_new_report, compare_to_last  # 👈 Import here

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)

# # Load model and feature names
# with open("logistic_regression.pkl", "rb") as f:
#     package = pickle.load(f)
#     model = package["model"]
#     feature_names = package["feature_names"]

# # Load SHAP background dataset
# try:
#     with open("background_data.pkl", "rb") as f:
#         background = pickle.load(f)
# except Exception as e:
#     st.error("❌ Failed to load background data. SHAP values may not be accurate.")
#     background = None

# # App title
# st.title("🧪 Semen Analysis Fertility Prediction")

# # Upload
# uploaded_file = st.file_uploader("Upload your semen analysis report (PDF/Image)", type=["pdf", "jpg", "png"])

# if uploaded_file:
#     file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name

#     st.info("🔍 Extracting data from uploaded file...")

#     try:
#         # Extract structured data
#         result = extract_structured_data(tmp_file_path)
#         extracted_data = result.model_dump()

#         # Save extracted JSON
#         json_path = tmp_file_path.replace(".pdf", ".json")
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(extracted_data, f, indent=2, ensure_ascii=False)

#         st.success("✅ Data extracted successfully!")

#         # Extract values
#         volume = extracted_data['semen_analysis']['volume']['value']
#         concentration = extracted_data['semen_analysis']['concentration']['value']
#         motility = extracted_data['semen_analysis']['motility']['value']

#         # Create input DataFrame
#         input_data = pd.DataFrame([{
#             feature_names[0]: volume,
#             feature_names[1]: concentration,
#             feature_names[2]: motility
#         }])

#         # Predict fertility probability
#         probability = model.predict_proba(input_data)[0][1]

#         st.subheader("🎯 Fertility Score")
#         st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{probability*100:.2f}%</h1>", unsafe_allow_html=True)

#         # Sidebar interpretation
#         st.sidebar.title("📊 Interpretation Guide")
#         st.sidebar.subheader("0-40%: Low Fertility Probability")
#         st.sidebar.write("Unfavorable indicators. Recommend follow-up consultation.")
#         st.sidebar.subheader("40-70%: Borderline")
#         st.sidebar.write("Results are inconclusive; consider retesting.")
#         st.sidebar.subheader("70-100%: High Fertility Probability")
#         st.sidebar.write("Strong indicators of fertility.")

#         # Table of input values
#         st.table(pd.DataFrame({
#             "Label Name": ["Volume", "Concentration", "Motility"],
#             "Value": [volume, concentration, motility]
#         }).set_index("Label Name"))

#         # SHAP Analysis
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

#             # Feature details
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

#             # Prepare JSON for RAG and Saving
#             shap_json = {
#                 "fertility_score": round(probability * 100, 2),
#                 "features": {}
#             }

#             for feature, value in zip(feature_names, input_data.iloc[0]):
#                 shap_json["features"][feature] = {
#                     "value": float(value),
#                     "impact": float(shap_values[0][feature_names.index(feature)])
#                 }

#             # ✅ Save report
#             save_new_report(shap_json, user_id="me")

#             # ✅ Compare to previous report
#             comparison = compare_to_last(shap_json, user_id="me")
#             st.markdown(comparison)

#             # ✅ RAG Explanation
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

# # Footer
# st.caption("🧠 Disclaimer: This tool is for educational use only. Not a substitute for professional diagnosis.")
# st.markdown("[Terms of Service](https://herafertility.co/policies/terms-of-service) | [Privacy Policy](https://herafertility.co/policies/privacy-policy)")


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
from run_rag_from_json import generate_rag_explanation
import google.generativeai as genai
from report_manager import save_new_report, compare_to_last

# Streamlit page configuration
st.set_page_config(
    page_title="Fertility Tracker",
    page_icon="🧪",
    layout="wide"
)

# Create directories for cloud deployment
def setup_directories():
    """Create necessary directories for Streamlit Cloud"""
    directories = ["user_reports", "temp_files"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

setup_directories()

# Handle API keys for cloud deployment
def get_google_api_key():
    """Get Google API key from secrets or user input"""
    try:
        # Try Streamlit secrets first
        api_key = st.secrets["GOOGLE_API_KEY"]
        return api_key
    except:
        # Fall back to user input
        with st.sidebar:
            st.header("🔑 API Configuration")
            api_key = st.text_input("Enter Google API Key:", type="password", help="Required for AI analysis")
            if api_key:
                st.success("✅ API Key configured!")
            else:
                st.warning("⚠️ Please enter your Google API Key to use AI features")
        return api_key

# Configure API
api_key = get_google_api_key()
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("❌ Google API Key required for fertility analysis")
    st.stop()

# Load model and feature names
@st.cache_resource
def load_model():
    """Load the ML model with caching for better performance"""
    try:
        with open("logistic_regression.pkl", "rb") as f:
            package = pickle.load(f)
            return package["model"], package["feature_names"]
    except FileNotFoundError:
        st.error("❌ Model file 'logistic_regression.pkl' not found. Please upload it to your repository.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, feature_names = load_model()

# Load SHAP background dataset
@st.cache_resource
def load_background_data():
    """Load SHAP background data with caching"""
    try:
        with open("background_data.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Background data file not found. SHAP values may be less accurate.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading background data: {e}")
        return None

background = load_background_data()

# Session state for user management
if 'user_id' not in st.session_state:
    st.session_state.user_id = "user_" + str(abs(hash(str(st.session_state))))[:8]

# App title and info
st.title("🧪 Semen Analysis Fertility Prediction")
st.markdown("Upload your semen analysis report to get AI-powered fertility insights and track your progress over time.")

# Sidebar with user info and history
with st.sidebar:
    st.header("👤 Your Session")
    st.info(f"User ID: {st.session_state.user_id}")
    
    # Display user history
    try:
        from report_manager import REPORTS_DIR
        if os.path.exists(REPORTS_DIR):
            user_files = [
                f for f in os.listdir(REPORTS_DIR)
                if f.startswith(st.session_state.user_id) and f.endswith(".json") and "latest" not in f
            ]
            
            if user_files:
                st.header("📊 Your History")
                st.write(f"Total reports: {len(user_files)}")
                
                # Show recent reports
                user_files.sort(reverse=True)
                for i, filename in enumerate(user_files[:3]):  # Show last 3
                    try:
                        filepath = os.path.join(REPORTS_DIR, filename)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        with st.expander(f"Report {i+1}"):
                            st.metric("Fertility Score", f"{data.get('fertility_score', 0):.1f}%")
                            # Extract date from filename
                            date_part = filename.split('_')[1:3]
                            if len(date_part) >= 2:
                                st.write(f"Date: {date_part[0]} {date_part[1].replace('-', ':')}")
                    except:
                        continue
            else:
                st.info("📝 No previous reports found. Upload your first report to start tracking!")
    except Exception as e:
        st.error(f"Error loading history: {e}")

# Main upload section
uploaded_file = st.file_uploader(
    "Upload your semen analysis report (PDF/Image)", 
    type=["pdf", "jpg", "png"],
    help="Supported formats: PDF, JPG, PNG"
)

if uploaded_file and api_key:
    # Create temporary file
    file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info("🔍 Extracting data from uploaded file...")

    try:
        # Extract structured data
        result = extract_structured_data(tmp_file_path)
        extracted_data = result.model_dump()

        # Save extracted JSON temporarily
        json_path = tmp_file_path.replace(file_suffix, ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        st.success("✅ Data extracted successfully!")

        # Extract values
        volume = extracted_data['semen_analysis']['volume']['value']
        concentration = extracted_data['semen_analysis']['concentration']['value']
        motility = extracted_data['semen_analysis']['motility']['value']

        # Display extracted values
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volume", f"{volume:.1f} mL")
        with col2:
            st.metric("Concentration", f"{concentration:.1f} M/mL")
        with col3:
            st.metric("Motility", f"{motility:.1f}%")

        # Create input DataFrame
        input_data = pd.DataFrame([{
            feature_names[0]: volume,
            feature_names[1]: concentration,
            feature_names[2]: motility
        }])

        # Predict fertility probability
        probability = model.predict_proba(input_data)[0][1]

        # Display fertility score
        st.subheader("🎯 Fertility Score")
        score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
        with score_col2:
            st.markdown(
                f"<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 1rem 0;'>"
                f"<h1 style='color: white; font-size: 4em; margin: 0;'>{probability*100:.1f}%</h1>"
                f"<p style='color: white; font-size: 1.2em; margin: 0;'>Fertility Probability</p>"
                f"</div>", 
                unsafe_allow_html=True
            )

        # Interpretation guide
        with st.expander("📊 How to Interpret Your Score"):
            st.markdown("""
            - **0-20%**: Very Low - Consider consulting a fertility specialist
            - **20-40%**: Low - Follow-up consultation recommended  
            - **40-60%**: Moderate - Monitor and consider lifestyle improvements
            - **60-80%**: Good - Positive indicators for fertility
            - **80-100%**: High - Strong fertility indicators
            """)

        # SHAP Analysis
        st.subheader("🔎 Feature Importance Analysis (SHAP)")
        if background is not None:
            with st.spinner("Calculating feature impacts..."):
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(input_data)

                feature_impact = pd.DataFrame({
                    "Feature": feature_names,
                    "Value": input_data.iloc[0].values,
                    "SHAP_Impact": shap_values[0]
                }).sort_values(by="SHAP_Impact")

                # Create SHAP plot
                colors = ['#ff4d4d' if x < 0 else '#28a745' for x in feature_impact["SHAP_Impact"]]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(feature_impact["Feature"], feature_impact["SHAP_Impact"], color=colors, edgecolor='black')

                for bar, impact in zip(bars, feature_impact["SHAP_Impact"]):
                    x_offset = 0.01 if impact < 0 else 0.01
                    ax.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height() / 2,
                            f'{impact:+.3f}', va='center', ha='left', fontsize=12, fontweight='bold')

                ax.axvline(0, color='black', linewidth=2)
                ax.set_title("Feature Contribution to Fertility Prediction", fontsize=16, fontweight='bold')
                ax.set_xlabel("SHAP Impact", fontsize=14)
                ax.grid(axis='x', linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Feature analysis details
                def get_impact_strength(abs_impact):
                    if abs_impact > 0.15:
                        return "Very Strong"
                    elif abs_impact > 0.10:
                        return "Strong"
                    elif abs_impact > 0.05:
                        return "Moderate"
                    else:
                        return "Mild"

                st.subheader("📋 Detailed Feature Analysis")
                
                positive_factors = feature_impact[feature_impact['SHAP_Impact'] >= 0]
                negative_factors = feature_impact[feature_impact['SHAP_Impact'] < 0]
                
                if not positive_factors.empty:
                    st.markdown("### ✅ **Positive Influences (Supporting Fertility)**")
                    for _, row in positive_factors.iterrows():
                        st.markdown(f"💪 **{row['Feature']}** = {row['Value']:.2f}")
                        st.markdown(f"→ {get_impact_strength(abs(row['SHAP_Impact']))} **positive** impact (**+{row['SHAP_Impact']:.2f}**)")

                if not negative_factors.empty:
                    st.markdown("### ⚠️ **Negative Influences (Areas for Improvement)**")
                    for _, row in negative_factors.iterrows():
                        st.markdown(f"🎯 **{row['Feature']}** = {row['Value']:.2f}")
                        st.markdown(f"→ {get_impact_strength(abs(row['SHAP_Impact']))} **negative** impact (**{row['SHAP_Impact']:.2f}**)  \n💡 Consider improving this parameter if possible.")

                # Prepare JSON for saving and RAG
                shap_json = {
                    "fertility_score": round(probability * 100, 2),
                    "features": {}
                }

                for feature, value in zip(feature_names, input_data.iloc[0]):
                    shap_json["features"][feature] = {
                        "value": float(value),
                        "impact": float(shap_values[0][feature_names.index(feature)])
                    }

                # Save report with user-specific ID
                try:
                    save_new_report(shap_json, user_id=st.session_state.user_id)
                    st.success("✅ Report saved to your history!")
                except Exception as e:
                    st.warning(f"⚠️ Could not save report: {e}")

                # Compare to previous report
                try:
                    comparison = compare_to_last(shap_json, user_id=st.session_state.user_id)
                    st.markdown(comparison)
                except Exception as e:
                    st.info("📊 No previous report found for comparison.")

                # Generate AI explanation
                try:
                    st.subheader("📖 AI Fertility Explanation")
                    with st.spinner("Generating personalized explanation..."):
                        explanation = generate_rag_explanation()
                        st.markdown(explanation)
                    st.success("✅ AI explanation generated!")
                except Exception as e:
                    st.error(f"❌ Could not generate AI explanation: {e}")
                    st.info("💡 Make sure all required files (FAISS index, etc.) are available.")

        else:
            st.warning("⚠️ SHAP explanation not available due to missing background data.")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
        st.info("💡 Please ensure your file contains valid semen analysis data.")

    finally:
        # Clean up temporary files
        try:
            os.remove(tmp_file_path)
            if 'json_path' in locals():
                os.remove(json_path)
        except:
            pass

# Footer
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.caption("🧠 **Disclaimer:** This tool is for educational use only. Not a substitute for professional medical diagnosis.")
with col2:
    st.markdown(
        "**Links:** [Terms of Service](https://herafertility.co/policies/terms-of-service) | "
        "[Privacy Policy](https://herafertility.co/policies/privacy-policy)"
    )