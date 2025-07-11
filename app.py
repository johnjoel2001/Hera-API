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
import shutil
from datetime import datetime
from pdf_processor import extract_structured_data
import numpy as np
import shap
import matplotlib.pyplot as plt
from run_rag_from_json import generate_rag_explanation
import google.generativeai as genai
from report_manager import save_new_report, compare_to_last

# 🔧 Cloud-compatible API key handling
def get_api_key():
    """Get API key from Streamlit secrets, environment, or user input"""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            return api_key
        return st.text_input("Enter Google API Key:", type="password", help="Required for PDF processing")

# Configure API
api_key = get_api_key()
if not api_key:
    st.error("❌ Google API Key required")
    st.stop()

genai.configure(api_key=api_key)

# Load model and background data
@st.cache_resource
def load_model_and_background():
    """Load model and background data with error handling"""
    try:
        with open("logistic_regression.pkl", "rb") as f:
            package = pickle.load(f)
            model = package["model"]
            feature_names = package["feature_names"]
    except FileNotFoundError:
        st.error("❌ Model file 'logistic_regression.pkl' not found. Please upload it to your repository.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    try:
        with open("background_data.pkl", "rb") as f:
            background = pickle.load(f)
    except Exception as e:
        st.warning("⚠️ Background data not available. SHAP values may be less accurate.")
        background = None
    
    return model, feature_names, background

model, feature_names, background = load_model_and_background()

# Ensure directories exist
os.makedirs("user_reports", exist_ok=True)
os.makedirs("uploaded_pdfs", exist_ok=True)

# PDF Management Functions
def save_uploaded_pdf(uploaded_file):
    """Save uploaded PDF with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"me_{timestamp}_{uploaded_file.name}"
    filepath = os.path.join("uploaded_pdfs", filename)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return filepath, filename

def get_uploaded_pdfs():
    """Get list of uploaded PDFs"""
    if not os.path.exists("uploaded_pdfs"):
        return []
    
    pdf_files = [f for f in os.listdir("uploaded_pdfs") if f.endswith(('.pdf', '.jpg', '.png'))]
    pdf_files.sort(reverse=True)  # Most recent first
    
    pdf_list = []
    for filename in pdf_files:
        filepath = os.path.join("uploaded_pdfs", filename)
        try:
            # Extract timestamp from filename
            parts = filename.split('_')
            if len(parts) >= 3:
                date_part = parts[1]
                time_part = parts[2]
                timestamp = f"{date_part} {time_part.replace('-', ':')}"
            else:
                timestamp = "Unknown"
            
            file_size = os.path.getsize(filepath)
            pdf_list.append({
                'filename': filename,
                'filepath': filepath,
                'timestamp': timestamp,
                'size': file_size
            })
        except:
            continue
    
    return pdf_list

def delete_all_pdfs_and_reports():
    """Delete all PDFs and corresponding reports"""
    try:
        # Delete uploaded PDFs
        if os.path.exists("uploaded_pdfs"):
            shutil.rmtree("uploaded_pdfs")
            os.makedirs("uploaded_pdfs", exist_ok=True)
        
        # Delete user reports
        if os.path.exists("user_reports"):
            user_files = [f for f in os.listdir("user_reports") if f.startswith("me_")]
            for filename in user_files:
                filepath = os.path.join("user_reports", filename)
                os.remove(filepath)
        
        return True
    except Exception as e:
        st.error(f"Error deleting files: {e}")
        return False

# Sidebar for PDF Management
with st.sidebar:
    st.header("📁 PDF Management")
    
    # Show uploaded PDFs
    uploaded_pdfs = get_uploaded_pdfs()
    
    if uploaded_pdfs:
        st.subheader(f"📄 Uploaded PDFs ({len(uploaded_pdfs)})")
        
        for i, pdf_info in enumerate(uploaded_pdfs):
            with st.expander(f"PDF {i+1}: {pdf_info['filename'][:20]}..."):
                st.write(f"**Date:** {pdf_info['timestamp']}")
                st.write(f"**Size:** {pdf_info['size']/1024:.1f} KB")
                st.write(f"**Full name:** {pdf_info['filename']}")
        
        # Delete all button
        st.markdown("---")
        if st.button("🗑️ Delete All PDFs & Reports", type="secondary", use_container_width=True):
            if delete_all_pdfs_and_reports():
                st.success("✅ All PDFs and reports deleted!")
                st.rerun()  # Refresh the page
            else:
                st.error("❌ Failed to delete files")
    else:
        st.info("📄 No PDFs uploaded yet")
    
    # Show current session info
    st.markdown("---")
    st.subheader("ℹ️ Session Info")
    st.write("User ID: **me**")
    
    # Show interpretation guide
    st.markdown("---")
    st.subheader("📊 Interpretation Guide")
    st.write("**0-20%:** Very Low")
    st.write("**20-40%:** Low") 
    st.write("**40-60%:** Moderate")
    st.write("**60-80%:** Good")
    st.write("**80-100%:** High")

# Main App
st.title("🧪 Semen Analysis Fertility Prediction")

# Upload section
uploaded_file = st.file_uploader("Upload your semen analysis report (PDF/Image)", type=["pdf", "jpg", "png"])

if uploaded_file:
    # Save the uploaded PDF
    pdf_path, pdf_filename = save_uploaded_pdf(uploaded_file)
    st.success(f"✅ PDF saved: {pdf_filename}")
    
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

        # Table of input values
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volume", f"{volume:.2f} mL")
        with col2:
            st.metric("Concentration", f"{concentration:.2f} M/mL")
        with col3:
            st.metric("Motility", f"{motility:.2f}%")

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

            # Save report
            save_new_report(shap_json, user_id="me")

            # Compare to previous report
            comparison = compare_to_last(shap_json, user_id="me")
            st.markdown(comparison)

            # RAG Explanation
            try:
                st.subheader("📖 AI Fertility Explanation")
                with st.spinner("Generating explanation..."):
                    explanation = generate_rag_explanation()
                    st.markdown(explanation)
                    try:
                        with open("rag_output.txt", "w") as f:
                            f.write(explanation)
                    except:
                        pass
                st.success("✅ Explanation complete and saved!")
            except Exception as e:
                st.error(f"❌ Failed to generate explanation: {e}")

        else:
            st.warning("⚠️ SHAP explanation not available due to missing background data.")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")

    # Clean up temporary files
    try:
        os.remove(tmp_file_path)
        if 'json_path' in locals():
            os.remove(json_path)
    except:
        pass

# Footer
st.caption("🧠 Disclaimer: This tool is for educational use only. Not a substitute for professional diagnosis.")
st.markdown("[Terms of Service](https://herafertility.co/policies/terms-of-service) | [Privacy Policy](https://herafertility.co/policies/privacy-policy)")