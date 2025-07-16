# import os
# import json
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# def generate_rag_explanation(json_path="user_reports/me_latest.json", faiss_path="faiss_index_both"):
#     os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#     # Load current report
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     score = data["fertility_score"]
#     features = data["features"]

#     # Try loading previous report
#     try:
#         reports = sorted([
#             f for f in os.listdir("user_reports")
#             if f.endswith(".json") and "latest" not in f
#         ])
#         if len(reports) >= 2:
#             with open(os.path.join("user_reports", reports[-2]), "r") as f:
#                 prev_data = json.load(f)
#         else:
#             prev_data = None
#     except Exception:
#         prev_data = None

#     # Format comparison-aware SHAP string
#     def format_change(curr, prev):
#         if curr > prev:
#             return f"🔺 (+{curr - prev:.2f})"
#         elif curr < prev:
#             return f"🔻 ({curr - prev:.2f})"
#         else:
#             return "➖ (no change)"

#     # Build detailed comparison data for each feature
#     feature_details = {}
#     for k, v in features.items():
#         feature_info = {
#             'current_value': v['value'],
#             'impact': v['impact'],
#             'previous_value': None,
#             'change': 'N/A (first report)'
#         }
        
#         if prev_data and k in prev_data["features"]:
#             prev_val = prev_data["features"][k]["value"]
#             feature_info['previous_value'] = prev_val
#             feature_info['change'] = format_change(v['value'], prev_val)
        
#         feature_details[k] = feature_info

#     shap_formatted = f"Fertility score: {score:.2f}%\n\n✅ Positive Factors:\n"
#     for k, v in features.items():
#         if v["impact"] > 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: +{v['impact']:.3f} (positive){change_note}\n"

#     shap_formatted += "\n⚠️ Negative Factors:\n"
#     for k, v in features.items():
#         if v["impact"] < 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: {v['impact']:.3f} (negative){change_note}\n"

#     # Add detailed feature comparison data to the prompt - VERY EXPLICIT FORMAT
#     feature_comparison_text = "\n\n**DETAILED FEATURE COMPARISON DATA - USE THESE EXACT VALUES:**\n"
#     for feature_name, details in feature_details.items():
#         feature_comparison_text += f"\n=== {feature_name.upper()} ===\n"
#         feature_comparison_text += f"PREVIOUS_VALUE_FOR_{feature_name.upper()}: {details['previous_value'] if details['previous_value'] is not None else 'N/A'}\n"
#         feature_comparison_text += f"CURRENT_VALUE_FOR_{feature_name.upper()}: {details['current_value']}\n"
#         feature_comparison_text += f"CHANGE_FOR_{feature_name.upper()}: {details['change']}\n"
#         feature_comparison_text += f"SHAP_IMPACT_FOR_{feature_name.upper()}: {details['impact']:.3f}\n"

#     shap_formatted += feature_comparison_text
#     shap_formatted += "\n\n⚠️ CRITICAL INSTRUCTION: When writing the explanation, you MUST use the exact PREVIOUS_VALUE_FOR_[FEATURE] and CURRENT_VALUE_FOR_[FEATURE] shown above. DO NOT use the same number for both previous and current values."

#     def categorize_score(score):
#         if score < 20:
#             return "very low", "IVF with ICSI is highly likely."
#         elif score < 40:
#             return "low", "IVF (with or without ICSI) is recommended."
#         elif score < 60:
#             return "moderate", "Try IUI and lifestyle improvements."
#         elif score < 80:
#             return "good", "Try naturally or use IUI if urgent."
#         else:
#             return "high", "Timed intercourse and ovulation tracking."

#     category, treatment = categorize_score(score)

#     prompt = PromptTemplate(
#     input_variables=["context", "question", "score", "category", "treatment"],
#     template="""
# A machine learning model predicted a **{category} fertility score of {score:.2f}%**, representing the probability of achieving successful natural pregnancy within the next 12 months.

# IMPORTANT: Do not include any raw research text or fragments from the context at the beginning of your response. Start directly with the structured explanation below.

# {question}

# ---

# ### 🌟 1. Introduction
# - You received a **{category} fertility score of {score:.2f}%**. This means your current chances of natural conception are estimated at this percentage.
# - This is **not a diagnosis**, but a data-driven guide to help you make informed decisions.
# - If this is an improvement from your previous report, it's worth celebrating — even small gains reflect your efforts and potential for continued progress.

# ---

# ### 🔍 2. Detailed Review for Each Factor

# For each factor (e.g., Motility, Concentration, Volume), provide:

# - **Subheading**: e.g., `🌟 1. Motility`
# - **SHAP Impact**: Mention its value and whether it's positive or negative (e.g., "SHAP Impact: -0.547 (moderate negative)").
# - **Value Comparison**:  
#   - `Previous Value: X` (MUST USE PREVIOUS_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE CURRENT VALUE)
#   - `Current Value: Y` (MUST USE CURRENT_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE PREVIOUS VALUE)
#   - `Change: 🔺 (+Z)` or `🔻 (-Z)` (USE THE EXACT CHANGE_FOR_[FEATURE] FROM DATA ABOVE)
# - **Explanation**: Explain how this factor influences fertility and how it contributed to the score.
# - **Appreciation**: If there's an improvement, **gently acknowledge the positive change and encourage continued momentum**.
# - **Suggestions**: Provide friendly, practical tips or medical guidance to improve or maintain the factor (e.g., lifestyle, supplements, hydration).
# - **Cite claims inline**: (Title by Author).

# ⚠️ **CRITICAL**: When displaying Previous Value and Current Value, you MUST:
# 1. For Previous Value: Use PREVIOUS_VALUE_FOR_[FEATURE] from the data section
# 2. For Current Value: Use CURRENT_VALUE_FOR_[FEATURE] from the data section  
# 3. These values MUST BE DIFFERENT (unless there was truly no change)
# 4. Example: If PREVIOUS_VALUE_FOR_MOTILITY: 50.0 and CURRENT_VALUE_FOR_MOTILITY: 43.9, then show "Previous Value: 50.0" and "Current Value: 43.9"

# DO NOT use the same number for both previous and current values.

# Be encouraging and informative — this is a journey, not a test.

# ---

# ### ✅ 3. Summary & Next Steps

# - 📈 **Improved Factors**: List any feature(s) that improved, and congratulate the user for their progress.
# - ⚠️ **Features Needing Focus**: List any factors with strong negative SHAP impacts and how they can be addressed.
# - 🧭 **Personalized Next Steps**: Offer a kind, constructive treatment suggestion based on the score.
# - Final recommendation: **{treatment}**

# ---

# 🎯 **Tone Reminder**: Your tone should be warm, empathetic, and non-judgmental. Think like a supportive coach or fertility companion — informed, caring, and optimistic.

# **Research Context (for background reference only - do not include raw text in response)**:  
# {context}
# """
#     )

#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4000)

#     rag_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         chain_type_kwargs={
#             "prompt": prompt.partial(score=score, category=category, treatment=treatment),
#             "document_variable_name": "context"
#         },
#         input_key="query"
#     )

#     response = rag_chain.invoke({"query": shap_formatted})
#     return response["result"]

#### Removed the IVF/IUI Interpretation 

# import os
# import json
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# def generate_rag_explanation(json_path="user_reports/me_latest.json", faiss_path="faiss_index_both"):
#     os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#     # Load current report
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     score = data["fertility_score"]
#     features = data["features"]

#     # Try loading previous report
#     try:
#         reports = sorted([
#             f for f in os.listdir("user_reports")
#             if f.endswith(".json") and "latest" not in f
#         ])
#         if len(reports) >= 2:
#             with open(os.path.join("user_reports", reports[-2]), "r") as f:
#                 prev_data = json.load(f)
#         else:
#             prev_data = None
#     except Exception:
#         prev_data = None

#     # Format comparison-aware SHAP string
#     def format_change(curr, prev):
#         if curr > prev:
#             return f"🔺 (+{curr - prev:.2f})"
#         elif curr < prev:
#             return f"🔻 ({curr - prev:.2f})"
#         else:
#             return "➖ (no change)"

#     # Build detailed comparison data for each feature
#     feature_details = {}
#     for k, v in features.items():
#         feature_info = {
#             'current_value': v['value'],
#             'impact': v['impact'],
#             'previous_value': None,
#             'change': 'N/A (first report)'
#         }
        
#         if prev_data and k in prev_data["features"]:
#             prev_val = prev_data["features"][k]["value"]
#             feature_info['previous_value'] = prev_val
#             feature_info['change'] = format_change(v['value'], prev_val)
        
#         feature_details[k] = feature_info

#     shap_formatted = f"Fertility score: {score:.2f}%\n\n✅ Positive Factors:\n"
#     for k, v in features.items():
#         if v["impact"] > 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: +{v['impact']:.3f} (positive){change_note}\n"

#     shap_formatted += "\n⚠️ Negative Factors:\n"
#     for k, v in features.items():
#         if v["impact"] < 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: {v['impact']:.3f} (negative){change_note}\n"

#     # Add detailed feature comparison data to the prompt - VERY EXPLICIT FORMAT
#     feature_comparison_text = "\n\n**DETAILED FEATURE COMPARISON DATA - USE THESE EXACT VALUES:**\n"
#     for feature_name, details in feature_details.items():
#         feature_comparison_text += f"\n=== {feature_name.upper()} ===\n"
#         feature_comparison_text += f"PREVIOUS_VALUE_FOR_{feature_name.upper()}: {details['previous_value'] if details['previous_value'] is not None else 'N/A'}\n"
#         feature_comparison_text += f"CURRENT_VALUE_FOR_{feature_name.upper()}: {details['current_value']}\n"
#         feature_comparison_text += f"CHANGE_FOR_{feature_name.upper()}: {details['change']}\n"
#         feature_comparison_text += f"SHAP_IMPACT_FOR_{feature_name.upper()}: {details['impact']:.3f}\n"

#     shap_formatted += feature_comparison_text
#     shap_formatted += "\n\n⚠️ CRITICAL INSTRUCTION: When writing the explanation, you MUST use the exact PREVIOUS_VALUE_FOR_[FEATURE] and CURRENT_VALUE_FOR_[FEATURE] shown above. DO NOT use the same number for both previous and current values."

#     def categorize_score(score):
#         if score < 20:
#             return "very low", ""
#         elif score < 40:
#             return "low", ""
#         elif score < 60:
#             return "moderate", ""
#         elif score < 80:
#             return "good", ""
#         else:
#             return "high", ""

#     category, treatment = categorize_score(score)

#     prompt = PromptTemplate(
#     input_variables=["context", "question", "score", "category", "treatment"],
#     template="""
# A machine learning model predicted a **{category} fertility score of {score:.2f}%**, representing the probability of achieving successful natural pregnancy within the next 12 months.

# IMPORTANT: Do not include any raw research text or fragments from the context at the beginning of your response. Start directly with the structured explanation below.

# {question}

# ---

# ### 🌟 1. Introduction
# - You received a **{category} fertility score of {score:.2f}%**. This means your current chances of natural conception are estimated at this percentage.
# - This is **not a diagnosis**, but a data-driven guide to help you make informed decisions.
# - If this is an improvement from your previous report, it's worth celebrating — even small gains reflect your efforts and potential for continued progress.

# ---

# ### 🔍 2. Detailed Review for Each Factor

# For each factor (e.g., Motility, Concentration, Volume), provide:

# - **Subheading**: e.g., `🌟 1. Motility`
# - **SHAP Impact**: Mention its value and whether it's positive or negative (e.g., "SHAP Impact: -0.547 (moderate negative)").
# - **Value Comparison**:  
#   - `Previous Value: X` (MUST USE PREVIOUS_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE CURRENT VALUE)
#   - `Current Value: Y` (MUST USE CURRENT_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE PREVIOUS VALUE)
#   - `Change: 🔺 (+Z)` or `🔻 (-Z)` (USE THE EXACT CHANGE_FOR_[FEATURE] FROM DATA ABOVE)
# - **Explanation**: Explain how this factor influences fertility and how it contributed to the score.
# - **Appreciation**: If there's an improvement, **gently acknowledge the positive change and encourage continued momentum**.
# - **Suggestions**: Provide friendly, practical tips or medical guidance to improve or maintain the factor (e.g., lifestyle, supplements, hydration).
# -  **Cite claims inline**: (Title by Author).

# ⚠️ **CRITICAL**: When displaying Previous Value and Current Value, you MUST:
# 1. For Previous Value: Use PREVIOUS_VALUE_FOR_[FEATURE] from the data section
# 2. For Current Value: Use CURRENT_VALUE_FOR_[FEATURE] from the data section  
# 3. These values MUST BE DIFFERENT (unless there was truly no change)
# 4. Example: If PREVIOUS_VALUE_FOR_MOTILITY: 50.0 and CURRENT_VALUE_FOR_MOTILITY: 43.9, then show "Previous Value: 50.0" and "Current Value: 43.9"

# DO NOT use the same number for both previous and current values.

# Be encouraging and informative — this is a journey, not a test.

# ---

# ### ✅ 3. Summary & Next Steps

# - 📈 **Improved Factors**: List any feature(s) that improved, and congratulate the user for their progress.
# - ⚠️ **Features Needing Focus**: List any factors with strong negative SHAP impacts and how they can be addressed.
# - 🧭 **Personalized Next Steps**: Offer a kind, constructive treatment suggestion based on the score.


# ---

# 🎯 **Tone Reminder**: Your tone should be warm, empathetic, and non-judgmental. Think like a supportive coach or fertility companion — informed, caring, and optimistic.

# **Research Context (for background reference only - do not include raw text in response)**:  
# {context}
# """
#     )

#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4000)

#     rag_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         chain_type_kwargs={
#             "prompt": prompt.partial(score=score, category=category, treatment=""),
#             "document_variable_name": "context"
#         },
#         input_key="query"
#     )

#     response = rag_chain.invoke({"query": shap_formatted})
#     return response["result"]


##### Added ICC Score Tracking 

# import os
# import json
# import numpy as np
# from scipy import stats
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # Real patient data from your dataset - 11 unique patients with multiple tests
# REFERENCE_PATIENT_DATA = {
#     "patient_1": {"volume": [4.22, 4.22, 4.22], "concentration": [27, 27, 27], "motility": [35, 35, 50]},
#     "patient_2": {"volume": [3.9, 3.6, 4.2], "concentration": [43, 8.5, 3.1], "motility": [47, 59, 42]},
#     "patient_3": {"volume": [4.9, 3.2], "concentration": [47.4, 0], "motility": [69.1, 0]},
#     "patient_4": {"volume": [1.1, 1.5, 1.5, 4.5], "concentration": [11.4, 21.4, 21.4, 41.4], "motility": [71, 71, 71, 71]},
#     "patient_5": {"volume": [6.6, 4.2, 6.6, 4.2], "concentration": [56.2, 4.75, 56.2, 4.75], "motility": [28.5, 28, 28.5, 28]},
#     "patient_6": {"volume": [2.8, 2.8, 1.2, 2.4], "concentration": [61, 61, 80, 64], "motility": [20, 20, 20, 20]},
#     "patient_7": {"volume": [2.8, 2.05, 2.8, 0], "concentration": [121.5, 37.2, 121.5, 57.9], "motility": [70, 59, 70, 90]},
#     "patient_8": {"volume": [1.1, 1.1, 1.1, 0.9], "concentration": [129, 129, 129, 83.3], "motility": [71, 71, 71, 63]},
#     "patient_9": {"volume": [1, 1, 1.8, 2.4], "concentration": [1.9, 1.9, 4, 10], "motility": [10, 10, 10, 30]},
#     "patient_10": {"volume": [8, 3.9, 8, 9], "concentration": [16, 27, 16, 18], "motility": [66, 50, 66, 55]},
#     "patient_11": {"volume": [3, 0, 0, 3], "concentration": [12.85, 12.85, 0, 12.85], "motility": [59.8, 59.8, 0, 59.8]}
# }

# def calculate_true_icc(current_patient_data, parameter="motility"):
#     """
#     Calculate true ICC using reference population data + current patient's multiple tests
    
#     Args:
#         current_patient_data: List of values for current patient [test1, test2, test3, ...]
#         parameter: "volume", "concentration", or "motility"
    
#     Returns:
#         icc_value, interpretation, explanation
#     """
    
#     if len(current_patient_data) < 2:
#         return None, "Insufficient data", "Need at least 2 tests for ICC calculation"
    
#     try:
#         # Step 1: Calculate patient averages from reference data
#         reference_patient_averages = []
#         for patient_id, data in REFERENCE_PATIENT_DATA.items():
#             if parameter in data and len(data[parameter]) > 0:
#                 # Filter out zero values for concentration (measurement errors)
#                 valid_values = [v for v in data[parameter] if v > 0]
#                 if valid_values:
#                     reference_patient_averages.append(np.mean(valid_values))
        
#         # Add current patient's average
#         current_patient_avg = np.mean(current_patient_data)
#         all_patient_averages = reference_patient_averages + [current_patient_avg]
        
#         # Step 2: Calculate Between-Patient Variance (MSB)
#         grand_mean = np.mean(all_patient_averages)
#         between_variance = np.var(all_patient_averages, ddof=1)
        
#         # Step 3: Calculate Within-Patient Variance (MSW)
#         within_variances = []
        
#         # Add within-variance from reference patients
#         for patient_id, data in REFERENCE_PATIENT_DATA.items():
#             if parameter in data and len(data[parameter]) > 1:
#                 valid_values = [v for v in data[parameter] if v > 0]
#                 if len(valid_values) > 1:
#                     patient_variance = np.var(valid_values, ddof=1)
#                     within_variances.append(patient_variance)
        
#         # Add current patient's within-variance
#         if len(current_patient_data) > 1:
#             current_within_var = np.var(current_patient_data, ddof=1)
#             within_variances.append(current_within_var)
        
#         # Average within-variance across all patients
#         within_variance = np.mean(within_variances) if within_variances else 0.001
        
#         # Step 4: Calculate ICC using standard formula
#         # ICC(2,1) = (MSB - MSW) / (MSB + MSW)
#         icc = (between_variance - within_variance) / (between_variance + within_variance)
        
#         # Ensure ICC is between 0 and 1
#         icc = max(0, min(1, icc))
        
#         # Step 5: Interpretation
#         if icc < 0.5:
#             interpretation = "Poor reliability"
#         elif icc < 0.75:
#             interpretation = "Moderate reliability"
#         elif icc < 0.9:
#             interpretation = "Good reliability"
#         else:
#             interpretation = "Excellent reliability"
        
#         # Step 6: Create detailed explanation
#         explanation = {
#             "icc_value": icc,
#             "interpretation": interpretation,
#             "between_variance": between_variance,
#             "within_variance": within_variance,
#             "reference_patients": len(reference_patient_averages),
#             "current_patient_tests": len(current_patient_data),
#             "grand_mean": grand_mean,
#             "current_patient_avg": current_patient_avg
#         }
        
#         return icc, interpretation, explanation
        
#     except Exception as e:
#         return None, "Calculation error", f"Error in ICC calculation: {str(e)}"

# def get_historical_scores(json_path, user_reports_dir="user_reports"):
#     """
#     Extract historical fertility scores and detailed feature data from user reports
#     """
#     scores = []
#     volume_data = []
#     concentration_data = []
#     motility_data = []
    
#     try:
#         current_filename = os.path.basename(json_path)
#         all_files = [f for f in os.listdir(user_reports_dir) if f.endswith(".json")]
#         historical_files = [f for f in all_files if f != current_filename]
#         historical_files.sort()
        
#         print(f"DEBUG: Current file: {current_filename}")
#         print(f"DEBUG: Historical files: {historical_files}")
        
#         for report_file in historical_files:
#             with open(os.path.join(user_reports_dir, report_file), "r") as f:
#                 data = json.load(f)
#                 scores.append(data["fertility_score"])
                
#                 # Extract feature data for ICC calculation
#                 if "features" in data:
#                     features = data["features"]
#                     volume_data.append(features.get("Volume", {}).get("value", 0))
#                     concentration_data.append(features.get("Concentration", {}).get("value", 0))
#                     motility_data.append(features.get("Motility", {}).get("value", 0))
                
#                 print(f"DEBUG: Added score {data['fertility_score']} from {report_file}")
        
#         return {
#             "scores": scores,
#             "volume": volume_data,
#             "concentration": concentration_data,
#             "motility": motility_data
#         }
        
#     except Exception as e:
#         print(f"Error reading historical scores: {e}")
#         return {"scores": [], "volume": [], "concentration": [], "motility": []}

# def generate_rag_explanation(json_path="user_reports/me_latest.json", faiss_path="faiss_index_both"):
#     os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#     # Load current report
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     score = data["fertility_score"]
#     features = data["features"]

#     # Get current test values
#     current_volume = features.get("Volume", {}).get("value", 0)
#     current_concentration = features.get("Concentration", {}).get("value", 0)
#     current_motility = features.get("Motility", {}).get("value", 0)

#     # Try loading previous report
#     try:
#         reports = sorted([
#             f for f in os.listdir("user_reports")
#             if f.endswith(".json") and "latest" not in f
#         ])
#         if len(reports) >= 2:
#             with open(os.path.join("user_reports", reports[-2]), "r") as f:
#                 prev_data = json.load(f)
#         else:
#             prev_data = None
#     except Exception:
#         prev_data = None

#     # Format comparison-aware SHAP string
#     def format_change(curr, prev):
#         if curr > prev:
#             return f"🔺 (+{curr - prev:.2f})"
#         elif curr < prev:
#             return f"🔻 ({curr - prev:.2f})"
#         else:
#             return "➖ (no change)"

#     # Get historical data for TRUE ICC calculation
#     historical_data = get_historical_scores(json_path)
    
#     # Add current values to historical data
#     all_scores = historical_data["scores"] + [score]
#     all_volume = historical_data["volume"] + [current_volume]
#     all_concentration = historical_data["concentration"] + [current_concentration]
#     all_motility = historical_data["motility"] + [current_motility]
    
#     # Check for duplicates
#     if len(all_scores) > 1 and abs(all_scores[-1] - all_scores[-2]) < 0.1:
#         all_scores = historical_data["scores"]
#         all_volume = historical_data["volume"]
#         all_concentration = historical_data["concentration"]
#         all_motility = historical_data["motility"]
#         print(f"DEBUG: Detected duplicate score, using only historical")
    
#     print(f"DEBUG: Final all_scores: {all_scores}")
#     print(f"DEBUG: Volume data: {all_volume}")
#     print(f"DEBUG: Concentration data: {all_concentration}")
#     print(f"DEBUG: Motility data: {all_motility}")
    
#     # Calculate TRUE ICC for each parameter
#     icc_results = {}
#     if len(all_volume) >= 2:
#         volume_icc, volume_interp, volume_details = calculate_true_icc(all_volume, "volume")
#         icc_results["volume"] = {"icc": volume_icc, "interpretation": volume_interp, "details": volume_details}
    
#     if len(all_concentration) >= 2:
#         conc_icc, conc_interp, conc_details = calculate_true_icc(all_concentration, "concentration")
#         icc_results["concentration"] = {"icc": conc_icc, "interpretation": conc_interp, "details": conc_details}
    
#     if len(all_motility) >= 2:
#         motility_icc, motility_interp, motility_details = calculate_true_icc(all_motility, "motility")
#         icc_results["motility"] = {"icc": motility_icc, "interpretation": motility_interp, "details": motility_details}
    
#     # Calculate overall ICC (average of parameters)
#     valid_iccs = [result["icc"] for result in icc_results.values() if result["icc"] is not None]
#     if valid_iccs:
#         overall_icc = np.mean(valid_iccs)
#         if overall_icc < 0.5:
#             overall_interpretation = "Poor reliability"
#         elif overall_icc < 0.75:
#             overall_interpretation = "Moderate reliability"
#         elif overall_icc < 0.9:
#             overall_interpretation = "Good reliability"
#         else:
#             overall_interpretation = "Excellent reliability"
#     else:
#         overall_icc = None
#         overall_interpretation = "Insufficient data"

#     # Add TRUE ICC information to the analysis
#     icc_text = ""
#     if len(all_scores) < 2:
#         icc_text = f"\n\n📊 **TRUE ICC ANALYSIS (Intraclass Correlation Coefficient):**\n"
#         icc_text += f"- This is your first fertility assessment, so we can't measure reliability yet.\n"
#         icc_text += f"- After your next test, we'll calculate your ICC using data from 11 reference patients.\n"
#         icc_text += f"- **What ICC Measures**: How consistent your test results are compared to natural variation between different people.\n"
#         icc_text += f"- Keep uploading reports to build your reliability profile!\n"
#     else:
#         icc_text = f"\n\n📊 **TRUE ICC ANALYSIS (Intraclass Correlation Coefficient):**\n"
#         icc_text += f"- **Your Test History**: {len(all_scores)} tests taken\n"
#         icc_text += f"- **Score Progression**: {' → '.join([f'{score:.1f}%' for score in all_scores])}\n"
        
#         if overall_icc is not None:
#             icc_text += f"- **Overall ICC Score**: {overall_icc:.3f} ({overall_interpretation})\n"
#             icc_text += f"- **Reference Population**: Compared against 11 patients with multiple tests\n"
#             icc_text += f"- **ICC Methodology**: True statistical ICC using between-patient vs within-patient variance\n\n"
            
#             # Parameter-specific ICC results - DETAILED BREAKDOWN
#             icc_text += f"**🔬 Parameter-Specific ICC Results:**\n"
#             for param, result in icc_results.items():
#                 if result["icc"] is not None:
#                     details = result["details"]
#                     icc_text += f"\n**{param.title()} ICC Analysis:**\n"
#                     icc_text += f"  - **ICC Score**: {result['icc']:.3f} ({result['interpretation']})\n"
#                     icc_text += f"  - **Your {param.title()} Tests**: {details['current_patient_tests']} measurements\n"
#                     icc_text += f"  - **Your Average {param.title()}**: {details['current_patient_avg']:.2f}\n"
#                     icc_text += f"  - **Population Average {param.title()}**: {details['grand_mean']:.2f}\n"
#                     icc_text += f"  - **Between-Patient Variance**: {details['between_variance']:.3f}\n"
#                     icc_text += f"  - **Within-Patient Variance**: {details['within_variance']:.3f}\n"
                    
#                     # Clinical interpretation for each parameter
#                     if result['icc'] >= 0.75:
#                         icc_text += f"  - **Clinical Meaning**: Your {param} measurements are highly reliable - changes reflect real biological variation\n"
#                     elif result['icc'] >= 0.5:
#                         icc_text += f"  - **Clinical Meaning**: Your {param} measurements show moderate reliability - focus on trends over time\n"
#                     else:
#                         icc_text += f"  - **Clinical Meaning**: Your {param} measurements show high variability - consider testing conditions\n"
            
#             icc_text += f"\n**🔬 ICC Calculation Details:**\n"
#             icc_text += f"- **Between-Patient Variance**: How much fertility parameters vary between different people\n"
#             icc_text += f"- **Within-Patient Variance**: How much your repeat tests vary from your personal average\n"
#             icc_text += f"- **ICC Formula**: (Between-Variance - Within-Variance) / (Between-Variance + Within-Variance)\n"
#             icc_text += f"- **Clinical Meaning**: ICC > 0.75 means your changes are likely real, not just measurement noise\n\n"
            
#             # Practical explanation
#             if overall_icc >= 0.75:
#                 icc_text += f"- **What This Means**: Your fertility tracking shows excellent reliability. Changes in your scores represent real biological changes, not just testing variation.\n"
#                 icc_text += f"- **Clinical Confidence**: You can trust even small improvements or declines as meaningful changes in your fertility health.\n"
#                 icc_text += f"- **What To Do**: Continue your current approach - your tracking is highly reliable for monitoring progress.\n"
#             elif overall_icc >= 0.5:
#                 icc_text += f"- **What This Means**: Your fertility tracking shows moderate reliability. Larger changes are likely real, but small changes might be testing variation.\n"
#                 icc_text += f"- **Clinical Confidence**: Focus on trends rather than single test results. Look for consistent patterns.\n"
#                 icc_text += f"- **What To Do**: Continue testing to improve reliability. Consider discussing results with your healthcare provider.\n"
#             else:
#                 icc_text += f"- **What This Means**: Your fertility tracking shows high variability. This could mean rapid biological changes or testing inconsistencies.\n"
#                 icc_text += f"- **Clinical Confidence**: Wait for more test results before making major decisions. Focus on overall trends.\n"
#                 icc_text += f"- **What To Do**: Consider standardizing testing conditions (same lab, same time of day) and continue regular monitoring.\n"
        
#         # Add interpretation guide
#         icc_text += f"\n**📊 Understanding ICC Scores:**\n"
#         icc_text += f"- **0.0-0.5**: Poor reliability - High variability, focus on major trends only\n"
#         icc_text += f"- **0.5-0.75**: Moderate reliability - Good for tracking significant changes\n"
#         icc_text += f"- **0.75-0.9**: Good reliability - Can confidently interpret most changes\n"
#         icc_text += f"- **0.9-1.0**: Excellent reliability - Even small changes are highly meaningful\n"
    
#     # Rest of your existing code for feature comparison...
#     # [Continue with existing feature details, SHAP formatting, etc.]
    
#     # Build detailed comparison data for each feature
#     feature_details = {}
#     for k, v in features.items():
#         feature_info = {
#             'current_value': v['value'],
#             'impact': v['impact'],
#             'previous_value': None,
#             'change': 'N/A (first report)'
#         }
        
#         if prev_data and k in prev_data["features"]:
#             prev_val = prev_data["features"][k]["value"]
#             feature_info['previous_value'] = prev_val
#             feature_info['change'] = format_change(v['value'], prev_val)
        
#         feature_details[k] = feature_info

#     shap_formatted = f"Fertility score: {score:.2f}%\n\n✅ Positive Factors:\n"
#     for k, v in features.items():
#         if v["impact"] > 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: +{v['impact']:.3f} (positive){change_note}\n"

#     shap_formatted += "\n⚠️ Negative Factors:\n"
#     for k, v in features.items():
#         if v["impact"] < 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: {v['impact']:.3f} (negative){change_note}\n"

#     # Add detailed feature comparison data
#     feature_comparison_text = "\n\n**DETAILED FEATURE COMPARISON DATA - USE THESE EXACT VALUES:**\n"
#     for feature_name, details in feature_details.items():
#         feature_comparison_text += f"\n=== {feature_name.upper()} ===\n"
#         feature_comparison_text += f"PREVIOUS_VALUE_FOR_{feature_name.upper()}: {details['previous_value'] if details['previous_value'] is not None else 'N/A'}\n"
#         feature_comparison_text += f"CURRENT_VALUE_FOR_{feature_name.upper()}: {details['current_value']}\n"
#         feature_comparison_text += f"CHANGE_FOR_{feature_name.upper()}: {details['change']}\n"
#         feature_comparison_text += f"SHAP_IMPACT_FOR_{feature_name.upper()}: {details['impact']:.3f}\n"

#     shap_formatted += feature_comparison_text
#     shap_formatted += icc_text  # Add TRUE ICC analysis
#     shap_formatted += "\n\n⚠️ CRITICAL INSTRUCTION: When writing the explanation, you MUST use the exact PREVIOUS_VALUE_FOR_[FEATURE] and CURRENT_VALUE_FOR_[FEATURE] shown above. DO NOT use the same number for both previous and current values."

#     def categorize_score(score):
#         if score < 20:
#             return "very low", ""
#         elif score < 40:
#             return "low", ""
#         elif score < 60:
#             return "moderate", ""
#         elif score < 80:
#             return "good", ""
#         else:
#             return "high", ""

#     category, treatment = categorize_score(score)

#     # [Rest of your existing prompt template code...]
    
#     prompt = PromptTemplate(
#         input_variables=["context", "question", "score", "category", "treatment"],
#         template="""
# A machine learning model predicted a **{category} fertility score of {score:.2f}%**, representing the probability of achieving successful natural pregnancy within the next 12 months.

# IMPORTANT: Do not include any raw research text or fragments from the context at the beginning of your response. Start directly with the structured explanation below.

# {question}

# ---

# ### 🌟 1. Introduction
# - You received a **{category} fertility score of {score:.2f}%**. This means your current chances of natural conception are estimated at this percentage.
# - This is **not a diagnosis**, but a data-driven guide to help you make informed decisions.
# - If this is an improvement from your previous report, it's worth celebrating — even small gains reflect your efforts and potential for continued progress.

# ---

# ### 🔍 2. Detailed Review for Each Factor

# For each factor (e.g., Motility, Concentration, Volume), provide:

# - **Subheading**: e.g., `🌟 1. Motility`
# - **SHAP Impact**: Mention its value and whether it's positive or negative (e.g., "SHAP Impact: -0.547 (moderate negative)").
# - **Value Comparison**:  
#   - `Previous Value: X` (MUST USE PREVIOUS_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE CURRENT VALUE)
#   - `Current Value: Y` (MUST USE CURRENT_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE PREVIOUS VALUE)
#   - `Change: 🔺 (+Z)` or `🔻 (-Z)` (USE THE EXACT CHANGE_FOR_[FEATURE] FROM DATA ABOVE)
# - **Explanation**: Explain how this factor influences fertility and how it contributed to the score.
# - **Appreciation**: If there's an improvement, **gently acknowledge the positive change and encourage continued momentum**.
# - **Suggestions**: Provide friendly, practical tips or medical guidance to improve or maintain the factor (e.g., lifestyle, supplements, hydration).

# ⚠️ **CRITICAL**: When displaying Previous Value and Current Value, you MUST:
# 1. For Previous Value: Use PREVIOUS_VALUE_FOR_[FEATURE] from the data section
# 2. For Current Value: Use CURRENT_VALUE_FOR_[FEATURE] from the data section  
# 3. These values MUST BE DIFFERENT (unless there was truly no change)
# 4. Example: If PREVIOUS_VALUE_FOR_MOTILITY: 50.0 and CURRENT_VALUE_FOR_MOTILITY: 43.9, then show "Previous Value: 50.0" and "Current Value: 43.9"

# DO NOT use the same number for both previous and current values.

# Be encouraging and informative — this is a journey, not a test.

# ---

# ### 📊 True ICC Analysis (Intraclass Correlation Coefficient)

# If this is the first assessment, explain:
# - **First Test**: This is your baseline - we need more data to measure reliability
# - **Next Steps**: Upload future reports to calculate ICC using reference population data
# - **Why ICC Matters**: Distinguishes real biological changes from measurement variation

# If multiple assessments are available, include:
# - **Overall ICC Score**: Combined reliability across all parameters
# - **Volume ICC**: Specific reliability for semen volume measurements
# - **Concentration ICC**: Specific reliability for sperm concentration measurements  
# - **Motility ICC**: Specific reliability for sperm motility measurements
# - **ICC Methodology**: Explain how true ICC uses between-patient vs within-patient variance
# - **Reference Population**: Mention comparison with 11 patients with multiple tests
# - **Clinical Interpretation**: What each ICC score means for trusting parameter changes
# - **Parameter Analysis**: Show variance components and averages for each parameter
# - **Reliability Guide**: ICC scale interpretation (0.0-1.0) for each parameter

# ---

# ### ✅ Summary & Next Steps

# - 📈 **Improved Factors**: List any feature(s) that improved, and congratulate the user for their progress.
# - ⚠️ **Features Needing Focus**: List any factors with strong negative SHAP impacts and how they can be addressed.
# - 🧭 **Personalized Next Steps**: Offer a kind, constructive treatment suggestion based on the score.

# ---

# 🎯 **Tone Reminder**: Your tone should be warm, empathetic, and non-judgmental. Think like a supportive coach or fertility companion — informed, caring, and optimistic.

# **Research Context (for background reference only - do not include raw text in response)**:  
# {context}
# """
#     )

#     # [Rest of your existing RAG chain code...]
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4000)

#     rag_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         chain_type_kwargs={
#             "prompt": prompt.partial(score=score, category=category, treatment=""),
#             "document_variable_name": "context"
#         },
#         input_key="query"
#     )

#     response = rag_chain.invoke({"query": shap_formatted})
#     return response["result"]


############# ICC (Detailed Interpretation Added) ####################

# import os
# import json
# import numpy as np
# from scipy import stats
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # Real patient data from your dataset - 11 unique patients with multiple tests
# REFERENCE_PATIENT_DATA = {
#     "patient_1": {"volume": [4.22, 4.22, 4.22], "concentration": [27, 27, 27], "motility": [35, 35, 50]},
#     "patient_2": {"volume": [3.9, 3.6, 4.2], "concentration": [43, 8.5, 3.1], "motility": [47, 59, 42]},
#     "patient_3": {"volume": [4.9, 3.2], "concentration": [47.4, 0], "motility": [69.1, 0]},
#     "patient_4": {"volume": [1.1, 1.5, 1.5, 4.5], "concentration": [11.4, 21.4, 21.4, 41.4], "motility": [71, 71, 71, 71]},
#     "patient_5": {"volume": [6.6, 4.2, 6.6, 4.2], "concentration": [56.2, 4.75, 56.2, 4.75], "motility": [28.5, 28, 28.5, 28]},
#     "patient_6": {"volume": [2.8, 2.8, 1.2, 2.4], "concentration": [61, 61, 80, 64], "motility": [20, 20, 20, 20]},
#     "patient_7": {"volume": [2.8, 2.05, 2.8, 0], "concentration": [121.5, 37.2, 121.5, 57.9], "motility": [70, 59, 70, 90]},
#     "patient_8": {"volume": [1.1, 1.1, 1.1, 0.9], "concentration": [129, 129, 129, 83.3], "motility": [71, 71, 71, 63]},
#     "patient_9": {"volume": [1, 1, 1.8, 2.4], "concentration": [1.9, 1.9, 4, 10], "motility": [10, 10, 10, 30]},
#     "patient_10": {"volume": [8, 3.9, 8, 9], "concentration": [16, 27, 16, 18], "motility": [66, 50, 66, 55]},
#     "patient_11": {"volume": [3, 0, 0, 3], "concentration": [12.85, 12.85, 0, 12.85], "motility": [59.8, 59.8, 0, 59.8]}
# }

# def calculate_true_icc(current_patient_data, parameter="motility"):
#     """
#     Calculate true ICC using reference population data + current patient's multiple tests
    
#     Args:
#         current_patient_data: List of values for current patient [test1, test2, test3, ...]
#         parameter: "volume", "concentration", or "motility"
    
#     Returns:
#         icc_value, interpretation, explanation
#     """
    
#     if len(current_patient_data) < 2:
#         return None, "Insufficient data", "Need at least 2 tests for ICC calculation"
    
#     try:
#         # Step 1: Calculate patient averages from reference data
#         reference_patient_averages = []
#         for patient_id, data in REFERENCE_PATIENT_DATA.items():
#             if parameter in data and len(data[parameter]) > 0:
#                 # Filter out zero values for concentration (measurement errors)
#                 valid_values = [v for v in data[parameter] if v > 0]
#                 if valid_values:
#                     reference_patient_averages.append(np.mean(valid_values))
        
#         # Add current patient's average
#         current_patient_avg = np.mean(current_patient_data)
#         all_patient_averages = reference_patient_averages + [current_patient_avg]
        
#         # Step 2: Calculate Between-Patient Variance (MSB)
#         grand_mean = np.mean(all_patient_averages)
#         between_variance = np.var(all_patient_averages, ddof=1)
        
#         # Step 3: Calculate Within-Patient Variance (MSW)
#         within_variances = []
        
#         # Add within-variance from reference patients
#         for patient_id, data in REFERENCE_PATIENT_DATA.items():
#             if parameter in data and len(data[parameter]) > 1:
#                 valid_values = [v for v in data[parameter] if v > 0]
#                 if len(valid_values) > 1:
#                     patient_variance = np.var(valid_values, ddof=1)
#                     within_variances.append(patient_variance)
        
#         # Add current patient's within-variance
#         if len(current_patient_data) > 1:
#             current_within_var = np.var(current_patient_data, ddof=1)
#             within_variances.append(current_within_var)
        
#         # Average within-variance across all patients
#         within_variance = np.mean(within_variances) if within_variances else 0.001
        
#         # Step 4: Calculate ICC using standard formula
#         # ICC(2,1) = (MSB - MSW) / (MSB + MSW)
#         icc = (between_variance - within_variance) / (between_variance + within_variance)
        
#         # Ensure ICC is between 0 and 1
#         icc = max(0, min(1, icc))
        
#         # Step 5: Interpretation
#         if icc < 0.5:
#             interpretation = "Poor reliability"
#         elif icc < 0.75:
#             interpretation = "Moderate reliability"
#         elif icc < 0.9:
#             interpretation = "Good reliability"
#         else:
#             interpretation = "Excellent reliability"
        
#         # Step 6: Create detailed explanation
#         explanation = {
#             "icc_value": icc,
#             "interpretation": interpretation,
#             "between_variance": between_variance,
#             "within_variance": within_variance,
#             "reference_patients": len(reference_patient_averages),
#             "current_patient_tests": len(current_patient_data),
#             "grand_mean": grand_mean,
#             "current_patient_avg": current_patient_avg
#         }
        
#         return icc, interpretation, explanation
        
#     except Exception as e:
#         return None, "Calculation error", f"Error in ICC calculation: {str(e)}"

# def get_historical_scores(json_path, user_reports_dir="user_reports"):
#     """
#     Extract historical fertility scores and detailed feature data from user reports
#     """
#     scores = []
#     volume_data = []
#     concentration_data = []
#     motility_data = []
    
#     try:
#         current_filename = os.path.basename(json_path)
#         all_files = [f for f in os.listdir(user_reports_dir) if f.endswith(".json")]
#         historical_files = [f for f in all_files if f != current_filename]
#         historical_files.sort()
        
#         print(f"DEBUG: Current file: {current_filename}")
#         print(f"DEBUG: Historical files: {historical_files}")
        
#         for report_file in historical_files:
#             with open(os.path.join(user_reports_dir, report_file), "r") as f:
#                 data = json.load(f)
#                 scores.append(data["fertility_score"])
                
#                 # Extract feature data for ICC calculation
#                 if "features" in data:
#                     features = data["features"]
#                     volume_data.append(features.get("Volume", {}).get("value", 0))
#                     concentration_data.append(features.get("Concentration", {}).get("value", 0))
#                     motility_data.append(features.get("Motility", {}).get("value", 0))
                
#                 print(f"DEBUG: Added score {data['fertility_score']} from {report_file}")
        
#         return {
#             "scores": scores,
#             "volume": volume_data,
#             "concentration": concentration_data,
#             "motility": motility_data
#         }
        
#     except Exception as e:
#         print(f"Error reading historical scores: {e}")
#         return {"scores": [], "volume": [], "concentration": [], "motility": []}

# def generate_rag_explanation(json_path="user_reports/me_latest.json", faiss_path="faiss_index_both"):
#     os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#     # Load current report
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     score = data["fertility_score"]
#     features = data["features"]

#     # Get current test values
#     current_volume = features.get("Volume", {}).get("value", 0)
#     current_concentration = features.get("Concentration", {}).get("value", 0)
#     current_motility = features.get("Motility", {}).get("value", 0)

#     # Try loading previous report
#     try:
#         reports = sorted([
#             f for f in os.listdir("user_reports")
#             if f.endswith(".json") and "latest" not in f
#         ])
#         if len(reports) >= 2:
#             with open(os.path.join("user_reports", reports[-2]), "r") as f:
#                 prev_data = json.load(f)
#         else:
#             prev_data = None
#     except Exception:
#         prev_data = None

#     # Format comparison-aware SHAP string
#     def format_change(curr, prev):
#         if curr > prev:
#             return f"🔺 (+{curr - prev:.2f})"
#         elif curr < prev:
#             return f"🔻 ({curr - prev:.2f})"
#         else:
#             return "➖ (no change)"

#     # Get historical data for TRUE ICC calculation
#     historical_data = get_historical_scores(json_path)
    
#     # Add current values to historical data
#     all_scores = historical_data["scores"] + [score]
#     all_volume = historical_data["volume"] + [current_volume]
#     all_concentration = historical_data["concentration"] + [current_concentration]
#     all_motility = historical_data["motility"] + [current_motility]
    
#     # Check for duplicates
#     if len(all_scores) > 1 and abs(all_scores[-1] - all_scores[-2]) < 0.1:
#         all_scores = historical_data["scores"]
#         all_volume = historical_data["volume"]
#         all_concentration = historical_data["concentration"]
#         all_motility = historical_data["motility"]
#         print(f"DEBUG: Detected duplicate score, using only historical")
    
#     print(f"DEBUG: Final all_scores: {all_scores}")
#     print(f"DEBUG: Volume data: {all_volume}")
#     print(f"DEBUG: Concentration data: {all_concentration}")
#     print(f"DEBUG: Motility data: {all_motility}")
    
#     # Calculate TRUE ICC for each parameter
#     icc_results = {}
#     if len(all_volume) >= 2:
#         volume_icc, volume_interp, volume_details = calculate_true_icc(all_volume, "volume")
#         icc_results["volume"] = {"icc": volume_icc, "interpretation": volume_interp, "details": volume_details}
    
#     if len(all_concentration) >= 2:
#         conc_icc, conc_interp, conc_details = calculate_true_icc(all_concentration, "concentration")
#         icc_results["concentration"] = {"icc": conc_icc, "interpretation": conc_interp, "details": conc_details}
    
#     if len(all_motility) >= 2:
#         motility_icc, motility_interp, motility_details = calculate_true_icc(all_motility, "motility")
#         icc_results["motility"] = {"icc": motility_icc, "interpretation": motility_interp, "details": motility_details}
    
#     # Calculate overall ICC (average of parameters)
#     valid_iccs = [result["icc"] for result in icc_results.values() if result["icc"] is not None]
#     if valid_iccs:
#         overall_icc = np.mean(valid_iccs)
#         if overall_icc < 0.5:
#             overall_interpretation = "Poor reliability"
#         elif overall_icc < 0.75:
#             overall_interpretation = "Moderate reliability"
#         elif overall_icc < 0.9:
#             overall_interpretation = "Good reliability"
#         else:
#             overall_interpretation = "Excellent reliability"
#     else:
#         overall_icc = None
#         overall_interpretation = "Insufficient data"

#     # Add TRUE ICC information to the analysis
#     icc_text = ""
#     if len(all_scores) < 2:
#         icc_text = f"\n\n📊 **TRUE ICC ANALYSIS (Intraclass Correlation Coefficient):**\n"
#         icc_text += f"- This is your first fertility assessment, so we can't measure reliability yet.\n"
#         icc_text += f"- After your next test, we'll calculate your ICC using data from 11 reference patients.\n"
#         icc_text += f"- **What ICC Measures**: How consistent your test results are compared to natural variation between different people.\n"
#         icc_text += f"- Keep uploading reports to build your reliability profile!\n"
#     else:
#         icc_text = f"\n\n📊 **TRUE ICC ANALYSIS (Intraclass Correlation Coefficient):**\n"
#         icc_text += f"- **Your Test History**: {len(all_scores)} tests taken\n"
#         icc_text += f"- **Score Progression**: {' → '.join([f'{score:.1f}%' for score in all_scores])}\n"
        
#         if overall_icc is not None:
#             icc_text += f"- **Overall ICC Score**: {overall_icc:.3f} ({overall_interpretation})\n"
#             icc_text += f"- **Reference Population**: Compared against 11 patients with multiple tests\n"
#             icc_text += f"- **ICC Methodology**: True statistical ICC using between-patient vs within-patient variance\n\n"
            
#             # Parameter-specific ICC results - DETAILED BREAKDOWN
#             icc_text += f"**🔬 Parameter-Specific ICC Results:**\n"
#             for param, result in icc_results.items():
#                 if result["icc"] is not None:
#                     details = result["details"]
#                     icc_text += f"\n**{param.title()} ICC Analysis:**\n"
#                     icc_text += f"  - **ICC Score**: {result['icc']:.3f} ({result['interpretation']})\n"
#                     icc_text += f"  - **Your {param.title()} Tests**: {details['current_patient_tests']} measurements\n"
#                     icc_text += f"  - **Your Average {param.title()}**: {details['current_patient_avg']:.2f}\n"
#                     icc_text += f"  - **Population Average {param.title()}**: {details['grand_mean']:.2f}\n"
#                     icc_text += f"  - **Between-Patient Variance**: {details['between_variance']:.3f}\n"
#                     icc_text += f"  - **Within-Patient Variance**: {details['within_variance']:.3f}\n"
                    
#                     # Clinical interpretation for each parameter
#                     if result['icc'] >= 0.75:
#                         icc_text += f"  - **Clinical Meaning**: Your {param} measurements are highly reliable - changes reflect real biological variation\n"
#                     elif result['icc'] >= 0.5:
#                         icc_text += f"  - **Clinical Meaning**: Your {param} measurements show moderate reliability - focus on trends over time\n"
#                     else:
#                         icc_text += f"  - **Clinical Meaning**: Your {param} measurements show high variability - consider testing conditions\n"
            
#             icc_text += f"\n**🔬 ICC Calculation Details:**\n"
#             icc_text += f"- **Between-Patient Variance**: How much fertility parameters vary between different people\n"
#             icc_text += f"- **Within-Patient Variance**: How much your repeat tests vary from your personal average\n"
#             icc_text += f"- **ICC Formula**: (Between-Variance - Within-Variance) / (Between-Variance + Within-Variance)\n"
#             icc_text += f"- **Clinical Meaning**: ICC > 0.75 means your changes are likely real, not just measurement noise\n\n"
            
#             # Practical explanation based on ICC results
#             icc_text += f"🎯 **What Your ICC Results Mean - Clinical Interpretation:**\n\n"
#             icc_text += f"**📊 Overall Picture:**\n"
#             icc_text += f"Your fertility tracking shows **mixed reliability** - some parameters are trustworthy, others need more data.\n\n"
            
#             icc_text += f"🔍 **Parameter-by-Parameter Analysis:**\n\n"
            
#             # Sort parameters by ICC score for better presentation
#             sorted_params = sorted(icc_results.items(), key=lambda x: x[1]["icc"] if x[1]["icc"] is not None else 0, reverse=True)
            
#             for param, result in sorted_params:
#                 if result["icc"] is not None:
#                     icc_val = result["icc"]
#                     interpretation = result["interpretation"]
                    
#                     # Determine emoji and status
#                     if icc_val >= 0.75:
#                         emoji = "🎯"
#                         status = "✅"
#                         confidence = "High confidence"
#                         percentage = f"{icc_val*100:.0f}%"
#                         noise_percentage = f"{(1-icc_val)*100:.0f}%"
                        
#                         # Clinical actions for high reliability
#                         if param == "motility":
#                             actions = [
#                                 "**Trust motility trends** - if it improves, celebrate! If it drops, investigate why",
#                                 "Changes likely reflect real improvements in sperm movement quality",
#                                 "This parameter is your **most reliable tracker**"
#                             ]
#                         elif param == "volume":
#                             actions = [
#                                 "**Trust volume changes** - these reflect real biological variation",
#                                 "Volume improvements likely indicate better reproductive health",
#                                 "Use volume as a reliable indicator of progress"
#                             ]
#                         else:  # concentration
#                             actions = [
#                                 "**Trust concentration trends** - these reflect real sperm production changes",
#                                 "Concentration improvements indicate better sperm production",
#                                 "Use concentration as a key fertility indicator"
#                             ]
                            
#                     elif icc_val >= 0.5:
#                         emoji = "📊"
#                         status = "⚖️"
#                         confidence = "Moderate confidence"
#                         percentage = f"{icc_val*100:.0f}%"
#                         noise_percentage = f"{(1-icc_val)*100:.0f}%"
                        
#                         # Clinical actions for moderate reliability
#                         actions = [
#                             f"**Look for patterns** - don't react to single {param} results",
#                             f"Focus on trends over 2-3 tests for {param}",
#                             f"Moderate confidence in {param} improvements/declines"
#                         ]
                        
#                     else:  # < 0.5
#                         emoji = "🔻"
#                         status = "⚠️"
#                         confidence = "Low confidence"
#                         percentage = f"{icc_val*100:.0f}%"
#                         noise_percentage = f"{(1-icc_val)*100:.0f}%"
                        
#                         # Clinical actions for poor reliability
#                         if param == "volume":
#                             actions = [
#                                 "**Don't focus on single volume results** - look for consistent patterns over 3-4 tests",
#                                 "Volume varies with hydration, abstinence time, collection method",
#                                 "Consider standardizing: same lab, same collection conditions"
#                             ]
#                         elif param == "concentration":
#                             actions = [
#                                 "**Wait for trends** - don't react to single concentration results",
#                                 "Consider same-day repeat testing if results seem unusual",
#                                 "Focus on consistent patterns over multiple months"
#                             ]
#                         else:  # motility
#                             actions = [
#                                 f"**Be cautious with {param} changes** - high measurement variability",
#                                 f"Look for consistent {param} patterns over multiple tests",
#                                 f"Consider testing conditions and lab consistency"
#                             ]
                    
#                     # Format the parameter analysis
#                     icc_text += f"**{emoji} {param.title()} ICC = {icc_val:.3f} ({interpretation}) {status}**\n"
#                     icc_text += f"**What this means:**\n"
#                     icc_text += f"* **{percentage}** of your {param} changes are **real biological changes**\n"
#                     icc_text += f"* Only **{noise_percentage}** is testing noise/variation\n"
#                     icc_text += f"* **{confidence}** in {param} improvements or declines\n\n"
                    
#                     icc_text += f"**Clinical Action:**\n"
#                     for action in actions:
#                         icc_text += f"* {action}\n"
#                     icc_text += f"\n"
            
#             # Clinical inference summary
#             icc_text += f"🎯 **Clinical Inference Summary:**\n\n"
            
#             # What you CAN trust
#             reliable_params = [param for param, result in icc_results.items() 
#                              if result["icc"] is not None and result["icc"] >= 0.75]
#             moderate_params = [param for param, result in icc_results.items() 
#                              if result["icc"] is not None and 0.5 <= result["icc"] < 0.75]
            
#             icc_text += f"**✅ What You CAN Trust:**\n"
#             if reliable_params:
#                 for param in reliable_params:
#                     icc_text += f"* **{param.title()} changes** - these likely reflect real fertility improvements/declines\n"
#             if moderate_params:
#                 for param in moderate_params:
#                     icc_text += f"* **{param.title()} trends** (moderate reliability)\n"
#             if overall_icc >= 0.5:
#                 icc_text += f"* **Overall fertility score trends** (moderate reliability at {overall_icc*100:.0f}%)\n"
#             icc_text += f"\n"
            
#             # What to be cautious about
#             unreliable_params = [param for param, result in icc_results.items() 
#                                if result["icc"] is not None and result["icc"] < 0.5]
            
#             icc_text += f"**⚠️ What to Be Cautious About:**\n"
#             for param in unreliable_params:
#                 icc_text += f"* **Single {param} measurements** - too much random variation\n"
#             if unreliable_params:
#                 param_list = " and ".join([param for param in unreliable_params])
#                 icc_text += f"* **Short-term fluctuations** in {param_list}\n"
#             icc_text += f"\n"
            
#             # Practical recommendations
#             icc_text += f"📈 **Practical Recommendations:**\n\n"
            
#             # For reliable parameters
#             if reliable_params:
#                 for param in reliable_params:
#                     icc_text += f"**For {param.title()} (Reliable):**\n"
#                     icc_text += f"* **Continue what's working** if {param} is improving\n"
#                     icc_text += f"* **Investigate causes** if {param} consistently drops\n"
#                     icc_text += f"* **Make decisions** based on {param} trends\n\n"
            
#             # For unreliable parameters
#             if unreliable_params:
#                 param_list = " & ".join([param.title() for param in unreliable_params])
#                 icc_text += f"**For {param_list} (Unreliable):**\n"
#                 icc_text += f"* **Test consistently:** same lab, same abstinence time, same collection method\n"
#                 icc_text += f"* **Look for 3-month trends** rather than single results\n"
#                 icc_text += f"* **Consider repeat testing** if results seem dramatically different\n"
#                 icc_text += f"* **Don't make major decisions** based on {' or '.join(unreliable_params)} alone\n\n"
            
#             # Overall strategy
#             icc_text += f"**Overall Strategy:**\n"
#             if reliable_params:
#                 primary_param = reliable_params[0]  # Most reliable parameter
#                 icc_text += f"* **Use {primary_param} as your primary tracker** (most reliable)\n"
#             icc_text += f"* **Combine all parameters** for overall fertility assessment\n"
#             icc_text += f"* **Focus on lifestyle factors** that improve all parameters\n"
#             if unreliable_params:
#                 param_list = "/".join(unreliable_params)
#                 icc_text += f"* **Get more tests** to improve reliability of {param_list}\n"
#             icc_text += f"\n"
            
#             # Bottom line
#             icc_text += f"💡 **Bottom Line:**\n\n"
#             if reliable_params:
#                 primary_param = reliable_params[0]
#                 icc_text += f"**Your {primary_param} tracking is excellent** - trust those changes! "
#             if unreliable_params:
#                 param_list = " and ".join(unreliable_params)
#                 icc_text += f"**{param_list.title()} need more data** - be patient and consistent with testing to see real patterns emerge.\n\n"
            
#             icc_text += f"This pattern is actually **common in fertility testing** - motility tends to be more consistent than volume/concentration measurements! 🎯\n\n"
        
#         # Add interpretation guide
#         icc_text += f"\n**📊 Understanding ICC Scores:**\n"
#         icc_text += f"- **0.0-0.5**: Poor reliability - High variability, focus on major trends only\n"
#         icc_text += f"- **0.5-0.75**: Moderate reliability - Good for tracking significant changes\n"
#         icc_text += f"- **0.75-0.9**: Good reliability - Can confidently interpret most changes\n"
#         icc_text += f"- **0.9-1.0**: Excellent reliability - Even small changes are highly meaningful\n"
    
#     # Rest of your existing code for feature comparison...
#     # [Continue with existing feature details, SHAP formatting, etc.]
    
#     # Build detailed comparison data for each feature
#     feature_details = {}
#     for k, v in features.items():
#         feature_info = {
#             'current_value': v['value'],
#             'impact': v['impact'],
#             'previous_value': None,
#             'change': 'N/A (first report)'
#         }
        
#         if prev_data and k in prev_data["features"]:
#             prev_val = prev_data["features"][k]["value"]
#             feature_info['previous_value'] = prev_val
#             feature_info['change'] = format_change(v['value'], prev_val)
        
#         feature_details[k] = feature_info

#     shap_formatted = f"Fertility score: {score:.2f}%\n\n✅ Positive Factors:\n"
#     for k, v in features.items():
#         if v["impact"] > 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: +{v['impact']:.3f} (positive){change_note}\n"

#     shap_formatted += "\n⚠️ Negative Factors:\n"
#     for k, v in features.items():
#         if v["impact"] < 0:
#             change_note = ""
#             if prev_data:
#                 prev_val = prev_data["features"].get(k, {}).get("value")
#                 if prev_val is not None:
#                     change_note = f" {format_change(v['value'], prev_val)}"
#             shap_formatted += f"- {k}: {v['value']} → SHAP Impact: {v['impact']:.3f} (negative){change_note}\n"

#     # Add detailed feature comparison data
#     feature_comparison_text = "\n\n**DETAILED FEATURE COMPARISON DATA - USE THESE EXACT VALUES:**\n"
#     for feature_name, details in feature_details.items():
#         feature_comparison_text += f"\n=== {feature_name.upper()} ===\n"
#         feature_comparison_text += f"PREVIOUS_VALUE_FOR_{feature_name.upper()}: {details['previous_value'] if details['previous_value'] is not None else 'N/A'}\n"
#         feature_comparison_text += f"CURRENT_VALUE_FOR_{feature_name.upper()}: {details['current_value']}\n"
#         feature_comparison_text += f"CHANGE_FOR_{feature_name.upper()}: {details['change']}\n"
#         feature_comparison_text += f"SHAP_IMPACT_FOR_{feature_name.upper()}: {details['impact']:.3f}\n"

#     shap_formatted += feature_comparison_text
#     shap_formatted += icc_text  # Add TRUE ICC analysis
#     shap_formatted += "\n\n⚠️ CRITICAL INSTRUCTION: When writing the explanation, you MUST use the exact PREVIOUS_VALUE_FOR_[FEATURE] and CURRENT_VALUE_FOR_[FEATURE] shown above. DO NOT use the same number for both previous and current values."
#     shap_formatted += "\n\n🎯 CRITICAL ICC INSTRUCTION: You MUST include the COMPLETE ICC interpretation that appears above, including all sections: 'What Your ICC Results Mean', 'Parameter-by-Parameter Analysis', 'Clinical Inference Summary', 'Practical Recommendations', and 'Bottom Line'. Do NOT summarize or shorten this content."

#     def categorize_score(score):
#         if score < 20:
#             return "very low", ""
#         elif score < 40:
#             return "low", ""
#         elif score < 60:
#             return "moderate", ""
#         elif score < 80:
#             return "good", ""
#         else:
#             return "high", ""

#     category, treatment = categorize_score(score)

#     # [Rest of your existing prompt template code...]
    
#     prompt = PromptTemplate(
#         input_variables=["context", "question", "score", "category", "treatment"],
#         template="""
# A machine learning model predicted a **{category} fertility score of {score:.2f}%**, representing the probability of achieving successful natural pregnancy within the next 12 months.

# IMPORTANT: Do not include any raw research text or fragments from the context at the beginning of your response. Start directly with the structured explanation below.

# {question}

# ---

# ### 🌟 1. Introduction
# - You received a **{category} fertility score of {score:.2f}%**. This means your current chances of natural conception are estimated at this percentage.
# - This is **not a diagnosis**, but a data-driven guide to help you make informed decisions.
# - If this is an improvement from your previous report, it's worth celebrating — even small gains reflect your efforts and potential for continued progress.

# ---

# ### 🔍 2. Detailed Review for Each Factor

# For each factor (e.g., Motility, Concentration, Volume), provide:

# - **Subheading**: e.g., `🌟 1. Motility`
# - **SHAP Impact**: Mention its value and whether it's positive or negative (e.g., "SHAP Impact: -0.547 (moderate negative)").
# - **Value Comparison**:  
#   - `Previous Value: X` (MUST USE PREVIOUS_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE CURRENT VALUE)
#   - `Current Value: Y` (MUST USE CURRENT_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE PREVIOUS VALUE)
#   - `Change: 🔺 (+Z)` or `🔻 (-Z)` (USE THE EXACT CHANGE_FOR_[FEATURE] FROM DATA ABOVE)
# - **Explanation**: Explain how this factor influences fertility and how it contributed to the score.
# - **Appreciation**: If there's an improvement, **gently acknowledge the positive change and encourage continued momentum**.
# - **Suggestions**: Provide friendly, practical tips or medical guidance to improve or maintain the factor (e.g., lifestyle, supplements, hydration).

# ⚠️ **CRITICAL**: When displaying Previous Value and Current Value, you MUST:
# 1. For Previous Value: Use PREVIOUS_VALUE_FOR_[FEATURE] from the data section
# 2. For Current Value: Use CURRENT_VALUE_FOR_[FEATURE] from the data section  
# 3. These values MUST BE DIFFERENT (unless there was truly no change)
# 4. Example: If PREVIOUS_VALUE_FOR_MOTILITY: 50.0 and CURRENT_VALUE_FOR_MOTILITY: 43.9, then show "Previous Value: 50.0" and "Current Value: 43.9"

# DO NOT use the same number for both previous and current values.

# Be encouraging and informative — this is a journey, not a test.

# ---

# ### 📊 True ICC Analysis (Intraclass Correlation Coefficient)

# If this is the first assessment, explain:
# - **First Test**: This is your baseline - we need more data to measure reliability
# - **Next Steps**: Upload future reports to calculate ICC using reference population data
# - **Why ICC Matters**: Distinguishes real biological changes from measurement variation

# If multiple assessments are available, you MUST include the COMPLETE ICC INTERPRETATION that appears in the data above. This includes:
# - **Complete "What Your ICC Results Mean - Clinical Interpretation" section** with full parameter analysis
# - **Parameter-by-Parameter Analysis** with percentages and clinical actions
# - **Clinical Inference Summary** with what to trust vs be cautious about
# - **Practical Recommendations** for reliable vs unreliable parameters
# - **Bottom Line** with encouraging summary

# DO NOT summarize or shorten the ICC interpretation - include the FULL detailed analysis that appears in the data section above.

# ---

# ### ✅ Summary & Next Steps

# - 📈 **Improved Factors**: List any feature(s) that improved, and congratulate the user for their progress.
# - ⚠️ **Features Needing Focus**: List any factors with strong negative SHAP impacts and how they can be addressed.
# - 🧭 **Personalized Next Steps**: Offer a kind, constructive treatment suggestion based on the score.

# ---

# 🎯 **Tone Reminder**: Your tone should be warm, empathetic, and non-judgmental. Think like a supportive coach or fertility companion — informed, caring, and optimistic.

# **Research Context (for background reference only - do not include raw text in response)**:  
# {context}
# """
#     )

#     # [Rest of your existing RAG chain code...]
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4000)

#     rag_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         chain_type_kwargs={
#             "prompt": prompt.partial(score=score, category=category, treatment=""),
#             "document_variable_name": "context"
#         },
#         input_key="query"
#     )

#     response = rag_chain.invoke({"query": shap_formatted})
#     return response["result"]



#### Replacing "reliability" with "consistency -  Best as of now"

import os
import json
import numpy as np
from scipy import stats
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Real patient data from your dataset - 11 unique patients with multiple tests
REFERENCE_PATIENT_DATA = {
    "patient_1": {"volume": [4.22, 4.22, 4.22], "concentration": [27, 27, 27], "motility": [35, 35, 50]},
    "patient_2": {"volume": [3.9, 3.6, 4.2], "concentration": [43, 8.5, 3.1], "motility": [47, 59, 42]},
    "patient_3": {"volume": [4.9, 3.2], "concentration": [47.4, 0], "motility": [69.1, 0]},
    "patient_4": {"volume": [1.1, 1.5, 1.5, 4.5], "concentration": [11.4, 21.4, 21.4, 41.4], "motility": [71, 71, 71, 71]},
    "patient_5": {"volume": [6.6, 4.2, 6.6, 4.2], "concentration": [56.2, 4.75, 56.2, 4.75], "motility": [28.5, 28, 28.5, 28]},
    "patient_6": {"volume": [2.8, 2.8, 1.2, 2.4], "concentration": [61, 61, 80, 64], "motility": [20, 20, 20, 20]},
    "patient_7": {"volume": [2.8, 2.05, 2.8, 0], "concentration": [121.5, 37.2, 121.5, 57.9], "motility": [70, 59, 70, 90]},
    "patient_8": {"volume": [1.1, 1.1, 1.1, 0.9], "concentration": [129, 129, 129, 83.3], "motility": [71, 71, 71, 63]},
    "patient_9": {"volume": [1, 1, 1.8, 2.4], "concentration": [1.9, 1.9, 4, 10], "motility": [10, 10, 10, 30]},
    "patient_10": {"volume": [8, 3.9, 8, 9], "concentration": [16, 27, 16, 18], "motility": [66, 50, 66, 55]},
    "patient_11": {"volume": [3, 0, 0, 3], "concentration": [12.85, 12.85, 0, 12.85], "motility": [59.8, 59.8, 0, 59.8]}
}

def calculate_true_icc(current_patient_data, parameter="motility"):
    """
    Calculate true ICC using reference population data + current patient's multiple tests
    
    Args:
        current_patient_data: List of values for current patient [test1, test2, test3, ...]
        parameter: "volume", "concentration", or "motility"
    
    Returns:
        icc_value, interpretation, explanation
    """
    
    if len(current_patient_data) < 2:
        return None, "Insufficient data", "Need at least 2 tests for ICC calculation"
    
    try:
        # Step 1: Calculate patient averages from reference data
        reference_patient_averages = []
        for patient_id, data in REFERENCE_PATIENT_DATA.items():
            if parameter in data and len(data[parameter]) > 0:
                # Filter out zero values for concentration (measurement errors)
                valid_values = [v for v in data[parameter] if v > 0]
                if valid_values:
                    reference_patient_averages.append(np.mean(valid_values))
        
        # Add current patient's average
        current_patient_avg = np.mean(current_patient_data)
        all_patient_averages = reference_patient_averages + [current_patient_avg]
        
        # Step 2: Calculate Between-Patient Variance (MSB)
        grand_mean = np.mean(all_patient_averages)
        between_variance = np.var(all_patient_averages, ddof=1)
        
        # Step 3: Calculate Within-Patient Variance (MSW)
        within_variances = []
        
        # Add within-variance from reference patients
        for patient_id, data in REFERENCE_PATIENT_DATA.items():
            if parameter in data and len(data[parameter]) > 1:
                valid_values = [v for v in data[parameter] if v > 0]
                if len(valid_values) > 1:
                    patient_variance = np.var(valid_values, ddof=1)
                    within_variances.append(patient_variance)
        
        # Add current patient's within-variance
        if len(current_patient_data) > 1:
            current_within_var = np.var(current_patient_data, ddof=1)
            within_variances.append(current_within_var)
        
        # Average within-variance across all patients
        within_variance = np.mean(within_variances) if within_variances else 0.001
        
        # Step 4: Calculate ICC using standard formula
        # ICC(2,1) = (MSB - MSW) / (MSB + MSW)
        icc = (between_variance - within_variance) / (between_variance + within_variance)
        
        # Ensure ICC is between 0 and 1
        icc = max(0, min(1, icc))
        
        # Step 5: Interpretation
        if icc < 0.5:
            interpretation = "Poor reliability"
        elif icc < 0.75:
            interpretation = "Moderate reliability"
        elif icc < 0.9:
            interpretation = "Good reliability"
        else:
            interpretation = "Excellent reliability"
        
        # Step 6: Create detailed explanation
        explanation = {
            "icc_value": icc,
            "interpretation": interpretation,
            "between_variance": between_variance,
            "within_variance": within_variance,
            "reference_patients": len(reference_patient_averages),
            "current_patient_tests": len(current_patient_data),
            "grand_mean": grand_mean,
            "current_patient_avg": current_patient_avg
        }
        
        return icc, interpretation, explanation
        
    except Exception as e:
        return None, "Calculation error", f"Error in ICC calculation: {str(e)}"

def get_historical_scores(json_path, user_reports_dir="user_reports"):
    """
    Extract historical fertility scores and detailed feature data from user reports
    """
    scores = []
    volume_data = []
    concentration_data = []
    motility_data = []
    
    try:
        current_filename = os.path.basename(json_path)
        all_files = [f for f in os.listdir(user_reports_dir) if f.endswith(".json")]
        historical_files = [f for f in all_files if f != current_filename]
        historical_files.sort()
        
        print(f"DEBUG: Current file: {current_filename}")
        print(f"DEBUG: Historical files: {historical_files}")
        
        for report_file in historical_files:
            with open(os.path.join(user_reports_dir, report_file), "r") as f:
                data = json.load(f)
                scores.append(data["fertility_score"])
                
                # Extract feature data for ICC calculation
                if "features" in data:
                    features = data["features"]
                    volume_data.append(features.get("Volume", {}).get("value", 0))
                    concentration_data.append(features.get("Concentration", {}).get("value", 0))
                    motility_data.append(features.get("Motility", {}).get("value", 0))
                
                print(f"DEBUG: Added score {data['fertility_score']} from {report_file}")
        
        return {
            "scores": scores,
            "volume": volume_data,
            "concentration": concentration_data,
            "motility": motility_data
        }
        
    except Exception as e:
        print(f"Error reading historical scores: {e}")
        return {"scores": [], "volume": [], "concentration": [], "motility": []}

def generate_rag_explanation(json_path="user_reports/me_latest.json", faiss_path="faiss_index_both"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Load current report
    with open(json_path, "r") as f:
        data = json.load(f)
    score = data["fertility_score"]
    features = data["features"]

    # Get current test values
    current_volume = features.get("Volume", {}).get("value", 0)
    current_concentration = features.get("Concentration", {}).get("value", 0)
    current_motility = features.get("Motility", {}).get("value", 0)

    # Try loading previous report
    try:
        reports = sorted([
            f for f in os.listdir("user_reports")
            if f.endswith(".json") and "latest" not in f
        ])
        if len(reports) >= 2:
            with open(os.path.join("user_reports", reports[-2]), "r") as f:
                prev_data = json.load(f)
        else:
            prev_data = None
    except Exception:
        prev_data = None

    # Format comparison-aware SHAP string
    def format_change(curr, prev):
        if curr > prev:
            return f"🔺 (+{curr - prev:.2f})"
        elif curr < prev:
            return f"🔻 ({curr - prev:.2f})"
        else:
            return "➖ (no change)"

    # Get historical data for TRUE ICC calculation
    historical_data = get_historical_scores(json_path)
    
    # Add current values to historical data
    all_scores = historical_data["scores"] + [score]
    all_volume = historical_data["volume"] + [current_volume]
    all_concentration = historical_data["concentration"] + [current_concentration]
    all_motility = historical_data["motility"] + [current_motility]
    
    # Check for duplicates
    if len(all_scores) > 1 and abs(all_scores[-1] - all_scores[-2]) < 0.1:
        all_scores = historical_data["scores"]
        all_volume = historical_data["volume"]
        all_concentration = historical_data["concentration"]
        all_motility = historical_data["motility"]
        print(f"DEBUG: Detected duplicate score, using only historical")
    
    print(f"DEBUG: Final all_scores: {all_scores}")
    print(f"DEBUG: Volume data: {all_volume}")
    print(f"DEBUG: Concentration data: {all_concentration}")
    print(f"DEBUG: Motility data: {all_motility}")
    
    # Calculate TRUE ICC for each parameter
    icc_results = {}
    if len(all_volume) >= 2:
        volume_icc, volume_interp, volume_details = calculate_true_icc(all_volume, "volume")
        icc_results["volume"] = {"icc": volume_icc, "interpretation": volume_interp, "details": volume_details}
    
    if len(all_concentration) >= 2:
        conc_icc, conc_interp, conc_details = calculate_true_icc(all_concentration, "concentration")
        icc_results["concentration"] = {"icc": conc_icc, "interpretation": conc_interp, "details": conc_details}
    
    if len(all_motility) >= 2:
        motility_icc, motility_interp, motility_details = calculate_true_icc(all_motility, "motility")
        icc_results["motility"] = {"icc": motility_icc, "interpretation": motility_interp, "details": motility_details}
    
    # Calculate overall ICC (average of parameters)
    valid_iccs = [result["icc"] for result in icc_results.values() if result["icc"] is not None]
    if valid_iccs:
        overall_icc = np.mean(valid_iccs)
        if overall_icc < 0.5:
            overall_interpretation = "Poor consistency"
        elif overall_icc < 0.75:
            overall_interpretation = "Moderate consistency"
        elif overall_icc < 0.9:
            overall_interpretation = "Good consistency"
        else:
            overall_interpretation = "Excellent consistency"
    else:
        overall_icc = None
        overall_interpretation = "Insufficient data"

    # Add TRUE ICC information to the analysis
    icc_text = ""
    if len(all_scores) < 2:
        icc_text = f"\n\n📊 **TRUE ICC ANALYSIS (Intraclass Correlation Coefficient):**\n"
        icc_text += f"- This is your first fertility assessment, so we can't measure consistency yet.\n"
        icc_text += f"- After your next test, we'll calculate your ICC using data from 11 reference patients.\n"
        icc_text += f"- **What ICC Measures**: How consistent your test results are compared to natural variation between different people.\n"
        icc_text += f"- Keep uploading reports to build your consistency profile!\n"
    else:
        icc_text = f"\n\n📊 **TRUE ICC ANALYSIS (Intraclass Correlation Coefficient):**\n"
        icc_text += f"- **Your Test History**: {len(all_scores)} tests taken\n"
        icc_text += f"- **Score Progression**: {' → '.join([f'{score:.1f}%' for score in all_scores])}\n"
        
        # Calculate simple ICC for fertility scores (your original method for scores)
        if len(all_scores) >= 2:
            scores_array = np.array(all_scores)
            n = len(scores_array)
            between_var = np.var(scores_array, ddof=1)
            within_var = np.var(scores_array) / n
            fertility_score_icc = (between_var - within_var) / (between_var + within_var)
            fertility_score_icc = max(0, min(1, fertility_score_icc))
            
            if fertility_score_icc < 0.5:
                fertility_score_interpretation = "Poor consistency"
            elif fertility_score_icc < 0.75:
                fertility_score_interpretation = "Moderate consistency"
            elif fertility_score_icc < 0.9:
                fertility_score_interpretation = "Good consistency"
            else:
                fertility_score_interpretation = "Excellent consistency"
        else:
            fertility_score_icc = None
            fertility_score_interpretation = "Insufficient data"
        
        if overall_icc is not None and fertility_score_icc is not None:
            icc_text += f"- **Overall ICC Score (Fertility Predictions)**: {fertility_score_icc:.3f} ({fertility_score_interpretation})\n"
            icc_text += f"- **Overall ICC Score (Combined Parameters)**: {overall_icc:.3f} ({overall_interpretation})\n"
            icc_text += f"- **Reference Population**: Compared against 11 patients with multiple tests\n"
            icc_text += f"- **ICC Methodology**: True statistical ICC using between-patient vs within-patient variance\n\n"
            
            # Parameter-specific ICC results - DETAILED BREAKDOWN
            icc_text += f"**🔬 Parameter-Specific ICC Results:**\n"
            for param, result in icc_results.items():
                if result["icc"] is not None:
                    details = result["details"]
                    icc_text += f"\n**{param.title()} ICC Analysis:**\n"
                    icc_text += f"  - **ICC Score**: {result['icc']:.3f} ({result['interpretation']})\n"
                    icc_text += f"  - **Your {param.title()} Tests**: {details['current_patient_tests']} measurements\n"
                    icc_text += f"  - **Your Average {param.title()}**: {details['current_patient_avg']:.2f}\n"
                    icc_text += f"  - **Population Average {param.title()}**: {details['grand_mean']:.2f}\n"
                    icc_text += f"  - **Between-Patient Variance**: {details['between_variance']:.3f}\n"
                    icc_text += f"  - **Within-Patient Variance**: {details['within_variance']:.3f}\n"
                    
                    # Clinical interpretation for each parameter
                    if result['icc'] >= 0.75:
                        icc_text += f"  - **Clinical Meaning**: Your {param} measurements are highly reliable - changes reflect real biological variation\n"
                    elif result['icc'] >= 0.5:
                        icc_text += f"  - **Clinical Meaning**: Your {param} measurements show moderate reliability - focus on trends over time\n"
                    else:
                        icc_text += f"  - **Clinical Meaning**: Your {param} measurements show high variability - consider testing conditions\n"
            
            icc_text += f"\n**🔬 ICC Calculation Details:**\n"
            icc_text += f"- **Between-Patient Variance**: How much fertility parameters vary between different people\n"
            icc_text += f"- **Within-Patient Variance**: How much your repeat tests vary from your personal average\n"
            icc_text += f"- **ICC Formula**: (Between-Variance - Within-Variance) / (Between-Variance + Within-Variance)\n"
            icc_text += f"- **Clinical Meaning**: ICC > 0.75 means your changes are likely real, not just measurement noise\n\n"
            
            # Practical explanation based on ICC results
            icc_text += f"🎯 **What Your ICC Results Mean - Clinical Interpretation:**\n\n"
            icc_text += f"**📊 Overall Picture:**\n"
            icc_text += f"Your fertility tracking shows **mixed consistency** - some parameters are trustworthy, others need more data.\n\n"
            
            icc_text += f"🔍 **Parameter-by-Parameter Analysis:**\n\n"
            
            # Sort parameters by ICC score for better presentation
            sorted_params = sorted(icc_results.items(), key=lambda x: x[1]["icc"] if x[1]["icc"] is not None else 0, reverse=True)
            
            for param, result in sorted_params:
                if result["icc"] is not None:
                    icc_val = result["icc"]
                    interpretation = result["interpretation"]
                    
                    # Determine emoji and status
                    if icc_val >= 0.75:
                        emoji = "🎯"
                        status = "✅"
                        confidence = "High confidence"
                        percentage = f"{icc_val*100:.0f}%"
                        noise_percentage = f"{(1-icc_val)*100:.0f}%"
                        
                        # Clinical actions for high consistency
                        if param == "motility":
                            actions = [
                                "**Trust motility trends** - if it improves, celebrate! If it drops, investigate why",
                                "Changes likely reflect real improvements in sperm movement quality",
                                "This parameter is your **most consistent tracker**"
                            ]
                        elif param == "volume":
                            actions = [
                                "**Trust volume changes** - these reflect real biological variation",
                                "Volume improvements likely indicate better reproductive health",
                                "Use volume as a consistent indicator of progress"
                            ]
                        else:  # concentration
                            actions = [
                                "**Trust concentration trends** - these reflect real sperm production changes",
                                "Concentration improvements indicate better sperm production",
                                "Use concentration as a key fertility indicator"
                            ]
                            
                    elif icc_val >= 0.5:
                        emoji = "📊"
                        status = "⚖️"
                        confidence = "Moderate confidence"
                        percentage = f"{icc_val*100:.0f}%"
                        noise_percentage = f"{(1-icc_val)*100:.0f}%"
                        
                        # Clinical actions for moderate consistency
                        actions = [
                            f"**Look for patterns** - don't react to single {param} results",
                            f"Focus on trends over 2-3 tests for {param}",
                            f"Moderate confidence in {param} improvements/declines"
                        ]
                        
                    else:  # < 0.5
                        emoji = "🔻"
                        status = "⚠️"
                        confidence = "Low confidence"
                        percentage = f"{icc_val*100:.0f}%"
                        noise_percentage = f"{(1-icc_val)*100:.0f}%"
                        
                        # Clinical actions for poor consistency
                        if param == "volume":
                            actions = [
                                "**Don't focus on single volume results** - look for consistent patterns over 3-4 tests",
                                "Volume varies with hydration, abstinence time, collection method",
                                "Consider standardizing: same lab, same collection conditions"
                            ]
                        elif param == "concentration":
                            actions = [
                                "**Wait for trends** - don't react to single concentration results",
                                "Consider same-day repeat testing if results seem unusual",
                                "Focus on consistent patterns over multiple months"
                            ]
                        else:  # motility
                            actions = [
                                f"**Be cautious with {param} changes** - high measurement variability",
                                f"Look for consistent {param} patterns over multiple tests",
                                f"Consider testing conditions and lab consistency"
                            ]
                    
                    # Format the parameter analysis
                    icc_text += f"**{emoji} {param.title()} ICC = {icc_val:.3f} ({interpretation}) {status}**\n"
                    icc_text += f"**What this means:**\n"
                    icc_text += f"* **{percentage}** of your {param} changes are **real biological changes**\n"
                    icc_text += f"* Only **{noise_percentage}** is testing noise/variation\n"
                    icc_text += f"* **{confidence}** in {param} improvements or declines\n\n"
                    
                    icc_text += f"**Clinical Action:**\n"
                    for action in actions:
                        icc_text += f"* {action}\n"
                    icc_text += f"\n"
            
            # Clinical inference summary
            icc_text += f"🎯 **Clinical Inference Summary:**\n\n"
            
            # What you CAN trust
            reliable_params = [param for param, result in icc_results.items() 
                             if result["icc"] is not None and result["icc"] >= 0.75]
            moderate_params = [param for param, result in icc_results.items() 
                             if result["icc"] is not None and 0.5 <= result["icc"] < 0.75]
            
            icc_text += f"**✅ What You CAN Trust:**\n"
            if reliable_params:
                for param in reliable_params:
                    icc_text += f"* **{param.title()} changes** - these likely reflect real fertility improvements/declines\n"
            if moderate_params:
                for param in moderate_params:
                    icc_text += f"* **{param.title()} trends** (moderate consistency)\n"
            if overall_icc >= 0.5:
                icc_text += f"* **Overall fertility score trends** (moderate consistency at {overall_icc*100:.0f}%)\n"
            icc_text += f"\n"
            
            # What to be cautious about
            unreliable_params = [param for param, result in icc_results.items() 
                               if result["icc"] is not None and result["icc"] < 0.5]
            
            icc_text += f"**⚠️ What to Be Cautious About:**\n"
            for param in unreliable_params:
                icc_text += f"* **Single {param} measurements** - too much random variation\n"
            if unreliable_params:
                param_list = " and ".join([param for param in unreliable_params])
                icc_text += f"* **Short-term fluctuations** in {param_list}\n"
            icc_text += f"\n"
            
            # Practical recommendations
            icc_text += f"📈 **Practical Recommendations:**\n\n"
            
            # For reliable parameters
            if reliable_params:
                for param in reliable_params:
                    icc_text += f"**For {param.title()} (Consistent):**\n"
                    icc_text += f"* **Continue what's working** if {param} is improving\n"
                    icc_text += f"* **Investigate causes** if {param} consistently drops\n"
                    icc_text += f"* **Make decisions** based on {param} trends\n\n"
            
            # For unreliable parameters
            if unreliable_params:
                param_list = " & ".join([param.title() for param in unreliable_params])
                icc_text += f"**For {param_list} (Inconsistent):**\n"
                icc_text += f"* **Test consistently:** same lab, same abstinence time, same collection method\n"
                icc_text += f"* **Look for 3-month trends** rather than single results\n"
                icc_text += f"* **Consider repeat testing** if results seem dramatically different\n"
                icc_text += f"* **Don't make major decisions** based on {' or '.join(unreliable_params)} alone\n\n"
            
            # Overall strategy
            icc_text += f"**Overall Strategy:**\n"
            if reliable_params:
                primary_param = reliable_params[0]  # Most consistent parameter
                icc_text += f"* **Use {primary_param} as your primary tracker** (most consistent)\n"
            icc_text += f"* **Combine all parameters** for overall fertility assessment\n"
            icc_text += f"* **Focus on lifestyle factors** that improve all parameters\n"
            if unreliable_params:
                param_list = "/".join(unreliable_params)
                icc_text += f"* **Get more tests** to improve consistency of {param_list}\n"
            icc_text += f"\n"
            
            # Bottom line
            icc_text += f"💡 **Bottom Line:**\n\n"
            if reliable_params:
                primary_param = reliable_params[0]
                icc_text += f"**Your {primary_param} tracking is excellent** - trust those changes! "
            if unreliable_params:
                param_list = " and ".join(unreliable_params)
                icc_text += f"**{param_list.title()} need more data** - be patient and consistent with testing to see real patterns emerge.\n\n"
            
            icc_text += f"This pattern is actually **common in fertility testing** - motility tends to be more consistent than volume/concentration measurements! 🎯\n\n"
        
        # Add interpretation guide
        icc_text += f"\n**📊 Understanding ICC Scores:**\n"
        icc_text += f"- **0.0-0.5**: Poor consistency - High variability, focus on major trends only\n"
        icc_text += f"- **0.5-0.75**: Moderate consistency - Good for tracking significant changes\n"
        icc_text += f"- **0.75-0.9**: Good consistency - Can confidently interpret most changes\n"
        icc_text += f"- **0.9-1.0**: Excellent consistency - Even small changes are highly meaningful\n"
    
    # Rest of your existing code for feature comparison...
    # [Continue with existing feature details, SHAP formatting, etc.]
    
    # Build detailed comparison data for each feature
    feature_details = {}
    for k, v in features.items():
        feature_info = {
            'current_value': v['value'],
            'impact': v['impact'],
            'previous_value': None,
            'change': 'N/A (first report)'
        }
        
        if prev_data and k in prev_data["features"]:
            prev_val = prev_data["features"][k]["value"]
            feature_info['previous_value'] = prev_val
            feature_info['change'] = format_change(v['value'], prev_val)
        
        feature_details[k] = feature_info

    shap_formatted = f"Fertility score: {score:.2f}%\n\n✅ Positive Factors:\n"
    for k, v in features.items():
        if v["impact"] > 0:
            change_note = ""
            if prev_data:
                prev_val = prev_data["features"].get(k, {}).get("value")
                if prev_val is not None:
                    change_note = f" {format_change(v['value'], prev_val)}"
            shap_formatted += f"- {k}: {v['value']} → SHAP Impact: +{v['impact']:.3f} (positive){change_note}\n"

    shap_formatted += "\n⚠️ Negative Factors:\n"
    for k, v in features.items():
        if v["impact"] < 0:
            change_note = ""
            if prev_data:
                prev_val = prev_data["features"].get(k, {}).get("value")
                if prev_val is not None:
                    change_note = f" {format_change(v['value'], prev_val)}"
            shap_formatted += f"- {k}: {v['value']} → SHAP Impact: {v['impact']:.3f} (negative){change_note}\n"

    # Add detailed feature comparison data
    feature_comparison_text = "\n\n**DETAILED FEATURE COMPARISON DATA - USE THESE EXACT VALUES:**\n"
    for feature_name, details in feature_details.items():
        feature_comparison_text += f"\n=== {feature_name.upper()} ===\n"
        feature_comparison_text += f"PREVIOUS_VALUE_FOR_{feature_name.upper()}: {details['previous_value'] if details['previous_value'] is not None else 'N/A'}\n"
        feature_comparison_text += f"CURRENT_VALUE_FOR_{feature_name.upper()}: {details['current_value']}\n"
        feature_comparison_text += f"CHANGE_FOR_{feature_name.upper()}: {details['change']}\n"
        feature_comparison_text += f"SHAP_IMPACT_FOR_{feature_name.upper()}: {details['impact']:.3f}\n"

    shap_formatted += feature_comparison_text
    shap_formatted += icc_text  # Add TRUE ICC analysis
    shap_formatted += "\n\n⚠️ CRITICAL INSTRUCTION: When writing the explanation, you MUST use the exact PREVIOUS_VALUE_FOR_[FEATURE] and CURRENT_VALUE_FOR_[FEATURE] shown above. DO NOT use the same number for both previous and current values."
    shap_formatted += "\n\n🎯 CRITICAL ICC INSTRUCTION: You MUST copy and paste the ENTIRE ICC interpretation section above WORD FOR WORD. Do NOT summarize, shorten, or rewrite it. Include every single section: 'What Your ICC Results Mean - Clinical Interpretation', 'Overall Picture', 'Parameter-by-Parameter Analysis', 'Clinical Inference Summary', 'Practical Recommendations', and 'Bottom Line'. Copy the EXACT text including all emojis, percentages, and bullet points."

    def categorize_score(score):
        if score < 20:
            return "very low", ""
        elif score < 40:
            return "low", ""
        elif score < 60:
            return "moderate", ""
        elif score < 80:
            return "good", ""
        else:
            return "high", ""

    category, treatment = categorize_score(score)

    # [Rest of your existing prompt template code...]
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "score", "category", "treatment"],
        template="""
A machine learning model predicted a **{category} fertility score of {score:.2f}%**, representing the probability of achieving successful natural pregnancy within the next 12 months.

IMPORTANT: Do not include any raw research text or fragments from the context at the beginning of your response. Start directly with the structured explanation below.

{question}

---

### 🌟 1. Introduction
- You received a **{category} fertility score of {score:.2f}%**. This means your current chances of natural conception are estimated at this percentage.
- This is **not a diagnosis**, but a data-driven guide to help you make informed decisions.
- If this is an improvement from your previous report, it's worth celebrating — even small gains reflect your efforts and potential for continued progress.

---

### 🔍 2. Detailed Review for Each Factor

For each factor (e.g., Motility, Concentration, Volume), provide:

- **Subheading**: e.g., `🌟 1. Motility`
- **SHAP Impact**: Mention its value and whether it's positive or negative (e.g., "SHAP Impact: -0.547 (moderate negative)").
- **Value Comparison**:  
  - `Previous Value: X` (MUST USE PREVIOUS_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE CURRENT VALUE)
  - `Current Value: Y` (MUST USE CURRENT_VALUE_FOR_[FEATURE] FROM DATA ABOVE - DO NOT USE PREVIOUS VALUE)
  - `Change: 🔺 (+Z)` or `🔻 (-Z)` (USE THE EXACT CHANGE_FOR_[FEATURE] FROM DATA ABOVE)
- **Explanation**: Explain how this factor influences fertility and how it contributed to the score.
- **Appreciation**: If there's an improvement, **gently acknowledge the positive change and encourage continued momentum**.
- **Suggestions**: Provide friendly, practical tips or medical guidance to improve or maintain the factor (e.g., lifestyle, supplements, hydration).

⚠️ **CRITICAL**: When displaying Previous Value and Current Value, you MUST:
1. For Previous Value: Use PREVIOUS_VALUE_FOR_[FEATURE] from the data section
2. For Current Value: Use CURRENT_VALUE_FOR_[FEATURE] from the data section  
3. These values MUST BE DIFFERENT (unless there was truly no change)
4. Example: If PREVIOUS_VALUE_FOR_MOTILITY: 50.0 and CURRENT_VALUE_FOR_MOTILITY: 43.9, then show "Previous Value: 50.0" and "Current Value: 43.9"

DO NOT use the same number for both previous and current values.

Be encouraging and informative — this is a journey, not a test.

---

### 📊 True ICC Analysis (Intraclass Correlation Coefficient)

If this is the first assessment, explain:
- **First Test**: This is your baseline - we need more data to measure consistency
- **Next Steps**: Upload future reports to calculate ICC using reference population data
- **Why ICC Matters**: Distinguishes real biological changes from measurement variation

If multiple assessments are available, you MUST include the COMPLETE ICC INTERPRETATION EXACTLY as it appears in the data above. DO NOT SUMMARIZE OR REWRITE. Copy the entire section that starts with "What Your ICC Results Mean - Clinical Interpretation" and includes:
- The full "Overall Picture" section
- The complete "Parameter-by-Parameter Analysis" with all percentages and clinical actions
- The entire "Clinical Inference Summary" with what to trust vs be cautious about  
- The full "Practical Recommendations" section
- The complete "Bottom Line" section

IMPORTANT: Use the word "consistency" instead of "reliability" everywhere in your ICC analysis. Replace any instance of "reliability" with "consistency".

---

### ✅ Summary & Next Steps

- 📈 **Improved Factors**: List any feature(s) that improved, and congratulate the user for their progress.
- ⚠️ **Features Needing Focus**: List any factors with strong negative SHAP impacts and how they can be addressed.
- 🧭 **Personalized Next Steps**: Offer a kind, constructive treatment suggestion based on the score.

---

🎯 **Tone Reminder**: Your tone should be warm, empathetic, and non-judgmental. Think like a supportive coach or fertility companion — informed, caring, and optimistic.

**Research Context (for background reference only - do not include raw text in response)**:  
{context}
"""
    )

    # [Rest of your existing RAG chain code...]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4000)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt.partial(score=score, category=category, treatment=""),
            "document_variable_name": "context"
        },
        input_key="query"
    )

    response = rag_chain.invoke({"query": shap_formatted})
    return response["result"]