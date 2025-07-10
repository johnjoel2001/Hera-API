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

import os
import json
import numpy as np
from scipy import stats
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def calculate_icc(scores_list):
    """
    Calculate Intraclass Correlation Coefficient for fertility scores
    scores_list: list of fertility scores from different time points
    """
    if len(scores_list) < 2:
        return None, "Insufficient data for ICC calculation"
    
    # Convert to numpy array for easier manipulation
    scores = np.array(scores_list)
    
    # Simple ICC calculation (ICC(1,1) - single measurement, absolute agreement)
    n = len(scores)
    mean_score = np.mean(scores)
    
    # Between-subject variance
    between_var = np.var(scores, ddof=1)
    
    # Within-subject variance (assuming single measurements)
    within_var = np.var(scores) / n
    
    # ICC calculation
    icc = (between_var - within_var) / (between_var + within_var)
    
    # ICC interpretation
    if icc < 0.5:
        interpretation = "Poor reliability"
    elif icc < 0.75:
        interpretation = "Moderate reliability"
    elif icc < 0.9:
        interpretation = "Good reliability"
    else:
        interpretation = "Excellent reliability"
    
    return icc, interpretation

def get_historical_scores(json_path, user_reports_dir="user_reports"):
    """
    Extract historical fertility scores from all user reports EXCEPT the current one
    """
    scores = []
    try:
        # Get the current file name to exclude it
        current_filename = os.path.basename(json_path)
        
        # Get all JSON files except the current one being processed
        all_files = [f for f in os.listdir(user_reports_dir) if f.endswith(".json")]
        historical_files = [f for f in all_files if f != current_filename]
        
        # Sort historical files to maintain chronological order
        historical_files.sort()
        
        print(f"DEBUG: Current file: {current_filename}")
        print(f"DEBUG: All files: {all_files}")
        print(f"DEBUG: Historical files: {historical_files}")
        
        for report_file in historical_files:
            with open(os.path.join(user_reports_dir, report_file), "r") as f:
                data = json.load(f)
                scores.append(data["fertility_score"])
                print(f"DEBUG: Added score {data['fertility_score']} from {report_file}")
        
        return scores
    except Exception as e:
        print(f"Error reading historical scores: {e}")
        return []

def generate_rag_explanation(json_path="user_reports/me_latest.json", faiss_path="faiss_index_both"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Load current report
    with open(json_path, "r") as f:
        data = json.load(f)
    score = data["fertility_score"]
    features = data["features"]

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

    # Calculate ICC for historical reliability
    historical_scores = get_historical_scores(json_path)
    
    # Check if current score is different from the last historical score
    if historical_scores and abs(historical_scores[-1] - score) < 0.1:
        # Current score is essentially the same as the last historical score
        # This means the "latest" file is a duplicate of the most recent timestamped file
        all_scores = historical_scores  # Don't add duplicate
        print(f"DEBUG: Detected duplicate score, using only historical: {all_scores}")
    else:
        # Current score is genuinely new
        all_scores = historical_scores + [score]
        print(f"DEBUG: New score detected, full timeline: {all_scores}")
    
    print(f"DEBUG: Historical scores: {historical_scores}")
    print(f"DEBUG: Current score: {score}")
    print(f"DEBUG: Final all_scores: {all_scores}")
    
    icc_value, icc_interpretation = calculate_icc(all_scores)
    
    # Add ICC information to the analysis
    icc_text = ""
    if len(all_scores) < 2:
        icc_text = f"\n\n📊 **CONSISTENCY TRACKING:**\n"
        icc_text += f"- This is your first fertility assessment, so we can't measure consistency yet.\n"
        icc_text += f"- After your next test, we'll show you how reliable your results are.\n"
        icc_text += f"- **Why this matters**: Consistency helps tell if score changes are real improvements or just normal ups and downs.\n"
        icc_text += f"- Keep uploading reports to build your fertility tracking history!\n"
    else:
        # Calculate the actual change between scores
        score_change = abs(all_scores[-1] - all_scores[-2])
        
        icc_text = f"\n\n📊 **CONSISTENCY TRACKING:**\n"
        icc_text += f"- **Your Test History**: {len(all_scores)} tests taken\n"
        icc_text += f"- **Score Progression**: {' → '.join([f'{score:.1f}%' for score in all_scores])}\n"
        icc_text += f"- **Reliability Score**: {icc_value:.2f} out of 1.0 ({icc_interpretation})\n"
        icc_text += f"- **What This Measures**: How predictable and trustworthy your fertility results are over time\n"
        icc_text += f"- **Technical Note**: This uses ICC (Intraclass Correlation Coefficient), a statistical method that measures how consistent your test results are\n"
        icc_text += f"- **Score Meaning**: 0.0 = Very unpredictable, 0.5 = Somewhat reliable, 1.0 = Highly consistent\n\n"
        
        # Give practical explanation based on their actual scores
        large_change_threshold = 30
        score_change = abs(all_scores[-1] - all_scores[-2]) if len(all_scores) >= 2 else 0
        
        if score_change > large_change_threshold:  # Large change
            if icc_value < 0.5:
                icc_text += f"- **What This Means**: You had a major change ({all_scores[-2]:.1f}% → {all_scores[-1]:.1f}%) and your scores vary quite a bit. This suggests you're making real changes to your fertility health - either through lifestyle improvements, treatments, or natural fluctuations.\n"
                icc_text += f"- **Why It Matters**: Big changes with low consistency often mean you're actively improving your fertility or something significant changed in your health.\n"
                icc_text += f"- **What To Do**: If your score improved dramatically (like yours did!), keep doing what you're doing! These changes are likely real and meaningful.\n"
            else:
                icc_text += f"- **What This Means**: You had a significant change ({all_scores[-2]:.1f}% → {all_scores[-1]:.1f}%), and your results are becoming more predictable. This change is very likely real and important.\n"
                icc_text += f"- **Why It Matters**: When someone with consistent results shows a big change, it's almost certainly a genuine improvement or concern.\n"
                icc_text += f"- **What To Do**: Trust this change completely - your tracking pattern shows this improvement is real, not just random variation.\n"
        else:  # Small change
            if icc_value < 0.5:
                icc_text += f"- **What This Means**: Your scores show some variation, which is normal early in fertility tracking. Changes like ({all_scores[-2]:.1f}% → {all_scores[-1]:.1f}%) might be natural ups and downs rather than true improvements.\n"
                icc_text += f"- **Why It Matters**: With unpredictable scores, you need more data points to separate real progress from normal fluctuations.\n"
                icc_text += f"- **What To Do**: Focus on healthy habits and don't worry about small changes yet. Wait for more tests to see clearer trends.\n"
            else:
                icc_text += f"- **What This Means**: Your fertility tracking is becoming reliable, so even smaller changes like ({all_scores[-2]:.1f}% → {all_scores[-1]:.1f}%) are probably meaningful.\n"
                icc_text += f"- **Why It Matters**: Consistent tracking means you can trust even subtle improvements or declines as real changes.\n"
                icc_text += f"- **What To Do**: Pay attention to this change - your reliable tracking history suggests it reflects actual shifts in your fertility health.\n"
        
        # Add interpretation guide
        icc_text += f"\n**📈 Understanding Your Reliability Score:**\n"
        icc_text += f"- **0.0-0.5**: Early tracking phase - focus on building healthy habits\n"
        icc_text += f"- **0.5-0.7**: Moderate tracking - changes are becoming more trustworthy\n"
        icc_text += f"- **0.7-0.9**: Good tracking - you can confidently interpret score changes\n"
        icc_text += f"- **0.9-1.0**: Excellent tracking - even small changes are highly meaningful\n"
    
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

    # Add detailed feature comparison data to the prompt - VERY EXPLICIT FORMAT
    feature_comparison_text = "\n\n**DETAILED FEATURE COMPARISON DATA - USE THESE EXACT VALUES:**\n"
    for feature_name, details in feature_details.items():
        feature_comparison_text += f"\n=== {feature_name.upper()} ===\n"
        feature_comparison_text += f"PREVIOUS_VALUE_FOR_{feature_name.upper()}: {details['previous_value'] if details['previous_value'] is not None else 'N/A'}\n"
        feature_comparison_text += f"CURRENT_VALUE_FOR_{feature_name.upper()}: {details['current_value']}\n"
        feature_comparison_text += f"CHANGE_FOR_{feature_name.upper()}: {details['change']}\n"
        feature_comparison_text += f"SHAP_IMPACT_FOR_{feature_name.upper()}: {details['impact']:.3f}\n"

    shap_formatted += feature_comparison_text
    shap_formatted += icc_text  # Add ICC analysis
    shap_formatted += "\n\n⚠️ CRITICAL INSTRUCTION: When writing the explanation, you MUST use the exact PREVIOUS_VALUE_FOR_[FEATURE] and CURRENT_VALUE_FOR_[FEATURE] shown above. DO NOT use the same number for both previous and current values."

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

### 📊 Intraclass Correlation Coefficient (ICC) Score Tracking

If this is the first assessment, explain:
- **First Test**: This is your baseline - we need more data to measure consistency
- **Next Steps**: Upload future reports to see how reliable your scores are over time
- **Why It Matters**: Consistency helps distinguish real changes from normal variation


If multiple assessments are available, include:
- **Consistency Level**: Mention the ICC result and explain how it was calculated
- **What This Means**: How confident you can be in score changes
- **Your Journey**: Show the progression of scores over time


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