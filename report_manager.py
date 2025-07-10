import os
import json
import datetime

REPORTS_DIR = "user_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def save_new_report(shap_json, user_id="default_user"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{user_id}_{timestamp}.json"
    file_path = os.path.join(REPORTS_DIR, file_name)

    # Save current report
    with open(file_path, "w") as f:
        json.dump(shap_json, f, indent=2)

    # Save a copy as 'latest'
    latest_path = os.path.join(REPORTS_DIR, f"{user_id}_latest.json")
    with open(latest_path, "w") as f:
        json.dump(shap_json, f, indent=2)

    return file_path

def compare_to_last(current_json, user_id="default_user"):
    # Get sorted list of user reports excluding latest
    report_files = sorted([
        f for f in os.listdir(REPORTS_DIR)
        if f.startswith(user_id) and f.endswith(".json") and "latest" not in f
    ])

    if len(report_files) < 2:
        return "🕓 This is the first report saved. No previous data to compare with."

    last_report_path = os.path.join(REPORTS_DIR, report_files[-2])
    try:
        with open(last_report_path, "r") as f:
            previous_json = json.load(f)
    except Exception as e:
        return f"⚠️ Could not load previous report: {e}"

    def fmt(val):
        return f"{val:.2f}" if isinstance(val, (float, int)) else str(val)

    lines = []
    lines.append("### 🔄 Comparison with Previous Report")
    lines.append(f"- **Previous Fertility Score:** {fmt(previous_json['fertility_score'])}%")
    lines.append(f"- **Current Fertility Score:** {fmt(current_json['fertility_score'])}%")

    # Fertility Score change
    diff = current_json["fertility_score"] - previous_json["fertility_score"]
    if diff > 0:
        lines.append(f"✅ **Improved by {diff:.2f}%**")
    elif diff < 0:
        lines.append(f"⚠️ **Decreased by {abs(diff):.2f}%**")
    else:
        lines.append("➖ **No change in fertility score**")

    # Feature-wise changes
    lines.append("\n#### 📊 Feature-wise Changes:")
    for feature in current_json["features"]:
        old_feature = previous_json["features"].get(feature, {})
        new_feature = current_json["features"].get(feature, {})

        old_val = old_feature.get("value", None)
        new_val = new_feature.get("value", None)

        if old_val is not None and new_val is not None:
            diff_val = float(new_val) - float(old_val)
            symbol = "🔺" if diff_val > 0 else "🔻" if diff_val < 0 else "➖"
            lines.append(f"- **{feature}**: {fmt(old_val)} → {fmt(new_val)} {symbol} ({diff_val:+.2f})")

    return "\n".join(lines)
