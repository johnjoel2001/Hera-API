# history_tracker.py
import os
import json
from datetime import datetime

HISTORY_FOLDER = "report_history"

# Ensure history folder exists
os.makedirs(HISTORY_FOLDER, exist_ok=True)

def save_new_report(report_data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"report_{timestamp}.json"
    path = os.path.join(HISTORY_FOLDER, filename)
    with open(path, "w") as f:
        json.dump(report_data, f, indent=2)
    return path

def get_last_report():
    files = sorted(os.listdir(HISTORY_FOLDER))
    if len(files) < 1:
        return None
    last_file = os.path.join(HISTORY_FOLDER, files[-1])
    with open(last_file, "r") as f:
        return json.load(f)

def compare_to_last(current):
    previous = get_last_report()
    if not previous:
        return "📌 This is your first report. No previous report to compare."

    comparison = []
    for feature in current["features"]:
        curr_val = current["features"][feature]["value"]
        prev_val = previous["features"].get(feature, {}).get("value", None)
        if prev_val is not None:
            diff = curr_val - prev_val
            arrow = "🔼" if diff > 0 else "🔽" if diff < 0 else "⏺️"
            comparison.append(f"- **{feature}**: {prev_val} → {curr_val} ({arrow} {diff:+.2f})")
    return "\n".join(comparison)
