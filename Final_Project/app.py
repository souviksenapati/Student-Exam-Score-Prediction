# ============================
# Imports
# ============================
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import shap
import os

# ============================
# Flask App Initialization
# ============================
app = Flask(__name__)

# ============================
# Load Both Models and SHAP Explainers
# ============================
model_dict = {
    "lgbm": pickle.load(open("model/student_score_lgbm_model.pkl", "rb")),
    "xgb": pickle.load(open("model/student_score_xgb_model.pkl", "rb"))
}

explainer_dict = {
    "lgbm": shap.TreeExplainer(model_dict["lgbm"]),
    "xgb": shap.Explainer(model_dict["xgb"])  # works with XGBoost's tree-based models
}

# ============================
# Utility: Dynamic SHAP Threshold
# ============================
def find_shap_threshold(feature_name, user_data, scan_range, explainer):
    for val in scan_range:
        test_data = user_data.copy()
        test_data[feature_name] = val
        shap_vals = explainer.shap_values(test_data)
        shap_vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
        feat_index = list(user_data.columns).index(feature_name)
        if shap_vals[0][feat_index] >= 0:
            return val
    return max(scan_range)

# ============================
# Routes
# ============================

@app.route("/")
def landing():
    return render_template("index.html")

@app.route("/predictor")
def index():
    return render_template("predictor.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Step 1: Input & Preprocessing
        input_data = [
            float(request.form["Gender"]),
            float(request.form["Hours_Studied"]),
            float(request.form["Previous_Score"]),
            float(request.form["Attendance"]),
            float(request.form["Sleep"]),
            float(request.form["Health"]),
            float(request.form["Extra"]),            
            
        ]
        model_choice = request.form["model_choice"]  # 'lgbm' or 'xgb'

        if model_choice not in model_dict:
            return "Invalid model selected"

        model = model_dict[model_choice]
        explainer = explainer_dict[model_choice]

        feature_names = [
            'Gender', 'Hours_Studied', 'Previous_Exam_Score', 'Attendance',
            'Sleep_Hours', 'Health_Issues', 'Extra_Curricular_Hours'
        ]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        data = dict(zip(feature_names, input_data))

        # Step 2: Prediction
        prediction = model.predict(input_df)[0]
        prediction = min(100, max(0, prediction))  # Clamp to 0–100

        # Step 3: SHAP Explanation
        shap_values = explainer.shap_values(input_df)
        shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap_array = shap_vals[0]
        explanation_sorted = list(zip(feature_names, shap_array))
        explanation_sorted.sort(key=lambda x: abs(x[1]), reverse=True)

        # Step 4: Status Tag
        if prediction >= 90:
            status = "Outstanding"
        elif prediction >= 80:
            status = "Excellent"
        elif prediction >= 70:
            status = "Very Good"
        elif prediction >= 60:
            status = "Good"
        elif prediction >= 50:
            status = "Average"
        elif prediction >= 40:
            status = "Below Average"
        else:
            status = "Needs Serious Attention"

        # Step 5: Feedback & Top 2 Negative SHAP Features
        negative_impact_features = [feat for feat, val in explanation_sorted if val < 0][:2]
        feedback = ", ".join(negative_impact_features) if negative_impact_features else "No major concerns"

        # Step 6: SHAP-Neutral Thresholds & Smart Suggestions
        raw_suggestions = []
        if prediction < 90:
            scan_ranges = {
                "Sleep_Hours": np.arange(3, 11.1, 0.5),
                "Hours_Studied": np.arange(1, 12.1, 0.5),
                "Extra_Curricular_Hours": np.arange(0, 8.1, 0.5),
                "Attendance": np.arange(40, 100.1, 1),
            }

            used_hours = data["Sleep_Hours"] + data["Hours_Studied"] + data["Extra_Curricular_Hours"]
            remaining_hours = max(0, 24 - used_hours)

            for feat in negative_impact_features:
                val = data[feat]
                shap_val = shap_array[feature_names.index(feat)]

                if feat in scan_ranges:
                    threshold = find_shap_threshold(feat, input_df, scan_ranges[feat], explainer)
                    delta = round(abs(val - threshold), 1)

                    if delta >= 0.0001 or shap_val < -1.0:
                        suggest_delta = round(max(delta, 0.5), 1)

                        if feat in ["Sleep_Hours", "Hours_Studied", "Extra_Curricular_Hours"] and val < threshold:
                            suggest_delta = min(suggest_delta, remaining_hours)
                            if suggest_delta <= 0:
                                continue
                            remaining_hours -= suggest_delta

                        if feat == "Sleep_Hours":
                            if val > threshold:
                                raw_suggestions.append(f"reduce sleep by {suggest_delta} hr{'s' if suggest_delta != 1 else ''}")
                            elif val < threshold:
                                raw_suggestions.append(f"increase sleep by {suggest_delta} hr{'s' if suggest_delta != 1 else ''}")

                        elif feat == "Hours_Studied":
                            if val > threshold:
                                raw_suggestions.append(f"reduce study time by {suggest_delta} hr{'s' if suggest_delta != 1 else ''} to avoid burnout")
                            elif val < threshold:
                                raw_suggestions.append(f"increase study time by {suggest_delta} hr{'s' if suggest_delta != 1 else ''}")

                        elif feat == "Extra_Curricular_Hours":
                            if val > threshold:
                                raw_suggestions.append(f"reduce extra-curricular hours by {suggest_delta} hr{'s' if suggest_delta != 1 else ''}")
                            elif val < threshold:
                                raw_suggestions.append(f"increase extra-curricular hours by {suggest_delta} hr{'s' if suggest_delta != 1 else ''}")

                        elif feat == "Attendance":
                            if val < threshold:
                                raw_suggestions.append(f"improve attendance by at least {suggest_delta}%")
                            elif val > threshold:
                                raw_suggestions.append(f"maintain attendance below {threshold}% to balance schedule")

                elif feat == "Previous_Exam_Score" and val < 50:
                    raw_suggestions.append("focus on basic concepts to improve your score")

                elif feat == "Health_Issues" and val == 1:
                    raw_suggestions.append("focus on your health and manage study load accordingly")

        # Step 7: Format Suggestions
        if raw_suggestions:
            if len(raw_suggestions) == 1:
                suggestions = [raw_suggestions[0].capitalize() + "."]
            else:
                final_suggestion = ", ".join(raw_suggestions[:-1]) + " and " + raw_suggestions[-1] + "."
                suggestions = [final_suggestion.capitalize()]
        else:
            suggestions = ["You're doing great! Keep maintaining your current routine."]

        return render_template(
            "result.html",
            prediction=round(prediction, 2),
            status=status,
            explanation=explanation_sorted,
            feedback=feedback,
            suggestions=suggestions
        )

    except Exception as e:
        return str(e)

# ============================
# Run App
# ============================
if __name__ == "__main__":
    app.run(debug=True)
