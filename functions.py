import numpy as np
import shap

def explain_sample(index, model, explainer, dataset, standardizer):
    # Extract one sample
    instance = dataset.iloc[[index]]

    print(f"Index: {index}")
    print(f"Predicted class: {model.predict(instance)}")

    # SHAP explanation
    shap_value = explainer(instance)

    shap.plots.waterfall(shap_value[0])
