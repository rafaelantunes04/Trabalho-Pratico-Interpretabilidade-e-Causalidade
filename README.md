# ML Interpretability @ UBI 25/26

##### This was a project done for a subject about Interpretability & Causality in ML, featuring metrics and explanation methods to explain decisions from a CNN to differentiate Cats from Dogs and from a Wine Quality Gradient Boosted Tree Classifier.

## Sensitivity-N

Method implemented by hand, changed from official implementation, resulting on a graph, plotting quantity of important pixels removed against model confidence in each iteration (CNN). For this example it was used Integrated Gradients, Gradient x Input and Smooth Gradients as explanation methods.

For the Gradient Boosted Tree Classifier, also implemented by hand, it was removed each important feature iteratively, measuring the confidence for each example, resulting in a similar graph, being used only Shap values as explanation methods.

## ROAD (Incomplete)

For this method i tried using Quantus, but unfortunately it didnt work, used the same cat-dog CNN classifier and has three explanation methods, integrated gradients, gradient x input and a saliency map, remaining incomplete.
