import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    from tf_explain.core.gradients_inputs import GradientsInputs
    from tf_explain.core.integrated_gradients import IntegratedGradients
    print("tf_explain imported successfully")
except ImportError:
    print("Please install tf_explain: pip install tf-explain")

try:
    import quantus
    print("quantus imported successfully")
except ImportError:
    print("Please install quantus: pip install quantus")

# Import your CNN_MNIST class
# Assuming it's in the same directory or in your path
from your_module import CNN_MNIST  # Replace 'your_module' with actual module name


class XAIAnalyzer:
    def __init__(self, cnn_model: CNN_MNIST):
        """
        Initialize XAI Analyzer with your CNN model
        
        Args:
            cnn_model: Instance of your CNN_MNIST class
        """
        self.cnn = cnn_model
        self.model = cnn_model.modelo
        
        # Get test data
        self.test_images, self.test_labels = self._extract_test_data()
        
    def _extract_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract test images and labels from test dataset"""
        test_images = []
        test_labels = []
        
        for batch in self.cnn.test_ds.unbatch():
            test_images.append(batch[0].numpy())
            test_labels.append(batch[1].numpy())
        
        return np.array(test_images), np.array(test_labels)
    
    def get_sample_data(self, n_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Get a sample of test data for XAI analysis"""
        indices = np.random.choice(len(self.test_images), n_samples, replace=False)
        return self.test_images[indices], self.test_labels[indices]
    
    def compute_integrated_gradients(self, 
                                    images: np.ndarray,
                                    labels: np.ndarray,
                                    n_steps: int = 50) -> np.ndarray:
        """
        Compute Integrated Gradients attributions
        
        Args:
            images: Input images
            labels: True labels
            n_steps: Number of steps for integration
        
        Returns:
            Integrated gradients attributions
        """
        explainer = IntegratedGradients()
        
        # Convert labels to one-hot if needed
        if len(labels.shape) == 1:
            labels_one_hot = tf.one_hot(labels, depth=10)
        else:
            labels_one_hot = labels
            
        # Compute integrated gradients
        explanations = explainer.explain(
            validation_data=(images, labels_one_hot),
            model=self.model,
            n_steps=n_steps,
            output_layer_index=-1  # Use last layer
        )
        
        return explanations
    
    def compute_gradients_input(self,
                               images: np.ndarray,
                               labels: np.ndarray) -> np.ndarray:
        """
        Compute Gradient * Input attributions
        
        Args:
            images: Input images
            labels: True labels
        
        Returns:
            Gradient * Input attributions
        """
        explainer = GradientsInputs()
        
        # Convert labels to one-hot if needed
        if len(labels.shape) == 1:
            labels_one_hot = tf.one_hot(labels, depth=10)
        else:
            labels_one_hot = labels
        
        # Compute gradients * input
        explanations = explainer.explain(
            validation_data=(images, labels_one_hot),
            model=self.model,
            output_layer_index=-1
        )
        
        return explanations
    
    def compute_deeplift(self,
                        images: np.ndarray,
                        labels: np.ndarray,
                        baseline: np.ndarray = None) -> np.ndarray:
        """
        Compute DeepLIFT attributions using tf-explain
        
        Note: tf-explain doesn't have DeepLIFT built-in, so we'll use
        a custom implementation or alternative approach
        """
        # Since tf-explain doesn't have DeepLIFT, we'll use a simple alternative
        # You might want to install a dedicated library for DeepLIFT
        print("Note: tf-explain doesn't have DeepLIFT implementation")
        print("Consider using captum or innvestigate for DeepLIFT")
        
        # Fallback to integrated gradients for demonstration
        print("Using Integrated Gradients as alternative...")
        return self.compute_integrated_gradients(images, labels)
    
    def compute_sensitivity_n(self,
                             attribution_method: str,
                             images: np.ndarray,
                             labels: np.ndarray,
                             n_samples: int = 200,
                             abs_values: bool = True) -> Dict[str, Any]:
        """
        Compute Sensitivity-n metric using Quantus
        
        Args:
            attribution_method: 'integrated_gradients', 'gradients_input', or 'deeplift'
            images: Input images
            labels: True labels
            n_samples: Number of samples for Sensitivity-n
            abs_values: Whether to use absolute values
        
        Returns:
            Dictionary with Sensitivity-n results
        """
        # Get attributions based on method
        if attribution_method == 'integrated_gradients':
            attributions = self.compute_integrated_gradients(images, labels)
        elif attribution_method == 'gradients_input':
            attributions = self.compute_gradients_input(images, labels)
        elif attribution_method == 'deeplift':
            attributions = self.compute_deeplift(images, labels)
        else:
            raise ValueError(f"Unknown attribution method: {attribution_method}")
        
        # Define prediction function for Quantus
        def predict_func(x: np.ndarray) -> np.ndarray:
            return self.model.predict(x, verbose=0)
        
        # Compute Sensitivity-n metric
        metric = quantus.SensitivityN(
            n_samples=n_samples,
            abs=abs_values,
            normalise=True,
            disable_warnings=True
        )
        
        scores = metric(
            model=predict_func,
            x_batch=images,
            y_batch=labels,
            a_batch=attributions,
            explain_func=self._quantus_explain_func,
            explain_func_kwargs={'attribution_method': attribution_method}
        )
        
        return {
            'method': attribution_method,
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'attributions': attributions
        }
    
    def _quantus_explain_func(self,
                             model: callable,
                             inputs: np.ndarray,
                             targets: np.ndarray,
                             attribution_method: str) -> np.ndarray:
        """Explanation function wrapper for Quantus"""
        if attribution_method == 'integrated_gradients':
            return self.compute_integrated_gradients(inputs, targets)
        elif attribution_method == 'gradients_input':
            return self.compute_gradients_input(inputs, targets)
        elif attribution_method == 'deeplift':
            return self.compute_deeplift(inputs, targets)
        else:
            raise ValueError(f"Unknown attribution method: {attribution_method}")
    
    def compare_xai_methods(self,
                           n_images: int = 3,
                           n_samples_sensitivity: int = 100) -> Dict[str, Any]:
        """
        Compare different XAI methods and compute Sensitivity-n for each
        
        Args:
            n_images: Number of images to analyze
            n_samples_sensitivity: Samples for Sensitivity-n metric
        
        Returns:
            Dictionary with comparison results
        """
        # Get sample data
        sample_images, sample_labels = self.get_sample_data(n_images)
        
        results = {}
        
        # Compute for each method
        methods = ['integrated_gradients', 'gradients_input', 'deeplift']
        
        for method in methods:
            print(f"\nComputing {method}...")
            
            # Get attributions
            if method == 'integrated_gradients':
                attributions = self.compute_integrated_gradients(sample_images, sample_labels)
            elif method == 'gradients_input':
                attributions = self.compute_gradients_input(sample_images, sample_labels)
            else:  # deeplift
                attributions = self.compute_deeplift(sample_images, sample_labels)
            
            # Compute Sensitivity-n
            sensitivity_result = self.compute_sensitivity_n(
                method,
                sample_images,
                sample_labels,
                n_samples=n_samples_sensitivity
            )
            
            results[method] = {
                'attributions': attributions,
                'sensitivity': sensitivity_result
            }
            
            print(f"{method}: Sensitivity-n = {sensitivity_result['mean_score']:.4f} ± {sensitivity_result['std_score']:.4f}")
        
        return results
    
    def visualize_attributions(self,
                              results: Dict[str, Any],
                              max_images: int = 3):
        """
        Visualize attributions from different methods
        
        Args:
            results: Dictionary from compare_xai_methods
            max_images: Maximum number of images to visualize
        """
        methods = list(results.keys())
        n_methods = len(methods)
        
        # Get sample images
        sample_images, sample_labels = self.get_sample_data(max_images)
        
        # Create figure
        fig, axes = plt.subplots(max_images, n_methods + 1, figsize=(15, 4 * max_images))
        
        if max_images == 1:
            axes = axes.reshape(1, -1)
        
        for img_idx in range(max_images):
            # Show original image
            axes[img_idx, 0].imshow(sample_images[img_idx].squeeze(), cmap='gray')
            axes[img_idx, 0].set_title(f"Original\nLabel: {sample_labels[img_idx]}")
            axes[img_idx, 0].axis('off')
            
            # Show attributions for each method
            for method_idx, method in enumerate(methods):
                attributions = results[method]['attributions'][img_idx]
                
                # Normalize attributions for visualization
                norm_attr = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-10)
                
                axes[img_idx, method_idx + 1].imshow(norm_attr.squeeze(), cmap='hot')
                axes[img_idx, method_idx + 1].set_title(f"{method}\nSens-n: {results[method]['sensitivity']['mean_score']:.3f}")
                axes[img_idx, method_idx + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate a summary report of XAI analysis"""
        print("\n" + "="*60)
        print("XAI ANALYSIS REPORT")
        print("="*60)
        
        # Model information
        print(f"\nModel Accuracy: {self.cnn.model_accuracy():.4f}")
        
        # Sensitivity-n comparison
        print("\nSensitivity-n Scores (higher is better):")
        print("-"*40)
        
        for method, data in results.items():
            sens_result = data['sensitivity']
            print(f"{method:25s}: {sens_result['mean_score']:.4f} ± {sens_result['std_score']:.4f}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Find best method
        best_method = max(results.keys(), 
                         key=lambda x: results[x]['sensitivity']['mean_score'])
        
        print(f"\nRecommended method: {best_method}")
        print(f"Reason: Highest Sensitivity-n score ({results[best_method]['sensitivity']['mean_score']:.4f})")
        print("\nInterpretation:")
        print("- Sensitivity-n measures how sensitive attributions are to input perturbations")
        print("- Higher scores suggest more robust and reliable explanations")
        print("- Methods with scores close to 0 may not be capturing relevant features")


def main():
    """Main execution function"""
    # Initialize your CNN model
    print("Initializing CNN model...")
    cnn = CNN_MNIST()
    cnn.setup()
    
    # Display model info
    cnn.display_model_info()
    
    # Initialize XAI analyzer
    print("\nInitializing XAI Analyzer...")
    analyzer = XAIAnalyzer(cnn)
    
    # Show some example predictions
    print("\nModel predictions on sample images:")
    cnn.predict_n_images(3)
    
    # Compare XAI methods
    print("\nComparing XAI methods...")
    results = analyzer.compare_xai_methods(n_images=3, n_samples_sensitivity=50)
    
    # Visualize results
    print("\nVisualizing attributions...")
    analyzer.visualize_attributions(results, max_images=3)
    
    # Generate report
    analyzer.generate_report(results)
    
    # Additional analysis: Test on specific cases
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS: Edge Cases")
    print("="*60)
    
    # Get images where model is confident vs uncertain
    test_batch = next(iter(cnn.test_ds))
    test_images, test_labels = test_batch
    
    predictions = cnn.modelo.predict(test_images, verbose=0)
    confidences = np.max(predictions, axis=1)
    
    # Most confident prediction
    most_confident_idx = np.argmax(confidences)
    most_confident_img = test_images[most_confident_idx:most_confident_idx+1]
    most_confident_label = test_labels[most_confident_idx:most_confident_idx+1]
    
    print(f"\nMost confident prediction (confidence: {confidences[most_confident_idx]:.2%})")
    
    # Analyze this case
    ig_attr = analyzer.compute_integrated_gradients(
        most_confident_img.numpy(),
        most_confident_label.numpy()
    )
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(most_confident_img.numpy().squeeze(), cmap='gray')
    axes[0].set_title(f"Original Image\nPredicted: {np.argmax(predictions[most_confident_idx])}")
    axes[0].axis('off')
    
    axes[1].imshow(ig_attr.squeeze(), cmap='hot')
    axes[1].set_title("Integrated Gradients Attribution")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()