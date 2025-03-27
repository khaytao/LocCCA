import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

from source.models.music import SoundLocalization
# from source.models.cca import LinearCCA, DeepCCA


def evaluate_localization(loc_pred, loc_true, threshold):
    """
    Evaluate localization predictions against ground truth.
    
    Args:
        loc_pred (ndarray): Predicted locations, shape (N,2)
        loc_true (ndarray): Ground truth locations, shape (N,2) 
        threshold (float): Distance threshold for counting failures
        
    Returns:
        tuple: (num_failures, mse)
            - num_failures: Number of predictions with distance > threshold
            - mse: Mean squared error for predictions within threshold
    """
    # Calculate distances between predicted and true locations
    distances = np.sqrt(np.sum((loc_pred - loc_true)**2, axis=1))
    
    # Count failures (distances > threshold)
    num_failures = np.sum(distances > threshold)
    
    # Calculate MSE for successful predictions only
    successful_mask = distances <= threshold
    # Calculate average distance for all points
    avg_with_failures = np.mean(distances)
    
    # Calculate average distance for successful points only
    if np.any(successful_mask):
        avg_without_failures = np.mean(distances[successful_mask])
    else:
        avg_without_failures = float('inf')  # All predictions failed
        
    return num_failures, avg_with_failures, avg_without_failures


def evaluate_model_on_data(args):
    """
    Evaluate a model on a dataset.
    
    Args:
        args: Namespace containing:
            - model: Model type to evaluate
            - data_dir: Path to dataset
            - threshold: Distance threshold for evaluation
            - plot: Whether to plot results
            - device: Device to run on (cpu/cuda)
    """
    model = args.model
    data_dir = args.data_dir
    threshold = args.threshold
    plot = args.plot if hasattr(args, 'plot') else True
    device = args.device if hasattr(args, 'device') else 'cpu'

    # Initialize model based on model_name
    if model == "music" or model == "srp-phat":
        model_obj = SoundLocalization(algorithm=model, device=device)
    elif model == "linear-cca":
        # model_obj = LinearCCA()
        pass
    elif model == "deep-cca":
        # model_obj = DeepCCA()
        pass
    else:
        raise ValueError(f"Unknown model type: {model}. Must be one of: music, srp-phat, linear-cca, deep-cca")

    # Preprocess data
    X, y = model_obj.preprocess(data_dir)

    # Get predictions
    predictions = model_obj.process(X)

    # Average over N dimension for each prediction if needed
    if isinstance(predictions[0], torch.Tensor):
        avg_predictions = [p.mean(dim=1).squeeze() for p in predictions]
        predictions_array = np.array(avg_predictions)[:,:2]
    else:
        predictions_array = np.array(predictions)[:,:2]

    # Convert ground truth to numpy if needed
    if isinstance(y, torch.Tensor):
        y_array = y.numpy()
    else:
        y_array = np.array(y)

    # Evaluate predictions
    num_failures, avg_with_failures, avg_without_failures = evaluate_localization(predictions_array, y_array, threshold)

    # Plot predictions
    if plot:
        plot_predictions(predictions_array, y_array)

    print(f"Number of failures: {num_failures}/{len(predictions)}")
    print(f"Average distance with failures: {avg_with_failures}")
    print(f"Average distance without failures: {avg_without_failures}")

    num_speakers = len(predictions)

    return num_speakers, num_failures, avg_with_failures, avg_without_failures

def plot_predictions(predictions, ground_truth):
    """
    Plot predictions against ground truth.
    """

    # Ensure predictions and ground_truth are 2D arrays
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    # If first dimension is 2, transpose to make second dimension 2
    if predictions.shape[0] == 2:
        predictions = predictions.T
    if ground_truth.shape[0] == 2:
        ground_truth = ground_truth.T

    # Assert arrays are 2D with second dimension of size 2
    assert len(predictions.shape) == 2 and predictions.shape[1] == 2, \
        f"Predictions must be 2D array with shape (N,2), got shape {predictions.shape}"
    assert len(ground_truth.shape) == 2 and ground_truth.shape[1] == 2, \
        f"Ground truth must be 2D array with shape (N,2), got shape {ground_truth.shape}"
    
    # Check shapes match
    assert predictions.shape == ground_truth.shape, \
        f"Predictions shape {predictions.shape} does not match ground truth shape {ground_truth.shape}"
    
    plt.figure()
    # Plot predictions
    plt.scatter(predictions[:,0], predictions[:,1], c='blue', marker='o', label='Predictions')

    # Plot ground truth
    plt.scatter(ground_truth[:,0], ground_truth[:,1], c='red', marker='^', label='Ground Truth')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Predictions vs Ground Truth')
    plt.grid(True)
    plt.legend()
    plt.show()

def main(args):
    num_speakers, num_failures, avg_with_failures, avg_without_failures = evaluate_model_on_data(args)

    # Calculate failure percentage
    fail_percentage = (num_failures / num_speakers) * 100

    # Create performance dataframe if it doesn't exist, otherwise load it
    performance_file = args.performance_file
    if os.path.exists(performance_file):
        perf_df = pd.read_excel(performance_file)
    else:
        perf_df = pd.DataFrame(columns=["model_name", "dataset_path", "num_speakers", 
                                      "threshold", "fail_percentage", "avg_with_failures", "avg_without_failures"])

    # Add new row with current results
    new_row = {
        "model_name": args.model,
        "dataset_path": args.data_dir,
        "num_speakers": num_speakers,
        "threshold": args.threshold,
        "fail_percentage": fail_percentage,
        "avg_with_failures": avg_with_failures,
        "avg_without_failures": avg_without_failures
    }
    
    perf_df = pd.concat([perf_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated dataframe
    perf_df.to_excel(performance_file, index=False)

    print(f"\nResults saved to {performance_file}")
    print(f"Model: {args.model}")
    print(f"Failure percentage: {fail_percentage:.2f}%")
    print(f"Average distance with failures: {avg_with_failures:.4f}")
    print(f"Average distance without failures: {avg_without_failures:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate localization model')
    parser.add_argument('--model', type=str, default='music',
                      help='Model type (music, srp-phat, linear-cca, deep-cca)')
    parser.add_argument('--data_dir', type=str, default='data/generated/TIMIT_sample',
                      help='Directory containing dataset')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Distance threshold for failure detection')
    parser.add_argument('--plot', type=bool, default=True,
                      help='Plot predictions')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      help='Device to run model on (cuda if available, otherwise cpu)')
    parser.add_argument('--performance_file', type=str, default='performance.xlsx',
                    help='Path to Excel file for saving performance metrics')

    args = parser.parse_args()
    main(args)