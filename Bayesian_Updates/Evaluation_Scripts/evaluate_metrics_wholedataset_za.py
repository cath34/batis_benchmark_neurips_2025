import torch
import pandas as pd
import numpy as np

def custom_topk_single(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Compute the fraction of indices that match between the top-k of `prediction`
    and the top-k of `ground_truth`, where k is the number of non-zero entries
    in `ground_truth`.

    Args:
        prediction (torch.Tensor): 1D tensor of predicted scores.
        ground_truth (torch.Tensor): 1D tensor of ground-truth scores/labels
                                     (often binary, but not necessarily).

    Returns:
        float: The fraction of overlap between the top-k predicted indices
               and top-k ground-truth indices. Returns 0.0 if there are
               no non-zero entries in the ground_truth.
    """
    # Ensure both have the same shape
    assert prediction.shape == ground_truth.shape, "Shapes must match!"

    # Determine k based on the number of non-zero elements in ground_truth
    k = torch.count_nonzero(ground_truth).item()
    
    # If k == 0, there is nothing to compare; return 0
    if k == 0:
        return 0.0

    # Get the indices of the top-k predictions
    _, pred_indices = torch.topk(prediction, k=k)
    
    # Get the indices of the top-k ground_truth
    _, gt_indices = torch.topk(ground_truth, k=k)
    
    # Compute how many indices are in both sets (the overlap)
    # One straightforward way is to convert indices to sets and compute intersection
    pred_set = set(pred_indices.tolist())
    gt_set   = set(gt_indices.tolist())
    overlap_count = len(pred_set.intersection(gt_set))
    
    # Fraction of top-k matches
    overlap_fraction = overlap_count / k
    return overlap_fraction

def custom_top10_single(target: torch.Tensor, pred: torch.Tensor) -> float:
    """
    For a single (target, pred) pair, compute the fraction of overlapping indices
    in the top-10 predictions and top-(k or 10) of the target, where k is the
    number of non-zero entries in `target`.

    - If k >= 10, we compare top-10 of both `pred` and `target`.
    - If k < 10, we still take top-10 of `pred`, but only top-k from `target`.
      The score is then (#overlapping indices) / 10 if k >= 10, or (#overlapping indices) / k if k < 10.
    - If k == 0, we return 0.0 (no non-zero entries to compare).

    Args:
        target (torch.Tensor): 1D ground-truth tensor.
        pred   (torch.Tensor): 1D prediction tensor (same shape as `target`).

    Returns:
        float: The fraction of matching indices among the chosen top elements.
    """
    # Sanity check: both must be 1D and same shape
    assert target.shape == pred.shape, "Target and pred must have the same shape"
    assert len(target.shape) == 1, "This function expects 1D tensors"

    # Number of non-zero entries in the ground truth
    k = torch.count_nonzero(target).item()
    if k == 0:
        # No non-zero entries => no possible matches
        return 0.0

    # If k >= 10, compare top-10 of both
    # If k < 10, compare top-10 of pred with top-k of target
    if k >= 10:
        _, i_pred = torch.topk(pred,   k=10)
        _, i_targ = torch.topk(target, k=10)
        denom = 10
    else:
        _, i_pred = torch.topk(pred,   k=10)
        _, i_targ = torch.topk(target, k=k)
        denom = k

    # Convert indices to sets and compute overlap
    pred_set = set(i_pred.tolist())
    targ_set = set(i_targ.tolist())
    overlap  = len(pred_set.intersection(targ_set))

    # Fraction of overlap depends on whether k >= 10
    overlap_fraction = overlap / denom
    return overlap_fraction

def custom_top30_single(target: torch.Tensor, pred: torch.Tensor) -> float:
    """
    For a single (target, pred) pair, compute the fraction of overlapping indices
    in the top-10 predictions and top-(k or 10) of the target, where k is the
    number of non-zero entries in `target`.

    - If k >= 10, we compare top-10 of both `pred` and `target`.
    - If k < 10, we still take top-10 of `pred`, but only top-k from `target`.
      The score is then (#overlapping indices) / 10 if k >= 10, or (#overlapping indices) / k if k < 10.
    - If k == 0, we return 0.0 (no non-zero entries to compare).

    Args:
        target (torch.Tensor): 1D ground-truth tensor.
        pred   (torch.Tensor): 1D prediction tensor (same shape as `target`).

    Returns:
        float: The fraction of matching indices among the chosen top elements.
    """
    # Sanity check: both must be 1D and same shape
    assert target.shape == pred.shape, "Target and pred must have the same shape"
    assert len(target.shape) == 1, "This function expects 1D tensors"

    # Number of non-zero entries in the ground truth
    k = torch.count_nonzero(target).item()
    if k == 0:
        # No non-zero entries => no possible matches
        return 0.0

    # If k >= 10, compare top-10 of both
    # If k < 10, compare top-10 of pred with top-k of target
    if k >= 30:
        _, i_pred = torch.topk(pred,   k=30)
        _, i_targ = torch.topk(target, k=30)
        denom = 30
    else:
        _, i_pred = torch.topk(pred,   k=30)
        _, i_targ = torch.topk(target, k=k)
        denom = k

    # Convert indices to sets and compute overlap
    pred_set = set(i_pred.tolist())
    targ_set = set(i_targ.tolist())
    overlap  = len(pred_set.intersection(targ_set))

    # Fraction of overlap depends on whether k >= 10
    overlap_fraction = overlap / denom
    return overlap_fraction

def mae_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Mean Absolute Error (MAE) for a single (prediction, target) pair.

    Args:
        pred (torch.Tensor): The predicted values.
        target (torch.Tensor): The true/ground-truth values.

    Returns:
        float: The MAE for these two tensors.
    """
    # Ensure the shapes match
    assert pred.shape == target.shape, "Prediction and target must have the same shape"

    # Compute the absolute error and then the mean
    error = (pred - target).abs().mean()

    # Convert from a tensor to a Python float
    return error.item()

def mse_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Mean Squared Error (MSE) for a single (prediction, target) pair.

    Args:
        pred (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground-truth values.

    Returns:
        float: The MSE for these two tensors.
    """
    # Check that they have the same shape
    assert pred.shape == target.shape, "Prediction and target must have the same shape"

    # Compute (pred - target)^2 and take the mean
    error = (pred - target).pow(2).mean()

    # Convert from a tensor to a Python float
    return error.item()

test_data = pd.read_csv("/Volumes/My Passport/South_Africa/south_africa/test_filtered.csv")
hotspots_list = list(test_data['hotspot_id'])
n_hotspots = len(hotspots_list)


#seeds = ["seed_924/", "seed_178/", "seed_877/"]
#seeds = ["1/", "2/"]
seeds = ["predictions_mvn_za_1/", "predictions_mvn_za_2/", "predictions_mvn_za_3/"]
#seeds = ["res_hetreg_za_1/", "res_hetreg_za_2/"]
#seeds = ["deep_1/", "deep_2/", "deep_3/"]

#predictions_folder = "/Users/cathv/Desktop/PhD/Main Project/Testing_Env/predictions_za/"
#predictions_folder = "/Users/cathv/Desktop/PhD/Main Project/Testing_Env/evaluate_results/bayesian_exploration_scripts/preds_za_deepens/"
#predictions_folder = "/Users/cathv/Desktop/PhD/Main Project/Testing_Env/evaluate_results/bayesian_exploration_scripts/preds_za_5chk_hetreg/"
predictions_folder = "/Users/cathv/Desktop/PhD/Main Project/Testing_Env/evaluate_results/bayesian_exploration_scripts/"
#predictions_folder = "/Volumes/My Passport/results_may13/"

target_folder = "/Users/cathv/Desktop/PhD/South_Africa/Encounter_Rates/dataframes/"

mse_final = []
mae_final = []
topk_final = []
top10_final = []
top30_final = []

list_birds = 0
for seed in seeds :
    mse_df = []
    mae_df = []
    topk_df = []
    top10_df = []
    top30_df = []
    for hotspot in hotspots_list:
        print(hotspot)
        preds = np.load(predictions_folder + seed + hotspot + ".npy")
        
        target_df = pd.read_csv(target_folder + hotspot + ".csv")
        target_array = np.array(list(target_df['is_observed']))

        prediction = torch.tensor(preds)
        ground_truth = torch.tensor(target_array)

        mse_df.append(mse_single(prediction, ground_truth)*100)
        mae_df.append(mae_single(prediction, ground_truth)*100)
        top10_df.append(custom_top10_single(ground_truth, prediction)*100)
        top30_df.append(custom_top30_single(ground_truth, prediction)*100)
        topk_df.append(custom_topk_single(prediction, ground_truth)*100)

    mse_final.append(np.mean(mse_df))
    mae_final.append(np.mean(mae_df))
    topk_final.append(np.mean(topk_df))
    top10_final.append(np.mean(top10_df))
    top30_final.append(np.mean(top30_df))


print("Average MAE : " + str(np.mean(mae_final)))
print("STD MAE : " + str(np.std(mae_final)))
print("")

print("Average MSE : " + str(np.mean(mse_final)))
print("STD MSE : " + str(np.std(mse_final)))
print("")

print("Average Top-10 : " + str(np.mean(top10_final)))
print("STD Top-10 : " + str(np.std(top10_final)))
print("")

print("Average Top-30 : " + str(np.mean(top30_final)))
print("STD Top-30 : " + str(np.std(top30_final)))
print("")

print("Average Top-K : " + str(np.mean(topk_final)))
print("STD Top-K : " + str(np.std(topk_final)))
print("")


"""

Average MAE : 1.8195078298520269
Average MSE : 0.3578961830389921
Average Top-K : 67.62158278396114
Average Top-10 : 37.752293577981646
Average Top-30 : 50.4647832629484

Average MAE : 1.8532632156818112
Average MSE : 0.37738863011168894
Average Top-K : 65.45025658063258
Average Top-10 : 39.81651376146789
Average Top-30 : 49.59492257657394

Average MAE : 2.466651693019537
STD MAE : 0.02485120661504802

Average MSE : 0.6235447878031202
STD MSE : 0.007308905434465702

Average Top-10 : 41.78661967476659
STD Top-10 : 0.5930079061276884

Average Top-30 : 57.199722921152244
STD Top-30 : 0.3177295773611291

Average Top-K : 67.1074455646376
STD Top-K : 0.23176218500649462

Average MAE : 2.6618684029740263
STD MAE : 0.021109382487911565

Average MSE : 0.6475671470187105
STD MSE : 0.009040746238705696

Average Top-10 : 37.01276819390342
STD Top-10 : 1.1184200296115017

Average Top-30 : 53.201111688824476
STD Top-30 : 1.0598141911318275

Average Top-K : 64.60755916545092
STD Top-K : 0.2686369973872555

Average MAE : 2.6988245301796874
STD MAE : 0.02422105735538249

Average MSE : 0.6444942530785288
STD MSE : 0.0037721072565803926

Average Top-10 : 37.62938230383973
STD Top-10 : 0.5773511407902099

Average Top-30 : 53.650363547405355
STD Top-30 : 0.18300872852810102

Average Top-K : 64.83969582884737
STD Top-K : 0.17273861974682347
"""