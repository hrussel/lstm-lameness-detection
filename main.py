import os
import random
import numpy as np
from datetime import datetime
from contextlib import redirect_stdout

import torch
from torch import nn
from datasets.tsKeypointDataset import TSKeypointDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from models.KPLSTM import KPLSTM

import sklearn
from sklearn.model_selection import StratifiedGroupKFold

import utils.config


def init_wandb(config):
    wandb.init(
        # set the wandb project where this run will be logged
        project="lstm-gait-analysis",

        # track hyperparameters and run metadata
        config={
            "gait_scores_csv": config.gait_scores_csv,
            "keypoints_path": config.keypoints_path,
            "flat_cv": config.flat_cv,
            "n_folds": config.n_folds,
            "merging": config.merging,
            "use_kp": config.use_kp,
            "load_model": config.load_model,
            "save_path": config.save_path,
            "model_name": config.model_name,
            "model_params": config.model_params,
            "seed": config.seed,
            "learning_rate": config.lr,
            "optimizer": config.optimizer,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "seq_length": config.seq_length,
            "step_size": config.step_size,
            "gradient_clipping": config.gradient_clipping,
            "weight_decay": config.weight_decay

        }
        # config=config
    )

    wandb.define_metric("epoch")
    # define which metrics will be plotted against it
    wandb.define_metric("Fold*", step_metric="epoch")

    return wandb.config


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def balance_dataset_indices(dataset, indices):
    """
    Returns a list of dataset indices that are balanced across classes (Equivalent to oversampling).
    :param dataset: the dataset to balance
    :param indices: the indices in the dataset (populated by the dataloader)
    :return: balanced indices
    """
    train_labels = dataset.labels_list[indices]
    bins = np.bincount(train_labels)
    max_label, bin_max = np.argmax(bins), np.max(bins)  # label with the largest number of samples,
    min_label, bin_min = np.argmin(bins), np.min(bins)  # label with the least number of samples

    if bin_max == bin_min:
        return indices

    min_indices = np.argwhere(train_labels == min_label).flatten()
    others = indices[min_indices]
    additional_indices = np.random.permutation(others)[:bin_max-bin_min]

    indices = np.append(indices, additional_indices)

    # debug
    # train_labels = dataset.labels_list[indices]
    # bins = np.bincount(train_labels)
    # print(bins, np.sum(bins))

    return indices


def get_optimizer(model, config):
    """
    Returns the optimizer with parameters defined in the config
    :param model: the model
    :param config: the config values
    :return: the initialized optimizer
    """
    amsgrad = 'AMS' in config.optimizer
    if 'AdamW' in config.optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay, amsgrad=amsgrad)
    elif 'Adam' in config.optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                     weight_decay=config.weight_decay, amsgrad=amsgrad)
    else:
        return None
    return optimizer


def get_model(config, device):
    """
    Returns the model with parameters defined in the config
    :param config: config values
    :param device: cpu or cuda
    :return: The model
    """
    if config.model_name == 'LSTM':
        model = KPLSTM(seq_length=config.seq_length, input_dim=9 * 2, **config.model_params).to(device)
    else:
        print(f"ERROR: model type {config.model_name} unknown.")
        return None

    return model


def eval_predictions(y_true, y_pred, average='macro', verbose=False):
    """
    Computes relevant metrics for the predictions
    :param y_true: Labels
    :param y_pred: Predictions
    :param average: Averaging strategy (default: macro)
    :param verbose: Whether to print the metrics (default: False)
    :return: A dictionary of the evaluation metrics
    """
    metrics = {'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
               f'f1_{average}': sklearn.metrics.f1_score(y_true, y_pred, average=average, zero_division=0),
               f'precision_{average}': sklearn.metrics.precision_score(y_true, y_pred, average=average,
                                                                       zero_division=0),
               f'recall_{average}': sklearn.metrics.recall_score(y_true, y_pred, average=average, zero_division=0),
               f'ROC_AUC_{average}': sklearn.metrics.roc_auc_score(y_true, y_pred, average=average),
               'mcc': sklearn.metrics.matthews_corrcoef(y_true, y_pred),
               'cohen': sklearn.metrics.cohen_kappa_score(y_true, y_pred),
               'cohen_w': sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights='linear'),
               'c_matrix': sklearn.metrics.confusion_matrix(y_true, y_pred),
               'f1_binary': sklearn.metrics.f1_score(y_true, y_pred, average='binary', pos_label=1),
               'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(y_true, y_pred),
               'zero_one_loss': sklearn.metrics.zero_one_loss(y_true, y_pred),
               'sensitivity': sklearn.metrics.recall_score(y_true, y_pred, pos_label=1),
               'specificity': sklearn.metrics.recall_score(y_true, y_pred, pos_label=0)}

    if verbose:
        for key, val in metrics.items():
            print(f"{key}:\n{val}")

    return metrics


def train_loop(dataloader, model, loss_fn, optimizer, scheduler=None, gradient_clipping=0):
    """
    Training loop: train the model for one epoch

    :param dataloader: the dataloader handles the batches of the training set
    :param model: the model to train
    :param loss_fn: loss funtion
    :param optimizer: optimizer
    :param scheduler: learning-rate scheduler (default: None)
    :param gradient_clipping: value of the gradient-clipping (default: 0)
    :return: the training loss
    """

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    loss = 0

    # For each batch in the dataset
    for batch, (x, y, _, _) in enumerate(dataloader):
        # Compute prediction and loss
        x = x.to(device)
        y = y.to(device, dtype=torch.float64)
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        if gradient_clipping > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        optimizer.zero_grad()

    # Apply scheduler after one epoch
    if scheduler is not None:
        scheduler.step()

    return loss


def test_loop(dataloader, model, loss_fn):
    """
    Test loop: evaluate the model
    :param dataloader: the dataloader handles the batches of the evaluation set
    :param model: the model to evaluate
    :param loss_fn: loss function
    :return: evalutation metrics (dict), loss, evaluation results (list of predictions)
    """

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    y_true, y_pred = [], []
    test_results = [["videos_test", "id_test", "y_test", "y_pred"]]

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y, id, vid_name in dataloader:
            x = x.to(device)
            y = y.to(device, dtype=torch.float64)
            pred = model(x)
            test_loss += loss_fn(pred, y)

            pred = torch.round(torch.sigmoid(pred))
            correct += (pred == y).type(torch.float).sum().item()
            y_true.extend(y.numpy(force=True))
            y_pred.extend(pred.numpy(force=True))

            test_results.append([vid_name, id.numpy(force=True), y.type(torch.int).numpy(force=True),
                                 pred.type(torch.int).numpy(force=True)])

    test_loss /= num_batches

    eval_metrics = eval_predictions(y_true, y_pred, verbose=False)

    return eval_metrics, test_loss, test_results


def train_test_loop(train_subset, val_subset, config, shuffle, device, fold=-1):
    """
    Completes a full training and test loop on the given training and validation subsets
    :param train_subset: training dataset
    :param val_subset: validation dataset
    :param config: config values
    :param shuffle: whether to shuffle the training set
    :param device: cpu or gpu
    :param fold: index of the fold (default: -1)
    :return: test metrics (dict) and test results (predictions)
    """

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=config.batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False)

    model = get_model(config, device)
    optimizer = get_optimizer(model, config)

    loss_fn = nn.BCEWithLogitsLoss()

    if config.step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.5)
    else:
        scheduler = None

    # For each epoch
    for t in range(config.epochs):
        # Train the model
        _ = train_loop(train_loader, model, loss_fn, optimizer, scheduler, config.gradient_clipping)

        # Evaluate on the training set
        train_results, train_loss, _ = test_loop(train_loader, model, loss_fn)

        # Evaluate on validation set
        val_results, val_loss, _ = test_loop(val_loader, model, loss_fn)

        if fold >= 0 and config.wandb:
            wandb.log({
                f"Fold {fold} Train Accuracy": train_results['accuracy'],
                f"Fold {fold} Train F1_macro": train_results['f1_macro'],
                f"Fold {fold} Train loss": train_loss,
                f"Fold {fold} Val Accuracy": val_results['accuracy'],
                f"Fold {fold} Val F1_macro": val_results['f1_macro'],
                f"Fold {fold} Val loss": val_loss,
                "epoch": t
            })

    # Test the model once the training is completed.
    test_metrics, _, test_results = test_loop(val_loader, model, loss_fn)
    return test_metrics, test_results


def train_test_k_fold(train_dataset, val_dataset, config, device, log_output=None):
    """
    Performs K fold cross validation
    :param train_dataset: the training dataset
    :param val_dataset: the validation dataset
    :param config: config values
    :param device: cpu or gpu
    :param log_output: path to save the logs (default: None)
    :return: None
    """

    sk_dataset = train_dataset.get_dataset_sklearn_style(pos_thr=2)

    # Divide dataset in n folds, with stratified grouping
    # set seed to 42 to match the folds used in Russello et al., 2024
    sgkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=42)

    fold_accs = []
    fold_f1 = []
    fold_roc = []
    fold_spe = []
    fold_sen = []

    if config.wandb:
        run_id = wandb.run.id
    else:
        run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Loop through the folds
    for i, (train_idx, val_idx) in enumerate(sgkf.split(sk_dataset.data, sk_dataset.labels, sk_dataset.ids)):

        print(f"Fold {i + 1}\n-------------------------------")

        # If the fold_id is set in config, only perform train-val on that specific fold
        # Only use for hyper-parameter tuning
        if config.fold_id > -1 and i != config.fold_id:
            print("skip")
            continue

        # Balance the training fold
        balanced_train_idx = balance_dataset_indices(train_dataset, train_idx)
        train_subset = Subset(train_dataset, balanced_train_idx)

        val_subset = Subset(val_dataset, val_idx)

        # reset the seed for each fold
        set_seed(config.seed)

        # Train and test for this fold
        test_metrics, test_results = train_test_loop(train_subset, val_subset, config, True, device, fold=i)

        # Save fold metrics
        fold_accs.append(test_metrics['accuracy'])
        fold_f1.append(test_metrics['f1_macro'])
        fold_roc.append(test_metrics['ROC_AUC_macro'])
        fold_spe.append(test_metrics['sensitivity'])
        fold_sen.append(test_metrics['specificity'])

        if log_output != None:
            header = np.array(test_results[0])
            filename = f'test_fold_{i}_{config.model_params["num_layers"]}x{config.model_params["num_hidden"]}-{config.seq_length}_{run_id}.csv'
            res_path = os.path.join(log_output, filename)
            np_result = np.array(test_results[1:]).squeeze()
            np_result = np.vstack((header, np_result))
            np.savetxt(res_path, np_result, fmt='%s', delimiter=',')

    if config.wandb:
        wandb.log({"f1": np.mean(fold_f1)})
        summary_table = wandb.Table(columns=["Metric", "Mean", "Std"],
                                    data=[["acc", np.mean(fold_accs), np.std(fold_accs)],
                                          ["f1", np.mean(fold_f1), np.std(fold_f1)],
                                          ["roc", np.mean(fold_roc), np.std(fold_roc)]])
        wandb.log({"Folds summary": summary_table})


    # Save output
    with open(f'./saves/test_eval_{config.model_params["num_layers"]}x{config.model_params["num_hidden"]}-{config.seq_length}_{run_id}.txt', 'w') as f:
        with redirect_stdout(f):
            print(config)
            print("======")
            print(f'\n****\nMean CV accuracy: {np.mean(fold_accs)}, +/- {np.std(fold_accs)}')
            print(fold_accs)
            print(f'\n****\nMean CV F1: {np.mean(fold_f1)}, +/- {np.std(fold_f1)}')
            print(fold_f1)
            print(f'\n****\nMean CV roc: {np.mean(fold_roc)}, +/- {np.std(fold_roc)}')
            print(fold_roc)
            print(f'\n****\nMean Sensitivity: {np.mean(fold_sen)}, +/- {np.std(fold_sen)}')
            print(fold_sen)
            print(f'\n****\nMean Specificity: {np.mean(fold_spe)}, +/- {np.std(fold_spe)}')
            print(fold_spe)


if __name__ == '__main__':
    """
    Main execution
    """

    # Use nvidia GPU if available, otherwise, use cpu
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # parse the config
    config, _ = utils.config.parse_args('Train TS-GA')

    set_seed(config.seed)

    if config.wandb:
        import wandb
        wandb_config = init_wandb(config)

    seq_length=config.seq_length

    # Transformations applied to the training data
    # Random sequence, Keypoint jitter, Normalize kp coordinates
    train_transform = transforms.Compose([
        TSKeypointDataset.TrimKeypoints(seq_length=seq_length, random=True),
        TSKeypointDataset.RandomXYJitter(percentage=config.jitter_percentage),
        TSKeypointDataset.RelativeKeypoints(kp_index=4)
    ])

    # Transformations applied to the validation data
    # Fixed sequence taken from the middle portion of the video, normalize kp coordinates
    val_transform = transforms.Compose([
        TSKeypointDataset.TrimKeypoints(seq_length=seq_length, random=False),
        TSKeypointDataset.RelativeKeypoints(kp_index=4)
    ])

    # Training and validation datasets
    # Note that the division is make through the cross validation, so both dataset objects are populated from the same csv file
    # We use two dataset objects because the transformations applied to the data differ if it's in training or val
    kp_dataset = TSKeypointDataset(config.gait_scores_csv, config.keypoints_path, config.use_kp, seq_length=seq_length, transform=train_transform, target_transform=TSKeypointDataset.binarize_label, device=device)
    val_dataset = TSKeypointDataset(config.gait_scores_csv, config.keypoints_path, config.use_kp, seq_length=seq_length, transform=val_transform, target_transform=TSKeypointDataset.binarize_label, device=device)

    # K-fold cross validation
    if config.n_folds > 0:
        if config.save_path != None and not os.path.exists(config.save_path):
            os.mkdir(config.save_path)
        train_test_k_fold(kp_dataset, val_dataset, config, device, log_output=config.save_path)

    # Regular train-test
    else:
        # shuffle dataset index
        rng = np.random.default_rng(config.seed)
        shuffled_dataset = rng.permutation(len(kp_dataset))

        # train / val split
        train_frac = int(len(kp_dataset) * 0.5)
        train_idx, val_idx = shuffled_dataset[0:train_frac], shuffled_dataset[train_frac:]
        train_ds = Subset(kp_dataset, train_idx)
        val_ds = Subset(val_dataset, val_idx)

        train_test_loop(train_ds, val_ds, config, True, device)

    if config.wandb:
        wandb.finish()

