from statap_code.vision_uncertainty.fraud_classifier import FraudIDClassifier
from statap_code.vision_uncertainty.dataset_loader import FraudIDDataset
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch


def make_subset_loader(root_dir, split, max_samples, batch_size=64):
    ds = FraudIDDataset(root_dir=root_dir, split=split)
    if max_samples is not None and len(ds) > max_samples:
        indices = np.linspace(0, len(ds)-1, max_samples, dtype=int)
        subset = Subset(ds, indices.tolist())
    else:
        subset = ds
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


if __name__ == '__main__':
    root = 'datasets/EST'
    val_max = 200
    test_max = 200
    batch_size = 64
    max_batches = 4

    val_loader = make_subset_loader(root, 'val', val_max, batch_size=batch_size)
    test_loader = make_subset_loader(root, 'test', test_max, batch_size=batch_size)

    clf = FraudIDClassifier(model_name='mobilenet_v2', num_classes=7, device='cpu', pretrained=False)

    # Calibrate using only first max_batches batches
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            logits = clf.model(images.to(clf.device))
            all_logits.append(logits)
            all_labels.append(labels)
            if i + 1 >= max_batches:
                break
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).to(clf.device)
    clf.conformal_method.calibrate(all_logits, all_labels)
    print(f'Calibrated conformal quantile={clf.conformal_method.quantile_val:.4f} on {all_logits.shape[0]} samples')

    # Evaluate on first max_batches batches of test_loader
    all_preds = []
    all_labels = []
    all_entropies = []
    all_log_probs = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            res = clf.predict_batch(images, return_uncertainties=True)
            all_preds.append(res['predictions'])
            all_labels.append(labels.numpy())
            all_entropies.append(res['entropy'])
            all_log_probs.append(res['log_prob'])
            if i + 1 >= max_batches:
                break
    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    mean_entropy = np.concatenate(all_entropies).mean()
    mean_logp = np.concatenate(all_log_probs).mean()

    print('\nQuick evaluation results (limited batches):')
    print(f'  Samples evaluated ~ {all_preds.shape[0]}')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  Mean entropy: {mean_entropy:.4f}')
    print(f'  Mean log-prob: {mean_logp:.4f}')

    # Show few sample predictions
    images, labels = next(iter(test_loader))
    res = clf.predict_batch(images[:6], return_uncertainties=True)
    print('\nSample predictions:')
    for i in range(min(6, len(labels))):
        print(f'  idx {i}: true={labels[i].item()}, pred={int(res["predictions"][i])}, entropy={res["entropy"][i]:.4f}, logp={res["log_prob"][i]:.4f}, perp_conf={res["perplexity_confidence"][i]:.4f}')
