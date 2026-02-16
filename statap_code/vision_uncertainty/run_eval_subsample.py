from statap_code.vision_uncertainty.fraud_classifier import FraudIDClassifier
from statap_code.vision_uncertainty.dataset_loader import FraudIDDataset
from torch.utils.data import DataLoader, Subset
import numpy as np


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
    val_max = 2000
    test_max = 2000
    batch_size = 64

    print(f'Creating val loader with up to {val_max} samples')
    val_loader = make_subset_loader(root, 'val', val_max, batch_size=batch_size)
    print(f'Creating test loader with up to {test_max} samples')
    test_loader = make_subset_loader(root, 'test', test_max, batch_size=batch_size)

    clf = FraudIDClassifier(model_name='resnet50', num_classes=7, device='cpu', pretrained=False)

    print('\n-- Calibrating conformal on val subset --')
    clf.calibrate_conformal(val_loader)

    print('\n-- Evaluating on test subset --')
    metrics = clf.evaluate(test_loader, return_uncertainties=True)
    print('\nMetrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')

    # sample predictions
    images, labels = next(iter(test_loader))
    res = clf.predict_batch(images[:6], return_uncertainties=True)
    print('\nSample predictions (6):')
    for i in range(min(6, len(labels))):
        print(f'  idx {i}: true={labels[i].item()}, pred={int(res["predictions"][i])}, entropy={res["entropy"][i]:.4f}, logp={res["log_prob"][i]:.4f}, perp_conf={res["perplexity_confidence"][i]:.4f}')
