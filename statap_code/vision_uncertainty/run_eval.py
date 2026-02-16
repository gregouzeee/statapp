from statap_code.vision_uncertainty.fraud_classifier import FraudIDClassifier
from statap_code.vision_uncertainty.dataset_loader import create_dataloaders

if __name__ == '__main__':
    est_dataset_path = 'datasets/EST'
    dataloaders = create_dataloaders(root_dir=est_dataset_path, batch_size=64, num_workers=0)
    clf = FraudIDClassifier(model_name='resnet50', num_classes=7, device='cpu', pretrained=False)
    print('Calibrating conformal on val...')
    clf.calibrate_conformal(dataloaders['val'])
    print('Evaluating on test...')
    metrics = clf.evaluate(dataloaders['test'], return_uncertainties=True)
    print('\nMetrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    images, labels = next(iter(dataloaders['test']))
    res = clf.predict_batch(images[:6], return_uncertainties=True)
    print('\nSample predictions (6):')
    for i in range(min(6, len(labels))):
        print(f'  idx {i}: true={labels[i].item()}, pred={int(res["predictions"][i])}, entropy={res["entropy"][i]:.4f}, logp={res["log_prob"][i]:.4f}, perp_conf={res["perplexity_confidence"][i]:.4f}')
