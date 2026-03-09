#!/usr/bin/env python3
"""
MIDV Fine-Grained Country Classification with LLM as a Judge
Using Gemini to validate CNN predictions on 10-class country identification.
Tests LLM uncertainty on HARD task: Identify exact country, not just ID vs Passport.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os
from contextlib import suppress
from sklearn.metrics import confusion_matrix, classification_report

# Get script directory and construct paths
SCRIPT_DIR = Path(__file__).parent
MIDV_ROOT = SCRIPT_DIR.parent
DATASET_DIR = MIDV_ROOT / "datasets" / "MIDV"
RESULTS_DIR = MIDV_ROOT / "results"

# Try loading environment variables
with suppress(ImportError):
    from dotenv import load_dotenv
    load_dotenv()

# Try importing Gemini (optional)
GEMINI_AVAILABLE = False
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠ Warning: google-genai not available. Install with: pip install google-genai")


# ============================================================================
# Vision-based Classification (CNN) - 10 Classes
# ============================================================================

class MIDVDocumentClassifier:
    """CNN-based classifier for 10-class country identification (FINE-GRAINED)."""
    
    DOCUMENT_TYPES = {
        # IDs (5 classes)
        "alb_id": "Albanian ID",
        "esp_id": "Spanish ID",
        "est_id": "Estonian ID",
        "fin_id": "Finnish ID",
        "svk_id": "Slovak ID",
        # Passports (5 classes)
        "aze_passport": "Azerbaijani Passport",
        "grc_passport": "Greek Passport",
        "lva_passport": "Latvian Passport",
        "rus_internalpassport": "Russian Internal Passport",
        "srb_passport": "Serbian Passport",
    }
    
    # Mapping to indices
    COUNTRY_TO_IDX = {k: i for i, k in enumerate(sorted(DOCUMENT_TYPES.keys()))}
    IDX_TO_COUNTRY = {v: k for k, v in COUNTRY_TO_IDX.items()}
    NUM_CLASSES = len(DOCUMENT_TYPES)
    
    def __init__(self, device="cpu"):
        self.device = device
        
        # Load ResNet18 with 10 output classes
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.NUM_CLASSES)
        self.model.to(device)

        # Load fine-tuned checkpoint (photo dataset)
        model_path = RESULTS_DIR / "midv_finegrained_photo_model.pt"
        if model_path.exists():
            state = torch.load(str(model_path), map_location=device)
            # Support both raw state_dict and wrapped dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)
            print(f"✓ Loaded fine-tuned weights from {model_path.name}")
        else:
            print(f"⚠ Fine-tuned model not found at {model_path} — using ImageNet weights only")

        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"✓ Initialized ResNet18 classifier (10 classes) on {device}")
    
    @torch.no_grad()
    def predict_image(self, img_path: str) -> Dict:
        """
        Predict document country from image using ResNet18 features.
        
        Returns:
            country_code: One of 10 country codes
            country_name: Full country name
            confidence: probability
            probabilities: Array of [10] class probabilities
        """
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
            pred_idx = torch.argmax(probs)
            pred_country_code = self.IDX_TO_COUNTRY[pred_idx.item()]
            pred_country_name = self.DOCUMENT_TYPES[pred_country_code]
            confidence = probs[pred_idx].item()
            
            return {
                "country_code": pred_country_code,
                "country_name": pred_country_name,
                "confidence": confidence,
                "probabilities": probs.cpu().numpy(),
                "top_3": [
                    (self.DOCUMENT_TYPES[self.IDX_TO_COUNTRY[idx]], float(probs[idx].item()))
                    for idx in torch.topk(probs, 3)[1].cpu().numpy()
                ],
            }
        except Exception as e:
            print(f"  ✗ Error processing {img_path}: {e}")
            return None
    
    def analyze_batch(self, img_paths: List[str]) -> List[Dict]:
        """Analyze a batch of images."""
        results = []
        for path in img_paths:
            result = self.predict_image(path)
            if result:
                results.append(result)
        return results


# ============================================================================
# LLM-based Judge (Gemini)
# ============================================================================

class GeminiDocumentJudge:
    """Use Gemini API to identify which country by document analysis."""
    
    COUNTRIES = {
        "alb_id": "Albanian ID",
        "esp_id": "Spanish ID",
        "est_id": "Estonian ID",
        "fin_id": "Finnish ID",
        "svk_id": "Slovak ID",
        "aze_passport": "Azerbaijani Passport",
        "grc_passport": "Greek Passport",
        "lva_passport": "Latvian Passport",
        "rus_internalpassport": "Russian Internal Passport",
        "srb_passport": "Serbian Passport",
    }
    
    def __init__(self, api_key: str = None):
        if not GEMINI_AVAILABLE:
            print("⚠ Gemini not available")
            self.client = None
            return
        
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        print("✓ Initialized Gemini Judge (Fine-grained)")
    
    def judge_prediction(
        self,
        img_path: str,
        cnn_country_code: str,
        cnn_confidence: float,
    ) -> Dict:
        """
        Use Gemini to identify which country the document is from.
        
        Returns:
            gemini_country: Detected country code
            is_agreement: True if Gemini agrees with CNN
            confidence: Gemini's confidence
            reasoning: Explanation
        """
        if not self.client:
            return {"error": "Gemini not available"}
        
        try:
            # Load image
            with open(img_path, 'rb') as f:
                image_data = f.read()
            
            # Prepare prompt for fine-grained classification
            countries_str = ", ".join([f"{code} ({name})" for code, name in self.COUNTRIES.items()])
            
            prompt = f"""You are an expert in document identification and authentication.

Analyze this document image and identify which specific country it is from.

POSSIBLE COUNTRIES:
{countries_str}

Respond with ONLY a JSON object (no other text):
{{
  "country_code": "One of: alb_id, esp_id, est_id, fin_id, svk_id, aze_passport, grc_passport, lva_passport, rus_internalpassport, srb_passport",
  "confidence": 0.0-1.0 (your confidence in this identification),
  "key_features": ["feature1", "feature2", "feature3"],
  "reasoning": "Brief explanation of how you identified the country",
  "alternative_options": ["country_code1", "country_code2"]
}}

Key distinguishing features:
- Text/Language: Cyrillic (Russian), Latin alphabet, specific spelling
- Colors: Different color schemes per country
- Emblems/Seals: National symbols
- Layout: Specific document design per country
- Numbers: Format of ID numbers
"""
            
            # Call Gemini with vision
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/tiff" if str(img_path).endswith(".tif") else "image/jpeg",
                            "data": image_data,
                        }
                    },
                ],
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=300,
                ),
            )
            
            # Parse response
            try:
                response_text = response.text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    judge_result = json.loads(json_match.group())
                else:
                    judge_result = {"error": "Could not parse response"}
            except Exception as e:
                judge_result = {"error": f"Parse error: {e}"}
            
            # Compare with CNN
            if "error" not in judge_result:
                gemini_country = judge_result.get("country_code", "unknown").lower()
                is_agreement = (gemini_country == cnn_country_code)
                
                return {
                    "gemini_country": gemini_country,
                    "gemini_confidence": judge_result.get("confidence", 0.0),
                    "cnn_country": cnn_country_code,
                    "cnn_confidence": cnn_confidence,
                    "agreement": is_agreement,
                    "reasoning": judge_result.get("reasoning", ""),
                    "features": judge_result.get("key_features", []),
                    "alternatives": judge_result.get("alternative_options", []),
                }
            else:
                return judge_result
        
        except Exception as e:
            return {"error": f"Gemini error: {str(e)}"}

# ============================================================================
# Calibration & Uncertainty Analysis
# ============================================================================

class UncertaintyAnalysis:
    """Analyze LLM uncertainty calibration and behavior."""
    
    @staticmethod
    def expected_calibration_error(predictions, confidences, targets):
        """Compute ECE - measures if confidence matches accuracy."""
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
    
    @staticmethod
    def identify_hardest_pairs(predictions_true, predictions_pred, country_codes):
        """Identify which country pairs the LLM confuses the most."""
        confusion = {}
        for true, pred, code in zip(predictions_true, predictions_pred, country_codes):
            if true != pred:
                pair = f"{true} → {pred}"
                confusion[pair] = confusion.get(pair, 0) + 1
        
        sorted_pairs = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:5]  # Top 5 confused pairs


# ============================================================================
# Comparison Analysis
# ============================================================================

def analyze_midv_documents(
    dataset_dir: str = None,
    max_per_country: int = 10,
    use_gemini: bool = False,
):
    """
    Analyze MIDV documents with CNN and optionally LLM Judge.
    FINE-GRAINED: 10-class country identification (not just ID vs Passport)
    """
    # Use default path if not provided
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR / "photo" / "images")
    
    print("=" * 80)
    print("MIDV FINE-GRAINED COUNTRY CLASSIFICATION - CNN vs LLM")
    print("Task: Identify exact country (10 classes) with uncertainty analysis")
    print("=" * 80)
    
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize classifiers
    print(f"\n1️⃣  Initializing classifiers on {device}...")
    print("-" * 80)
    
    cnn_classifier = MIDVDocumentClassifier(device=device)
    
    gemini_judge = None
    if use_gemini:
        gemini_judge = GeminiDocumentJudge()
    
    # Analyze documents
    print(f"\n2️⃣  Analyzing MIDV documents (max {max_per_country} per country)...")
    print("-" * 80)
    print(f"   Dataset directory: {dataset_dir}")
    
    root_dir = Path(dataset_dir)
    if not root_dir.exists():
        print(f"❌ ERROR: Dataset directory not found: {root_dir}")
        return
    
    all_results = []
    all_true_countries = []
    all_pred_countries = []
    all_confidences = []
    
    for country_code in sorted(MIDVDocumentClassifier.DOCUMENT_TYPES.keys()):
        country_name = MIDVDocumentClassifier.DOCUMENT_TYPES[country_code]
        country_dir = root_dir / country_code
        
        if not country_dir.exists():
            print(f"⚠️  {country_code}: directory not found ({country_dir})")
            continue
        
        # Get images
        image_files = sorted([
            f for f in country_dir.iterdir()
            if f.suffix.lower() in ['.tif', '.jpg', '.jpeg', '.png']
        ])[:max_per_country]
        
        print(f"\n  📌 {country_code} ({country_name}): {len(image_files)} images")
        
        for img_path in image_files:
            # CNN prediction
            cnn_result = cnn_classifier.predict_image(str(img_path))
            
            if not cnn_result:
                continue
            
            prediction = {
                "image": img_path.name,
                "true_country": country_code,
                "cnn_country": cnn_result["country_code"],
                "cnn_confidence": cnn_result["confidence"],
                "cnn_name": cnn_result["country_name"],
                "cnn_top3": cnn_result["top_3"],
                "gemini": None,
            }
            
            # Gemini verification (optional)
            if gemini_judge:
                gemini_result = gemini_judge.judge_prediction(
                    str(img_path),
                    cnn_result["country_code"],
                    cnn_result["confidence"],
                )
                prediction["gemini"] = gemini_result
                
                # Check agreement
                if "error" not in gemini_result:
                    agreement = gemini_result.get("agreement", False)
                    symbol = "✓" if agreement else "✗"
                    print(f"    {symbol} {img_path.name}: CNN={cnn_result['country_code']} "
                          f"({cnn_result['confidence']:.0%}), "
                          f"LLM={gemini_result.get('gemini_country', 'error')}")
                else:
                    print(f"    ? {img_path.name}: {gemini_result.get('error')}")
            else:
                match = "✓" if cnn_result["country_code"] == country_code else "✗"
                print(f"    {match} {img_path.name}: CNN={cnn_result['country_code']} "
                      f"({cnn_result['confidence']:.0%})")
            
            all_results.append(prediction)
            all_true_countries.append(country_code)
            all_pred_countries.append(cnn_result["country_code"])
            all_confidences.append(cnn_result["confidence"])
    
    # Convert to numpy arrays
    all_true_countries = np.array(all_true_countries)
    all_pred_countries = np.array(all_pred_countries)
    all_confidences = np.array(all_confidences)
    
    # Summary & Analysis
    print("\n3️⃣  Analysis & Metrics")
    print("=" * 80)
    
    # Check if we have results
    if len(all_true_countries) == 0:
        print("❌ No images found! Check datasetpath.")
        return
    
    # CNN Accuracy
    cnn_correct = np.sum(all_pred_countries == all_true_countries)
    cnn_total = len(all_true_countries)
    cnn_accuracy = 100 * cnn_correct / cnn_total if cnn_total > 0 else 0
    
    print(f"\n📊 CNN ACCURACY (10-class fine-grained):")
    print(f"    {cnn_correct}/{cnn_total} correct ({cnn_accuracy:.1f}%)")
    print(f"    Mean confidence: {np.mean(all_confidences)*100:.1f}%")
    print(f"    Min confidence: {np.min(all_confidences)*100:.1f}%")
    print(f"    Max confidence: {np.max(all_confidences)*100:.1f}%")
    print(f"    Std confidence: {np.std(all_confidences)*100:.1f}%")
    
    # Gemini stats
    if use_gemini:
        gemini_agrees = sum(1 for r in all_results if r["gemini"] and r["gemini"].get("agreement"))
        gemini_total = sum(1 for r in all_results if r["gemini"] and "error" not in r["gemini"])
        if gemini_total > 0:
            print(f"\n🤖 GEMINI LLM AGREEMENT:")
            print(f"    {gemini_agrees}/{gemini_total} agree with CNN ({100*gemini_agrees/gemini_total:.1f}%)")
    
    # Confusion Matrix
    print(f"\n📈 CONFUSION MATRIX (CNN):")
    cm = confusion_matrix(
        [MIDVDocumentClassifier.COUNTRY_TO_IDX[c] for c in all_true_countries],
        [MIDVDocumentClassifier.COUNTRY_TO_IDX[c] for c in all_pred_countries]
    )
    print(f"    Shape: {cm.shape[0]} countries")
    print(f"    Diagonal sum (correct): {np.trace(cm)}")
    
    # Hardest confusions
    analysis = UncertaintyAnalysis()
    hardest_pairs = analysis.identify_hardest_pairs(
        all_true_countries, all_pred_countries, all_true_countries
    )
    
    if hardest_pairs:
        print(f"\n❌ CONFUSED PAIRS (hardest for CNN):")
        for pair, count in hardest_pairs:
            print(f"    {pair}: {count} times")
    
    # Calibration
    print(f"\n📊 CONFIDENCE CALIBRATION:")
    correct_mask = all_pred_countries == all_true_countries
    avg_conf_correct = np.mean(all_confidences[correct_mask]) if np.any(correct_mask) else 0
    avg_conf_wrong = np.mean(all_confidences[~correct_mask]) if np.any(~correct_mask) else 0
    
    print(f"    Avg confidence when CORRECT: {avg_conf_correct*100:.1f}%")
    print(f"    Avg confidence when WRONG: {avg_conf_wrong*100:.1f}%")
    print(f"    Calibration gap: {(avg_conf_correct - avg_conf_wrong)*100:.1f}%")
    
    # Rejection strategy
    print(f"\n🚫 REJECTION STRATEGY:")
    thresholds = [0.5, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        accepted = all_confidences >= threshold
        if np.sum(accepted) > 0:
            accuracy_accepted = np.sum(correct_mask[accepted]) / np.sum(accepted)
            coverage = 100 * np.sum(accepted) / len(all_confidences)
            print(f"    Threshold {threshold:.0%}: "
                  f"Coverage={coverage:.0f}%, Accuracy={accuracy_accepted*100:.0f}%")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "task": "MIDV Fine-Grained Country Classification (10 classes)",
        "config": {
            "max_per_country": max_per_country,
            "use_gemini": use_gemini,
            "device": device,
            "num_classes": MIDVDocumentClassifier.NUM_CLASSES,
        },
        "summary": {
            "cnn_accuracy": float(cnn_accuracy / 100),
            "cnn_mean_confidence": float(np.mean(all_confidences)),
            "total_samples": int(cnn_total),
            "correct_predictions": int(cnn_correct),
            "confusion_pairs": hardest_pairs,
        },
        "calibration": {
            "avg_conf_correct": float(avg_conf_correct),
            "avg_conf_wrong": float(avg_conf_wrong),
            "calibration_gap": float(avg_conf_correct - avg_conf_wrong),
        },
        "predictions": all_results,
    }
    
    results_path = RESULTS_DIR / "midv_finegrained_llm_judge_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Fine-grained analysis completed!")
    print("=" * 80)
    print("\n📌 KEY INSIGHTS:")
    print("   - This is a HARD task (10-class) vs easy binary classification")
    print("   - Real uncertainty should be HIGH (~50-80% confidence)")
    print("   - Look for confused country pairs")
    print("   - Rejection thresholds can improve production accuracy")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MIDV Fine-Grained Country Classification with LLM Uncertainty"
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini LLM for verification (requires GEMINI_API_KEY)"
    )
    parser.add_argument(
        "--max-per-country",
        type=int,
        default=10,
        help="Max images per country to analyze"
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/MIDV/photo/images",
        help="Path to MIDV dataset (default: photo/images)"
    )
    
    args = parser.parse_args()
    
    analyze_midv_documents(
        dataset_dir=args.dataset_dir,
        max_per_country=args.max_per_country,
        use_gemini=args.gemini,
    )
