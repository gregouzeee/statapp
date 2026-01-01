import numpy as np


class ConformalPredictor:    
    """
    Classe pour la prédiction conforme avec LAC ou APS.

    Attributs:
        score_fn: Fonction de score conforme ("lac" ou "aps")
        alpha: Taux d'erreur souhaité (ex: 0.1 pour 90% de couverture)
        Q_hat: Seuil calculé sur l'ensemble de calibration
    """
    def __init__(self, score_fn="lac", alpha=0.1):
        if score_fn not in ["lac", "aps"]:
            raise ValueError("score_fn must be 'lac' or 'aps'")
        self.score_fn = score_fn
        self.alpha = alpha
        self.q_hat = None  # Calculé après calibration

    def lac_score(self, probs, y):
        """
        LAC (Least Ambiguous set-valued Classifier) score.

        Args:
            probs: Vecteur de probabilités de shape (K,) pour K classes
            y: Index de la classe

        Returns:
            Score conforme s(X, Y) = 1 - f(X)_Y
        """
        return 1.0 - probs[y]


    def aps_score(self, probs, y):
        """
        APS (Adaptive Prediction Sets) score.

        Args:
            probs: Vecteur de probabilités de shape (K,) pour K classes
            y: Index de la classe

        Returns:
            Score conforme s(X, Y) = somme des probas des classes avec proba >= proba de y
        """
        threshold = probs[y]
        return float(np.sum(probs[probs >= threshold]))


    def lac_score_batch(self, probs, y):
        """
        LAC score pour un batch d'exemples.

        Args:
            probs: Matrice de probabilités de shape (n, K)
            y: Vecteur des classes de shape (n,)

        Returns:
            Vecteur des scores de shape (n,)
        """
        return 1.0 - probs[np.arange(len(y)), y]


    def aps_score_batch(self, probs, y):
        """
        APS score pour un batch d'exemples.

        Args:
            probs: Matrice de probabilités de shape (n, K)
            y: Vecteur des classes de shape (n,)

        Returns:
            Vecteur des scores de shape (n,)
        """
        n = len(y)
        scores = np.zeros(n)
        for i in range(n):
            threshold = probs[i, y[i]]
            scores[i] = np.sum(probs[i, probs[i] >= threshold])
        return scores


    def compute_quantile(self, scores):
        """
        Calcule le seuil q_hat pour la prédiction conforme.

        Args:
            scores: Scores sur l'ensemble de calibration
            alpha: Taux d'erreur souhaité (ex: 0.1 pour 90% de couverture)

        Returns:
            Quantile q_hat = quantile((n+1)(1-alpha)/n) des scores
        """
        n = len(scores)
        quantile_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        return float(np.quantile(scores, quantile_level))


    def build_prediction_set(self, probs, q_hat):
        """
        Construit l'ensemble de prédiction C(X) = {Y' : s(X, Y') <= q_hat}.

        Args:
            probs: Vecteur de probabilités de shape (K,)
            q_hat: Seuil calculé sur l'ensemble de calibration

        Returns:
            Liste des indices des classes dans l'ensemble de prédiction
        """
        k = len(probs)
        prediction_set = []

        score_method = getattr(self, f"{self.score_fn}_score")
        for y_prime in range(k):
            if score_method(probs, y_prime) <= q_hat:
                prediction_set.append(y_prime)

        return prediction_set

###Étapes principales de la conformal prediction##

    def calibrate(self, probs_cal, y_cal):
        """
        Étapes 2-3 du PDF : calcule les scores sur D_cal et le seuil q̂.

        Args:
            probs_cal: Matrice de probabilités de shape (n_cal, K)
            y_cal: Vecteur des vraies classes de shape (n_cal,)

        Returns:
            self (pour chaînage)
        """
        probs_cal = np.array(probs_cal)
        y_cal = np.array(y_cal)

        # Calcul des scores sur D_cal
        score_batch_method = getattr(self, f"{self.score_fn}_score_batch")
        scores = score_batch_method(probs_cal, y_cal)

        # Calcul du quantile
        self.q_hat = self.compute_quantile(scores)

        return self

    def predict(self, probs):
        """
        Étape 4 du PDF : construit C(X) pour un nouvel exemple.

        Args:
            probs: Vecteur de probabilités de shape (K,)

        Returns:
            Liste des indices des classes dans C(X)
        """
        if self.q_hat is None:
            raise ValueError("Calibrer d'abord")
        return self.build_prediction_set(np.array(probs), self.q_hat)

    def predict_batch(self, probs_test):
        """
        Prédit pour plusieurs exemples.

        Args:
            probs_test: Matrice de probabilités de shape (n_test, K)

        Returns:
            Liste de listes d'indices
        """
        return [self.predict(p) for p in probs_test]

##Métriques d'évaluation##

    def evaluate(self, probs_test, y_test):
        """
        Calcule les 3 métriques du PDF : Accuracy, Set Size, Coverage Rate.

        Args:
            probs_test: Matrice de probabilités de shape (n_test, K)
            y_test: Vecteur des vraies classes de shape (n_test,)

        Returns:
            dict avec 'accuracy', 'set_size', 'coverage_rate'
        """
        probs_test = np.array(probs_test)
        y_test = np.array(y_test)
        n = len(y_test)

        prediction_sets = self.predict_batch(probs_test)

        # Accuracy (Acc) : Y_pred == Y_true (prédiction = classe avec max proba)
        y_pred = np.argmax(probs_test, axis=1)
        accuracy = np.mean(y_pred == y_test)

        # Set Size (SS) : taille moyenne des ensembles
        set_size = np.mean([len(s) for s in prediction_sets])

        # Coverage Rate (CR) : proportion où Y_true ∈ C(X)
        coverage = np.mean([y_test[i] in prediction_sets[i] for i in range(n)])

        return {
            'accuracy': accuracy,
            'set_size': set_size,
            'coverage_rate': coverage
        }
