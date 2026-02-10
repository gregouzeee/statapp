import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
#  ConformalPredictor object
# ============================================================

class ConformalPredictor:
    """
    Split-conformal for classification using:
      - 'lac':  s(x,y)=1-p_y
      - 'aps':  s(x,y)=sum_{j: p_j >= p_y} p_j
    """
    def __init__(self, score_fn: str = "aps", alpha: float = 0.1):
        if score_fn not in ("lac", "aps"):
            raise ValueError("score_fn must be 'lac' or 'aps'")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        self.score_fn = score_fn
        self.alpha = float(alpha)
        self.q_hat: Optional[float] = None

    def lac_score_batch(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs)
        y = np.asarray(y, dtype=int)
        return 1.0 - probs[np.arange(len(y)), y]

    def aps_score_batch(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs)
        y = np.asarray(y, dtype=int)
        n, K = probs.shape
        order = np.argsort(-probs, axis=1)
        probs_sorted = np.take_along_axis(probs, order, axis=1)
        cums = np.cumsum(probs_sorted, axis=1)
        pos = np.argmax(order == y[:, None], axis=1)
        return cums[np.arange(n), pos]

    def compute_quantile(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=float)
        n = len(scores)
        if n == 0:
            raise ValueError("No calibration scores provided.")
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))  # 1-based
        k = max(1, min(k, n))
        scores_sorted = np.sort(scores)
        return float(scores_sorted[k - 1])

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        probs_cal = np.asarray(probs_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        if self.score_fn == "lac":
            scores = self.lac_score_batch(probs_cal, y_cal)
        else:
            scores = self.aps_score_batch(probs_cal, y_cal)
        self.q_hat = self.compute_quantile(scores)
        return self

    def build_prediction_set(self, probs: np.ndarray, q_hat: float) -> List[int]:
        probs = np.asarray(probs, dtype=float)
        K = probs.shape[0]
        if self.score_fn == "lac":
            scores = 1.0 - probs
        else:
            scores = np.zeros(K)
            for j in range(K):
                scores[j] = float(np.sum(probs[probs >= probs[j]]))
        return [int(j) for j in range(K) if scores[j] <= q_hat]

    def predict(self, probs: np.ndarray) -> List[int]:
        if self.q_hat is None:
            raise ValueError("Must calibrate first.")
        return self.build_prediction_set(np.asarray(probs, dtype=float), float(self.q_hat))

    def predict_batch(self, probs_test: np.ndarray) -> List[List[int]]:
        probs_test = np.asarray(probs_test, dtype=float)
        return [self.predict(p) for p in probs_test]

    def evaluate_sets(self, probs_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        probs_test = np.asarray(probs_test, dtype=float)
        y_test = np.asarray(y_test, dtype=int)
        sets = self.predict_batch(probs_test)
        coverage = float(np.mean([int(y_test[i] in sets[i]) for i in range(len(y_test))])) if len(y_test) else float("nan")
        avg_size = float(np.mean([len(s) for s in sets])) if len(sets) else float("nan")
        return coverage, avg_size


# ============================================================
#  Grid + selection objects
# ============================================================

@dataclass
class AlphaGridResult:
    score_fn: str
    alphas: List[float]
    q_hats: List[float]
    coverages: List[float]
    avg_sizes: List[float]
    miscoverage: List[float]
    losses_by_name: Dict[str, List[float]]  # ex: {"loss_lambda_5": [...]}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlphaSelection:
    score_fn: str
    coverage_target: float

    alpha_cov_max: Optional[float] = None
    alpha_min_size_cov: Optional[float] = None

    alpha_min_loss: Dict[str, Optional[float]] = None  # name -> alpha
    alpha_elbow: Optional[float] = None
    alpha_stable: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["alpha_min_loss"] is None:
            d["alpha_min_loss"] = {}
        return d


# ============================================================
#  AlphaTuner
# ============================================================

class AlphaTuner:
    def __init__(self, score_fn: str = "aps"):
        if score_fn not in ("lac", "aps"):
            raise ValueError("score_fn must be 'lac' or 'aps'")
        self.score_fn = score_fn

    @staticmethod
    def _loss_set_size_plus_lambda_miscoverage(lambda_: float) -> Callable[[float, float], float]:
        # loss = size + lambda * miscoverage
        def f(avg_size: float, miscov: float) -> float:
            return float(avg_size + lambda_ * miscov)
        return f

    def run_grid(
        self,
        probs_cal: np.ndarray, y_cal: np.ndarray,
        probs_val: np.ndarray, y_val: np.ndarray,
        alphas: np.ndarray,
        loss_lambdas: List[float] = (5.0,),
    ) -> AlphaGridResult:
        probs_cal = np.asarray(probs_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        probs_val = np.asarray(probs_val, dtype=float)
        y_val = np.asarray(y_val, dtype=int)

        q_hats, covs, sizes, miscovs = [], [], [], []
        losses_by_name: Dict[str, List[float]] = {f"loss_lambda_{lam}": [] for lam in loss_lambdas}

        for a in alphas:
            cp = ConformalPredictor(score_fn=self.score_fn, alpha=float(a)).calibrate(probs_cal, y_cal)
            q_hats.append(float(cp.q_hat))
            cov, avg_size = cp.evaluate_sets(probs_val, y_val)
            covs.append(float(cov))
            sizes.append(float(avg_size))
            mis = float(1.0 - cov) if not np.isnan(cov) else float("nan")
            miscovs.append(mis)

            for lam in loss_lambdas:
                name = f"loss_lambda_{lam}"
                loss = self._loss_set_size_plus_lambda_miscoverage(lam)(avg_size, mis)
                losses_by_name[name].append(float(loss))

        return AlphaGridResult(
            score_fn=self.score_fn,
            alphas=[float(x) for x in alphas],
            q_hats=q_hats,
            coverages=covs,
            avg_sizes=sizes,
            miscoverage=miscovs,
            losses_by_name=losses_by_name,
        )

    @staticmethod
    def _select_cov_max_alpha(alphas: np.ndarray, coverages: np.ndarray, cov_target: float) -> Optional[float]:
        ok = np.where(coverages >= cov_target)[0]
        return float(alphas[ok[-1]]) if len(ok) else None

    @staticmethod
    def _select_min_size_under_cov(alphas: np.ndarray, coverages: np.ndarray, sizes: np.ndarray, cov_target: float) -> Optional[float]:
        ok = np.where(coverages >= cov_target)[0]
        if not len(ok):
            return None
        idx = ok[np.argmin(sizes[ok])]
        return float(alphas[idx])

    @staticmethod
    def _select_min_loss(alphas: np.ndarray, loss: np.ndarray) -> Optional[float]:
        if len(loss) == 0:
            return None
        return float(alphas[int(np.argmin(loss))])

    @staticmethod
    def _select_elbow(alphas: np.ndarray, sizes: np.ndarray, miscov: np.ndarray) -> Optional[float]:
        # Heuristique simple: normalise size et miscov, puis prend point le plus proche de l’origine
        # (compromis “petit set” + “peu d’erreurs”)
        if len(alphas) == 0:
            return None
        s = np.asarray(sizes, dtype=float)
        m = np.asarray(miscov, dtype=float)
        if np.any(np.isnan(s)) or np.any(np.isnan(m)):
            return None
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
        m_norm = (m - m.min()) / (m.max() - m.min() + 1e-12)
        dist = np.sqrt(s_norm**2 + m_norm**2)
        return float(alphas[int(np.argmin(dist))])

    def select_alphas(
        self,
        grid: AlphaGridResult,
        coverage_target: float = 0.95
    ) -> AlphaSelection:
        alphas = np.asarray(grid.alphas, dtype=float)
        cov = np.asarray(grid.coverages, dtype=float)
        size = np.asarray(grid.avg_sizes, dtype=float)
        mis = np.asarray(grid.miscoverage, dtype=float)

        sel = AlphaSelection(score_fn=grid.score_fn, coverage_target=float(coverage_target))
        sel.alpha_cov_max = self._select_cov_max_alpha(alphas, cov, coverage_target)
        sel.alpha_min_size_cov = self._select_min_size_under_cov(alphas, cov, size, coverage_target)

        sel.alpha_min_loss = {}
        for name, arr in grid.losses_by_name.items():
            loss = np.asarray(arr, dtype=float)
            sel.alpha_min_loss[name] = self._select_min_loss(alphas, loss)

        sel.alpha_elbow = self._select_elbow(alphas, size, mis)
        return sel

    @staticmethod
    def plot(
        grid,
        selection=None,
        save_dir: Optional[str] = None,
        tag: str = "",
        max_marks: int = 3,
        zoom_to_cov: bool = True,
        dpi: int = 200,
        show: bool = False,                 # <-- NEW: show or not
        subfolder: str = "graphe",          # <-- NEW: save into graphe/
    ) -> None:
        alphas = np.asarray(grid.alphas, dtype=float)
        cov = np.asarray(grid.coverages, dtype=float)
        size = np.asarray(grid.avg_sizes, dtype=float)

        # --------- create save folder (save_dir/graphe) ----------
        if save_dir is None:
            base = Path(".")
        else:
            base = Path(save_dir)
        out_dir = base / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

        # --------- pick a few marks ----------
        marks: List[Tuple[str, float]] = []
        if selection is not None:
            if selection.alpha_cov_max is not None:
                marks.append(("cov_max", selection.alpha_cov_max))
            if selection.alpha_min_size_cov is not None and selection.alpha_min_size_cov != selection.alpha_cov_max:
                marks.append(("min_size_cov", selection.alpha_min_size_cov))

            # take one loss marker (lambda=5 if exists)
            if selection.alpha_min_loss:
                key = "loss_lambda_5.0" if "loss_lambda_5.0" in selection.alpha_min_loss else next(iter(selection.alpha_min_loss.keys()), None)
                if key and selection.alpha_min_loss.get(key) is not None:
                    a = selection.alpha_min_loss[key]
                    if a not in [x[1] for x in marks]:
                        marks.append((key, a))

            if selection.alpha_elbow is not None and selection.alpha_elbow not in [x[1] for x in marks]:
                marks.append(("elbow", selection.alpha_elbow))

        marks = marks[:max_marks]

        def add_vlines():
            for name, a in marks:
                plt.axvline(a, linestyle=":", linewidth=1.2, label=name)

        def maybe_zoom_x():
            if not (zoom_to_cov and selection is not None and selection.alpha_cov_max is not None):
                return
            xmax = min(1.0, selection.alpha_cov_max + 0.08)
            plt.xlim(0.0, xmax)

        # --------- Color choices (simple + consistent) ----------
        # You can change these if you want
        color_cov  = "tab:blue"    if "aps" in tag else "tab:green"
        color_size = "tab:orange"  if "aps" in tag else "tab:red"
        loss_colors = ["tab:purple", "tab:brown", "tab:gray", "tab:pink"]

        # --------- Coverage plot ----------
        plt.figure(figsize=(9, 4))
        plt.plot(alphas, cov, marker="o", color=color_cov)
        if selection is not None:
            plt.axhline(selection.coverage_target, linestyle="--", linewidth=1, label="target_cov", color="black")
        add_vlines()
        plt.xlabel("alpha")
        plt.ylabel("coverage")
        plt.title(f"Coverage vs alpha {tag}".strip())
        plt.grid(True)
        maybe_zoom_x()
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"alpha_{tag}_coverage.png", bbox_inches="tight", dpi=dpi)
        if show:
            plt.show()
        plt.close()

        # --------- Size plot ----------
        plt.figure(figsize=(9, 4))
        plt.plot(alphas, size, marker="o", color=color_size)
        add_vlines()
        plt.xlabel("alpha")
        plt.ylabel("avg set size")
        plt.title(f"Avg set size vs alpha {tag}".strip())
        plt.grid(True)
        maybe_zoom_x()
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"alpha_{tag}_size.png", bbox_inches="tight", dpi=dpi)
        if show:
            plt.show()
        plt.close()

        # --------- Loss plots ----------
        for idx, (name, arr) in enumerate(grid.losses_by_name.items()):
            loss = np.asarray(arr, dtype=float)
            plt.figure(figsize=(9, 4))
            plt.plot(alphas, loss, marker="o", color=loss_colors[idx % len(loss_colors)])
            add_vlines()
            plt.xlabel("alpha")
            plt.ylabel(name)
            plt.title(f"{name} vs alpha {tag}".strip())
            plt.grid(True)
            maybe_zoom_x()
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"alpha_{tag}_{name}.png", bbox_inches="tight", dpi=dpi)
            if show:
                plt.show()
            plt.close()

    @staticmethod
    def summarize_alphas(selection) -> Dict[str, Optional[float]]:
        """Renvoie tous les alphas 'optimaux' sélectionnés, proprement."""
        out = {
            "cov_max": selection.alpha_cov_max,
            "min_size_cov": selection.alpha_min_size_cov,
            "elbow": selection.alpha_elbow,
        }
        for k, v in (selection.alpha_min_loss or {}).items():
            out[k] = v
        return out

    @staticmethod
    def plot_useful_one_figure(
        grid,
        selection,
        save_dir: str,
        tag: str,
        loss_key: str = "loss_lambda_5.0",
        dpi: int = 200,
        show: bool = False,
        subfolder: str = "graphe",
        zoom_to_cov: bool = True,
    ) -> str:
        """
        Une seule figure utile: coverage + size + loss(lambda choisi)
        + lignes verticales (cov_max, min_size_cov, elbow, min_loss_*)
        Couleurs différentes pour chaque option.
        Retourne le chemin du fichier image.
        """
        alphas = np.asarray(grid.alphas, dtype=float)
        cov = np.asarray(grid.coverages, dtype=float)
        size = np.asarray(grid.avg_sizes, dtype=float)

        losses = grid.losses_by_name
        if loss_key not in losses:
            # fallback: première loss dispo
            loss_key = next(iter(losses.keys()))
        loss = np.asarray(losses[loss_key], dtype=float)

        # dossier de sortie
        out_dir = Path(save_dir) / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"alpha_{tag}_useful.png"

        # --- marks (alpha optimaux) ---
        marks: List[Tuple[str, float]] = []
        alpha_map = AlphaTuner.summarize_alphas(selection)

        # On ne garde que ceux qui existent
        for name in ["cov_max", "min_size_cov", "elbow", loss_key]:
            a = alpha_map.get(name)
            if a is not None:
                marks.append((name, float(a)))

        # couleurs par option (lisible)
        mark_colors = {
            "cov_max": "tab:green",
            "min_size_cov": "tab:blue",
            "elbow": "tab:orange",
            loss_key: "tab:red",
        }

        # --- figure ---
        fig, ax1 = plt.subplots(figsize=(11, 5))

        # Coverage (axe gauche)
        ax1.plot(alphas, cov, marker="o", label="coverage", color="tab:blue")
        ax1.set_xlabel("alpha")
        ax1.set_ylabel("coverage", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True)

        # ligne target
        ax1.axhline(selection.coverage_target, linestyle="--", linewidth=1, color="black", label="target_cov")

        # Size (axe droit 1)
        ax2 = ax1.twinx()
        ax2.plot(alphas, size, marker="o", label="avg set size", color="tab:purple")
        ax2.set_ylabel("avg set size", color="tab:purple")
        ax2.tick_params(axis="y", labelcolor="tab:purple")

        # Loss (axe droit 2, décalé)
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 55))
        ax3.plot(alphas, loss, marker="o", label=loss_key, color="tab:brown")
        ax3.set_ylabel(loss_key, color="tab:brown")
        ax3.tick_params(axis="y", labelcolor="tab:brown")

        # Zoom optionnel sur zone utile (jusqu’à cov_max)
        if zoom_to_cov and selection.alpha_cov_max is not None:
            xmax = min(1.0, float(selection.alpha_cov_max) + 0.08)
            ax1.set_xlim(0.0, xmax)

        # lignes verticales (couleurs différentes)
        for name, a in marks:
            ax1.axvline(a, linestyle=":", linewidth=2, color=mark_colors.get(name, "tab:gray"), label=f"alpha_{name}")

        # Légende unique (on fusionne les handles)
        handles, labels = [], []
        for ax in [ax1, ax2, ax3]:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        # supprimer doublons
        seen = set()
        uniq = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append(h); uniq_labels.append(l); seen.add(l)
        ax1.legend(uniq, uniq_labels, loc="best", fontsize=8)

        plt.title(f"Alpha tuning (useful) — {tag} — marks: cov_max/min_size/elbow/{loss_key}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        return str(out_path)