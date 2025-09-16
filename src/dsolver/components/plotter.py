import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def plot_k_curve(
    per_k_objs: List[Tuple[int, Optional[float]]],
    k_star: Optional[int] = None,
    title: str = "HALDA: k vs objective",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot k (x-axis) vs objective/latency (y-axis) using the last per-k sweep.
    - per_k_objs: list of (k, objective_or_None)
    - k_star: optional vertical marker for the chosen k*
    - save_path: if provided, saves a PNG instead of (or in addition to) showing it
    """
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return

    # Keep only feasible points and sort by k
    pairs = sorted([(k, v) for k, v in per_k_objs if v is not None], key=lambda t: t[0])
    if not pairs:
        print("No feasible k values to plot.")
        return

    ks = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    plt.figure()
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k (number of segments)")
    plt.xticks(ks)
    plt.ylabel("Objective (estimated latency)")
    plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    # Try to show (will no-op on headless backends)
    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close()


def plot_batch_tpot(
    tpots: List[float],
    batches: List[int],
    title: str = "HALDA: batch size vs TPOT",
    save_path: Optional[str] = None,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return

    plt.figure()
    plt.plot(batches, tpots, marker="o")
    plt.xlabel("Batch size")
    plt.xticks(batches)
    plt.ylabel("Time to generate one token")
    plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    # Try to show (will no-op on headless backends)
    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close()
