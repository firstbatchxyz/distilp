import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties


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
    # plt.grid(True, which="both", axis="both", linewidth=0.6, alpha=0.5)


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


# def plot_k_curves_by_batch(
#     results: Dict[int, List[Tuple[int, Optional[float]]]],
#     k_stars: Optional[Dict[int, int]] = None,
#     title: str = "HALDA: k vs objective (per batch)",
#     save_path: Optional[str] = None,
#     show: bool = False,
# ):
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#
#     for batch, per_k_objs in sorted(results.items()):
#         pairs = sorted([(k, v) for k, v in per_k_objs if v is not None], key=lambda t: t[0])
#         if not pairs:
#             continue
#
#         ks = [p[0] for p in pairs]
#         vals = [p[1] for p in pairs]
#
#         # draw main curve with label
#         line, = ax.plot(ks, vals, marker="o", label=f"batch={batch}")
#
#         # highlight k* with same color as the curve
#         if k_stars and batch in k_stars and k_stars[batch] in ks:
#             k_star = k_stars[batch]
#             v_star = vals[ks.index(k_star)]
#             ax.plot(
#                 [k_star], [v_star],
#                 marker="D", markersize=8,
#                 linestyle="None",
#                 color=line.get_color()
#             )
#
#     ax.set_xlabel("k (number of rounds)")
#     ax.set_ylabel("Objective (estimated latency)")
#     ax.set_title(title)
#     ax.grid(True, which="both", axis="both", linewidth=0.6, alpha=0.5)
#
#     ax.legend()
#     plt.tight_layout()
#
#     if save_path:
#         fig.savefig(save_path, dpi=150)
#         print(f"Saved plot to {save_path}")
#
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#     return fig, ax


def plot_k_curves_by_batch(
    results: Dict[int, List[Tuple[int, Optional[float]]]],
    k_stars: Optional[Dict[int, int]] = None,
    title: str = "HALDA: k vs objective (multi-batch)",
    save_path: Optional[str] = None,
    show: bool = False,
    top_pad_frac: float = 0.08,
    label_fontsize: int = 9,
):
    fig, ax = plt.subplots()

    # 1) Curves + colors
    batch_color = {}
    all_vals, all_ks = [], set()
    for batch in sorted(results.keys()):
        pairs = sorted([(k, v) for k, v in results[batch] if v is not None], key=lambda t: t[0])
        if not pairs:
            continue
        ks = [k for k, _ in pairs]
        ys = [float(v) for _, v in pairs]
        all_vals.extend(ys); all_ks.update(ks)
        (line,) = ax.plot(ks, ys, marker="o", label=f"batch={batch}")
        batch_color[batch] = line.get_color()

    if not all_vals:
        print("No feasible points to plot.")
        plt.close(fig); return None, None

    # 2) Diamonds at k* (same color)
    if k_stars:
        for b, kstar in k_stars.items():
            if b not in results or b not in batch_color or kstar is None:
                continue
            pairs = sorted([(k, v) for k, v in results[b] if v is not None], key=lambda t: t[0])
            ks = [k for k, _ in pairs]; ys = [float(v) for _, v in pairs]
            if kstar in ks:
                yi = ys[ks.index(kstar)]
                ax.plot([kstar], [yi],
                        marker="D", markersize=7, linestyle="None",
                        color=batch_color[b], zorder=5,
                        path_effects=[pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])

    # Axes cosmetics
    ax.set_xlabel("k (number of rounds)")
    ax.set_ylabel("Objective (latency)")
    ax.set_title(title)
    ax.grid(True, which="both", axis="both", linewidth=0.6, alpha=0.5)
    ax.set_xticks(sorted(all_ks))
    ax.legend()

    # 3) Banner: one label per x=k, colored batch numbers in brackets
    if k_stars:
        # group batches by chosen k*
        by_k = {}
        for b, kx in k_stars.items():
            if kx is None: continue
            by_k.setdefault(kx, []).append(b)
        if by_k:
            ymin, ymax = min(all_vals), max(all_vals)
            yr = ymax - ymin if ymax > ymin else 1.0
            y_base = ymax + top_pad_frac * yr
            # current auto limits (with bottom padding preserved)
            y0, y1 = ax.get_ylim()
            yr = max(y1 - y0, 1.0)

            # how much extra room you want above the tallest point
            extra_top = top_pad_frac * yr

            # place banner baseline just above the auto top
            y_base = y1 + extra_top * 0.6  # baseline for the "k*=[...]" text
            ax.set_ylim(y0, y1 + extra_top)  # <- only extend the TOP, keep bottom as-is

            # Ensure renderer is initialized
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # pixels per data-x unit
            x0_px = ax.transData.transform((0, 0))[0]
            x1_px = ax.transData.transform((1, 0))[0]
            data_per_px_x = 0.75 / max(x1_px - x0_px, 1e-9)

            # font properties to measure widths
            fp = FontProperties(size=label_fontsize)

            for kx in sorted(by_k.keys()):
                batches_here = sorted(by_k[kx])

                # Build tokens: ["k*=[", "b1", ",", "b2", ..., "]"]
                tokens = []
                tokens.append(("k*=[", "black"))
                for i, b in enumerate(batches_here):
                    tokens.append((f"{b}", batch_color.get(b, "black")))
                    if i < len(batches_here) - 1:
                        tokens.append((",", "black"))
                tokens.append(("]", "black"))

                # Measure total pixel width with renderer metrics (robust!)
                total_px = 0.0
                widths_px = []
                for text, _color in tokens:
                    w_px, _h_px, _des = renderer.get_text_width_height_descent(
                        text, fp, ismath=False
                    )
                    widths_px.append(w_px)
                    total_px += w_px

                # Starting x so the whole banner is centered above kx
                x_start = kx - (total_px * data_per_px_x) / 2.0
                x_cursor = x_start

                # Place each token sequentially
                for (text, color), w_px in zip(tokens, widths_px):
                    ax.text(x_cursor, y_base, text,
                            ha="left", va="bottom",
                            fontsize=label_fontsize, color=color)
                    x_cursor += w_px * data_per_px_x
    # After plotting your curves and diamonds:
    ax.margins(y=0.05)
    ax.relim()
    ax.autoscale_view()

    # Get auto-scaled limits
    y0, y1 = ax.get_ylim()
    yr = max(y1 - y0, 1.0)

    # baseline for the banner: just a little above the current top
    y_base = y1 + 0.02 * yr  # <-- 2% above current top

    # extend the ylim only enough to include the labels
    ax.set_ylim(y0, y_base + 0.05 * yr)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax