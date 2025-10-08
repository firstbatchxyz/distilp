"""
HALDA end-to-end (parameters + Algorithm 1 with Gurobi), using the same notation as the paper.

References to the paper (prima.cpp / Halda):
- LDA definition: Eqs. (1)-(5)                           [Sec. 3.2]
- Fixed-k ILP:    Eqs. (6)-(10)                          [Sec. 3.3]
- Vectorized a,b,c and selector matrices Pw, Pn          [App. A.3]
- RAM/VRAM bounds (z, z_gpu) and P^gpu_n                 [App. A.3]
- Case constraints (M1..M4) and GPU bounds: (28)-(37)    [App. A.3]
- Coefficients b', α_m, β_m, ξ_m and constants (Eq. 21)  [App. A.3]
- Algorithm 1 (HALDA), lines 1-17                        [Sec. 3.3]
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
import matplotlib as mpl

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple
from collections import defaultdict

import json
import re
import gurobipy as gp
from gurobipy import GRB

try:
    # Package context
    from .components.dataclasses import DeviceProfile, ModelProfile, QuantPerf
    from .components.plotter import plot_k_curve
except Exception:
    # Script context fallback
    from .components.dataclasses import DeviceProfile, ModelProfile, QuantPerf
    from .components.plotter import plot_k_curve


Node = Tuple[int, int]  # (expert_id like 12, layer_id like 0)
def kv_size(model: ModelProfile) -> int:
    # kv cache size: 2 * (model.hk * model.ek + model.hv * model.ev) * model.n_kv
    return 2 * (model.hk * model.ek + model.hv * model.ev) * model.n_kv

# def _sum_f_over_S(f_by_q: Dict[str, QuantPerf], S_by_q: Dict[str, QuantPerf], q: Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"], batch_size: int = 1) -> float:
#     """
#     Helper: ∑_{q ∈ Q} f_q / S_q   (used in α_m and β_m)
#     Now handles batch sizes in S_by_q (device performance data)
#     batch_size: integer batch size (e.g., 1, 2, 4) - will be converted to "b_1", "b_2", etc.
#     """
#     batch_key = f"b_{batch_size}"
#     total = 0.0
#     if batch_key in f_by_q and q in S_by_q:
#         # Handle new format where S_by_q[q] is a dict with batch sizes
#         if isinstance(S_by_q[q], dict):
#             if batch_key not in S_by_q[q]:
#                 raise ValueError(f"Batch size {batch_size} (key '{batch_key}') not found in S_by_q[{q}]")
#             s_val = S_by_q[q][batch_key]
#         else:
#             # Old format compatibility - direct float values
#             # Get rid of this branch once all data is updated
#             s_val = S_by_q[q]
#
#         # Handle f_by_q which might also have batch sizes in future
#         if isinstance(f_by_q, dict):
#             if batch_key not in f_by_q:
#                 raise ValueError(f"Batch size {batch_size} (key '{batch_key}') not found in f_by_q[{q}]")
#             f_val = f_by_q[batch_key]
#
#         if s_val > 0:
#             total += f_val / s_val
#     return total




def load_transitions(json_path: str):
    """
    Returns:
      transitions: Dict[Node, Dict[Node, float]]
        e.g. transitions[(0, -1)][(5, 0)] = 0.0157
      layers: sorted list of layer ids present (ints)
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # Parse keys like "layer_-1_to_0"
    pat = re.compile(r"^layer_(-?\d+)_to_(-?\d+)$")

    # Parse expert ids like 'expert_12' -> 12
    _expert_pat = re.compile(r"^expert_(\d+)$")
    def parse_expert_id(s: str) -> int:
        m = _expert_pat.match(s)
        if not m:
            raise ValueError(f"Unexpected expert id format: {s}")
        return int(m.group(1))

    transitions: Dict[Node, Dict[Node, float]] = defaultdict(dict)
    layers_set = set()

    for k, mapping in raw.items():
        m = pat.match(k)
        if not m:
            # ignore any other keys
            continue
        src_layer = int(m.group(1))
        dst_layer = int(m.group(2))
        layers_set.update([src_layer, dst_layer])

        # mapping: {src_expert: {dst_expert: prob, ...}, ...}
        for src_expert, dests in mapping.items():
            src_node: Node = (parse_expert_id(src_expert), src_layer)
            for dst_expert, p in dests.items():
                dst_node: Node = (parse_expert_id(dst_expert), dst_layer)
                transitions[src_node][dst_node] = float(p)

    layers = sorted(layers_set)
    return transitions, layers

def predecessors(transitions: Dict[Node, Dict[Node, float]]) -> Dict[Node, Dict[Node, float]]:
    """Inverse adjacency with probabilities."""
    prevs: Dict[Node, Dict[Node, float]] = defaultdict(dict)
    for u, nbrs in transitions.items():
        for v, p in nbrs.items():
            prevs[v][u] = p
    return prevs

def compute_visit_probabilities(
    transitions: Dict[Node, Dict[Node, float]], layers: Iterable[int],
    source_visits: Dict[Node, float] | None = None
) -> Dict[Node, float]:
    """
    Dynamic pass over layers: for each layer L>min_layer,
    pi[v] = sum_{u in previous layer(s)} pi[u] * P(u->v).
    If source_visits provided, it seeds visit prob(s) for the earliest layer.
    Otherwise, if exactly one node exists in the earliest layer, it is set to 1.0.
    """
    layers = list(sorted(layers))
    pi: Dict[Node, float] = defaultdict(float)
    prevs = predecessors(transitions)

    # Find nodes present per layer
    nodes_by_layer: Dict[int, set[Node]] = defaultdict(set)
    for u in transitions:
        nodes_by_layer[u[1]].add(u)
        for v in transitions[u]:
            nodes_by_layer[v[1]].add(v)

    # Seed the earliest layer
    first_layer = layers[0]
    if source_visits:
        for n, val in source_visits.items():
            if n[1] != first_layer:
                raise ValueError("source_visits must be for the earliest layer only.")
            pi[n] = float(val)
    else:
        # If there is exactly one node in the first layer, assume visit prob 1.0
        if len(nodes_by_layer[first_layer]) == 1:
            (only_node,) = tuple(nodes_by_layer[first_layer])
            pi[only_node] = 1.0
        # Else leave as zeros unless you provide source_visits

    # Forward pass across layers
    for li in range(1, len(layers)):
        L_prev = layers[li - 1]
        L_curr = layers[li]
        # For every node v in current layer, sum incoming from any u in previous layer
        for v in nodes_by_layer[L_curr]:
            inc = 0.0
            for u, p in prevs.get(v, {}).items():
                if u[1] == L_prev:
                    inc += pi[u] * p
            pi[v] = inc  # overwrite; if multiple prev layers exist, extend logic accordingly

    return dict(pi)

def classify_device_case(
    dev: DeviceProfile,
) -> int:
    """
    Decide Case 1..4 for device m given tentative (w_m, n_m, k),
    following the inequality structure in Eqs. (28)-(33).

    Case 1 (M1): macOS, Metal disabled
    Case 2 (M2): macOS, Metal enabled
    Case 3 (M3): Linux/Android

    """
    if dev.os_type == "mac_no_metal":
        return 1
    elif dev.os_type == "mac_metal":
        return 2
    else:
        return 3

def assign_sets(
    devs: List[DeviceProfile],
) -> Dict[str, List[int]]:
    """
    Partition devices into M1..M3
    """
    M1: List[int] = []
    M2: List[int] = []
    M3: List[int] = []
    for i, d in enumerate(devs):
        case = classify_device_case(d)
        if case == 1:
            M1.append(i)
        elif case == 2:
            M2.append(i)
        else:
            M3.append(i)

    return {"M1": M1, "M2": M2, "M3": M3}

def _gpu_table(dev: DeviceProfile) -> Optional[QuantPerf]:
    """
    Pick the GPU FLOPS table for this device (Metal preferred when present).
    """
    if dev.has_metal and dev.sgpu_metal:
        return dev.sgpu_metal
    if dev.has_cuda and dev.sgpu_cuda:
        return dev.sgpu_cuda
    return None

def _pick_T_gpu(dev: DeviceProfile) -> Optional[float]:
    """
    Pick the GPU register-loading throughput T^{gpu}_m (bytes/s).
    """
    if dev.has_metal and dev.T_metal:
        return dev.T_metal
    if dev.has_cuda and dev.T_cuda:
        return dev.T_cuda
    return None

def alpha_beta(
    dev: DeviceProfile, model: ModelProfile
) -> Tuple[float, float]:
    """
    α_m, β_m, ξ_m exactly as defined under Assumption 1.  [App. A.3, Eq. 21 block]
      α_m =  Σ_q f_q/scpu_q  + t^{kv_cpy,cpu}_m + b'/T^{cpu}_m
      β_m =  Σ_q f_q/sgpu_q  - Σ_q f_q/scpu_q + (t^{kv_cpy,gpu}_m - t^{kv_cpy,cpu}_m)
             + (b'/T^{gpu}_m - b'/T^{cpu}_m)    (0 if no GPU path)
      ξ_m =  (t^{ram->vram}_m + t^{vram->ram}_m)·(1 - I_{UMA}) + t^{comm}_m
    """
    # bprime = b_prime(model)
    # α_m (CPU path)
    req_flops_by_router = model.router_flops["1"]
    req_flops_by_attention = model.attn_flops["decode"]["b_1"][0]
    req_flops_by_expert = model.flops_per_active_expert_per_token["1"]
    flops_by_device_cpu = dev.scpu["F16"]["b_1"]

    # comp_cpu = _sum_f_over_S(model.f_by_quant, dev.scpu, model.Q)
    comp_cpu = (req_flops_by_expert + req_flops_by_router + req_flops_by_attention)/flops_by_device_cpu
    alpha = (comp_cpu)
             # + dev.t_kvcpy_cpu + (bprime / dev.T_cpu))

    # β_m (GPU minus CPU path), 0 if no GPU available
    S_gpu = _gpu_table(dev)
    T_gpu = _pick_T_gpu(dev)
    if S_gpu is not None and T_gpu is not None:
        flops_by_device_gpu = S_gpu["F16"]["b_1"]
        # comp_gpu = _sum_f_over_S(model.f_by_quant, S_gpu, model.Q)
        comp_gpu = (req_flops_by_expert+req_flops_by_router+req_flops_by_attention)/flops_by_device_gpu
        beta = (
            comp_gpu
            # + (dev.t_kvcpy_gpu - dev.t_kvcpy_cpu)
            # + (bprime / T_gpu - bprime / dev.T_cpu)
        )
    else:
        beta = 0.0

    # # ξ_m (traffic + comm)
    # # dev.t_ram2vram + dev.t_vram2ram is done once per round as it is done for sequence of layers within a window.
    # xi = (dev.t_ram2vram + dev.t_vram2ram) * (
    #     0 if dev.is_unified_mem else 1
    # ) + dev.t_comm
    return alpha, beta

def objective_vectors(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sets: Dict[str, List[int]],
) -> Tuple[List[float], List[float]]:
    """
    Build a, b, c as in the vectorized form (block right after ILFP).  [App. A.3]
       a_m = α_m + b'/s_disk   (m ∈ M1)
             α_m + b /s_disk   (m ∈ M2)
             α_m + b'/s_disk   (m ∈ M3)
             α_m              (m ∈ M4)
       b_m = 0                 (m ∈ M1)
             β_m               (m ∈ M2)
             β_m - b'/s_disk   (m ∈ M3)
             β_m               (m ∈ M4)
       c_m = ξ_m  (all)
    """
    a: List[float] = [0.0] * len(devs)
    b: List[float] = [0.0] * len(devs)

    for i, d in enumerate(devs):
        alpha, beta = alpha_beta(d, model)
        print(f"alpha: {alpha}, beta: {beta}")

        if i in sets["M1"]:
            a[i] = alpha
            b[i] = 0.0
        elif i in sets["M2"]:
            a[i] = alpha
            b[i] = beta
        elif i in sets["M3"]:
            a[i] = alpha
            b[i] = beta
    return a, b


@dataclass
class MINLPResult:
    x: List[int]
    g: List[int]
    r: List[int]
    obj_value: float

# --- Visualization helper ----------------------------------------------------
# Discrete heatmap (E x L) colored by device and intensity (x darkest, g middle,
# r lightest). Uses the variable ordering used in x_index_list: (d, l, e).



def _shade_rgb(rgb: Sequence[float], factor: float) -> tuple[float, float, float]:
    """
    Blend rgb toward white by `factor` in [0,1]. factor=0 -> original (darkest),
    larger factor -> lighter. We ensure values stay in [0,1].
    """
    r, g, b = rgb[:3]
    return (
        1.0 - (1.0 - r) * (1.0 - factor),
        1.0 - (1.0 - g) * (1.0 - factor),
        1.0 - (1.0 - b) * (1.0 - factor),
    )


def plot_solution_heatmap(
    result: MINLPResult,
    E: int,
    L: int,
    M: int,
    title: str | None = None,
    show_legend: bool = True,
):
    """
    Plot an E x L heatmap summarizing the ILP solution.

    Coloring rule (per cell e,l):
      - Choose device d and intensity by priority x > g > r.
      - For that (e,l), if any x[e,l,d]==1 for some d, use that device's color (darkest).
        Else if any g[e,l,d]==1, use same device color (medium).
        Else if any r[e,l,d]==1, use same device color (lightest).
        If multiple devices tie within a level, the smallest d is used.
      - Cells with no assignment remain blank.

    Discrete colors: each device has its own hue; intensity encodes x/g/r.

    Parameters
    ----------
    result : ILPResult (with x, g, r as flat lists in (d,l,e) order)
    E, L, M : problem sizes
    title : optional title for the plot
    show_legend : whether to draw a legend mapping devices and intensities
    """
    # Rebuild 3-D arrays with index order (e, l, d) for convenience
    x3 = np.zeros((E, L, M), dtype=int)
    g3 = np.zeros((E, L, M), dtype=int)
    r3 = np.zeros((E, L, M), dtype=int)

    pos = 0
    for d in range(M):
        for l in range(L):
            for e in range(E):
                x3[e, l, d] = int(result.x[pos])
                r3[e, l, d] = int(result.r[pos])
                g3[e, l, d] = int(result.g[pos])
                pos += 1

    # Build E x L matrix of discrete codes: code = 3*d + level (0=r lightest, 1=g, 2=x darkest)
    codes = -np.ones((E, L), dtype=int)
    for e in range(E):
        for l in range(L):
            # Priority X > G > R
            ds = np.where(x3[e, l, :] == 1)[0]
            if ds.size > 0:
                d = int(ds.min())
                codes[e, l] = 3 * d + 2  # darkest
                continue
            ds = np.where(g3[e, l, :] == 1)[0]
            if ds.size > 0:
                d = int(ds.min())
                codes[e, l] = 3 * d + 1  # medium
                continue
            ds = np.where(r3[e, l, :] == 1)[0]
            if ds.size > 0:
                d = int(ds.min())
                codes[e, l] = 3 * d + 0  # lightest
                continue
            # else remains -1 (unassigned)

    # High-contrast categorical palette for devices
    HC_COLORS = [
        # ColorBrewer Set1 (9)
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999",
        # Dark2 (8)
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
        "#66a61e", "#e6ab02", "#a6761d", "#666666",
        # Set2 (8)
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
    ]

    if M <= len(HC_COLORS):
        base = np.array([mcolors.to_rgb(c) for c in HC_COLORS[:M]])
    else:
        # Fallback for many devices: evenly spaced hues in HSV
        base = np.array([
            mpl.colors.hsv_to_rgb((h, 0.75, 0.95))
            for h in np.linspace(0, 1, M, endpoint=False)
        ])
    light_f, mid_f, dark_f = 0.85, 0.45, 0.2

    colors: list[tuple[float, float, float]] = []
    for d in range(M):
        rgb = tuple(base[d % len(base)][:3])
        # Order of levels must match codes above: 0=r (light), 1=g (mid), 2=x (dark)
        colors.append(_shade_rgb(rgb, light_f))  # r
        colors.append(_shade_rgb(rgb, mid_f))    # g
        colors.append(_shade_rgb(rgb, dark_f))   # x

    cmap = ListedColormap(colors, name="dev_xgr")

    # Mask unassigned
    masked = np.ma.masked_where(codes < 0, codes)

    # Figure size scaled to problem; keep sane bounds
    fig_w = max(6.0, L * 0.35)
    fig_h = max(4.0, E * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(masked, origin="lower", interpolation="nearest", aspect="auto", cmap=cmap)

    ax.set_xlabel("Layer (l)")
    ax.set_ylabel("Expert (e)")
    ax.set_xticks(np.arange(L))
    ax.set_yticks(np.arange(E))

    if title:
        ax.set_title(title)
    else:
        our_title = "MOE allocation with average latency: " + str(round(result.obj_value, 6)) + "s"
        ax.set_title(our_title)

    # Build a compact legend
    if show_legend:
        import matplotlib.patches as mpatches
        legend_handles = []
        for d in range(M):
            rgb = base[d % len(base)][:3]
            # show the three intensities for this device
            legend_handles.append(
                mpatches.Patch(color=_shade_rgb(rgb, 0.2), label=f"D{d}: CPU")
            )
            legend_handles.append(
                mpatches.Patch(color=_shade_rgb(rgb, 0.45), label=f"D{d}: GPU")
            )
            legend_handles.append(
                mpatches.Patch(color=_shade_rgb(rgb, 0.85), label=f"D{d}: Disk")
            )
        # To avoid overly large legends, collapse if too many devices
        if len(legend_handles) > 24:
            # Show only device IDs (dark patch) and a separate shade key
            legend_handles = []
            for d in range(M):
                rgb = base[d % len(base)][:3]
                legend_handles.append(
                    mpatches.Patch(color=_shade_rgb(rgb, 0.2), label=f"D{d}")
                )
            shade_handles = [
                mpatches.Patch(color=(0.2, 0.2, 0.2), label="x (dark)"),
                mpatches.Patch(color=(0.45, 0.45, 0.45), label="g (mid)"),
                mpatches.Patch(color=(0.85, 0.85, 0.85), label="r (light)"),
            ]
            ax.legend(handles=legend_handles[:10] + shade_handles, ncol=4, fontsize=8, loc="upper right")
        else:
            ax.legend(handles=legend_handles, ncol=min(6, 3 * M), fontsize=8, loc="upper right")

    plt.tight_layout()
    return fig, ax

def gurobi_solve(
    path: str,
    devs: List[DeviceProfile],
    model: ModelProfile,
    mip_gap: float = 0.0005,
    time_limit: float = 120.0,
)-> MINLPResult:
    M = len(devs)
    L = model.L
    E = model.n_routed_experts
    # E = 20
    sets = assign_sets(devs)
    kv_cache_size = kv_size(model)
    router_size = model.router_bytes["1"]
    expert_size = model.bytes_per_expert["1"]
    attention_size = model.attn_bytes[0]

    alpha, beta = objective_vectors(devs, model, sets)

    transitions, layers = load_transitions(path)
    pi = compute_visit_probabilities(transitions, layers)

    edges: List[Tuple[Node, Node, float]] = []
    for u, nbrs in transitions.items():
        if u[1] != -1:
            for v, p in nbrs.items():
                # if u[0] <=E-1 and v[0]<=E-1:
                    f = pi[u[0], u[1]] * float(p)
                    if f > 0.0:
                        edges.append((u, v, f))

    x_index_list = []
    for d in range(M):
        for l in range(L):
            for e in range(E):
                x_index_list.append((e, l, d))

    m = gp.Model("MoeDelo")

    # Add termination criterion for Gurobi
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    # m.Params.LogToConsole = False
    # Decision variables

    x = m.addVars(x_index_list, vtype=GRB.BINARY, name="x")  # executed on CPU
    y = m.addVars(x_index_list, vtype=GRB.BINARY, name="y")  # resident in device
    # r = m.addVars(x_index_list, vtype=GRB.BINARY, name="r")  # disk-load if not in RAM
    g = m.addVars(x_index_list, vtype=GRB.BINARY, name="g")  # executed on GPU
    zc = m.addVars(L, M, vtype=GRB.BINARY, name="zc")     # layer at device ram
    zg = m.addVars(L, M, vtype=GRB.BINARY, name="zg")  # layer at device vram

    # Each expert have to be executed in at least one device
    for l in range(L):
        for e in range(E):
            m.addConstr(gp.quicksum(y[e, l, d] for d in range(M)) >= 1, name=f"exec_unique[{e}]")

    # # Disk load linking
    # for l in range(L):
    #     for e in range(E):
    #         for d in range(M):
    #             m.addConstr(r[e, l, d] >= y[e, l, d] - x[e, l, d] - g[e, l, d], name=f"r_lb[{e},{l},{d}]")
    #             m.addConstr(r[e, l, d] <= y[e, l, d], name=f"r_ub[{e},{l},{d}]")

    # Layer-DeviceRAM assignment
    for l in range(L):
        for d in range(M):
            if d not in sets["M2"]:
                m.addConstr(gp.quicksum(x[e, l, d] for e in range(E)) <= E * zc[l,d], name=f"layer_allocation[{l},{d}]")

    # Layer-DeviceVRAM assignment
    for l in range(L):
        for d in range(M):
            if d not in sets["M2"]:
                m.addConstr(gp.quicksum(g[e, l, d] for e in range(E)) <= E * zg[l,d], name=f"layer_allocation[{l},{d}]")

    for l in range(L):
        for d in sets["M2"]:
            m.addConstr(gp.quicksum(g[e, l, d] + x[e, l, d] for e in range(E)) <= E * zc[l,d], name=f"unified_layer_allocation[{l},{d}]")

    # CPU and GPU linking
    for l in range(L):
        for e in range(E):
            for d in range(M):
                m.addConstr(x[e, l, d] + g[e, l, d]  <= y[e, l, d], name=f"g_le_x[{e},{l},{d}]")

    # # GPU linking
    # for l in range(L):
    #     for e in range(E):
    #         for d in range(M):
    #             m.addConstr(g[e, l, d] <= y[e, l, d], name=f"g_le_y[{e},{l},{d}]")

    # RAM constraints
    for d in sets["M1"]:
        rhs = devs[d].d_avail_ram
        m.addConstr(
            gp.quicksum(expert_size * x[e, l, d] for l in range(L) for e in range(E)) <= rhs - gp.quicksum(
                zc[l, d] * (kv_cache_size + router_size + attention_size) for l in range(L)), name=f"M1_ram[{d}]")

    for d in sets["M2"]:
        dev_metal = float(devs[d].d_avail_metal or 0)
        rhs = dev_metal - devs[d].c_gpu
        m.addConstr(
            gp.quicksum(expert_size * (x[e, l, d]+g[e, l, d]) for l in range(L) for e in range(E)) <= rhs - gp.quicksum(
                zc[l, d] * (kv_cache_size + router_size + attention_size) for l in range(L)), name=f"M2_ram[{d}]")

    # # VRAM
    # for d in range(M):
    #     i = devs[d]
    #     # CUDA VRAM bound
    #     # if d.has_cuda and d.d_avail_cuda is not None:
    #     #     rhs = d.d_avail_cuda - d.c_gpu
    #     #     m.addConstr(n[i] <= rhs, name=f"cuda_vram[{i}]")
    #     # Metal shared-memory bound
    #     if i.has_metal and i.d_avail_metal is not None:
    #         rhs = i.d_avail_metal - i.c_gpu
    #         m.addConstr(gp.quicksum(expert_size * g[e, l, d] for l in range(L) for e in range(E)) <= rhs - gp.quicksum(
    #             zg[l, d] * (kv_cache_size + router_size + attention_size) for l in range(L)), name=f"metal_shared[{d}]")

    # No GPU
    for d in range(M):
        has_cuda = bool(devs[d].has_cuda and devs[d].d_avail_cuda is not None)
        has_metal = bool(devs[d].has_metal and devs[d].d_avail_metal is not None)
        if not (has_cuda or has_metal):
            m.addConstr(gp.quicksum(g[e, l, d] for e in range(E) for l in range(L)) == 0, name=f"no_gpu[{d}]")

    # #Comm linearization
    # w = {}
    # for (u, v, _) in edges:
    #     for d1 in range(M):
    #         for d2 in  range(M):
    #             w[(u[0], v[0], u[1], v[1], d1, d2)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
    #                                              name=f"w[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")
    #             # Full McCormick (tight) is 3 constraints:
    #             m.addConstr(w[(u[0], v[0], u[1], v[1], d1, d2)] <= y[u[0], u[1], d1],
    #                         name=f"w_le_yu[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")
    #             m.addConstr(w[(u[0], v[0], u[1], v[1], d1, d2)] <= y[v[0], v[1], d2],
    #                         name=f"w_le_yv[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")
    #             m.addConstr(w[(u[0], v[0], u[1], v[1], d1, d2)] >= y[u[0], u[1], d1] + y[v[0], v[1], d2] - 1,
    #                         name=f"w_ge_sum-1[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")

    # m.update()


    # Objective components
    compute_cost = gp.quicksum(
        pi[e, l] * ((g[e, l, d] * beta[d]) + (y[e, l, d]- g[e, l, d]- x[e, l, d])* alpha[d]) + alpha[d] * x[e, l, d]
        for l in range(L) for e in range(E) for d in range(M))

    # comm_cost = gp.quicksum(
    #     flow * 0.002 * w[(u[0], v[0], u[1], v[1], d1, d2)]
    #     for (u, v, flow) in edges for d1 in range(M) for d2 in range(M)
    # )

    comm_cost = gp.quicksum(
        flow * 0.02 * (y[u[0], u[1], d1] * y[v[0], v[1], d2] )
        for (u, v, flow) in edges for d1 in range(M) for d2 in range(M)
    )


    load_cost = gp.quicksum(
        pi[e, l] * (expert_size/devs[d].s_disk) * (y[e, l, d] - x[e, l, d] - g[e, l, d]) for e in range(E) for l in range(L) for d in range(M))

    m.setObjective(compute_cost + comm_cost + load_cost, GRB.MINIMIZE)

    m.optimize()
    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # m.write("model.lp")
        # input("Paused. Press Enter to continue...")
        raise RuntimeError(f"Gurobi status {m.status}.")

    x_sol = [int(round(x[i].X)) for i in x_index_list]
    g_sol = [int(round(g[i].X)) for i in x_index_list]
    r_sol = [1 if x[i].X + g[i].X < y[i].X else 0  for i in x_index_list]

    for i in m.getVars():
        if i.X > 0 and ("x" in i.varName or "g" in i.varName or "w" in i.varName):
            print(i.varName, i.X)

    result = MINLPResult(x=x_sol, g=g_sol, r=r_sol, obj_value=float(m.ObjVal))
    fig, ax = plot_solution_heatmap(result, E=E, L=L, M=M)
    plt.show()

    # ---------------- Structured summary of assignments (MILP) ----------------
    # Classify per device:
    #  GPU if g[e,l,d] = 1
    #  CPU if x[e,l,d] = 1
    #  Disk if y[e,l,d] = 1 and x=g=0
    per_device = {d: {"CPU": [], "GPU": [], "Disk": []} for d in range(M)}

    for l in range(L):
        for e in range(E):
            for d in range(M):
                yv = float(y[e, l, d].X) if (e, l, d) in y else 0.0
                xv = float(x[e, l, d].X) if (e, l, d) in x else 0.0
                gv = float(g[e, l, d].X) if (e, l, d) in g else 0.0
                if gv >= 0.5:
                    per_device[d]["GPU"].append((e, l))
                elif xv >= 0.5:
                    per_device[d]["CPU"].append((e, l))
                elif yv >= 0.5:
                    per_device[d]["Disk"].append((e, l))
                # else: not present on this device

    print("\n================ Residency & Execution Summary (MILP) ================")
    try:
        print(f"Total objective (model ObjVal): {m.ObjVal:.6f}s\n")
    except Exception:
        pass

    grand_counts = {"CPU": 0, "GPU": 0, "Disk": 0}
    for d in range(M):
        print(f"Device {d}:")
        for mode in ("CPU", "GPU", "Disk"):
            pairs = per_device[d][mode]
            grand_counts[mode] += len(pairs)
            tokens = [f"e{e}-l{l}" for (e, l) in pairs]
            if not tokens:
                print(f"  {mode:<4}: (none)")
            else:
                print(f"  {mode:<4}: {len(tokens)} item(s)")
                row = []
                for i, t in enumerate(tokens, 1):
                    row.append(t)
                    if i % 12 == 0:
                        print("          " + ", ".join(row))
                        row = []
                if row:
                    print("          " + ", ".join(row))
        print("")

    print("Totals:")
    print(f"  CPU : {grand_counts['CPU']}")
    print(f"  GPU : {grand_counts['GPU']}")
    print(f"  Disk: {grand_counts['Disk']}")
    print("===================================================================\n")

    return result


def gurobi_solve_SA(
        path: str,
        devs: List[DeviceProfile],
        model: ModelProfile,
        mip_gap: float = 0.0005,  # kept for API symmetry; unused in SA
        time_limit: float = 120.0,
) -> MINLPResult:
    """
    Simulated annealing on the residency assignment y (exactly ONE device per (e,l)).
    For each candidate y, we solve a fast greedy subproblem (no IP) to decide CPU/GPU/Disk
    for each (e,l) on its assigned device, respecting RAM/VRAM and per-(layer,device)
    first-use overheads (KV+router+attn).

    Objective evaluated per solution:
        total = Compute + Load + Comm
      where:
        Compute + Load are computed from the greedy subproblem, and
        Comm(y) = sum_{(u->v)} flow(u->v)*COMM_COEFF * I{device(u)!=device(v)}.

    Returns
    -------
    MINLPResult with x,g,r flattened in (d,l,e) order and obj_value = total latency.
    """
    import time
    import random
    from collections import defaultdict

    start_time = time.time()
    M = len(devs)
    L = model.L
    E = model.n_routed_experts

    sets = assign_sets(devs)
    kv_cache_size = kv_size(model)
    router_size = model.router_bytes["1"]
    expert_size = model.bytes_per_expert["1"]
    attention_size = model.attn_bytes[0]

    alpha_vec, beta_vec = objective_vectors(devs, model, sets)

    # Transitions and visit probabilities
    transitions, layers = load_transitions(path)
    pi = compute_visit_probabilities(transitions, layers)

    # Edges with nonzero flow
    edges: List[Tuple[Node, Node, float]] = []
    for u, nbrs in transitions.items():
        if u[1] != -1:
            for v, p in nbrs.items():
                f = pi.get((u[0], u[1]), 0.0) * float(p)
                if f > 0.0:
                    edges.append((u, v, f))

    # el_weight = defaultdict(float)
    # for (u, v, f) in edges:
    #     el_weight[(u[0], u[1])] += float(f)
    #     el_weight[(v[0], v[1])] += float(f)
    #
    # # Build fixed index mapping and weighted CDF for fast sampling
    # el_index = {}
    # el_keys = []
    # weights = []
    # eps_w = 1e-12
    # k = 0
    # for l_idx in range(L):
    #     for e_idx in range(E):
    #         el_index[(e_idx, l_idx)] = k
    #         el_keys.append((e_idx, l_idx))
    #         weights.append(el_weight.get((e_idx, l_idx), 0.0) + eps_w)
    #         k += 1
    # # Normalize to CDF
    # total_w = float(sum(weights)) if weights else 1.0
    # cdf = []
    # acc = 0.0
    # for w in weights:
    #     acc += (w / total_w)
    #     cdf.append(acc)

    # def _sample_el_edge_aware(rng_local: random.Random) -> tuple[int, int]:
    #     """Inverse-CDF sample of (e,l) proportional to incident flow weight."""
    #     r = rng_local.random()
    #     # binary search in cdf
    #     lo, hi = 0, len(cdf) - 1
    #     while lo < hi:
    #         mid = (lo + hi) // 2
    #         if r <= cdf[mid]:
    #             hi = mid
    #         else:
    #             lo = mid + 1
    #     return el_keys[lo]

    # Communication coefficient (match your Benders setup for apples-to-apples)
    COMM_COEFF = 0.02

    # ---- Exact-one y state: map (e,l) -> device id ----
    # Greedy initializer: place higher-π first, prefer feasible GPU (VRAM), else CPU (RAM), else any.
    y_state: Dict[Tuple[int, int], int] = {}

    # Soft capacity trackers for initializer only (the greedy subproblem below enforces capacities again)
    ram_cpu = [float(devs[d].d_avail_ram or 0.0) for d in range(M)]
    ram_vram = [
        float((devs[d].d_avail_metal or 0.0) - (devs[d].c_gpu or 0.0))
        if devs[d].has_metal and devs[d].d_avail_metal is not None else 0.0
        for d in range(M)
    ]
    zc_used0 = [[0 for _ in range(M)] for _ in range(L)]
    zg_used0 = [[0 for _ in range(M)] for _ in range(L)]

    def _can_gpu_init(l_idx: int, d_idx: int) -> bool:
        # device supports GPU path?
        if not ((devs[d_idx].has_metal and devs[d_idx].d_avail_metal is not None) or
                (devs[d_idx].has_cuda and devs[d_idx].d_avail_cuda is not None)):
            return False
        need = expert_size
        overhead = (kv_cache_size + router_size + attention_size) if zg_used0[l_idx][d_idx] == 0 else 0
        return ram_vram[d_idx] >= need + overhead

    def _place_gpu_init(e_idx: int, l_idx: int, d_idx: int):
        overhead = (kv_cache_size + router_size + attention_size) if zg_used0[l_idx][d_idx] == 0 else 0
        ram_vram[d_idx] -= (expert_size + overhead)
        zg_used0[l_idx][d_idx] = 1
        y_state[(e_idx, l_idx)] = d_idx

    def _can_cpu_init(l_idx: int, d_idx: int) -> bool:
        need = expert_size
        overhead = (kv_cache_size + router_size + attention_size) if zc_used0[l_idx][d_idx] == 0 else 0
        return ram_cpu[d_idx] >= need + overhead

    def _place_cpu_init(e_idx: int, l_idx: int, d_idx: int):
        overhead = (kv_cache_size + router_size + attention_size) if zc_used0[l_idx][d_idx] == 0 else 0
        ram_cpu[d_idx] -= (expert_size + overhead)
        zc_used0[l_idx][d_idx] = 1
        y_state[(e_idx, l_idx)] = d_idx

    el_list = [(e, l, pi.get((e, l), 0.0)) for l in range(L) for e in range(E)]
    el_list.sort(key=lambda t: t[2], reverse=True)

    for (e_i, l_i, _) in el_list:
        gpu_cands = [(beta_vec[d], d) for d in range(M) if _can_gpu_init(l_i, d)]
        cpu_cands = [(alpha_vec[d], d) for d in range(M) if _can_cpu_init(l_i, d)]
        if gpu_cands:
            gpu_cands.sort()
            _place_gpu_init(e_i, l_i, gpu_cands[0][1])
        elif cpu_cands:
            cpu_cands.sort()
            _place_cpu_init(e_i, l_i, cpu_cands[0][1])
        else:
            # fallback: choose fastest disk; break ties by current load (#items already assigned)
            # Build per-device counts from current y_state
            counts = [0] * M
            for (_, _), dd in y_state.items():
                counts[dd] += 1
            best_d = 0 if M > 0 else 0
            best_key = None
            for d in range(M):
                speed = float(devs[d].s_disk or 1.0)
                key = (-speed, counts[d])  # faster disk first (larger speed => smaller -speed), then fewer assigned
                if best_key is None or key < best_key:
                    best_key = key
                    best_d = d
            y_state[(e_i, l_i)] = best_d

    # ---- Helpers for fast subproblem (no IP) ----
    def _device_cap_cpu(d_idx: int) -> float:
        if d_idx in sets["M1"]:
            return float(devs[d_idx].d_avail_ram)
        if d_idx in sets["M2"]:
            return float(devs[d_idx].d_avail_metal - devs[d_idx].c_gpu)
        return float(devs[d_idx].d_avail_ram)

    def _device_cap_gpu(d_idx: int) -> float:
        di = devs[d_idx]
        if di.has_metal and di.d_avail_metal is not None:
            return float(di.d_avail_metal - di.c_gpu)
        return 0.0

    OVERHEAD = float(kv_cache_size + router_size + attention_size)

    # Costs per (e,l,d) under your objective structure:
    # Disk: π*(α + expert_size/s_disk), CPU: α, GPU: π*β
    def _cost_disk(e: int, l: int, d: int) -> float:
        pi_el = float(pi.get((e, l), 0.0))
        ld = float(devs[d].s_disk or 1.0)
        return pi_el * (alpha_vec[d] + (expert_size / ld))

    def _cost_cpu(e: int, l: int, d: int) -> float:
        pi_el = float(pi.get((e, l), 0.0))
        return float(pi_el*alpha_vec[d])

    def _cost_gpu(e: int, l: int, d: int) -> float:
        pi_el = float(pi.get((e, l), 0.0))
        return pi_el * float(beta_vec[d])

    from collections import defaultdict

    def _solve_subproblem_fast(state: Dict[Tuple[int, int], int]):
        """
        Greedy choice CPU/GPU vs Disk per (e,l) on its assigned device, subject to capacities.
        - For devices in M2 (macOS with Metal shared memory): use ONE shared pool per device and a
          single first-use overhead flag per (layer, device), regardless of CPU/GPU mode.
        - For other devices: keep separate CPU and GPU pools and overhead flags as before.
        Returns:
            assign_x: {(e,l) -> d} for CPU,
            assign_g: {(e,l) -> d} for GPU,
            obj_compute_load: total compute+load,
            zc_active, zg_active: indicator dicts for which (l,d) were opened (for non-shared),
            z_shared_active: indicator for shared-memory open (for M2 devices).
        """
        assign_x: Dict[Tuple[int, int], int] = {}
        assign_g: Dict[Tuple[int, int], int] = {}
        zc_active = defaultdict(int)
        zg_active = defaultdict(int)
        z_shared_active = defaultdict(int)

        cap_cpu = {d: _device_cap_cpu(d) for d in range(M)}
        cap_gpu = {d: _device_cap_gpu(d) for d in range(M)}
        # Shared pool for M2 devices
        cap_shared = {}
        for d in range(M):
            if d in sets["M2"]:
                # unified memory size (Metal shared) minus reserved c_gpu
                di = devs[d]
                cap_shared[d] = float(di.d_avail_metal or 0.0) - float(di.c_gpu or 0.0)
            else:
                cap_shared[d] = 0.0  # unused for non-M2

        # Build candidates with dynamic reweighting after overhead flips.
        # We maintain a max-heap keyed by density and invalidate stale entries via version counters.
        from heapq import heappush, heappop

        # Version counters to invalidate stale candidates
        ver_ldm = defaultdict(int)   # key: (l,d,mem) where mem in {"cpu","gpu","shared"}
        ver_el  = defaultdict(int)   # key: (e,l)

        def _mk_candidate(e: int, l: int, d: int, kind: str):
            """Return tuple for heap: (-density, delta, kind, mem, e, l, d, need, v_ldm, v_el).
            kind in {"CPU","GPU"}. mem in {"cpu","gpu","shared"} indicates which pool is used.
            If not beneficial or not supported, return None.
            """
            disk_cost = _cost_disk(e, l, d)
            cpu_cost  = _cost_cpu(e, l, d)
            gpu_cost  = _cost_gpu(e, l, d)
            is_shared = (d in sets["M2"])  # UMA device => unified pool & single overhead per (l,d)

            if kind == "CPU":
                delta = max(0.0, disk_cost - cpu_cost)
                if delta <= 1e-12:
                    return None
                if is_shared:
                    mem = "shared"
                    need = expert_size + (OVERHEAD if z_shared_active[(l, d)] == 0 else 0.0)
                    # shared pool always exists for M2
                    if cap_shared[d] <= 0.0:
                        return None
                else:
                    mem = "cpu"
                    need = expert_size + (OVERHEAD if zc_active[(l, d)] == 0 else 0.0)
                    if cap_cpu[d] <= 0.0:
                        return None
            else:  # kind == "GPU"
                # device must actually support GPU path
                if _device_cap_gpu(d) <= 0.0:
                    return None
                delta = max(0.0, disk_cost - gpu_cost)
                if delta <= 1e-12:
                    return None
                if is_shared:
                    mem = "shared"
                    need = expert_size + (OVERHEAD if z_shared_active[(l, d)] == 0 else 0.0)
                    if cap_shared[d] <= 0.0:
                        return None
                else:
                    mem = "gpu"
                    need = expert_size + (OVERHEAD if zg_active[(l, d)] == 0 else 0.0)
                    if cap_gpu[d] <= 0.0:
                        return None

            density = delta / max(1.0, need)
            return (-density, delta, kind, mem, e, l, d, need, ver_ldm[(l, d, mem)], ver_el[(e, l)])

        # Max-heap of candidates across all (e,l)
        heap = []
        used = set()  # (e,l) already upgraded off Disk

        for (e, l), d in state.items():
            c1 = _mk_candidate(e, l, d, "CPU")
            if c1 is not None:
                heappush(heap, c1)
            c2 = _mk_candidate(e, l, d, "GPU")
            if c2 is not None:
                heappush(heap, c2)

        # Pop best, accept if feasible, then update versions for affected keys so that
        # subsequent pops for same (l,d,mem) or (e,l) are recomputed when encountered.
        while heap:
            neg_density, delta, kind, mem, e, l, d, need, v_ldm_saved, v_el_saved = heappop(heap)
            if (e, l) in used:
                continue
            # Invalidate stale candidates (overhead flipped or (e,l) changed)
            if v_ldm_saved != ver_ldm[(l, d, mem)] or v_el_saved != ver_el[(e, l)]:
                # Recompute and push fresh version (if still beneficial)
                c_new = _mk_candidate(e, l, d, kind)
                if c_new is not None:
                    heappush(heap, c_new)
                continue

            # Check capacity and accept if feasible; otherwise skip
            if mem == "shared":
                if cap_shared[d] < need:
                    continue
                # Accept
                cap_shared[d] -= need
                if kind == "CPU":
                    assign_x[(e, l)] = d
                else:
                    assign_g[(e, l)] = d
                used.add((e, l))
                if z_shared_active[(l, d)] == 0:
                    z_shared_active[(l, d)] = 1
                    ver_ldm[(l, d, "shared")] += 1  # overhead flip => invalidate related candidates
                ver_el[(e, l)] += 1                   # invalidate other mode for this (e,l)

            elif mem == "cpu":
                if cap_cpu[d] < need:
                    continue
                cap_cpu[d] -= need
                assign_x[(e, l)] = d
                used.add((e, l))
                if zc_active[(l, d)] == 0:
                    zc_active[(l, d)] = 1
                    ver_ldm[(l, d, "cpu")] += 1
                ver_el[(e, l)] += 1

            else:  # mem == "gpu"
                if cap_gpu[d] < need:
                    continue
                cap_gpu[d] -= need
                assign_g[(e, l)] = d
                used.add((e, l))
                if zg_active[(l, d)] == 0:
                    zg_active[(l, d)] = 1
                    ver_ldm[(l, d, "gpu")] += 1
                ver_el[(e, l)] += 1

            # After acceptance, push updated candidates for OTHER (e',l') that might be impacted?
            # We rely on lazy invalidation via version counters; when their entries are popped,
            # they will be recomputed using the new overhead state. This keeps the loop efficient.

        # Compute + Load for this assignment
        total_compute_load = 0.0
        for (e, l), d in state.items():
            if assign_x.get((e, l), -1) == d:
                total_compute_load += _cost_cpu(e, l, d)
            elif assign_g.get((e, l), -1) == d:
                total_compute_load += _cost_gpu(e, l, d)
            else:
                total_compute_load += _cost_disk(e, l, d)

        return assign_x, assign_g, total_compute_load, zc_active, zg_active

    # Communication cost for exact-one y
    def _comm_cost_from_state(state: Dict[Tuple[int, int], int]) -> float:
        total = 0.0
        for (u, v, f) in edges:
            if state.get((u[0], u[1]), -1) != state.get((v[0], v[1]), -1):
                total += f * COMM_COEFF
        return total

    # ----- Evaluate current state -----
    assign_x, assign_g, sub_obj, zc_active, zg_active = _solve_subproblem_fast(y_state)

    def _extract_minlp_result() -> MINLPResult:
        x_sol, g_sol, r_sol = [], [], []
        for d in range(M):
            for l in range(L):
                for e in range(E):
                    on_d = 1 if y_state[(e, l)] == d else 0
                    xv = 1 if assign_x.get((e, l), -1) == d else 0
                    gv = 1 if assign_g.get((e, l), -1) == d else 0
                    x_sol.append(xv)
                    g_sol.append(gv)
                    r_sol.append(1 if on_d and (xv + gv == 0) else 0)
        return MINLPResult(x=x_sol, g=g_sol, r=r_sol, obj_value=0.0)

    def _full_obj_from_current() -> float:
        return sub_obj + _comm_cost_from_state(y_state)

    best_result = _extract_minlp_result()
    best_obj = _full_obj_from_current()
    # Track incumbent objective by iteration for plotting
    obj_history = [best_obj]

    # ----- Simulated annealing loop -----
    max_iters = 50000
    # T = max(1e-6, 0.1 * best_obj + 1e-6)
    T = 1000
    cooling = 0.98
    rng = random.Random(42)

    def _neighbor_move(state: Dict[Tuple[int, int], int]) -> Tuple[Tuple[int, int], int, int]:
        # Pure random: pick (e,l) uniformly at random.
        e = rng.randrange(E)
        l = rng.randrange(L)
        d_old = state[(e, l)]
        choices = [d for d in range(M) if d != d_old]
        d_new = rng.choice(choices) if choices else d_old
        return (e, l), d_old, d_new

    for it in range(max_iters):
        if time.time() - start_time >= time_limit - 0.25:
            break
        (e_mv, l_mv), d_old, d_new = _neighbor_move(y_state)

        # Apply move, resolve fast subproblem, compute objective
        y_state[(e_mv, l_mv)] = d_new
        assign_x_new, assign_g_new, sub_obj_new, _, _ = _solve_subproblem_fast(y_state)
        cand_obj = sub_obj_new + _comm_cost_from_state(y_state)

        improve = cand_obj < best_obj - 1e-9

        if not improve:
            p
        accept = improve or (rng.random() < np.exp(-(cand_obj - best_obj) / max(1e-9, T)))
        if accept:
            print("T: " + str(T))
            best_obj = cand_obj
            print(best_obj)
            assign_x, assign_g, sub_obj = assign_x_new, assign_g_new, sub_obj_new
            best_result = _extract_minlp_result()
        else:
            # revert move
            y_state[(e_mv, l_mv)] = d_old

        obj_history.append(best_obj)
        T *= cooling

    best_result.obj_value = float(best_obj)
    # Plot objective trajectory
    plt.figure(figsize=(8, 3.5))
    plt.plot(range(len(obj_history)), obj_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Incumbent objective")
    plt.title("SA incumbent objective by iteration")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()
    # Plot
    fig, ax = plot_solution_heatmap(
        best_result, E=E, L=L, M=M,
        title=f"MOE allocation (SA) — total latency ≈ {best_obj:.6f}s"
    )
    plt.show()

    # ---------------- Structured summary of assignments ----------------
    # Build per-device lists of (expert, layer) by mode using the final SA state
    per_device = {d: {"CPU": [], "GPU": [], "Disk": []} for d in range(M)}
    for l in range(L):
        for e in range(E):
            d = y_state[(e, l)]
            if assign_g.get((e, l), -1) == d:
                per_device[d]["GPU"].append((e, l))
            elif assign_x.get((e, l), -1) == d:
                per_device[d]["CPU"].append((e, l))
            else:
                per_device[d]["Disk"].append((e, l))

    # Pretty-print summary
    print("\n================ Residency & Execution Summary (SA) ================")
    print(f"Total objective (Compute+Load+Comm): {best_obj:.6f}s\n")
    grand_counts = {"CPU": 0, "GPU": 0, "Disk": 0}

    for d in range(M):
        print(f"Device {d}:")
        for mode in ("CPU", "GPU", "Disk"):
            pairs = per_device[d][mode]
            grand_counts[mode] += len(pairs)
            # Format as eX-lY compact tokens in rows of up to 12
            tokens = [f"e{e}-l{l}" for (e, l) in pairs]
            if not tokens:
                print(f"  {mode:<4}: (none)")
            else:
                print(f"  {mode:<4}: {len(tokens)} item(s)")
                row = []
                for i, t in enumerate(tokens, 1):
                    row.append(t)
                    if i % 12 == 0:
                        print("          " + ", ".join(row))
                        row = []
                if row:
                    print("          " + ", ".join(row))
        print("")

    print("Totals:")
    print(f"  CPU : {grand_counts['CPU']}")
    print(f"  GPU : {grand_counts['GPU']}")
    print(f"  Disk: {grand_counts['Disk']}")
    print("===================================================================\n")

    return best_result