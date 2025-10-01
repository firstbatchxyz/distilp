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
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Literal
from collections import defaultdict
import math
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
    req_flops_by_attention = model.attn_flops["decode"]["b1"][0]
    req_flops_by_expert = model.flops_per_active_expert_per_token[0]
    flops_by_device_cpu = dev.scpu["fp16"]["b_1"]

    # comp_cpu = _sum_f_over_S(model.f_by_quant, dev.scpu, model.Q)
    comp_cpu = (req_flops_by_expert + req_flops_by_router + req_flops_by_attention)/flops_by_device_cpu
    alpha = (comp_cpu)
             # + dev.t_kvcpy_cpu + (bprime / dev.T_cpu))

    # β_m (GPU minus CPU path), 0 if no GPU available
    S_gpu = _gpu_table(dev)
    T_gpu = _pick_T_gpu(dev)
    if S_gpu is not None and T_gpu is not None:
        flops_by_device_gpu = S_gpu["fp16"]["b_1"]
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
        else:  # M4
            a[i] = alpha
            b[i] = beta
    return a, b

@dataclass
class ILPResult:
    x: List[int]
    g: List[int]
    r: List[int]
    obj_value: float

def gurobi_solve(
    path: str,
    devs: List[DeviceProfile],
    model: ModelProfile,
)-> ILPResult:
    M = len(devs)
    L = model.L
    E = model.n_routed_experts

    sets = assign_sets(devs)
    kv_cache_size = kv_size(model)
    router_size = model.router_bytes["1"]
    expert_size = model.bytes_per_expert["1"]
    attention_size = model.attn_bytes["1"]

    alpha, beta = objective_vectors(devs, model, sets)

    transitions, layers = load_transitions(path)
    pi = compute_visit_probabilities(transitions, layers)

    edges: List[Tuple[Node, Node, float]] = []
    for u, nbrs in transitions.items():
        if u[1] != -1:
            for v, p in nbrs.items():
                f = pi.get(u[0], u[1]) * float(p)
                if f > 0.0:
                    edges.append((u, v, f))

    x_index_list = []
    for d in range(M):
        for l in range(L):
            for e in range(E):
                x_index_list.append((e, l, d))

    m = gp.Model("MoeDelo")

    # Decision variables

    x = m.addVars(x_index_list, vtype=GRB.BINARY, name="x")  # resident in RAM
    y = m.addVars(x_index_list, vtype=GRB.BINARY, name="y")  # executed on
    r = m.addVars(x_index_list, vtype=GRB.BINARY, name="r")  # disk-load if not resident
    g = m.addVars(x_index_list, vtype=GRB.BINARY, name="g")  # executed on GPU
    z = m.addVars(L, M, vtype=GRB.BINARY, name="z")     # layer at device

    # One execution device per expert
    for l in range(L):
        for e in range(E):
            m.addConstr(gp.quicksum(y[e, l, d] for d in range(M)) >= 1, name=f"exec_unique[{e}]")

    # Disk load linking
    for l in range(L):
        for e in range(E):
            for d in range(M):
                m.addConstr(r[e, l, d] >= y[e, l, d] - x[e, l, d], name=f"r_lb[{e},{l},{d}]")
                m.addConstr(r[e, l, d] <= y[e, l, d], name=f"r_ub[{e},{l},{d}]")

    # Layer-Device assignment
    for l in range(L):
        for d in range(M):
            m.addConstr(gp.quicksum(x[e, l, d] for e in range(E)) <= E * z[l,d], name=f"layer_allocation[{l},{d}]")

    # CPU linking
    for l in range(L):
        for e in range(E):
            for d in range(M):
                m.addConstr(x[e, l, d] <= y[e, l, d], name=f"g_le_x[{e},{l},{d}]")

    # GPU linking
    for l in range(L):
        for e in range(E):
            for d in range(M):
                m.addConstr(g[e, l, d] <= y[e, l, d], name=f"g_le_y[{e},{l},{d}]")

    # RAM constraints
    for d in sets["M1"]:
        rhs = devs[d].d_avail_ram
        m.addConstr(
            gp.quicksum(expert_size * x[e, l, d] for l in range(L) for e in range(E)) <= rhs - gp.quicksum(
                z[l, d] * (kv_cache_size + router_size + attention_size) for l in range(L)), name=f"M1_ram[{d}]")

    for d in sets["M2"]:
        dev_metal = float(devs[d].d_avail_metal or 0)
        rhs = dev_metal - devs[d].c_gpu
        m.addConstr(
            gp.quicksum(expert_size * x[e, l, d] for l in range(L) for e in range(E)) <= rhs - gp.quicksum(
                z[l, d] * (kv_cache_size + router_size + attention_size) for l in range(L)), name=f"M2_ram[{d}]")

    # VRAM
    for d in range(M):
        i = devs[d]
        # CUDA VRAM bound
        # if d.has_cuda and d.d_avail_cuda is not None:
        #     rhs = d.d_avail_cuda - d.c_gpu
        #     m.addConstr(n[i] <= rhs, name=f"cuda_vram[{i}]")
        # Metal shared-memory bound
        if i.has_metal and i.d_avail_metal is not None:
            rhs = i.d_avail_metal - i.c_gpu
            m.addConstr(gp.quicksum(expert_size * g[e, l, d] for l in range(L) for e in range(E)) <= rhs - gp.quicksum(
                z[l, d] * (kv_cache_size + router_size + attention_size) for l in range(L)), name=f"metal_shared[{i}]")

    # No GPU
    for d in range(M):
        has_cuda = bool(devs[d].has_cuda and devs[d].d_avail_cuda is not None)
        has_metal = bool(devs[d].has_metal and devs[d].d_avail_metal is not None)
        if not (has_cuda or has_metal):
            m.addConstr(gp.quicksum(g[e, l, d] for e in range(E) for l in range(L)) == 0, name=f"no_gpu[{d}]")

    # Comm linearization
    w = {}
    for (u, v, _) in edges:
        for d1 in range(M):
            for d2 in  range(M):
                w[(u[0], v[0], u[1], v[1], d1, d2)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                                 name=f"w[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")
                # Full McCormick (tight) is 3 constraints:
                m.addConstr(w[(u[0], v[0], u[1], v[1], d1, d2)] <= y[u[0], u[1], d1],
                            name=f"w_le_yu[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")
                m.addConstr(w[(u[0], v[0], u[1], v[1], d1, d2)] <= y[v[0], v[1], d2],
                            name=f"w_le_yv[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")
                m.addConstr(w[(u[0], v[0], u[1], v[1], d1, d2)] >= y[u[0], u[1], d1] + y[v[0], v[1], d2] - 1,
                            name=f"w_ge_sum-1[{u[0]},{v[0]},{u[1]},{v[1]}, {d1},{d2}]")

    m.update()


    # Objective components
    compute_cost = gp.quicksum(
        pi[e, l] * ((g[e, l, d] * beta[d]) + (y[e, l, d] - g[e, l, d]) * alpha[d])
        for l in range(L) for e in range(E) for d in range(M))


    comm_cost = gp.quicksum(
        flow * 0.02 * w[(u[0], v[0], u[1], v[1], d1, d2)]
        for (u, v, flow) in edges for d1 in range(M) for d2 in range(M)
    )

    load_cost = gp.quicksum(
        pi[e, l] * (expert_size/devs[d].s_disk) * r[e, l, d] for e in range(E) for l in range(L) for d in (M))

    m.setObjective(compute_cost + comm_cost + load_cost, GRB.MINIMIZE)

    m.optimize()
    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # m.write("model.lp")
        # input("Paused. Press Enter to continue...")
        raise RuntimeError(f"Gurobi status {m.status}.")

    x_sol = [int(round(x[i].X)) for i in x_index_list]
    r_sol = [int(round(r[i].X)) for i in x_index_list]
    g_sol = [int(round(g[i].X)) for i in x_index_list]

    for i in m.getVars():
        if i.X > 0:
            print(i.varName, i.X)

    return ILPResult(x=x_sol, g=g_sol, r=r_sol, obj_value=float(m.ObjVal))

