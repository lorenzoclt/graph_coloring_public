import argparse
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch_geometric.data import Data

from GNN import MMPN_torch
from functions import discrete_energy


def generate_planted_graph(num_nodes: int, conn: float, num_colors: int) -> Data:
    """Generate one planted colorable graph with target average connectivity."""
    num_edges = int(num_nodes * conn / 2)
    max_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_edges:
        raise ValueError(
            f"Too many edges for a simple graph: requested={num_edges}, max={max_edges}"
        )

    colors = [i % num_colors for i in range(num_nodes)]
    random.shuffle(colors)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    directed_edges = []

    added_edges = 0
    while added_edges < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        if u == v:
            continue
        if colors[u] == colors[v]:
            continue
        if adj[u, v]:
            continue

        adj[u, v] = True
        adj[v, u] = True
        directed_edges.append([u, v])
        directed_edges.append([v, u])
        added_edges += 1

    degree = adj.sum(dim=1).to(torch.float32)
    one_hot_colors = torch.nn.functional.one_hot(
        torch.tensor(colors, dtype=torch.long), num_classes=num_colors
    ).to(torch.float32)
    x = torch.cat([one_hot_colors, degree.unsqueeze(1)], dim=1)
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def build_model(
    checkpoint_path: Path,
    device: torch.device,
    latent_dim: int,
    num_colors: int,
    num_layers: int,
    dropout_rate: float,
    self_loops: bool,
    temp: float,
) -> MMPN_torch:
    model = MMPN_torch(latent_dim, num_colors, num_layers, dropout_rate, self_loops, temp)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run_demo(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(
        checkpoint_path=checkpoint_path,
        device=device,
        latent_dim=args.latent_dim,
        num_colors=args.num_colors,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        self_loops=not args.no_self_loops,
        temp=args.temp,
    )

    alpha_array = torch.linspace(
        args.alpha_min, args.alpha_max, args.num_iterations, device=device
    )

    best_energies = []
    solved_counter = 0
    total_denoising_time = 0.0

    amp_ctx = autocast(enabled=(device.type == "cuda")) if device.type == "cuda" else nullcontext()

    print("=" * 88)
    print(
        f"Demo denoising | device={device} | graphs={args.num_graphs} | "
        f"N={args.num_nodes} | conn={args.connectivity:.2f}"
    )
    print(
        f"iters={args.num_iterations} | alpha=[{args.alpha_min:.3f}, {args.alpha_max:.3f}] | "
        f"num_colors={args.num_colors}"
    )
    print("=" * 88)

    with torch.no_grad():
        for graph_id in range(args.num_graphs):
            graph = generate_planted_graph(
                num_nodes=args.num_nodes,
                conn=args.connectivity,
                num_colors=args.num_colors,
            ).to(device)

            # Initialize color channels with Gaussian noise, keep degree channel unchanged.
            graph.x[:, : args.num_colors] = torch.randn_like(graph.x[:, : args.num_colors])

            e_best = float("inf")
            graph_start = time.time()

            with amp_ctx:
                for alpha in alpha_array:
                    graph.x[:, : args.num_colors] = (
                        math.sqrt(alpha.item()) * graph.x[:, : args.num_colors]
                        + math.sqrt(1.0 - alpha.item())
                        * torch.randn_like(graph.x[:, : args.num_colors])
                    )

                    graph.x[:, : args.num_colors] = model(graph.x, graph.edge_index)
                    e_now = discrete_energy(graph, num_colors=args.num_colors).item()

                    if e_now < e_best:
                        e_best = e_now

                    if e_now <= 0.0:
                        solved_counter += 1
                        break

            elapsed = time.time() - graph_start
            total_denoising_time += elapsed
            best_energies.append(e_best)

            print(
                f"Graph {graph_id + 1:03d}/{args.num_graphs:03d} | "
                f"best_energy={e_best:.8f} | time={elapsed:.3f}s | solved={solved_counter}"
            )

    best_energy_tensor = torch.tensor(best_energies, dtype=torch.float32)
    e_mean = best_energy_tensor.mean().item()
    e_std = best_energy_tensor.std(unbiased=False).item()
    solved_ratio = solved_counter / args.num_graphs if args.num_graphs > 0 else 0.0
    avg_time = total_denoising_time / args.num_graphs if args.num_graphs > 0 else 0.0

    print("\nSummary metrics")
    print(f"- Connectivity: {args.connectivity:.2f}")
    print(f"- Num nodes: {args.num_nodes}")
    print(f"- Num graphs: {args.num_graphs}")
    print(f"- Iterations: {args.num_iterations}")
    print(f"- E_mean: {e_mean:.8f}")
    print(f"- E_std: {e_std:.8f}")
    print(f"- Num solved: {solved_counter}/{args.num_graphs} ({100.0 * solved_ratio:.2f}%)")
    print(f"- Avg time per graph: {avg_time:.3f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate planted graph-coloring instances and run denoising demo."
    )
    parser.add_argument("--num-nodes", type=int, default=1000, help="Number of nodes per graph")
    parser.add_argument(
        "--connectivity",
        type=float,
        default=12.0,
        help="Target average connectivity (num_edges = N * conn / 2)",
    )
    parser.add_argument("--num-graphs", type=int, default=10, help="How many graphs to generate")
    parser.add_argument(
        "--num-iterations", type=int, default=1000, help="Denoising steps per graph"
    )
    parser.add_argument("--alpha-min", type=float, default=0.4, help="Minimum alpha in schedule")
    parser.add_argument("--alpha-max", type=float, default=0.9, help="Maximum alpha in schedule")

    parser.add_argument("--num-colors", type=int, default=5, help="Number of graph colors")
    parser.add_argument("--latent-dim", type=int, default=32, help="MMPN latent dimension")
    parser.add_argument("--num-layers", type=int, default=5, help="MMPN number of layers")
    parser.add_argument("--dropout-rate", type=float, default=0.0, help="MMPN dropout")
    parser.add_argument("--temp", type=float, default=10.0, help="Softmax temperature")
    parser.add_argument(
        "--no-self-loops",
        action="store_true",
        help="Disable adding self-loops in the model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights_demo/checkpoint_demo.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())