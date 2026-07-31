#!/usr/bin/env python3
"""Generate graph-coloring datasets directly as PyTorch Geometric files.

This script replaces the old two-step flow:

    gcc -O2 -o create_graphs create_graphs.c -lm
    ./create_graphs
    python create_python_dataset.py

It generates planted q-partite or fully random graphs and saves a list of
torch_geometric.data.Data objects with planted coloring features and degree.
"""

from __future__ import annotations

import argparse
import random
import time
from itertools import product
from pathlib import Path
from typing import Iterable


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value.replace(",", "."))
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value.replace(",", "."))
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def format_connectivity(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def num_edges_for(num_nodes: int, connectivity: float) -> int:
    return int(0.5 * connectivity * num_nodes + 0.5)


def max_random_edges(num_nodes: int) -> int:
    return num_nodes * (num_nodes - 1) // 2


def split_counts(total: int, ratios: list[float]) -> tuple[int, int, int]:
    ratio_sum = sum(ratios)
    train_count = int(total * ratios[0] / ratio_sum)
    validation_count = int(total * ratios[1] / ratio_sum)
    test_count = total - train_count - validation_count
    return train_count, validation_count, test_count


def split_output_path(output: Path, split_name: str) -> Path:
    suffix = output.suffix or ".pt"
    return output.with_name(f"{output.stem}_{split_name}{suffix}")


def create_dataset_layout_path(args: argparse.Namespace) -> Path:
    size_dir = f"{args.num_nodes // 1000}k"
    num_edges = num_edges_for(args.num_nodes, args.connectivity)
    run_dir = f"G_{args.num_graphs}-E_{num_edges}-C_{args.connectivity:.2f}"
    return args.output_dir / size_dir / args.dataset_type / run_dir / "graph.pt"


def progress(
    items: Iterable[int],
    total: int,
    enabled: bool,
    description: str | None = None,
) -> Iterable[int]:
    if not enabled:
        return items

    try:
        from tqdm import tqdm
    except ImportError:
        return items

    return tqdm(
        items,
        total=total,
        desc=description,
        unit="graph",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


def import_graph_dependencies():
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install torch and torch_geometric before "
            "generating the .pt dataset."
        ) from exc

    return torch, Data


def max_cross_group_edges(num_nodes: int, num_colors: int) -> int:
    base_size = num_nodes // num_colors
    remainder = num_nodes % num_colors
    group_sizes = [
        base_size + (1 if group < remainder else 0) for group in range(num_colors)
    ]
    total_pairs = num_nodes * (num_nodes - 1) // 2
    same_group_pairs = sum(size * (size - 1) // 2 for size in group_sizes)
    return total_pairs - same_group_pairs


def node_group(node: int, num_nodes: int, num_colors: int) -> int:
    return min((node * num_colors) // num_nodes, num_colors - 1)


def generate_edges(
    num_nodes: int,
    num_colors: int,
    num_edges: int,
    graph_type: str,
    rng: random.Random,
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate unique undirected edges for planted or random graphs."""

    undirected_edges: set[tuple[int, int]] = set()
    directed_edges: list[list[int]] = []
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]

    while len(undirected_edges) < num_edges:
        first = rng.randrange(num_nodes)

        if graph_type == "plt":
            while True:
                second = rng.randrange(num_nodes)
                if node_group(first, num_nodes, num_colors) != node_group(
                    second, num_nodes, num_colors
                ):
                    break
        elif graph_type == "rdm":
            while True:
                second = rng.randrange(num_nodes)
                if first != second:
                    break
        else:
            raise ValueError("graph_type must be either 'plt' or 'rdm'")

        if first > second:
            first, second = second, first

        edge = (first, second)
        if edge in undirected_edges:
            continue

        undirected_edges.add(edge)
        directed_edges.append([first, second])
        directed_edges.append([second, first])
        neighbors[first].append(second)
        neighbors[second].append(first)

    return directed_edges, neighbors


def parse_neigh_dataset(path: Path) -> list[list[list[int]]]:
    graphs: list[list[list[int]]] = []
    raw_graphs = path.read_text().strip().split("\n\n\n")

    for raw_graph in raw_graphs:
        edges: list[list[int]] = []
        for source, line in enumerate(raw_graph.strip().splitlines()):
            if not line.strip():
                continue
            for target in map(int, line.split()):
                edges.append([source, target])
        graphs.append(edges)

    return graphs


def save_neighbors(path: Path, all_neighbors: list[list[list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for neighbors in all_neighbors:
            for node_neighbors in neighbors:
                if node_neighbors:
                    file.write(" ".join(map(str, node_neighbors)))
                    file.write(" ")
                file.write("\n")
            file.write("\n\n\n")


def create_torch_graphs(
    edge_lists: list[list[list[int]]],
    num_nodes: int,
    num_colors: int,
):
    torch, Data = import_graph_dependencies()

    torch_graphs = []
    for edges in edge_lists:
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        degree = torch.bincount(edge_index[0], minlength=num_nodes).unsqueeze(1).float()
        planted_colors = torch.tensor(
            [node_group(node, num_nodes, num_colors) for node in range(num_nodes)],
            dtype=torch.long,
        )
        color_features = torch.nn.functional.one_hot(
            planted_colors,
            num_classes=num_colors,
        ).float()
        x = torch.cat([color_features, degree], dim=1).float()
        torch_graphs.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes))

    return torch_graphs


def generate_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    num_edges = num_edges_for(args.num_nodes, args.connectivity)
    if args.graph_type == "plt":
        max_edges = max_cross_group_edges(args.num_nodes, args.num_colors)
        graph_description = f"{args.num_colors}-partite graph"
    else:
        max_edges = max_random_edges(args.num_nodes)
        graph_description = "random graph"

    if num_edges > max_edges:
        raise SystemExit(
            f"Too many edges requested: {num_edges}. Maximum for this "
            f"{graph_description} is {max_edges}."
        )

    start = time.time()
    edge_lists: list[list[list[int]]] = []
    all_neighbors: list[list[list[int]]] = []

    progress_label = f"N={args.num_nodes}, c={args.connectivity:g}"
    for graph_id in progress(
        range(args.num_graphs), args.num_graphs, args.progress, progress_label
    ):
        edges, neighbors = generate_edges(
            args.num_nodes, args.num_colors, num_edges, args.graph_type, rng
        )
        edge_lists.append(edges)

        if args.neigh_output is not None:
            all_neighbors.append(neighbors)

        if args.log_every and not args.progress and graph_id % args.log_every == 0:
            elapsed = time.time() - start
            print(f"Graph number: {graph_id} | Time elapsed: {elapsed:.2f} seconds")

    if args.neigh_output is not None:
        save_neighbors(args.neigh_output, all_neighbors)
        print(f"Saved neighbor dataset to {args.neigh_output}")

    save_pt_dataset(
        edge_lists,
        args.output,
        args.num_nodes,
        args.num_colors,
        args.split,
        args.create_dataset_layout,
    )


def convert_dataset(args: argparse.Namespace) -> None:
    edge_lists = parse_neigh_dataset(args.from_neigh)
    save_pt_dataset(
        edge_lists,
        args.output,
        args.num_nodes,
        args.num_colors,
        args.split,
        args.create_dataset_layout,
    )


def build_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output

    if args.create_dataset_layout:
        return create_dataset_layout_path(args)

    return args.output_dir / args.output_pattern.format(
        N=args.num_nodes,
        q=args.num_colors,
        c=format_connectivity(args.connectivity),
        g=args.num_graphs,
        type=args.graph_type,
    )


def build_neigh_output_path(args: argparse.Namespace) -> Path | None:
    if args.neigh_output is None:
        return None

    if args.total_jobs == 1 and args.neigh_output.suffix:
        return args.neigh_output

    return args.neigh_output / args.neigh_pattern.format(
        N=args.num_nodes,
        q=args.num_colors,
        c=format_connectivity(args.connectivity),
        g=args.num_graphs,
        type=args.graph_type,
    )


def iter_generation_jobs(args: argparse.Namespace):
    for job_index, (num_nodes, connectivity) in enumerate(
        product(args.num_nodes, args.connectivity)
    ):
        job_args = argparse.Namespace(**vars(args))
        job_args.num_nodes = num_nodes
        job_args.connectivity = connectivity
        if args.seed is not None:
            job_args.seed = args.seed + job_index * args.seed_step
        job_args.output = build_output_path(job_args)
        job_args.neigh_output = build_neigh_output_path(job_args)
        yield job_args


def save_pt_dataset(
    edge_lists: list[list[list[int]]],
    output: Path,
    num_nodes: int,
    num_colors: int,
    split: list[float] | None,
    create_dataset_layout: bool,
) -> None:
    torch, _ = import_graph_dependencies()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_graphs = create_torch_graphs(edge_lists, num_nodes, num_colors)

    if split is None:
        torch.save(torch_graphs, output)
        print(f"Saved {len(torch_graphs)} PyTorch Geometric graphs to {output}")
        return

    train_count, validation_count, test_count = split_counts(len(torch_graphs), split)
    split_ranges = [
        ("train", "train", 0, train_count),
        ("validation", "val", train_count, train_count + validation_count),
        ("test", "test", train_count + validation_count, len(torch_graphs)),
    ]

    for split_name, layout_dir, start, end in split_ranges:
        if create_dataset_layout:
            split_path = output.parent / layout_dir / "graph.pt"
        else:
            split_path = split_output_path(output, split_name)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch_graphs[start:end], split_path)
        print(
            f"Saved {end - start} {split_name} PyTorch Geometric graphs "
            f"to {split_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate planted q-partite or fully random graphs and save them as a "
            "PyTorch Geometric .pt dataset."
        )
    )
    parser.add_argument(
        "-N",
        "--num-nodes",
        type=positive_int,
        nargs="+",
        default=[10000],
        help="number of nodes per graph; accepts one or more values (default: 10000)",
    )
    parser.add_argument(
        "-q",
        "--num-colors",
        type=positive_int,
        default=5,
        help=(
            "number of planted color groups/color feature columns before adding "
            "degree as the last feature (default: 5)"
        ),
    )
    parser.add_argument(
        "-g",
        "--num-graphs",
        type=positive_int,
        default=16384,
        help="number of graphs to generate (default: 16384)",
    )
    parser.add_argument(
        "-c",
        "--connectivity",
        type=positive_float,
        nargs="+",
        default=[20.0],
        help=(
            "average connectivity; accepts one or more values; edges are "
            "round(c * N / 2) (default: 20.0)"
        ),
    )
    parser.add_argument(
        "--graph-type",
        choices=("plt", "rdm"),
        default="plt",
        help=(
            "graph generation type: 'plt' for planted q-partite graphs, "
            "'rdm' for random graphs with no planted edge constraint (default: plt)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .pt file; only allowed when generating one N/c combination",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory for generated .pt files when -o is not used (default: .)",
    )
    parser.add_argument(
        "--output-pattern",
        default="validation_graphs_N{N}_c{c}.pt",
        help=(
            "filename pattern for generated .pt files. Available placeholders: "
            "{N}, {q}, {c}, {g}, {type}"
        ),
    )
    parser.add_argument(
        "--create-dataset-layout",
        action="store_true",
        help=(
            "save using the same directory layout as create_dataset.py: "
            "data_test/{N//1000}k/{type}/G_{g}-E_{edges}-C_{c:.2f}/{train,val,test}/graph.pt"
        ),
    )
    parser.add_argument(
        "--dataset-type",
        help=(
            "dataset type directory used with --create-dataset-layout "
            "(default: same as --graph-type)"
        ),
    )
    parser.add_argument(
        "--neigh-output",
        type=Path,
        help=(
            "optional directory for compatibility outputs in neigh_dataset text "
            "format"
        ),
    )
    parser.add_argument(
        "--neigh-pattern",
        default="neigh_dataset_N{N}_c{c}.txt",
        help=(
            "filename pattern for --neigh-output files. Available placeholders: "
            "{N}, {q}, {c}, {g}, {type}"
        ),
    )
    parser.add_argument(
        "--from-neigh",
        type=Path,
        help="convert an existing neigh_dataset text file instead of generating",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducible graph structure and node features",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help=(
            "increment applied to --seed for each N/c combination; use 0 to reuse "
            "the same seed for every combination (default: 1)"
        ),
    )
    parser.add_argument(
        "--split",
        type=non_negative_float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help=(
            "optional train/validation/test split. Values are normalized and may "
            "include zero, so '0.8 0.2 0', '80 20 0', and '0,8 0,2 0' are valid"
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="print progress every N generated graphs; use 0 to disable (default: 10)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="disable tqdm progress bar",
    )
    parser.set_defaults(progress=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dataset_type is None:
        args.dataset_type = args.graph_type

    if args.from_neigh is not None and args.neigh_output is not None:
        parser.error("--from-neigh and --neigh-output cannot be used together")

    if args.split is not None and sum(args.split) <= 0:
        parser.error("--split must contain at least one value greater than zero")

    if args.create_dataset_layout and args.split is None:
        parser.error("--create-dataset-layout requires --split")

    if args.create_dataset_layout and args.output is not None:
        parser.error("--create-dataset-layout uses --output-dir; do not pass -o/--output")

    if args.from_neigh is not None:
        if len(args.num_nodes) != 1 or len(args.connectivity) != 1:
            parser.error("--from-neigh can only be used with one -N and one -c value")
        args.num_nodes = args.num_nodes[0]
        args.connectivity = args.connectivity[0]
        if args.output is None:
            if args.create_dataset_layout:
                args.output = create_dataset_layout_path(args)
            else:
                args.output = Path(f"validation_graphs_N{args.num_nodes}.pt")
        convert_dataset(args)
    else:
        total_jobs = len(args.num_nodes) * len(args.connectivity)
        args.total_jobs = total_jobs
        if args.output is not None and total_jobs > 1:
            parser.error(
                "-o/--output can only be used when generating one N/c combination. "
                "Use --output-dir for multiple combinations."
            )

        for job_number, job_args in enumerate(iter_generation_jobs(args), start=1):
            if total_jobs > 1:
                print(
                    f"Generating dataset {job_number}/{total_jobs}: "
                    f"N={job_args.num_nodes}, c={job_args.connectivity:g}, "
                    f"graphs={job_args.num_graphs}"
                )
            generate_dataset(job_args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
