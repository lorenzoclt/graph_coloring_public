"""Run one independent graph per CUDA stream during denoising inference.

This is an experimental alternative to PyG's merged disconnected-graph batch.
It preserves the model architecture and checkpoint format, but lets CUDA overlap
kernels from distinct graphs when one graph does not fill the GPU on its own.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Optional

import torch
from torch.cuda.amp import autocast
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops

from denoising import (
    DatasetRun,
    RunMetrics,
    discrete_energy_and_conflicts_per_graph,
    discover_graph_paths,
    format_duration,
    init_log_file,
    iter_denoising_parameters,
    mean_and_std,
    parse_args,
    parse_connectivities,
    planted_overlap_per_graph,
    torch_load,
    write_metrics,
)
from denoising_ottimizzato import compute_aggregation_degree, load_model_optimized


@dataclass
class StreamGraphState:
    graph: object
    edge_index: torch.Tensor
    aggregation_degree: torch.Tensor
    color_features: torch.Tensor
    planted_labels: torch.Tensor
    noise_buffer: torch.Tensor
    stream: torch.cuda.Stream
    best_energy: torch.Tensor
    best_conflicts: torch.Tensor
    best_overlap: torch.Tensor
    solved: bool = False
    pending_energy: Optional[torch.Tensor] = None
    pending_conflicts: Optional[torch.Tensor] = None


def _prepare_graph_states(
    graphs_cpu: list,
    streams: list,
    model_self_loops: bool,
    device: torch.device,
    num_colors: int,
) -> list:
    states = []
    for graph_idx, graph in enumerate(graphs_cpu):
        stream = streams[graph_idx % len(streams)]
        with torch.cuda.stream(stream):
            graph = graph.to(device, non_blocking=True)
            edge_index = graph.edge_index
            if model_self_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=graph.x.shape[0])

            aggregation_degree = compute_aggregation_degree(
                edge_index,
                num_nodes=graph.x.shape[0],
                device=device,
                dtype=graph.x.dtype,
            )
            color_features = graph.x[:, :num_colors]
            planted_labels = torch.argmax(color_features, dim=1).clone()
            noise_buffer = torch.empty_like(color_features)
            noise_buffer.normal_()
            color_features.copy_(noise_buffer)
            states.append(
                StreamGraphState(
                    graph=graph,
                    edge_index=edge_index,
                    aggregation_degree=aggregation_degree,
                    color_features=color_features,
                    planted_labels=planted_labels,
                    noise_buffer=noise_buffer,
                    stream=stream,
                    best_energy=torch.tensor(float("inf"), device=device),
                    best_conflicts=torch.tensor(float("inf"), device=device),
                    best_overlap=torch.tensor(float("nan"), device=device),
                )
            )
    return states


def run_single_dataset_multistream(
    dataset_run: DatasetRun,
    model: torch.nn.Module,
    device: torch.device,
    args,
    num_iterations: int,
) -> RunMetrics:
    dataset = torch_load(dataset_run.graph_path, map_location="cpu", weights_only=False)
    total_graphs = len(dataset)
    graph_offset = max(0, args.graph_offset)
    if graph_offset:
        dataset = dataset[graph_offset:]
    if args.max_graphs is not None:
        dataset = dataset[: args.max_graphs]
    if len(dataset) == 0:
        raise ValueError(
            f"No graphs selected from {dataset_run.graph_path}: graph_offset={graph_offset}, "
            f"max_graphs={args.max_graphs}, total_graphs={total_graphs}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    requested_streams = args.cuda_streams if args.cuda_streams > 0 else args.batch_size
    stream_count = max(1, min(requested_streams, args.batch_size))
    streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
    alpha_array = torch.linspace(args.alpha_min, args.alpha_max, num_iterations)
    sqrt_alpha_array = torch.sqrt(alpha_array).tolist()
    sqrt_one_minus_alpha_array = torch.sqrt(1.0 - alpha_array).tolist()
    check_every = max(1, args.check_every)
    progress_every = max(0, args.progress_every)
    model_self_loops = bool(getattr(model, "self_loops", False))
    best_energies = []
    best_conflicts = []
    best_overlaps = []
    solved_counter = 0
    total_time = 0.0
    num_nodes = int(dataset[0].num_nodes)

    print("\n" + "=" * 88)
    print(f"Processing file: {dataset_run.graph_path}")
    print(
        f"multistream=True | connectivity={dataset_run.connectivity:.2f} | "
        f"graphs={len(dataset)}/{total_graphs} | graph_offset={graph_offset} | "
        f"nodes={num_nodes} | iterations={num_iterations} | check_every={check_every}"
    )
    print(
        f"parallelism=batch_size={args.batch_size} | cuda_streams={stream_count} | "
        "each graph is submitted to an independent CUDA stream"
    )

    with torch.no_grad():
        with autocast() if device.type == "cuda" else nullcontext():
            for batch_idx, batch_cpu in enumerate(dataloader, start=1):
                states = _prepare_graph_states(
                    batch_cpu.to_data_list(),
                    streams,
                    model_self_loops,
                    device,
                    args.num_colors,
                )
                start_time = time()
                steps_run = 0
                last_checked_step = 0

                for step_idx, (sqrt_alpha, sqrt_one_minus_alpha) in enumerate(
                    zip(sqrt_alpha_array, sqrt_one_minus_alpha_array),
                    start=1,
                ):
                    steps_run = step_idx
                    for state in states:
                        with torch.cuda.stream(state.stream):
                            state.noise_buffer.normal_()
                            state.color_features.mul_(sqrt_alpha).add_(
                                state.noise_buffer,
                                alpha=sqrt_one_minus_alpha,
                            )
                            state.color_features.copy_(
                                model(
                                    state.graph.x,
                                    state.edge_index,
                                    edge_index_has_self_loops=model_self_loops,
                                    aggregation_degree=state.aggregation_degree,
                                )
                            )

                    should_check_energy = step_idx % check_every == 0 or step_idx == num_iterations
                    batch_is_done = False
                    if should_check_energy:
                        for state in states:
                            with torch.cuda.stream(state.stream):
                                (
                                    state.pending_energy,
                                    state.pending_conflicts,
                                ) = discrete_energy_and_conflicts_per_graph(
                                    state.graph,
                                    num_colors=args.num_colors,
                                )
                        current_stream = torch.cuda.current_stream(device)
                        for stream in streams:
                            current_stream.wait_stream(stream)

                        for state in states:
                            energy = state.pending_energy.item()
                            if energy < state.best_energy.item():
                                state.best_energy = state.pending_energy
                                state.best_conflicts = state.pending_conflicts
                                state.best_overlap = planted_overlap_per_graph(
                                    state.graph,
                                    state.planted_labels,
                                    num_colors=args.num_colors,
                                )[0]
                            state.solved = state.solved or energy <= 0.0
                        last_checked_step = step_idx
                        batch_is_done = all(state.solved for state in states)

                    if progress_every and (
                        step_idx % progress_every == 0 or step_idx == num_iterations
                    ):
                        torch.cuda.synchronize(device)
                        elapsed = time() - start_time
                        rate = step_idx / max(elapsed, 1e-9)
                        eta = (num_iterations - step_idx) / max(rate, 1e-9)
                        best_energy_mean = torch.stack([state.best_energy for state in states]).mean().item()
                        print(
                            f"Progress | batch {batch_idx:04d}/{len(dataloader):04d} | "
                            f"step={step_idx}/{num_iterations} ({100.0 * step_idx / num_iterations:.2f}%) | "
                            f"elapsed={format_duration(elapsed)} | rate={rate:.2f} steps/s | "
                            f"eta={format_duration(eta)} | best_E_mean={best_energy_mean:.8f} | "
                            f"last_energy_check={last_checked_step} | "
                            f"batch_solved={sum(state.solved for state in states)}/{len(states)}",
                            flush=True,
                        )
                    if batch_is_done:
                        break

                torch.cuda.synchronize(device)
                elapsed = time() - start_time
                total_time += elapsed
                solved_counter += sum(state.solved for state in states)
                best_energies.extend(state.best_energy.item() for state in states)
                best_conflicts.extend(state.best_conflicts.item() for state in states)
                best_overlaps.extend(state.best_overlap.item() for state in states)
                batch_energy_mean = torch.stack([state.best_energy for state in states]).mean().item()
                print(
                    f"Batch {batch_idx:04d}/{len(dataloader):04d} | graphs={len(states)} | "
                    f"processed={len(best_energies)}/{len(dataset)} | steps={steps_run} | "
                    f"time={elapsed:.1f}s | batch_E_mean={batch_energy_mean:.8f} | "
                    f"batch_solved={sum(state.solved for state in states)}/{len(states)} | "
                    f"solved={solved_counter}"
                )

    energies = torch.tensor(best_energies, dtype=torch.float32)
    conflicts = torch.tensor(best_conflicts, dtype=torch.float32)
    overlap_mean, overlap_std = mean_and_std(best_overlaps)
    return RunMetrics(
        connectivity=dataset_run.connectivity,
        num_iterations=num_iterations,
        energy_min=energies.min().item(),
        energy_mean=energies.mean().item(),
        energy_std=energies.std(unbiased=False).item(),
        conflicts_min=conflicts.min().item(),
        conflicts_mean=conflicts.mean().item(),
        conflicts_std=conflicts.std(unbiased=False).item(),
        overlap_mean=overlap_mean,
        overlap_std=overlap_std,
        graph_type=dataset_run.graph_type,
        algorithm=f"{args.algorithm}_multistream",
        avg_time=total_time / len(best_energies),
        num_solved=solved_counter,
        num_graphs=len(best_energies),
        num_nodes=num_nodes,
    )


def run_denoising_multistream(args) -> None:
    if args.cpu or not torch.cuda.is_available():
        raise RuntimeError("denoising_multistream.py requires a CUDA GPU")
    if args.compile_model:
        print("torch.compile disabled: CUDA Graph buffers are not safe for overlapping stream calls")
        args.compile_model = False

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dataset_runs = discover_graph_paths(
        Path(args.data_dir),
        parse_connectivities(args.connectivities),
        args.split,
    )
    log_path = Path(args.log_file)
    init_log_file(log_path)
    model = load_model_optimized(args, device)

    print(f"device={device} | checkpoint={args.checkpoint}")
    print(f"log_file={log_path}")
    print("using denoising_multistream.py")
    for num_iterations, alpha_min, alpha_max in iter_denoising_parameters(args):
        args.alpha_min = alpha_min
        args.alpha_max = alpha_max
        for dataset_run in dataset_runs:
            metrics = run_single_dataset_multistream(dataset_run, model, device, args, num_iterations)
            write_metrics(log_path, metrics, dataset_run.graph_path, args)
            print(
                f"Summary | conn={metrics.connectivity:.2f} | E_mean={metrics.energy_mean:.8f} | "
                f"alpha={args.alpha_min:.8f}-{args.alpha_max:.8f} | iter={metrics.num_iterations} | "
                f"E_std={metrics.energy_std:.8f} | solved={metrics.num_solved}/{metrics.num_graphs} | "
                f"overlap={metrics.overlap_mean:.8f} | overlap_std={metrics.overlap_std:.8f} | "
                f"solve_probability={metrics.solve_probability:.8f} | avg_time={metrics.avg_time:.2f}s"
            )


if __name__ == "__main__":
    run_denoising_multistream(parse_args())
