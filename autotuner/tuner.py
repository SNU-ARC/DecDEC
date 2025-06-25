import csv
import os
import nsys_parser
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import config
from itertools import product


def dump_to_csv(results, csv_output_path):
    csv_entries = []
    csv_entries.append(
        ('model', 'algo', 'bit_width', 'module', 'standalone/concurrent', 'n_tb', 'k_chunk', 
        'decdec_time_us', 'gemv_time_us', 'dec_time_us', 'orderings'))

    for model_name, model_results in results.items():
        for algo, bit_width_results in model_results.items():
            for bit_width, module_results in bit_width_results.items():
                for module_name, (standalone_quant_kernel_time, avg_times, gemv_times, dec_times, 
                                  orderings_per_tb) in module_results.items():
                    # standalone baseline row
                    csv_entries.append( (model_name, algo, bit_width, module_name, 'standalone', 0, 0,
                         '', f"{standalone_quant_kernel_time:.2f}", ''))

                    # concurrent timings
                    for n_tb, time_by_num_rows in avg_times.items():
                        for k_chunk, time in time_by_num_rows.items():
                            gemv_time = gemv_times.get(n_tb, {}).get(k_chunk, '')
                            dec_time = dec_times.get(n_tb, {}).get(k_chunk, '')
                            orderings = orderings_per_tb.get(n_tb, {}).get(k_chunk, '')
                            ordering_strings = []
                            for ordering in orderings:
                                ordering_strings.append('>'.join(nsys_parser.index_to_kernel_ordering(ordering)))
                            csv_entries.append(
                                (
                                    model_name, algo, bit_width, module_name,
                                    'concurrent', n_tb, k_chunk,
                                    f"{time:.2f}",
                                    f"{gemv_time:.2f}" if gemv_time else '',
                                    f"{dec_time:.2f}" if dec_time else '',
                                    ' | '.join(ordering_strings) if orderings else ''
                                 )
                            )

    with open(csv_output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_entries)


def plot_results_by_tb(prefix, fp_dtype, results):
    os.makedirs("./plots", exist_ok=True)
    for model_name, model_results in results.items():
        for algo, bit_width_results in model_results.items():
            for bit_width, module_results in bit_width_results.items():
                for module_name, (standalone_quant_kernel_time, decdec_times, _, _, _) in module_results.items():
                    plot_path = f"./plots/{prefix}_{model_name}_{fp_dtype}_{algo}_{bit_width}_{module_name}.png"
                    if os.path.exists(plot_path):
                        print(f"Plot {plot_path} already exists, skipping...")
                        continue

                    matrix_dim = config.model_configurations[model_name][module_name]
                    unique_n_tb = len(decdec_times.keys())
                    print(f"Plotting results for: {prefix} {model_name} {fp_dtype} {algo} {bit_width}-bit {module_name} ({matrix_dim[0]}x{matrix_dim[1]})")

                    graphs_per_row = 4
                    num_graph_rows = (unique_n_tb + graphs_per_row - 1) // graphs_per_row
                    fig, axs = plt.subplots(num_graph_rows, graphs_per_row, figsize=(20, 5 * num_graph_rows))

                    # Set title for the entire figure
                    fig.suptitle(f"{model_name}\n{prefix} {fp_dtype} {algo} {bit_width}-bit\nModule {module_name} ({matrix_dim[0]} x {matrix_dim[1]})\nBaseline: {standalone_quant_kernel_time:.2f} us",
                                fontsize=20)

                    # Ensure axs is a 2D array
                    if num_graph_rows == 1:
                        axs = np.array([axs])

                    limit = 4
                    hard_limit = max(next(iter(decdec_times.values())).keys())
                    cutoff = 2.0
                    # While the time at any n_tb at limit is less than 2x the standalone kernel, increase limit by 16
                    while True:
                        if any([time_by_num_rows[limit] < cutoff * standalone_quant_kernel_time for time_by_num_rows in
                                decdec_times.values()]):
                            limit += 4
                            if limit > hard_limit:
                                limit = hard_limit
                                break
                        else:
                            break

                    for j, (n_tb, time_by_num_rows) in enumerate(decdec_times.items()):
                        row = j // graphs_per_row
                        col = j % graphs_per_row
                        ax = axs[row, col]

                        k_chunk = list(range(1, limit + 1))
                        times = np.array([time_by_num_rows[num_rows] for num_rows in k_chunk])

                        rel_times = times / standalone_quant_kernel_time

                        ax.plot(k_chunk, rel_times, 'o-')
                        ax.set_title(f"Threadblocks={n_tb}")
                        ax.set_xlabel("k_chunk")
                        ax.set_ylabel("Relative execution time")

                        # Try grid step of 5%, if it's too small, increase it
                        grid_step = 0.05

                        # Add horizontal line at 1.0
                        ax.axhline(y=1.0, color='r', linestyle='-')

                        ax.yaxis.set_major_locator(plt.MultipleLocator(grid_step))
                        ax.xaxis.set_major_locator(plt.MultipleLocator(4))

                        # Limit the y-axis to 2x the standalone kernel
                        min_value = min(rel_times)
                        ax.set_ylim([min(0.9, min_value), cutoff])

                        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

                    # Hide any unused axes
                    for j in range(unique_n_tb, num_graph_rows * graphs_per_row):
                        fig.delaxes(axs.flatten()[j])

                    # Adjust layout to be tight with added y padding
                    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.97])  # rect reserves space for the suptitle
                    plt.savefig(plot_path)
                    plt.close()


def tuner(model_layer_config, results_by_module, target_slowdown=1.1):
    # ===================== Set targets =====================
    print("========================== Begin Tuning ===========================")

    # Compute baseline times for each module
    time_by_module_baseline = []
    for module_name in model_layer_config:
        baseline_time, _, _, _, _ = results_by_module[module_name]
        time_by_module_baseline.append(baseline_time)

    layer_baseline = sum(time_by_module_baseline)

    print(f"Layer baseline: {layer_baseline:.2f} us")
    for i, module_name in enumerate(model_layer_config):
        matrix_dim = model_layer_config[module_name]
        print(f"    {module_name} ({matrix_dim[0]} x {matrix_dim[1]}): {time_by_module_baseline[i]:.2f} us")

    layer_target = layer_baseline * target_slowdown
    print(f"Layer target: {layer_target:.2f} us (Slowdown: {target_slowdown:.2f})")

    # ===================== Begin Global Search =====================
    print("========================== Begin Global Search ===========================")

    # Initialize variables for global search
    best_max_n_tb = None
    global_best_k_chunk = 0
    best_layer_time = float('inf')

    # Initialize included matrices
    included_modules = list(model_layer_config.keys())
    excluded_modules = []

    # Find maximum k_chunk and n_tb across all modules
    max_k_chunk = 0
    for _, (_, avg_times_by_tb, _, _, _) in results_by_module.items():
        for num_tb_dict in avg_times_by_tb.values():
            max_k_chunk = max(max_k_chunk, *num_tb_dict.keys())

    min_n_tb = min(
        min(avg_times_by_tb.keys())
        for _, (_, avg_times_by_tb, _, _, _) in results_by_module.items()
    )

    max_n_tb = max(
        max(avg_times_by_tb.keys())
        for _, (_, avg_times_by_tb, _, _, _) in results_by_module.items()
    )

    # Sort matrices by size (smallest to largest) based on matrix_dim[0] * matrix_dim[1]
    sorted_modules = sorted(model_layer_config, key=lambda module_name: model_layer_config[module_name][0] * model_layer_config[module_name][1])

    while True:
        # Reset variables for each iteration
        best_max_n_tb = None
        global_best_k_chunk = 0
        best_layer_time = float('inf')

        # Iterate over possible num_max_threadblocks
        for num_max_threadblocks in range(min_n_tb, max_n_tb + 1):
            # For each num_max_threadblocks, find the highest k_chunk within layer_target
            best_k_chunk = 0
            layer_time_at_best_num_rows = layer_baseline

            for k_chunk in range(1, max_k_chunk + 1):
                total_time = 0
                for module_name in included_modules:
                    _, avg_times_by_tb, _, _, _ = results_by_module[module_name]
                    # Get the largest n_tb <= num_max_threadblocks
                    valid_num_tb = [num_tb for num_tb in avg_times_by_tb.keys() if num_tb <= num_max_threadblocks]
                    if not valid_num_tb:
                        print(f"No valid n_tb <= {num_max_threadblocks} for module {module_name}")
                        total_time += float('inf')
                        continue
                    n_tb = max(valid_num_tb)
                    avg_times_by_rows = avg_times_by_tb[n_tb]
                    if k_chunk in avg_times_by_rows:
                        time = avg_times_by_rows[k_chunk]
                        total_time += time
                    else:
                        print(f"No value for {k_chunk} rows at {n_tb} TB for module {module_name}")
                        print("This is potentially a missed optimization opportunity.")
                        total_time += float('inf')

                # Add baseline times for excluded matrices
                for module_name in excluded_modules:
                    baseline_time, _, _, _, _ = results_by_module[module_name]
                    total_time += baseline_time

                if total_time > layer_target:
                    break
                else:
                    best_k_chunk = k_chunk
                    layer_time_at_best_num_rows = total_time
            else:
                raise ValueError("Global Search did not converge - increase the range of k_chunk!")

            # Update best parameters if current configuration is better
            if best_k_chunk > global_best_k_chunk:
                global_best_k_chunk = best_k_chunk
                best_max_n_tb = num_max_threadblocks
                best_layer_time = layer_time_at_best_num_rows
            elif best_k_chunk == global_best_k_chunk:
                if layer_time_at_best_num_rows < best_layer_time:
                    best_max_n_tb = num_max_threadblocks
                    best_layer_time = layer_time_at_best_num_rows

        # Check if any fetching occurred
        if global_best_k_chunk > 0:
            # if fetching occurred, break out of the loop, ending the global search
            break
        else:
            # if no fetching occurred, exclude the smallest module and retry global search
            smallest_module = sorted_modules.pop(0)
            included_modules.remove(smallest_module)
            excluded_modules.append(smallest_module)
            if not included_modules:
                print("Unable to fetch any rows for any matrices within the target.")
                break
            else:
                print(f"Excluding smallest module {smallest_module} ({model_layer_config[smallest_module][0]} x {model_layer_config[smallest_module][1]}) and retrying global search...")


    # Compute time by module with the best global parameters
    time_by_module = []
    n_tb_per_module = []
    for module_name in model_layer_config:
        if module_name in included_modules:
            _, avg_times_by_tb, _, _, _ = results_by_module[module_name]
            valid_num_tb = [num_tb for num_tb in avg_times_by_tb.keys() if num_tb <= best_max_n_tb]
            n_tb = max(valid_num_tb)
            avg_times_by_rows = avg_times_by_tb[n_tb]
            time = avg_times_by_rows[global_best_k_chunk]
            n_tb_per_module.append(n_tb)
        else:
            # Use baseline time for excluded modules
            time, _, _, _, _ = results_by_module[module_name]
            n_tb_per_module.append(None)  # Indicate that this module is excluded
        time_by_module.append(time)

    print(f"Best num_max_threadblocks: {best_max_n_tb}")
    print(f"Global k_chunk: {global_best_k_chunk} / 1024")
    print(f"Layer time: {best_layer_time:.2f} us "
          f"(Slowdown: {best_layer_time / layer_baseline:.2f})")
    for i, module_name in enumerate(model_layer_config):
        matrix_dim = model_layer_config[module_name]
        status = "(excluded)" if module_name in excluded_modules else ""
        print(f"    {module_name} ({matrix_dim[0]} x {matrix_dim[1]}): {time_by_module[i]:.2f} us {status}")

    # ===================== Begin Fine-grained Search =====================
    print("========================== Fine-grained Search ===========================")

    # Initialize per-module parameters for fine-grained search
    k_chunk_per_module = []
    for module_name in model_layer_config:
        if module_name in included_modules:
            k_chunk_per_module.append(global_best_k_chunk)
        else:
            k_chunk_per_module.append(0)  # Indicate no fetching for excluded matrices

    while True:
        time_deltas_by_module = []
        for i, module_name in enumerate(model_layer_config):
            if module_name in included_modules:
                _, avg_times_by_tb, _, _, _ = results_by_module[module_name]
                n_tb = n_tb_per_module[i]
                avg_times_by_rows = avg_times_by_tb[n_tb]
                next_num_rows = k_chunk_per_module[i] + 1
                if next_num_rows in avg_times_by_rows:
                    time_delta = avg_times_by_rows[next_num_rows] - avg_times_by_rows[k_chunk_per_module[i]]
                    time_deltas_by_module.append(time_delta)
                else:
                    print(f"No value for {next_num_rows} rows at {n_tb} TB for module {module_name}")
                    print("This is potentially a missed optimization opportunity.")
                    time_deltas_by_module.append(float('inf'))
            else:
                time_deltas_by_module.append(float('inf'))

        # Convert to numpy array for argsort
        time_deltas_by_module = np.array(time_deltas_by_module)
        sorted_indices = np.argsort(time_deltas_by_module)

        update = False
        for idx in sorted_indices:
            delta = time_deltas_by_module[idx]
            if sum(time_by_module) + delta <= layer_target:
                k_chunk_per_module[idx] += 1
                time_by_module[idx] += delta
                update = True
            else:
                continue  # Try next module

        if not update:
            break

    print("Fine-grained k_chunk:")
    for i, module_name in enumerate(model_layer_config):
        n_tb = n_tb_per_module[i]
        n_rows = k_chunk_per_module[i]
        if module_name in included_modules:
            matrix_dim = model_layer_config[module_name]
            print(f"    {module_name} ({matrix_dim[0]} x {matrix_dim[1]}): {n_rows} / 1024 "
                  f"(n_tb={n_tb})")
        else:
            print(f"    {module_name} ({matrix_dim[0]} x {matrix_dim[1]}): 0 / 1024 (excluded)")

    fine_grained_layer_time = sum(time_by_module)
    print(f"Fine-grained layer time: {fine_grained_layer_time:.2f} us "
          f"(Slowdown: {fine_grained_layer_time / layer_baseline:.2f})")
    for i, module_name in enumerate(model_layer_config):
        matrix_dim = model_layer_config[module_name]
        status = "(excluded)" if module_name in excluded_modules else ""
        print(f"    {module_name} ({matrix_dim[0]} x {matrix_dim[1]}): {time_by_module[i]:.2f} us {status}")

    print("========================== End Tuning ===========================")

    results = {
        "model_layer_config": model_layer_config,
        "best_max_n_tb": best_max_n_tb,
        "threadblocks_per_module": n_tb_per_module,
        "k_chunk_per_module": k_chunk_per_module,
        "time_by_module": time_by_module,
        "layer_time": fine_grained_layer_time,
    }
    return results


if __name__ == '__main__':
    # Get prefix from command line
    if len(sys.argv) != 2:
        print("Usage: python tuner.py <prefix>")
        sys.exit(1)

    prefix = sys.argv[1]

    print("Processing profile results...")
    results = nsys_parser.get_all_results(prefix, config.fp_dtype)

    print("Dumping results to csv...")
    # dump results to csv for us humans to read, if interested
    dump_to_csv(results, f"{prefix}_timings.csv")

    print("Plotting results...")
    # plot results by n_tb
    plot_results_by_tb(prefix, config.fp_dtype, results)

    target_slowdowns = [1.025, 1.05, 1.1, 1.2]

    all_tuner_results = {}

    for model_name in results:
        for algo in results[model_name]:
            for bit_width in results[model_name][algo]:
                print(f"\n>>> Tuning {model_name} {algo} {bit_width}-bit <<<")
                for target_slowdown in target_slowdowns:
                    all_tuner_results[(model_name, algo, bit_width, target_slowdown)] = tuner(
                        config.model_configurations[model_name],
                        results[model_name][algo][bit_width],
                        target_slowdown,
                    )

    for model_name in results:
        for algo in results[model_name]:
            for bit_width in results[model_name][algo]:
                print(f"\n>>> {model_name} {algo} {bit_width}-bit <<<")
                for target_slowdown in target_slowdowns:
                    tuner_results = all_tuner_results[(model_name, algo, bit_width, target_slowdown)]
                    n_tb = tuner_results["best_max_n_tb"]
                    k_per_matrix = tuner_results["k_chunk_per_module"]
                    print(f"({','.join(map(str, k_per_matrix))}) x{n_tb}")
                print()
