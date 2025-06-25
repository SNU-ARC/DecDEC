import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from multiprocessing import Pool
import threadpoolctl


def get_module_by_name(layer, module_name):
    if module_name == '':
        return layer
    components = module_name.split('.')
    module = layer
    for comp in components:
        module = getattr(module, comp)
    return module


@torch.no_grad()
def _process_layer(args):
    idx, original_layer, quantized_layer, save_exact_residuals, extra_bit_widths = args
    cheatsheet_per_layer = {}
    exact_residuals_per_layer = {}
    quantized_residuals_per_layer = {}  # Keyed by bit_width

    # Get all Linear modules in the layer
    module_dict = {}
    for name, module in original_layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_dict[name] = module

    # Process each module sequentially
    for module_name, orig_module in module_dict.items():
        # Get the quantized module
        quant_module = get_module_by_name(quantized_layer, module_name)
        # Get the weights
        orig_weight = orig_module.weight.cpu()
        quant_weight = quant_module.weight.cpu()
        # Compute the error
        error = orig_weight - quant_weight
        # Now, compute the quantization step size for each row
        row_count = error.shape[0]
        col_count = error.shape[1]

        # Assert that row_count is a multiple of 8
        assert row_count % 8 == 0, f"row_count ({row_count}) is not a multiple of 8, cannot pack nibbles"

        # Define the bit widths to process
        if not extra_bit_widths:
            bit_widths = [4]
        else:
            bit_widths = [2, 4, 8]

        # Initialize storage for quantized values
        q_error_rows_dict = {}  # Keyed by bit_width
        scales_dict = {}  # Keyed by bit_width

        # For each bit width, perform quantization
        for bit_width in bit_widths:
            levels = 2 ** bit_width - 1
            positive_levels = levels // 2

            scales = torch.zeros(row_count, dtype=torch.float16, device=error.device)
            q_error_rows = torch.zeros((row_count, col_count), dtype=torch.float16, device=error.device)

            grid_size = 32  # The number of scales to try for each row

            # For each row, compute quantization step size and quantize
            for row_idx in range(row_count):
                error_row = error[row_idx]
                max_scale = torch.max(torch.abs(error_row)) / positive_levels

                # Grid search for the best scale
                scales_to_try = torch.linspace(max_scale / grid_size, max_scale, grid_size, device=error.device)
                error_row_expanded = error_row.unsqueeze(0).expand(grid_size, -1)
                scales_expanded = scales_to_try.unsqueeze(1)
                q_error_rows_candidates = torch.clamp(torch.round(error_row_expanded / scales_expanded), -positive_levels, positive_levels)
                reconstructed_error_rows = q_error_rows_candidates * scales_expanded
                # Use fp32 for MSE, as squaring small fp16 values can lead to underflow
                MSEs = torch.mean((error_row_expanded.float() - reconstructed_error_rows.float()) ** 2, dim=1)
                best_scale_idx = torch.argmin(MSEs)

                best_scale = scales_to_try[best_scale_idx]
                scales[row_idx] = best_scale
                q_error_row = q_error_rows_candidates[best_scale_idx]
                q_error_rows[row_idx] = q_error_row

            # Store scales and quantized error rows
            scales_dict[bit_width] = scales
            q_error_rows_dict[bit_width] = q_error_rows

        # Now, process the stored quantized values for each bit width
        for bit_width in bit_widths:
            levels = 2 ** bit_width - 1
            positive_levels = levels // 2
            scales = scales_dict[bit_width]
            q_error_rows = q_error_rows_dict[bit_width]

            if bit_width == 4:
                # Packing logic for 4-bit quantization
                nibbles = torch.zeros((row_count, col_count), dtype=torch.int32, device=error.device)
                for row_idx in range(row_count):
                    # Shift to positive values to prepare for packing
                    nibbles[row_idx] = q_error_rows[row_idx] + positive_levels + 1

                nibbles = nibbles.t()

                cheatsheet_col_dim = row_count // 8
                cheatsheet_row_dim = col_count

                # Pack 8 nibbles into each uint32
                cheatsheet = torch.zeros((cheatsheet_row_dim, cheatsheet_col_dim), dtype=torch.int32, device=error.device)
                for cheatsheet_col_idx in range(cheatsheet_col_dim):
                    for nibble_idx in range(8):
                        row_idx = cheatsheet_col_idx * 8 + nibble_idx
                        cheatsheet[:, cheatsheet_col_idx] |= (nibbles[:, row_idx] & 0xF) << (4 * nibble_idx)

                # Reorder scales
                reordered_scales = torch.zeros_like(scales)
                for cheatsheet_col_idx in range(cheatsheet_col_dim):
                    for nibble_idx in range(8):
                        col_idx = cheatsheet_col_idx * 8 + nibble_idx
                        reordered_scales[nibble_idx * cheatsheet_col_dim + cheatsheet_col_idx] = scales[col_idx]

                cheatsheet_per_layer[module_name] = {'reordered_scales': reordered_scales.cpu(), 'cheatsheet': cheatsheet.cpu()}

            else:
                # For other bit widths, store quantized residuals as fp16 tensors
                quantized_error_rows_fp16 = q_error_rows * scales.unsqueeze(1)
                if bit_width not in quantized_residuals_per_layer:
                    quantized_residuals_per_layer[bit_width] = {}
                quantized_residuals_per_layer[bit_width][module_name] = quantized_error_rows_fp16.cpu()

        if save_exact_residuals:
            exact_residuals_per_layer[module_name] = error.cpu()
        else:
            exact_residuals_per_layer = None

    return cheatsheet_per_layer, exact_residuals_per_layer, quantized_residuals_per_layer


def create_cheatsheet(
        original_model_path,
        quantized_model_path,
        save_path,
        num_workers=None,
        save_exact_residuals=False,
        save_extra_bit_widths=False,
):
    """
    Create a cheatsheet for the quantized model by comparing it with the original model.
    By default, the cheatsheet.pt will be generated in 4-bit quantized format.

    If save_extra_bit_widths is True, it will also generate 2-bit and 8-bit quantized residuals,
    although in fake-quantized fp16 format.
    If save_exact_residuals is True, it will also save the exact residuals.

    These optional residuals may be useful for analysis or debugging purposes.

    Args:
        original_model_path (str): Path to the original model.
        quantized_model_path (str): Path to the quantized model.
        save_path (str): Directory where the cheatsheet and residuals will be saved.
        num_workers (int, optional): Number of workers for multiprocessing. Defaults to None, which
            will use the number of layers in the model.
        save_exact_residuals (bool, optional): Whether to save the exact residuals. Defaults to False.
        save_extra_bit_widths (bool, optional): Whether to save quantized residuals for 2-bit and 8-bit. Defaults to False.

    Returns:
        None: The function saves the cheatsheet and residuals to the specified save_path.
    """
    # If save_path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        # Check that save_path is a directory
        if not os.path.isdir(save_path):
            raise ValueError(f"Provided save_path {save_path} is not a directory.")
            
    cheatsheet_save_path = os.path.join(save_path, 'cheatsheet.pt')

    # These paths will be used to save quantized residuals if extra_bit_widths is True
    quantized_residuals_paths = {
        2: save_path + '/quantized_residuals_2bit.pt',
        8: save_path + '/quantized_residuals_8bit.pt'
    }

    # Check if all files exist to avoid overwriting
    if os.path.exists(cheatsheet_save_path) and all(os.path.exists(path) for path in quantized_residuals_paths.values()):
        print(f"Cheatsheet and quantized residuals already exist for {quantized_model_path}. Skipping...")
        return

    # Load the models
    original_model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.float16)
    quantized_model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype=torch.float16)

    num_layers = len(original_model.model.layers)

    cheatsheets = []
    exact_residuals = []
    quantized_residuals = {}  # Keyed by bit_width

    # limit the number of workers to the number of layers
    num_workers = min(num_workers, num_layers) if num_workers is not None else num_layers

    # Prepare arguments for multiprocessing
    layer_indices = list(range(num_layers))
    args_list = [
        (idx, 
        original_model.model.layers[idx], 
        quantized_model.model.layers[idx], 
        save_exact_residuals, 
        save_extra_bit_widths) 
        for idx in layer_indices
    ]

    # Use multiprocessing with tqdm progress bar
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(_process_layer, args_list), total=num_layers))

    print("All layers processed. Saving cheatsheets and quantized residuals...")

    # Initialize the quantized_residuals dictionary for bit widths
    for bit_width in [2, 8]:
        quantized_residuals[bit_width] = []

    for result in results:
        cheatsheet_per_layer, exact_residuals_per_layer, quantized_residuals_per_layer = result
        cheatsheets.append(cheatsheet_per_layer)
        exact_residuals.append(exact_residuals_per_layer)
        for bit_width in quantized_residuals_per_layer:
            quantized_residuals[bit_width].append(quantized_residuals_per_layer[bit_width])

    # Save the cheatsheets
    if not os.path.exists(cheatsheet_save_path):
        torch.save(cheatsheets, cheatsheet_save_path)
    else:
        print(f"Cheatsheet already exists for {quantized_model_path}. Skipping...")

    # Save the extra residuals quantized for 2-bit and 8-bit if requested
    if save_extra_bit_widths:
        for bit_width in [2, 8]:
            quantized_residuals_save_path = quantized_residuals_paths[bit_width]
            if not os.path.exists(quantized_residuals_save_path):
                torch.save(quantized_residuals[bit_width], quantized_residuals_save_path)
            else:
                print(f"Quantized residuals for {bit_width}-bit already exist for {quantized_model_path}. Skipping...")

    # Save the exact residuals if requested
    if save_exact_residuals:
        exact_residuals_save_path = quantized_model_path + '/exact_residuals.pt'
        if not os.path.exists(exact_residuals_save_path):
            # Save the exact residuals
            torch.save(exact_residuals, exact_residuals_save_path)
        else:
            print(f"Exact residuals already exist for {quantized_model_path}. Skipping...")


if __name__ == '__main__':
    import argparse
    threadpoolctl.threadpool_limits(limits=1)

    parser = argparse.ArgumentParser(description="Generate cheatsheet for quantized model.")
    parser.add_argument('--original_model', type=str, required=True, help='Path to the original model')
    parser.add_argument('--quantized_model', type=str, required=True, help='Path to the quantized model')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save the cheatsheet and residuals')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes (optional)')
    parser.add_argument('--save_exact_residuals', action='store_true', help='Whether to save exact residuals')
    parser.add_argument('--save_extra_bit_widths', action='store_true', help='Whether to save 2-bit and 8-bit residuals')

    args = parser.parse_args()

    print(f"Creating cheatsheet for {args.quantized_model} based on {args.original_model}")
    create_cheatsheet(
        original_model_path=args.original_model,
        quantized_model_path=args.quantized_model,
        save_path=args.save_path,
        num_workers=args.num_workers,
        save_exact_residuals=args.save_exact_residuals,
        save_extra_bit_widths=args.save_extra_bit_widths
    )
