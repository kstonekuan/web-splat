// Scan phase for reduce-then-scan radix sort
// This shader computes the exclusive prefix sum of per-workgroup reductions
// and stores the prefix values for the scatter phase to read.
//
// This replaces the decoupled lookback mechanism which can deadlock on Apple GPUs.

struct GeneralInfo{
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
};

@group(0) @binding(0)
var<storage, read_write> infos: GeneralInfo;
@group(0) @binding(1)
var<storage, read_write> histograms : array<atomic<u32>>;
@group(0) @binding(2)
var<storage, read_write> keys : array<u32>;
@group(0) @binding(3)
var<storage, read_write> keys_b : array<u32>;
@group(0) @binding(4)
var<storage, read_write> payload_a : array<u32>;
@group(0) @binding(5)
var<storage, read_write> payload_b : array<u32>;

fn partitions_base_offset() -> u32 { return rs_keyval_size * rs_radix_size; }

// Safety limit to prevent infinite loops on any hardware
const MAX_WORKGROUPS: u32 = 65536u;

// Compute the exclusive prefix sum for a single digit across all workgroups
// Each thread handles one digit (0-255)
@compute @workgroup_size({prefix_wg_size})
fn scan_partitions_even(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cur_pass = infos.even_pass * 2u;

    // Calculate number of scatter workgroups
    let scatter_block_kvs = {scatter_wg_size}u * rs_scatter_block_rows;
    let num_scatter_wgs = (infos.keys_size + scatter_block_kvs - 1u) / scatter_block_kvs;

    // Each thread processes one digit, for both halves of the radix (like prefix_histogram)
    let digit1 = lid.x;
    let digit2 = lid.x + {prefix_wg_size}u;

    let partition_base = partitions_base_offset();
    let global_histogram_offset = cur_pass * rs_radix_size;

    // Read the global prefix for this digit (computed by prefix_histogram)
    var prefix1 = atomicLoad(&histograms[global_histogram_offset + digit1]);
    var prefix2 = atomicLoad(&histograms[global_histogram_offset + digit2]);

    // Process workgroups sequentially, computing exclusive prefix
    // Safety bound to prevent infinite loops
    let safe_num_wgs = min(num_scatter_wgs, MAX_WORKGROUPS);

    for (var wg = 0u; wg < safe_num_wgs; wg++) {
        let partition_offset = partition_base + wg * rs_radix_size;

        // Read the reduction for this workgroup
        let red1 = atomicLoad(&histograms[partition_offset + digit1]);
        let red2 = atomicLoad(&histograms[partition_offset + digit2]);

        // Store the exclusive prefix (prefix BEFORE this workgroup)
        atomicStore(&histograms[partition_offset + digit1], prefix1);
        atomicStore(&histograms[partition_offset + digit2], prefix2);

        // Accumulate for next workgroup's prefix
        prefix1 += red1;
        prefix2 += red2;
    }
}

@compute @workgroup_size({prefix_wg_size})
fn scan_partitions_odd(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cur_pass = infos.odd_pass * 2u + 1u;

    // Calculate number of scatter workgroups
    let scatter_block_kvs = {scatter_wg_size}u * rs_scatter_block_rows;
    let num_scatter_wgs = (infos.keys_size + scatter_block_kvs - 1u) / scatter_block_kvs;

    // Each thread processes one digit
    let digit1 = lid.x;
    let digit2 = lid.x + {prefix_wg_size}u;

    let partition_base = partitions_base_offset();
    let global_histogram_offset = cur_pass * rs_radix_size;

    // Read the global prefix for this digit
    var prefix1 = atomicLoad(&histograms[global_histogram_offset + digit1]);
    var prefix2 = atomicLoad(&histograms[global_histogram_offset + digit2]);

    // Safety bound to prevent infinite loops
    let safe_num_wgs = min(num_scatter_wgs, MAX_WORKGROUPS);

    for (var wg = 0u; wg < safe_num_wgs; wg++) {
        let partition_offset = partition_base + wg * rs_radix_size;

        let red1 = atomicLoad(&histograms[partition_offset + digit1]);
        let red2 = atomicLoad(&histograms[partition_offset + digit2]);

        atomicStore(&histograms[partition_offset + digit1], prefix1);
        atomicStore(&histograms[partition_offset + digit2], prefix2);

        prefix1 += red1;
        prefix2 += red2;
    }
}
