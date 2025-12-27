// Reduce phase for reduce-then-scan radix sort
// This shader computes per-workgroup histogram reductions and stores them
// to the partitions buffer for later scanning.
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

var<workgroup> smem : array<atomic<u32>, rs_radix_size>;
var<private> kv : array<u32, rs_scatter_block_rows>;
var<private> pv : array<u32, rs_scatter_block_rows>;
var<private> kr : array<u32, rs_scatter_block_rows>;

fn partitions_base_offset() -> u32 { return rs_keyval_size * rs_radix_size; }

fn zero_smem(lid: u32) {
    if lid < rs_radix_size {
        atomicStore(&smem[lid], 0u);
    }
}

fn histogram_load(digit: u32) -> u32 {
    return atomicLoad(&smem[digit]);
}

fn histogram_store(digit: u32, count: u32) {
    atomicStore(&smem[digit], count);
}

// Load keys for even pass (keys -> keys_b)
fn fill_kv_even(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals : u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_a[pos];
    }
}

// Load keys for odd pass (keys_b -> keys)
fn fill_kv_odd(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals : u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys_b[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_b[pos];
    }
}

// Compute and store the per-workgroup histogram reduction
fn reduce_pass(pass_: u32, lid: vec3<u32>, wid: vec3<u32>, nwg: vec3<u32>) {
    let subgroup_id = lid.x / histogram_sg_size;
    let subgroup_offset = subgroup_id * histogram_sg_size;
    let subgroup_tid = lid.x - subgroup_offset;
    let subgroup_count = {scatter_wg_size}u / histogram_sg_size;

    // Compute local ranks for each key
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let u_val = bitcast<u32>(kv[i]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicStore(&smem[lid.x], digit);
        workgroupBarrier();  // Ensure all threads have stored their digit before reading
        var count = 0u;
        var rank = 0u;

        for (var j = 0u; j < histogram_sg_size; j++) {
            if atomicLoad(&smem[subgroup_offset + j]) == digit {
                count += 1u;
                if j <= subgroup_tid {
                    rank += 1u;
                }
            }
        }

        kr[i] = (count << 16u) | rank;
    }

    zero_smem(lid.x);
    workgroupBarrier();

    // Accumulate per-workgroup histogram in shared memory
    for (var i = 0u; i < subgroup_count; i++) {
        if subgroup_id == i {
            for (var j = 0u; j < rs_scatter_block_rows; j++) {
                let v = bitcast<u32>(kv[j]);
                let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
                let prev = histogram_load(digit);
                let rank = kr[j] & 0xFFFFu;
                let count = kr[j] >> 16u;
                kr[j] = prev + rank;

                if rank == count {
                    histogram_store(digit, (prev + count));
                }
            }
        }
        workgroupBarrier();
    }

    // Store the per-workgroup reduction to the partitions buffer
    // Each workgroup stores its reduction value for each digit
    let partition_offset = lid.x + partitions_base_offset();
    let partition_base = wid.x * rs_radix_size;

    if lid.x < rs_radix_size {
        let red = histogram_load(lid.x);
        // Store reduction with status bits = 0 (indicates reduction, not prefix)
        atomicStore(&histograms[partition_offset + partition_base], red);
    }
}

@compute @workgroup_size({scatter_wg_size})
fn reduce_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cur_pass = infos.even_pass * 2u;
    fill_kv_even(wid.x, lid.x);
    reduce_pass(cur_pass, lid, wid, nwg);
}

@compute @workgroup_size({scatter_wg_size})
fn reduce_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cur_pass = infos.odd_pass * 2u + 1u;
    fill_kv_odd(wid.x, lid.x);
    reduce_pass(cur_pass, lid, wid, nwg);
}
