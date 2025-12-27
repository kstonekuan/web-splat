// Unit tests for the reduce-then-scan radix sort implementation

// Helper functions for potential future GPU testing
#[allow(dead_code)]
fn is_sorted(data: &[f32]) -> bool {
    for i in 1..data.len() {
        if data[i - 1] > data[i] {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
fn verify_sort_correctness(original: &[f32], sorted: &[f32]) -> bool {
    if original.len() != sorted.len() {
        return false;
    }

    // Check if sorted
    if !is_sorted(sorted) {
        return false;
    }

    // Check if permutation (same elements)
    let mut orig_sorted = original.to_vec();
    orig_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut result_sorted = sorted.to_vec();
    result_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    orig_sorted == result_sorted
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_radix_sort_small() {
        // This is a compile-time test to ensure the shaders are included correctly
        // The actual GPU sort test requires async runtime
        let shader_code = include_str!("../src/shaders/radix_sort.wgsl");
        assert!(shader_code.contains("scatter"));
        assert!(!shader_code.contains("while true")); // Verify lookback removed
    }

    #[test]
    fn test_reduce_shader_exists() {
        let shader_code = include_str!("../src/shaders/radix_sort_reduce.wgsl");
        assert!(shader_code.contains("reduce_even"));
        assert!(shader_code.contains("reduce_odd"));
    }

    #[test]
    fn test_scan_shader_exists() {
        let shader_code = include_str!("../src/shaders/radix_sort_scan.wgsl");
        assert!(shader_code.contains("scan_partitions_even"));
        assert!(shader_code.contains("scan_partitions_odd"));
        assert!(shader_code.contains("MAX_WORKGROUPS")); // Safety limit
    }

    #[test]
    fn test_no_infinite_loops_in_shaders() {
        // Verify that the problematic while true loop has been removed
        let scatter_shader = include_str!("../src/shaders/radix_sort.wgsl");
        let reduce_shader = include_str!("../src/shaders/radix_sort_reduce.wgsl");
        let scan_shader = include_str!("../src/shaders/radix_sort_scan.wgsl");

        // None of the shaders should contain unbounded while loops
        assert!(
            !scatter_shader.contains("while true"),
            "Scatter shader contains 'while true'"
        );
        assert!(
            !reduce_shader.contains("while true"),
            "Reduce shader contains 'while true'"
        );
        assert!(
            !scan_shader.contains("while true"),
            "Scan shader contains 'while true'"
        );
    }

    #[test]
    fn test_safety_limits_in_scan() {
        let scan_shader = include_str!("../src/shaders/radix_sort_scan.wgsl");

        // Verify the MAX_WORKGROUPS safety limit exists
        assert!(
            scan_shader.contains("MAX_WORKGROUPS"),
            "Scan shader missing MAX_WORKGROUPS safety limit"
        );

        // Verify the limit is used in the loop
        assert!(
            scan_shader.contains("safe_num_wgs"),
            "Scan shader not using safe workgroup limit"
        );
    }

    #[test]
    fn test_is_sorted_helper() {
        use super::is_sorted;
        assert!(is_sorted(&[1.0, 2.0, 3.0, 4.0]));
        assert!(is_sorted(&[1.0, 1.0, 2.0, 2.0]));
        assert!(is_sorted(&[]));
        assert!(is_sorted(&[1.0]));
        assert!(!is_sorted(&[2.0, 1.0, 3.0]));
    }

    #[test]
    fn test_verify_sort_correctness_helper() {
        use super::verify_sort_correctness;
        let original = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let sorted = vec![1.0, 1.0, 3.0, 4.0, 5.0, 9.0];
        assert!(verify_sort_correctness(&original, &sorted));

        let wrong = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(!verify_sort_correctness(&original, &wrong));
    }
}
