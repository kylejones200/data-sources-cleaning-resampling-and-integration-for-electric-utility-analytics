//! Hourly bucket aggregation (mean per bucket index).

pub fn bucket_hourly_means(values: &[f64], bucket_ids: &[i64], n_buckets: usize) -> Vec<f64> {
    assert_eq!(values.len(), bucket_ids.len());
    let mut sums = vec![0.0; n_buckets];
    let mut counts = vec![0usize; n_buckets];
    for (&v, &b) in values.iter().zip(bucket_ids) {
        let idx = b as usize;
        if idx < n_buckets {
            sums[idx] += v;
            counts[idx] += 1;
        }
    }
    sums.iter()
        .zip(counts)
        .map(|(&s, c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect()
}
