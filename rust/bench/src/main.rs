use data_sources_cleaning_resampling_and_integration_for_electric_utility_analytics_core::bucket_hourly_means;

fn main() {
    let n = 10000usize;
    let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin() + 50.0).collect();
    let buckets: Vec<i64> = (0..n).map(|i| (i % 24) as i64).collect();
    for _ in 0..5000 {
        let _ = bucket_hourly_means(&values, &buckets, 24);
    }
}
