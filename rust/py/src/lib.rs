use data_sources_cleaning_resampling_and_integration_for_electric_utility_analytics_core::bucket_hourly_means;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn bucket_hourly_means_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    bucket_ids: PyReadonlyArray1<i64>,
    n_buckets: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(bucket_hourly_means(values.as_slice()?, bucket_ids.as_slice()?, n_buckets).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (values, bucket_ids, n_buckets, iterations=5_000))]
fn bench_kernel_py(
    values: PyReadonlyArray1<f64>,
    bucket_ids: PyReadonlyArray1<i64>,
    n_buckets: usize,
    iterations: usize,
) -> PyResult<f64> {
    let v = values.as_slice()?.to_vec();
    let b = bucket_ids.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = bucket_hourly_means(&v, &b, n_buckets);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn data_sources_cleaning_resampling_and_integration_for_electric_utility_analytics_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bucket_hourly_means_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
