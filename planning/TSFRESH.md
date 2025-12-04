# TSFresh Feature Calculators - JAX Implementation Status

**Summary:** 12 of ~78 features implemented (~15%)

---

## ✅ Implemented (10 features - MinimalFCParameters)

- [x] `sum_values(x)` - Calculates the sum over the time series values
- [x] `median(x)` - Returns the median of x
- [x] `mean(x)` - Returns the mean of x
- [x] `length(x)` - Returns the length of x
- [x] `standard_deviation(x)` - Returns the standard deviation of x
- [x] `variance(x)` - Returns the variance of x
- [x] `root_mean_square(x)` - Returns the root mean square (rms) of the time series
- [x] `maximum(x)` - Calculates the highest value of the time series x
- [x] `absolute_maximum(x)` - Calculates the highest absolute value of the time series x
- [x] `minimum(x)` - Calculates the lowest value of the time series x

## ✅ Custom Features (2 additional)

- [x] `delta(x)` - Absolute difference between maximum and mean
- [x] `log_delta(x)` - Log-transformed delta for improved feature space separation

---

## ❌ Not Implemented

### Simple Statistics

- [ ] `abs_energy(x)` - Returns the absolute energy of the time series which is the sum over the squared values
- [ ] `kurtosis(x)` - Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2)
- [ ] `skewness(x)` - Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G1)
- [ ] `quantile(x, q)` - Calculates the q quantile of x
- [ ] `variation_coefficient(x)` - Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x

### Change/Difference Based

- [ ] `absolute_sum_of_changes(x)` - Returns the sum over the absolute value of consecutive changes in the series x
- [ ] `mean_abs_change(x)` - Average over first differences
- [ ] `mean_change(x)` - Average over time series differences
- [ ] `mean_second_derivative_central(x)` - Returns the mean value of a central approximation of the second derivative

### Counting

- [ ] `count_above(x, t)` - Returns the percentage of values in x that are higher than t — **selected_in: [Lorenz]**
- [ ] `count_above_mean(x)` - Returns the number of values in x that are higher than the mean of x
- [ ] `count_below(x, t)` - Returns the percentage of values in x that are lower than t — **selected_in: [Pendulum, Lorenz]**
- [ ] `count_below_mean(x)` - Returns the number of values in x that are lower than the mean of x

### Boolean

- [ ] `has_duplicate(x)` - Checks if any value in x occurs more than once — **selected_in: [Friction]**
- [ ] `has_duplicate_max(x)` - Checks if the maximum value of x is observed more than once
- [ ] `has_duplicate_min(x)` - Checks if the minimal value of x is observed more than once
- [ ] `has_variance_larger_than_standard_deviation(x)` - Is variance higher than the standard deviation? — **selected_in: [Friction]**
- [ ] `has_large_standard_deviation(x, r)` - Does time series have large standard deviation? — **selected_in: [Duffing, Friction]**

### Location Features

- [ ] `first_location_of_maximum(x)` - Returns the first location of the maximum value of x
- [ ] `first_location_of_minimum(x)` - Returns the first location of the minimal value of x
- [ ] `last_location_of_maximum(x)` - Returns the relative last location of the maximum value of x
- [ ] `last_location_of_minimum(x)` - Returns the last location of the minimal value of x
- [ ] `index_mass_quantile(x, param)` - Calculates the relative index i of time series x where q% of the mass of x lies left of i — **selected_in: [Friction]**

### Streak/Pattern Features

- [ ] `longest_strike_above_mean(x)` - Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x — **selected_in: [Pendulum, Duffing, Lorenz]**
- [ ] `longest_strike_below_mean(x)` - Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x — **selected_in: [Pendulum, Duffing, Lorenz]**
- [ ] `number_crossing_m(x, m)` - Calculates the number of crossings of x on m — **selected_in: [Pendulum, Duffing, Lorenz]**
- [ ] `number_peaks(x, n)` - Calculates the number of peaks of at least support n in the time series x — **selected_in: [Pendulum, Friction]**
- [ ] `number_cwt_peaks(x, n)` - Number of different peaks in x — **selected_in: [Duffing]**

### Autocorrelation

- [ ] `autocorrelation(x, lag)` - Calculates the autocorrelation of the specified lag
- [ ] `partial_autocorrelation(x, param)` - Calculates the value of the partial autocorrelation function at the given lag
- [ ] `agg_autocorrelation(x, param)` - Descriptive statistics on the autocorrelation of the time series

### Entropy/Complexity

- [ ] `approximate_entropy(x, m, r)` - Implements a vectorized Approximate entropy algorithm
- [ ] `sample_entropy(x)` - Calculate and return sample entropy of x
- [ ] `permutation_entropy(x, tau, dimension)` - Calculate the permutation entropy
- [ ] `binned_entropy(x, max_bins)` - First bins the values of x into max_bins equidistant bins
- [ ] `fourier_entropy(x, bins)` - Calculate the binned entropy of the power spectral density of the time series (using the welch method) — **selected_in: [Pendulum, Duffing, Friction]**
- [ ] `lempel_ziv_complexity(x, bins)` - Calculate a complexity estimate based on the Lempel-Ziv compression algorithm
- [ ] `cid_ce(x, normalize)` - This function calculator is an estimate for a time series complexity (A more complex time series has more peaks, valleys etc.)

### Frequency Domain (FFT/Spectral)

- [ ] `fft_coefficient(x, param)` - Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast fourier transformation algorithm — **selected_in: [Lorenz]**
- [ ] `fft_aggregated(x, param)` - Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum
- [ ] `spkt_welch_density(x, param)` - This feature calculator estimates the cross power spectral density of the time series x at different frequencies
- [ ] `cwt_coefficients(x, param)` - Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet"

### Trend/Regression

- [ ] `linear_trend(x, param)` - Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to length of the time series minus one
- [ ] `linear_trend_timewise(x, param)` - Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to length of the time series minus one
- [ ] `agg_linear_trend(x, param)` - Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus the sequence from 0 up to the number of chunks minus one
- [ ] `ar_coefficient(x, param)` - This feature calculator fits the unconditional maximum likelihood of an autoregressive AR(k) process
- [ ] `augmented_dickey_fuller(x, param)` - Does the time series have a unit root?

### Reoccurrence Features

- [ ] `percentage_of_reoccurring_datapoints_to_all_datapoints(x)` - Returns the percentage of non-unique data points
- [ ] `percentage_of_reoccurring_values_to_all_values(x)` - Returns the percentage of values that are present in the time series more than once
- [ ] `sum_of_reoccurring_data_points(x)` - Returns the sum of all data points, that are present in the time series more than once
- [ ] `sum_of_reoccurring_values(x)` - Returns the sum of all values, that are present in the time series more than once
- [ ] `ratio_value_number_to_time_series_length(x)` - Returns a factor which is 1 if all values in the time series occur only once, and below one if this is not the case

### Other Advanced

- [ ] `benford_correlation(x)` - Useful for anomaly detection applications. Returns the correlation from first digit distribution
- [ ] `c3(x, lag)` - Uses c3 statistics to measure non linearity in the time series
- [ ] `change_quantiles(x, ql, qh, isabs, f_agg)` - First fixes a corridor given by the quantiles ql and qh of the distribution of x
- [ ] `energy_ratio_by_chunks(x, param)` - Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole series
- [ ] `friedrich_coefficients(x, param)` - Coefficients of polynomial h(x), which has been fitted to the deterministic dynamics of Langevin model
- [ ] `max_langevin_fixed_point(x, r, m)` - Largest fixed point of dynamics estimated from polynomial h(x), which has been fitted to the deterministic dynamics of Langevin model
- [ ] `matrix_profile(x, param)` - Calculates the 1-D Matrix Profile and returns Tukey's Five Number Set plus the mean of that Matrix Profile
- [ ] `mean_n_absolute_max(x, number_of_maxima)` - Calculates the arithmetic mean of the n absolute maximum values of the time series
- [ ] `query_similarity_count(x, param)` - This feature calculator accepts an input query subsequence parameter, compares the query (under z-normalized Euclidean distance) to all subsequences within the time series
- [ ] `range_count(x, min, max)` - Count observed values within the interval [min, max) — **selected_in: [Lorenz]**
- [ ] `ratio_beyond_r_sigma(x, r)` - Ratio of values that are more than r * std(x) (so r times sigma) away from the mean of x
- [ ] `symmetry_looking(x, param)` - Boolean variable denoting if the distribution of x looks symmetric
- [ ] `time_reversal_asymmetry_statistic(x, lag)` - Returns the time reversal asymmetry statistic
- [ ] `value_count(x, value)` - Count occurrences of value in time series x