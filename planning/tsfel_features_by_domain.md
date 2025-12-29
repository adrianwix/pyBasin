# TSFEL Feature Extractors by Domain

Reference: https://tsfel.readthedocs.io/en/latest/_modules/tsfel/feature_extraction/features.html

Legend:

- ✅ = Implemented in torch_feature_calculators.py
- ⚠️ = Partially implemented or similar feature exists
- ❌ = Not implemented

## TEMPORAL DOMAIN

1. ✅ **autocorr** - Calculates the first 1/e crossing of the autocorrelation function (ACF) → `autocorrelation`
2. ❌ **calc_centroid** - Computes the centroid along the time axis
3. ❌ **negative_turning** - Computes number of negative turning points of the signal
4. ⚠️ **positive_turning** - Computes number of positive turning points of the signal → `number_peaks` (similar)
5. ✅ **mean_abs_diff** - Computes mean absolute differences of the signal → `mean_abs_change`
6. ✅ **mean_diff** - Computes mean of differences of the signal → `mean_change`
7. ❌ **median_abs_diff** - Computes median absolute differences of the signal
8. ❌ **median_diff** - Computes median of differences of the signal
9. ❌ **distance** - Computes signal traveled distance using the hypotenuse between 2 datapoints
10. ✅ **sum_abs_diff** - Computes sum of absolute differences of the signal → `absolute_sum_of_changes`
11. ✅ **zero_cross** - Computes Zero-crossing rate of the signal → `number_crossing_m`
12. ✅ **slope** - Computes the slope of the signal by fitting a linear equation → `linear_trend`
13. ❌ **auc** - Computes the area under the curve of the signal computed with trapezoid rule
14. ⚠️ **neighbourhood_peaks** - Computes the number of peaks from a defined neighbourhood of the signal → `number_peaks`
15. ✅ **lempel_ziv** - Computes the Lempel-Ziv's (LZ) complexity index, normalized by the signal's length → `lempel_ziv_complexity`

## STATISTICAL DOMAIN

1. ✅ **abs_energy** - Computes the absolute energy of the signal → `abs_energy`
2. ❌ **average_power** - Computes the average power of the signal
3. ⚠️ **entropy** - Computes the entropy of the signal using the Shannon Entropy → `binned_entropy`, `fourier_entropy`, `permutation_entropy`
4. ❌ **hist_mode** - Compute the mode of a histogram using a given number of (linearly spaced) bins
5. ⚠️ **interq_range** - Computes interquartile range of the signal → `quantile` (can calculate)
6. ✅ **kurtosis** - Computes kurtosis of the signal → `kurtosis`
7. ✅ **skewness** - Computes skewness of the signal → `skewness`
8. ✅ **calc_max** - Computes the maximum value of the signal → `maximum`
9. ✅ **calc_min** - Computes the minimum value of the signal → `minimum`
10. ✅ **calc_mean** - Computes mean value of the signal → `mean`
11. ✅ **calc_median** - Computes median of the signal → `median`
12. ❌ **mean_abs_deviation** - Computes mean absolute deviation of the signal
13. ❌ **median_abs_deviation** - Computes median absolute deviation of the signal
14. ✅ **rms** - Computes root mean square of the signal → `root_mean_square`
15. ✅ **calc_std** - Computes standard deviation (std) of the signal → `standard_deviation`
16. ✅ **calc_var** - Computes variance of the signal → `variance`
17. ⚠️ **pk_pk_distance** - Computes the peak to peak distance → `maximum` - `minimum`
18. ❌ **ecdf** - Computes the values of ECDF (empirical cumulative distribution function) along the time axis
19. ❌ **ecdf_slope** - Computes the slope of the ECDF between two percentiles
20. ❌ **ecdf_percentile** - Computes the percentile value of the ECDF
21. ❌ **ecdf_percentile_count** - Computes the cumulative sum of samples that are less than the percentile

## SPECTRAL DOMAIN

1. ❌ **spectral_distance** - Computes the signal spectral distance
2. ⚠️ **fundamental_frequency** - Computes fundamental frequency of the signal → `fft_coefficient` (can extract)
3. ⚠️ **max_power_spectrum** - Computes maximum power spectrum density of the signal → `spkt_welch_density`
4. ❌ **max_frequency** - Computes maximum frequency of the signal (0.95 of maximum frequency using cumsum)
5. ❌ **median_frequency** - Computes median frequency of the signal (0.50 of maximum frequency using cumsum)
6. ❌ **spectral_centroid** - Barycenter of the spectrum
7. ❌ **spectral_decrease** - Represents the amount of decreasing of the spectra amplitude
8. ❌ **spectral_kurtosis** - Measures the flatness of a distribution around its mean value
9. ❌ **spectral_skewness** - Measures the asymmetry of a distribution around its mean value
10. ❌ **spectral_spread** - Measures the spread of the spectrum around its mean value
11. ❌ **spectral_slope** - Computes the spectral slope by finding constants m and b of the function aFFT = mf + b
12. ❌ **spectral_variation** - Computes the amount of variation of the spectrum along time
13. ❌ **spectral_positive_turning** - Computes number of positive turning points of the fft magnitude signal
14. ❌ **spectral_roll_off** - Computes the spectral roll-off (frequency where 95% of the signal magnitude is contained)
15. ❌ **spectral_roll_on** - Computes the spectral roll-on (frequency where 5% of the signal magnitude is contained)
16. ❌ **human_range_energy** - Computes the human range energy ratio (energy in 0.6-2.5Hz / whole energy band)
17. ❌ **mfcc** - Computes the MEL cepstral coefficients
18. ❌ **power_bandwidth** - Computes power spectrum density bandwidth (width of frequency band with 95% of power)
19. ❌ **spectrogram_mean_coeff** - Calculates the average power spectral density (PSD) for each frequency
20. ❌ **lpcc** - Computes the linear prediction cepstral coefficients
21. ⚠️ **spectral_entropy** - Computes the spectral entropy of the signal based on Fourier transform → `fourier_entropy`
22. ❌ **wavelet_entropy** - Computes CWT entropy of the signal
23. ⚠️ **wavelet_abs_mean** - Computes CWT absolute mean value of each wavelet scale → `cwt_coefficients`
24. ❌ **wavelet_std** - Computes CWT std value of each wavelet scale
25. ❌ **wavelet_var** - Computes CWT variance value of each wavelet scale
26. ❌ **wavelet_energy** - Computes CWT energy of each wavelet scale

## FRACTAL DOMAIN

1. ❌ **dfa** - Computes the Detrended Fluctuation Analysis (DFA) of the signal
2. ❌ **hurst_exponent** - Computes the Hurst exponent through Rescaled range (R/S) analysis
3. ❌ **higuchi_fractal_dimension** - Computes the fractal dimension using Higuchi's method (HFD)
4. ❌ **maximum_fractal_length** - Computes the Maximum Fractal Length (MFL) using Higuchi's method
5. ❌ **petrosian_fractal_dimension** - Computes the Petrosian Fractal Dimension
6. ❌ **mse** - Computes the Multiscale entropy (MSE) that performs entropy analysis over multiple time scales

## Summary Statistics

- **Temporal Domain**: 15 features → 8 implemented ✅, 2 partially ⚠️, 5 missing ❌
- **Statistical Domain**: 21 features → 13 implemented ✅, 2 partially ⚠️, 6 missing ❌
- **Spectral Domain**: 26 features → 0 fully implemented, 4 partially ⚠️, 22 missing ❌
- **Fractal Domain**: 6 features → 0 implemented, 6 missing ❌
- **Total**: 68 features → 21 implemented (31%), 8 partial (12%), 39 missing (57%)

## Implementation Notes

### Already Implemented (21 core features):

**Temporal:** autocorrelation, mean_abs_change, mean_change, absolute_sum_of_changes, number_crossing_m, linear_trend, lempel_ziv_complexity

**Statistical:** abs_energy, kurtosis, skewness, maximum, minimum, mean, median, root_mean_square, standard_deviation, variance

**Additional non-TSFEL features we have:**

- Autocorrelation variants: partial_autocorrelation, agg_autocorrelation
- Entropy: approximate_entropy, sample_entropy
- FFT: fft_coefficient, fft_aggregated
- Wavelet: cwt_coefficients
- Dynamical systems: lyapunov_r, lyapunov_e, correlation_dimension

### Priority Missing Features (commonly used):

**Temporal:**

- calc_centroid
- negative_turning / positive_turning
- median_abs_diff / median_diff
- auc (area under curve)

**Statistical:**

- mean_abs_deviation / median_abs_deviation
- ECDF family (ecdf, ecdf_slope, ecdf_percentile, ecdf_percentile_count)

**Spectral:**

- spectral_centroid
- spectral_spread
- spectral_entropy (we have fourier_entropy but may differ)
- spectral_roll_off / spectral_roll_on
- mfcc
- spectral_distance

**Fractal:**

- dfa (Detrended Fluctuation Analysis)
- hurst_exponent
- higuchi_fractal_dimension
- petrosian_fractal_dimension
