------------------------------------------------------------------------------------------------------------------------
Feature                                    JAX warmup    JAX run      tsfresh   w/ compile  w/o compile
------------------------------------------------------------------------------------------------------------------------
  sum_values                                    27.57ms     1.68ms      22.35ms        0.81x       13.28x
  median                                      1079.10ms  1034.69ms     128.30ms        0.12x        0.12x
  mean                                          30.24ms     1.85ms      45.83ms        1.52x       24.78x
  length                                        34.10ms     0.26ms       0.96ms        0.03x        3.67x
  standard_deviation                            50.64ms    11.11ms     114.82ms        2.27x       10.34x
  variance                                      42.52ms    10.42ms      99.43ms        2.34x        9.54x
  root_mean_square                              27.99ms     9.21ms      55.47ms        1.98x        6.02x
  maximum                                       37.24ms     3.55ms      19.79ms        0.53x        5.58x
  absolute_maximum                              27.30ms    11.49ms      27.27ms        1.00x        2.37x
  minimum                                       36.39ms     4.81ms      19.80ms        0.54x        4.12x
  abs_energy                                     8.02ms     9.52ms       6.89ms        0.86x        0.72x
  kurtosis                                     111.53ms    26.07ms     479.53ms        4.30x       18.39x
  skewness                                      60.75ms    26.49ms     421.68ms        6.94x       15.92x
  variation_coefficient                         20.66ms    14.25ms     164.97ms        7.99x       11.58x
  absolute_sum_of_changes                       63.57ms    15.62ms      49.42ms        0.78x        3.16x
  mean_abs_change                               29.91ms    15.14ms      75.50ms        2.52x        4.99x
  mean_change                                  104.05ms     0.60ms       3.25ms        0.03x        5.43x
  mean_second_derivative_central                14.83ms     0.78ms       4.91ms        0.33x        6.30x
  count_above_mean                              86.78ms    11.86ms      69.53ms        0.80x        5.86x
  count_below_mean                              26.09ms    11.53ms      64.66ms        2.48x        5.61x
  has_duplicate                               1057.79ms  1059.62ms      84.80ms        0.08x        0.08x
  has_duplicate_max                             68.49ms    14.24ms      54.35ms        0.79x        3.82x
  has_duplicate_min                             46.95ms    15.72ms      55.91ms        1.19x        3.56x
  has_variance_larger_than_standard_devia       18.53ms    10.85ms     107.76ms        5.81x        9.94x
  first_location_of_maximum                     48.68ms     3.16ms      11.84ms        0.24x        3.75x
  first_location_of_minimum                     37.88ms     2.42ms      11.83ms        0.31x        4.89x
  last_location_of_maximum                      42.21ms     9.58ms      19.55ms        0.46x        2.04x
  last_location_of_minimum                      66.50ms    13.21ms      19.67ms        0.30x        1.49x
  longest_strike_above_mean                    135.75ms    61.64ms     277.32ms        2.04x        4.50x
  longest_strike_below_mean                     66.76ms    68.73ms     285.84ms        4.28x        4.16x
  percentage_of_reoccurring_datapoints_to     1039.50ms  1043.18ms    2288.83ms        2.20x        2.19x
  percentage_of_reoccurring_values_to_all     1073.22ms  1068.14ms     182.99ms        0.17x        0.17x
  sum_of_reoccurring_data_points              1079.37ms  1066.65ms     201.95ms        0.19x        0.19x
  sum_of_reoccurring_values                   1070.69ms  1069.11ms     215.10ms        0.20x        0.20x
  ratio_value_number_to_time_series_lengt     1050.70ms  1046.91ms      81.07ms        0.08x        0.08x
  benford_correlation                          473.09ms   202.24ms    9068.61ms       19.17x       44.84x
  time_reversal_asymmetry_statistic(lag=1      182.99ms    86.94ms     108.74ms        0.59x        1.25x
  c3(lag=1)                                     39.67ms    40.84ms      89.09ms        2.25x        2.18x
  cid_ce(normalize=True)                        40.96ms    29.12ms     207.25ms        5.06x        7.12x
  symmetry_looking(r=0.1)                     1033.50ms  1022.01ms     238.14ms        0.23x        0.23x
  large_standard_deviation(r=0.25)              18.67ms    18.54ms     157.64ms        8.44x        8.50x
  quantile(q=0.5)                             1070.10ms  1048.14ms     335.17ms        0.31x        0.32x
  autocorrelation(lag=1)                        58.82ms    45.60ms     320.81ms        5.45x        7.03x
  agg_autocorrelation(f_agg=mean, maxlag=      516.20ms   249.61ms    1441.06ms        2.79x        5.77x
  partial_autocorrelation(lag=1)                45.83ms    45.76ms    1326.30ms       28.94x       28.98x
  number_cwt_peaks(max_width=5)                181.46ms   177.86ms   75459.94ms      415.85x      424.27x
  number_peaks(n=3)                            146.97ms   110.46ms     163.84ms        1.11x        1.48x
  binned_entropy(max_bins=10)                  277.11ms   158.36ms     501.87ms        1.81x        3.17x
  fourier_entropy(bins=10)                     238.64ms    53.58ms    3339.58ms       13.99x       62.33x
  permutation_entropy(tau=1, dimension=3)     3345.49ms  3191.25ms    4544.03ms        1.36x        1.42x
  lempel_ziv_complexity(bins=2)               1090.94ms  1035.34ms   11133.54ms       10.21x       10.75x
  index_mass_quantile(q=0.5)                   137.18ms    55.41ms     120.68ms        0.88x        2.18x
  fft_coefficient(coeff=0, attr=abs)           129.79ms    26.55ms     127.70ms        0.98x        4.81x
  fft_aggregated(aggtype=centroid)             129.69ms    64.64ms     170.55ms        1.32x        2.64x
  spkt_welch_density(coeff=2)                  146.48ms     6.63ms    2639.52ms       18.02x      398.13x
  cwt_coefficients(widths=(2,), coeff=0)       332.29ms    86.90ms    4524.49ms       13.62x       52.07x
  ar_coefficient(coeff=0, k=10)               1038.90ms   413.25ms   16094.87ms       15.49x       38.95x
  linear_trend(attr=slope)                     239.99ms    44.97ms    2775.42ms       11.56x       61.71x
  linear_trend_timewise(attr=slope)             19.53ms    59.53ms        ERROR          N/A          N/A
  agg_linear_trend(attr=slope, chunk_size      267.58ms    10.13ms    6197.13ms       23.16x      611.92x