import os, numpy as np, os
import matplotlib.pyplot as plt
import tools21cm as t2c
import cupy as cp
import itertools
import time

import astropy.units as u
import astropy.constants as cst

from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from tqdm import tqdm
from matplotlib.colors import LogNorm
plt.rcParams["font.family"] = "serif"
from scipy import stats

from scipy.interpolate import make_interp_spline

from sys import getsizeof


def _splice_values_equal_bins_cupy(blcoord_sorted, blmag_sorted, vis_sorted, num_bins):
    """
    The provided arrays should be cupy arrays.
    """
    # Step 1: Calculate the number of elements per bin
    num_elements = len(blmag_sorted)
    elements_per_bin = num_elements // num_bins
    extra_elements = num_elements % num_bins

    # Step 2: Create an array of bin sizes
    bin_sizes = cp.full(num_bins, elements_per_bin)
    bin_sizes[:extra_elements] += 1

    # Step 3: Calculate the cumulative sum of bin sizes
    cumulative_sizes = cp.cumsum(bin_sizes)
    cumulative_sizes_np = cp.asnumpy(cumulative_sizes)  # Convert to NumPy array

    # Step 4: Initialize the bins
    blcoord_bins = cp.split(blcoord_sorted, cumulative_sizes_np[:-1])
    blmag_bins = cp.split(blmag_sorted, cumulative_sizes_np[:-1])
    vis_bins = cp.split(vis_sorted, cumulative_sizes_np[:-1])

    return blcoord_bins, blmag_bins, vis_bins



def _splice_values_function_fit_cupy(blmag_sorted_transformed, blcoord_sorted, blmag_sorted, vis_sorted, num_bins):
    bins = cp.linspace(np.min(blmag_sorted_transformed), np.max(blmag_sorted_transformed), num_bins)
    bin_indices = np.digitize(blmag_sorted_transformed, bins) - 1
    
    # Use histogram to count the occurrences of each bin
    hist, _ = cp.histogram(bin_indices, bins=cp.arange(0, num_bins + 1))

    # Calculate the cumulative counts to determine the indices where each bin's values start and end
    cum_counts = cp.cumsum(hist)
    cumulative_sizes_np = cp.asnumpy(cum_counts)  # Convert to NumPy array

    bin_blcoords = cp.split(blcoord_sorted, cumulative_sizes_np[:-1])
    bin_vis = cp.split(vis_sorted, cumulative_sizes_np[:-1])
    bin_blmags = cp.split(blmag_sorted, cumulative_sizes_np[:-1])

    return bin_blcoords, bin_blmags, bin_vis



def compute_vals_cupy(blcoord, vis, sigma_0, V_0, bin_number):

    bin_len = len(blcoord)
    if bin_len == 0 or bin_len == 1:
        print("Skipped bin:", bin_number)
        return 0, 0, 0

    # Create a meshgrid for vectorized pairwise operations
    idx_i, idx_j = np.triu_indices(bin_len, k=1)
    idx_i = cp.asarray(idx_i)
    idx_j = cp.asarray(idx_j)

    # Calculate pairwise distances
    delta= blcoord[idx_i] - blcoord[idx_j]
    distance_norm = cp.linalg.norm(delta, axis=1)
    
    combination_len_unfiltered= len(distance_norm)
    
    # Filter pairs where distance is less than or equal to sigma_0
    valid_pairs = distance_norm <= sigma_0
    
    if not cp.any(valid_pairs):
        print("Skipped bin:", bin_number)
        return 0, 0, 0

    distance_norm = distance_norm[valid_pairs]
    idx_i = idx_i[valid_pairs]
    idx_j = idx_j[valid_pairs]
    
    combination_len_filtered= len(distance_norm)
    print("Bin", bin_number, ":Theoretical combinations: ", combination_len_unfiltered,
         " - When filtered: ", combination_len_filtered)
    
    
    # Calculate weights
    weight = cp.exp(-(distance_norm**2) / sigma_0**2)

    # Calculate the visibility product
    visibility_product = vis[idx_i] * cp.conj(vis[idx_j])

    #Calclating the bare estimator
    be_numerator = cp.sum(weight * visibility_product)
    be_denominator = cp.sum(weight * V_0 * cp.exp(-(distance_norm**2) / (sigma_0**2)))

    #Calculating the mean of ell
    l_i = 2 * cp.pi * cp.sqrt(blcoord[idx_i, 0]**2 + blcoord[idx_i, 1]**2)
    exp_weight_distance = weight * cp.exp(-(distance_norm**2) / (sigma_0**2))

    ell_numerator = cp.sum(exp_weight_distance * l_i)
    ell_denominator = cp.sum(exp_weight_distance)

    #Computing arrays needed for the error (variance)
    #Eb_square_mean_numerator = cp.sum((exp_weight_distance * 5.13 * 1e-4 * (1000 / l_i) ** 2.34) ** 2)
    #Eb_mean_square_numerator = cp.sum(exp_weight_distance * 5.13 * 1e-4 * (1000 / l_i) ** 2.34)
    Eb_square_mean_numerator = cp.sum(exp_weight_distance * (513 * (1000 / l_i) ** 2.34) ** 2)
    Eb_mean_square_numerator = cp.sum(exp_weight_distance * 513 * (1000 / l_i) ** 2.34)
    #Eb_square_mean_numerator = cp.sum((exp_weight_distance * 513) ** 2)
    #Eb_mean_square_numerator = cp.sum(exp_weight_distance * 513)
    
    Eb_denominator = cp.sum(exp_weight_distance)

    bare_estimator = cp.abs(be_numerator) / be_denominator
    ell_mean = ell_numerator / ell_denominator
    var_squared = (Eb_square_mean_numerator / Eb_denominator - (Eb_mean_square_numerator / Eb_denominator) ** 2)

    # Convert results back to numpy arrays
    bare_estimator = cp.asnumpy(bare_estimator)
    ell_mean = cp.asnumpy(ell_mean)
    var_squared = cp.asnumpy(var_squared)
    
    # Explicitly delete the intermediate GPU arrays to free memory
    del idx_i, idx_j, delta, distance_norm, valid_pairs
    del weight, visibility_product, be_numerator, be_denominator
    del l_i, exp_weight_distance, ell_numerator, ell_denominator
    del Eb_square_mean_numerator, Eb_mean_square_numerator, Eb_denominator
    
    # Synchronize to ensure all operations are complete
    cp.cuda.Stream.null.synchronize()

    return bare_estimator, ell_mean, np.abs(var_squared)



def bare_estimator_func_cupy(baselines, visibilities, theta_fwhm, num_bins, bin_type, precision, max_dist, wavelength):
    # Set data types based on the precision parameter
    if precision == 'single':
        float_dtype = cp.float32
        complex_dtype = cp.complex64
    elif precision == 'double':
        float_dtype = cp.float64
        complex_dtype = cp.complex128
    else:
        raise ValueError("Unsupported precision type. Use 'single' or 'double'.")    
    
    theta_0 = 0.6 * theta_fwhm
    V_0 = np.pi * theta_0**2 / 2
    sigma_0 = 0.76 / theta_fwhm
    
    print("sigma_0: ", sigma_0)
    print("V_0:", V_0)

    start_time = time.time()
    # Step 0: turn arrays into cupy arrays
    baselines= cp.asarray(baselines, dtype= float_dtype)
    visibilities= cp.asarray(visibilities, dtype= complex_dtype)

    # Step 1: Calculate the baseline magnitudes
    bl_mag = cp.linalg.norm(baselines, axis=1)
    
    print("Length of input arrays: ", len(bl_mag))
    
    # Step 2: Apply filtering based on max_bl_mag if provided
    if max_dist is not None:
        max_bl_mag = max_dist / wavelength
        print('Max. uvw:', round(max_bl_mag,3))
        mask = bl_mag <= max_bl_mag
        baselines = baselines[mask]
        visibilities = visibilities[mask]
        bl_mag = bl_mag[mask]
        
    print("Length of input arrays after filtering: ", len(bl_mag))

    # Stack baselines, magnitudes, and visibilities into a single array for sorting
    #combined_array = cp.hstack((baselines[:,0, None], baselines[:,1, None], bl_mag[:, None], visibilities[:, None]))
    combined_array = cp.hstack((baselines[:], bl_mag[:, None], visibilities[:, None]))

    # Sort the combined array based on the magnitudes
    sorted_combined_array = combined_array[cp.argsort(combined_array[:,2])]

    # Split the sorted array back into individual components
    blcoord_sorted = sorted_combined_array[:, 0:2].real
    blmag_sorted = sorted_combined_array[:, 2].real
    vis_sorted = sorted_combined_array[:, -1]
        
    #This is for sorting the baselines into bins, depending on the log of the baseline norm, or just the baseline norm, 
    #or such that all the bins have roughly the same size
    if bin_type == "log":
        blmag_sorted_transformed = cp.log10(blmag_sorted)
        blcoord_bins, blmag_bins, vis_bins = _splice_values_function_fit_cupy(blmag_sorted_transformed, blcoord_sorted, blmag_sorted, vis_sorted, num_bins)

    elif bin_type == "linear":
        blmag_sorted_transformed = blmag_sorted
        blcoord_bins, blmag_bins, vis_bins = _splice_values_function_fit_cupy(blmag_sorted_transformed, blcoord_sorted, blmag_sorted, vis_sorted, num_bins)

    elif bin_type == "equal_length":
        blcoord_bins, blmag_bins, vis_bins = _splice_values_equal_bins_cupy(blcoord_sorted, blmag_sorted, vis_sorted, num_bins)
    else:
        raise ValueError("Unsupported bin_type")

    end_time = time.time()
    print('Time to sort and bin values: ', end_time-start_time, 's')

    # Verify the bins
    #for i, bin in enumerate(blcoord_bins):
    #    print(f"Bin {i+1}: {len(bin)} elements - Combinations: {int(len(bin)*(len(bin)-1) / 2)}")
    # Print to verify the results
    #print("Total elements after binning: ", cp.sum(cp.array([len(bin) for bin in blmag_bins])))

    bare_estimators = np.empty(num_bins, dtype=float_dtype)
    average_ells = np.empty(num_bins, dtype=float_dtype)
    square_variances = np.empty(num_bins, dtype=float_dtype)

    start_time = time.time()
    for i in range(num_bins):
        bin_number = i + 1
        BE, ell_avg, var_squared= compute_vals_cupy(blcoord_bins[i], vis_bins[i], sigma_0, V_0, bin_number) 

        bare_estimators[i] = BE
        average_ells[i] = ell_avg
        square_variances[i] = var_squared
    end_time = time.time()
    print('Time to run the bare estimator: ', end_time-start_time, 's')
    
    return bare_estimators, average_ells, square_variances




#visls = np.load('visibilities/SKA_all_visibilities_virgin_37.315diam_nobeam.npy')
#visls = np.load('visibilities/MWA_all_visibilities_gainandsystematic_4.51diam_nobeam.npy')
#visls = np.load('visibilities/MWA_all_visibilities_virgin_4.51diam_nobeam.npy')
#visls = np.load('visibilities/SKA_all_visibilities_gainandsystematic_37.315diam_nobeam.npy')
visls = np.load('visibilities/MWA_all_visibilities_gainandsystematic_4.51diam_nobeam.npy')
visls = visls.astype(np.complex64)

print('Memory of the visibilities array:', getsizeof(visls) / 1e6, 'Mb')
print('Shape of the visibilities:', visls.shape)
print(30*'-')

#all_uvw = np.load('bl_lc_256_train_130923_i0_dT_ch600_4h1d_256.npy', mmap_mode= 'r')
all_uvw = np.load('bl_MWA_syngf_ch600_1024.npy', mmap_mode= 'r')
uvw = all_uvw
uvw = uvw.astype(np.float32)
print('Memory of the uvw array:', getsizeof(uvw) / 1e6, 'Mb')
print('Shape of the uvw:', uvw.shape)
print(30*'-')

freq = 166000000
lam = cst.c.value / freq
z = t2c.nu_to_z(freq/ 1e6)
#D = 37.315 #Antenna base diameter
D = 4.51 #Antenna base diameter
print('frequency [Hz]:', freq)
print('wavelength [m]:', lam)
print('redshift:', z)
print('dish diameter [m]:', D)

theta_fwhm = (1.03 * lam / D) * u.rad #D is the diameter of the antenna base
print("theta_fwhm:", theta_fwhm)


#num_bins = 2000
num_bins = 20
binning_method = 'equal_length'
#binning_method = 'log'

"""
uvw_flat = uvw[0:360].reshape(-1, 2)
print(uvw_flat.shape)
visls_flat = visls[0:360].flatten()
print(visls_flat.shape)

print("Input arrays shape:", uvw_flat.shape)
bare_estimator, ell_mean, var_squared= bare_estimator_func_cupy(uvw_flat, visls_flat, theta_fwhm.value, num_bins, binning_method, 'single', 28743, 1.8059786626506025)

# Save all arrays in one file
np.savez('BE_results_SKA_0to360_2000bins.npz', bare_estimator=bare_estimator, ell_mean=ell_mean, var_squared=var_squared)
print("Arrays saved in a single .npz file.")
"""

# Define the time step ranges
#time_step_ranges = [(0, 360), (360, 720), (720, 1080), (1080, 1440)]
time_step_ranges = [(0, 90), (90, 180), (180, 270), (270, 360)]

# Iterate over each time step range
for i, (start, end) in enumerate(time_step_ranges):
    # Flatten the UVW and visibilities for the current time step range
    uvw_flat = uvw[start:end].reshape(-1, 2)
    visls_flat = visls[start:end].flatten()

    print(f"Processing time steps from {start} to {end}...")
    print("Input arrays shape:", uvw_flat.shape)

    # Compute the bare estimator, ell mean, and variance squared
    """
    bare_estimator, ell_mean, var_squared = bare_estimator_func_cupy(
        uvw_flat, visls_flat, theta_fwhm.value, num_bins, binning_method, 'single', 28743, 1.8059786626506025)
    """
    bare_estimator, ell_mean, var_squared = bare_estimator_func_cupy(
        uvw_flat, visls_flat, theta_fwhm.value, num_bins, binning_method, 'single', 2874, 1.8059786626506025)
    
    # Save the results in a separate file for each range
    output_filename = f'BE_results_MWA_{start}to{end}_20bins.npz'
    #output_filename = f'BE_results_SKA_{start}to{end}_20bins_virgin.npz'
    np.savez(output_filename, bare_estimator=bare_estimator, ell_mean=ell_mean, var_squared=var_squared)
    print(f"Arrays saved in {output_filename}.")