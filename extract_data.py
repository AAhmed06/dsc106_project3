"""
Script to extract CMIP6 data and convert to JSON format for D3.js visualization
"""
import pandas as pd
import xarray as xr
import zarr
import gcsfs
import numpy as np
import json
from datetime import datetime

def extract_temperature_data():
    """Extract air temperature data (ta) from CMIP6"""
    gcs = None
    ds = None
    try:
        print("Loading CMIP6 catalog...")
        df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        
        print("Querying temperature data...")
        subset = df.query("activity_id=='CMIP' & variable_id == 'ta' & table_id == 'Emon' & experiment_id=='historical'")
        
        if len(subset) == 0:
            print("No data found!")
            return None
        
        # Get the first available dataset
        zstore = subset.zstore.values[-1]
        print(f"Using zstore: {zstore}")
        
        # Create GCS filesystem
        gcs = gcsfs.GCSFileSystem(token='anon')
        mapper = gcs.get_mapper(zstore)
        
        # Open dataset
        print("Opening dataset...")
        ds = xr.open_zarr(mapper, consolidated=True)
        
        # Extract data for multiple time steps
        var = 'ta'
        
        # Check if there's a pressure level dimension
        if 'plev' in ds[var].dims:
            plev_index = 0  # Surface level
            da2d = ds[var].isel(time=0, plev=plev_index)
        else:
            da2d = ds[var].isel(time=0)
        
        # Get coordinate arrays
        lons_full = ds['lon'].values
        lats_full = ds['lat'].values
        times = ds['time'].values
        
        # Limit time steps to keep file size under 100MB
        # Current: 600 steps = 158MB, so ~380 steps ≈ 100MB (31-32 years)
        # Using 360 steps (30 years) to stay safely under 100MB
        max_times_for_100mb = 360  # 30 years of monthly data
        num_times = min(max_times_for_100mb, len(times))
        start_index = len(times) - num_times  # Start from the most recent time steps
        
        # Downsample spatial resolution to reduce file size (take every 2nd point)
        # This reduces file size by ~4x while maintaining good detail
        spatial_step = 2
        lons = lons_full[::spatial_step]
        lats = lats_full[::spatial_step]
        
        years = num_times / 12
        print(f"Extracting {num_times} time steps (most recent {years:.1f} years)...")
        print(f"  Time range: {times[start_index]} to {times[-1]}")
        print(f"  Spatial resolution: {len(lons)}x{len(lats)} (downsampled from {len(lons_full)}x{len(lats_full)})")
        
        data = {
            'variable': var,
            'longitude': lons.tolist(),
            'latitude': lats.tolist(),
            'timeSteps': []
        }
        
        # Extract in reverse order: most recent first (index 0 = most recent)
        # Load all time steps at once for better performance
        if 'plev' in ds[var].dims:
            # Select all time steps and pressure level at once
            da_subset = ds[var].isel(time=slice(start_index, start_index + num_times), plev=plev_index)
        else:
            # Select all time steps at once
            da_subset = ds[var].isel(time=slice(start_index, start_index + num_times))
        
        # Load data into memory (this is faster than loading one at a time)
        print("  Loading data into memory...")
        da_subset = da_subset.load()
        
        # Iterate backwards through the time steps
        for i in range(num_times - 1, -1, -1):  # From 59 down to 0
            t = start_index + i  # Actual time index in dataset (oldest to newest)
            time_idx = i  # Index in the subset array
            
            # Get values for this time step (already loaded)
            values = da_subset.isel(time=time_idx).values
            
            # Downsample spatial resolution (take every Nth point in both dimensions)
            values_downsampled = values[::spatial_step, ::spatial_step]
            
            # Vectorized conversion: Kelvin to Celsius
            # This is much faster than nested loops
            values_celsius = values_downsampled - 273.15
            
            # Convert to list and handle NaN (faster than nested loops)
            # Use object dtype to allow None values
            values_list = []
            for row in values_celsius:
                # Use list comprehension which is faster than nested loops with conditionals
                values_list.append([None if np.isnan(v) else float(v) for v in row])
            
            # Get time as string
            time_str = str(times[t])
            
            # Store with most recent at index 0 (append as we go backwards)
            data['timeSteps'].append({
                'timeIndex': num_times - 1 - i,  # 0 for most recent, 59 for oldest
                'time': time_str,
                'values': values_list
            })
            
            if (num_times - i) % 10 == 0:
                print(f"  Processed {num_times - i}/{num_times} time steps...")
        
        print(f"Saving data to temperature_data.json...")
        with open('temperature_data.json', 'w') as f:
            json.dump(data, f)
        
        print("Done!")
        return data
    finally:
        # Clean up resources
        if ds is not None:
            ds.close()
        if gcs is not None:
            try:
                gcs.close()
            except:
                pass

def extract_vegetation_data():
    """Extract vegetation carbon data (cVeg) from CMIP6"""
    gcs = None
    ds = None
    try:
        print("Loading CMIP6 catalog...")
        df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        
        print("Querying vegetation data...")
        # Try Lmon first (land monthly), then Emon (energy monthly)
        subset = df.query("activity_id=='CMIP' & variable_id == 'cVeg' & table_id == 'Lmon' & experiment_id=='historical'")
        
        if len(subset) == 0:
            # Fallback to Emon if Lmon not available
            subset = df.query("activity_id=='CMIP' & variable_id == 'cVeg' & table_id == 'Emon' & experiment_id=='historical'")
        
        if len(subset) == 0:
            print("No vegetation data found! Trying without table constraint...")
            # Try without table constraint
            subset = df.query("activity_id=='CMIP' & variable_id == 'cVeg' & experiment_id=='historical'")
        
        if len(subset) == 0:
            print("No vegetation data found!")
            return None
        
        print(f"Found {len(subset)} vegetation data record(s), using table_id: {subset.table_id.values[0]}")
        
        zstore = subset.zstore.values[-1]
        print(f"Using zstore: {zstore}")
        
        gcs = gcsfs.GCSFileSystem(token='anon')
        mapper = gcs.get_mapper(zstore)
        
        print("Opening dataset...")
        ds = xr.open_zarr(mapper, consolidated=True)
        
        var = 'cVeg'
        lons_full = ds['lon'].values
        lats_full = ds['lat'].values
        times = ds['time'].values
        
        # Limit to 360 time steps (30 years of monthly data) to keep file size manageable
        max_times = 360  # 30 years of monthly data
        num_times = min(max_times, len(times))
        start_index = len(times) - num_times  # Start from the most recent time steps
        
        # Downsample spatial resolution to reduce file size (take every 2nd point)
        # With spatial_step = 2, 400MB -> ~100MB (4x reduction)
        spatial_step = 2
        lons = lons_full[::spatial_step]
        lats = lats_full[::spatial_step]
        
        years = num_times / 12
        print(f"Extracting {num_times} time steps (most recent {years:.1f} years)...")
        print(f"  Spatial resolution: {len(lons)}x{len(lats)} (downsampled from {len(lons_full)}x{len(lats_full)})")
        
        data = {
            'variable': var,
            'longitude': lons.tolist(),
            'latitude': lats.tolist(),
            'timeSteps': []
        }
        
        # Extract in reverse order: most recent first (index 0 = most recent)
        # Load all time steps at once for better performance
        print("  Loading data into memory...")
        da_subset = ds[var].isel(time=slice(start_index, start_index + num_times))
        da_subset = da_subset.load()
        
        # Iterate backwards through the time steps
        for i in range(num_times - 1, -1, -1):  # From num_times-1 down to 0
            t = start_index + i  # Actual time index in dataset
            time_idx = i  # Index in the subset array
            
            # Get values for this time step (already loaded)
            values = da_subset.isel(time=time_idx).values
            
            # Downsample spatial resolution (take every Nth point in both dimensions)
            values_downsampled = values[::spatial_step, ::spatial_step]
            
            # Vectorized processing: convert to list and handle NaN
            values_list = []
            for row in values_downsampled:
                # Use list comprehension which is faster than nested loops with conditionals
                values_list.append([None if np.isnan(v) or v <= 1e-6 else float(v) for v in row])
            
            # Get time as string
            time_str = str(times[t])
            
            # Store with most recent at index 0 (append as we go backwards)
            data['timeSteps'].append({
                'timeIndex': num_times - 1 - i,  # 0 for most recent, num_times-1 for oldest
                'time': time_str,
                'values': values_list
            })
            
            if (num_times - i) % 10 == 0:
                print(f"  Processed {num_times - i}/{num_times} time steps...")
        
        print(f"Saving data to vegetation_data.json...")
        with open('vegetation_data.json', 'w') as f:
            json.dump(data, f)
        
        print("Done!")
        return data
    finally:
        # Clean up resources
        if ds is not None:
            ds.close()
        if gcs is not None:
            try:
                gcs.close()
            except:
                pass

def extract_ocean_temperature_data():
    """Extract ocean temperature data (bigthetao) from CMIP6"""
    gcs = None
    ds = None
    try:
        print("Loading CMIP6 catalog...")
        df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        
        print("Querying ocean temperature data...")
        subset = df.query("activity_id=='CMIP' & variable_id == 'bigthetao' & table_id == 'Omon' & experiment_id=='historical'")
        
        if len(subset) == 0:
            print("No ocean temperature data found!")
            return None
        
        zstore = subset.zstore.values[-1]
        print(f"Using zstore: {zstore}")
        
        gcs = gcsfs.GCSFileSystem(token='anon')
        mapper = gcs.get_mapper(zstore)
        
        print("Opening dataset...")
        ds = xr.open_zarr(mapper, consolidated=True)
        
        var = 'bigthetao'
        
        # Ocean data may use different coordinate names (i, j, lev instead of lon, lat, plev)
        # Check for coordinate variables
        if 'lon' in ds.coords or 'lon' in ds.data_vars:
            lons = ds['lon'].values
        elif 'i' in ds.coords:
            # May need to use i/j coordinates - try to find lon/lat equivalents
            print("Warning: Using i/j grid coordinates. Attempting to find lon/lat...")
            # For ocean models, coordinates might be 2D
            if 'lon' in ds.data_vars:
                lons = ds['lon'].values
            else:
                # Create synthetic coordinates based on grid
                i_size = ds[var].sizes.get('i', ds[var].sizes.get('x', 360))
                lons = np.linspace(-180, 180, i_size)
        else:
            raise ValueError("Cannot determine longitude coordinates for ocean data")
        
        # Convert longitude from 0-360 to -180-180 if needed (ocean models often use 0-360)
        # Also handle if data is centered differently than the map projection expects
        needs_data_roll = False
        roll_amount = 0
        if lons is not None and len(lons) > 0:
            lons_array = np.array(lons) if not isinstance(lons, np.ndarray) else lons
            
            # Check if longitude is in 0-360 range
            if np.max(lons_array) > 180:
                print("  Converting longitude from 0-360 to -180-180 range...")
                # Find where to split (at 180°)
                wrap_idx = np.argmin(np.abs(lons_array - 180))
                # Roll so that -180 is near the start
                lons_array = np.roll(lons_array, -wrap_idx)
                lons_array = np.where(lons_array > 180, lons_array - 360, lons_array)
                needs_data_roll = True
                roll_amount = wrap_idx
                print(f"  Rolling data by {wrap_idx} points to align with -180-180 coordinates")
            else:
                # Longitude is already in -180-180 range, but might not start at -180
                # Find the index closest to -180 and roll to start there
                min_lon_idx = np.argmin(lons_array)
                if min_lon_idx > 0 and lons_array[min_lon_idx] < -170:
                    # Roll to start near -180
                    roll_amount = min_lon_idx
                    lons_array = np.roll(lons_array, -roll_amount)
                    needs_data_roll = True
                    print(f"  Rolling data by {roll_amount} points to start at -180°")
            
            lons = lons_array
        
        if 'lat' in ds.coords or 'lat' in ds.data_vars:
            lats = ds['lat'].values
        elif 'j' in ds.coords:
            if 'lat' in ds.data_vars:
                lats = ds['lat'].values
            else:
                j_size = ds[var].sizes.get('j', ds[var].sizes.get('y', 180))
                lats = np.linspace(-90, 90, j_size)
        else:
            raise ValueError("Cannot determine latitude coordinates for ocean data")
        
        times = ds['time'].values
        # Downsample spatial resolution to reduce file size
        # Using spatial_step = 2 for better resolution
        # With spatial_step = 2, we can fit 68 time steps (~5.7 years) under 100MB
        spatial_step = 2
        max_times = 68  # ~5.7 years of monthly data (fits under 100MB with spatial_step=2)
        num_times = min(max_times, len(times))
        start_index = len(times) - num_times  # Start from the most recent time steps
        lons_full = lons
        lats_full = lats
        lons = lons_full[::spatial_step] if hasattr(lons_full, '__getitem__') else lons_full
        lats = lats_full[::spatial_step] if hasattr(lats_full, '__getitem__') else lats_full
        
        years = num_times / 12
        print(f"Extracting {num_times} time steps (most recent {years:.1f} years)...")
        if hasattr(lons_full, '__len__') and hasattr(lats_full, '__len__'):
            print(f"  Spatial resolution: {len(lons)}x{len(lats)} (downsampled from {len(lons_full)}x{len(lats_full)})")
        
        data = {
            'variable': var,
            'longitude': lons.tolist() if hasattr(lons, 'tolist') else lons.flatten().tolist() if hasattr(lons, 'flatten') else list(lons),
            'latitude': lats.tolist() if hasattr(lats, 'tolist') else lats.flatten().tolist() if hasattr(lats, 'flatten') else list(lats),
            'timeSteps': []
        }
        
        # Get surface level (lev=0 or depth=0)
        # Load all time steps at once for better performance
        print("  Loading data into memory...")
        if 'lev' in ds[var].dims:
            da_subset = ds[var].isel(time=slice(start_index, start_index + num_times), lev=0)
        elif 'olev' in ds[var].dims:
            da_subset = ds[var].isel(time=slice(start_index, start_index + num_times), olev=0)
        elif 'depth' in ds[var].dims:
            da_subset = ds[var].isel(time=slice(start_index, start_index + num_times), depth=0)
        else:
            da_subset = ds[var].isel(time=slice(start_index, start_index + num_times))
        
        da_subset = da_subset.load()
        
        # Extract in reverse order: most recent first (index 0 = most recent)
        # Iterate backwards through the time steps
        for i in range(num_times - 1, -1, -1):  # From num_times-1 down to 0
            t = start_index + i  # Actual time index in dataset
            time_idx = i  # Index in the subset array
            
            # Get values for this time step (already loaded)
            values = da_subset.isel(time=time_idx).values
            
            # Roll the data to match longitude coordinate conversion (if needed)
            if needs_data_roll and roll_amount > 0:
                # Determine which axis is longitude
                # For ocean data, check the shape and dimensions
                # Typically: (lat, lon) or (j, i) where i is longitude
                if len(values.shape) == 2:
                    # Assume (lat, lon) format - roll along the last axis (longitude)
                    values = np.roll(values, -roll_amount, axis=-1)
                else:
                    # Fallback: roll along axis 1
                    values = np.roll(values, -roll_amount, axis=1)
            
            # Downsample spatial resolution (take every Nth point in both dimensions)
            values_downsampled = values[::spatial_step, ::spatial_step]
            
            # Vectorized processing: convert to list and handle NaN
            values_list = []
            for row in values_downsampled:
                # Use list comprehension which is faster than nested loops with conditionals
                values_list.append([None if np.isnan(v) else float(v) for v in row])
            
            # Get time as string
            time_str = str(times[t])
            
            # Store with most recent at index 0 (append as we go backwards)
            data['timeSteps'].append({
                'timeIndex': num_times - 1 - i,  # 0 for most recent, num_times-1 for oldest
                'time': time_str,
                'values': values_list
            })
            
            if (num_times - i) % 10 == 0:
                print(f"  Processed {num_times - i}/{num_times} time steps...")
        
        print(f"Saving data to ocean_temperature_data.json...")
        with open('ocean_temperature_data.json', 'w') as f:
            json.dump(data, f)
        
        print("Done!")
        return data
    except Exception as e:
        print(f"Error extracting ocean temperature data: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if ds is not None:
            ds.close()
        if gcs is not None:
            try:
                gcs.close()
            except:
                pass

def extract_snow_melt_data():
    """Extract snow melt data (snm) from CMIP6"""
    gcs = None
    ds = None
    try:
        print("Loading CMIP6 catalog...")
        df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        
        print("Querying snow melt data...")
        subset = df.query("activity_id=='CMIP' & variable_id == 'snm' & table_id == 'LImon' & experiment_id=='historical'")
        
        if len(subset) == 0:
            print("No snow melt data found!")
            return None
        
        zstore = subset.zstore.values[-1]
        print(f"Using zstore: {zstore}")
        
        gcs = gcsfs.GCSFileSystem(token='anon')
        mapper = gcs.get_mapper(zstore)
        
        print("Opening dataset...")
        ds = xr.open_zarr(mapper, consolidated=True)
        
        var = 'snm'
        lons_full = ds['lon'].values
        lats_full = ds['lat'].values
        times = ds['time'].values
        
        # Limit to 360 time steps (30 years of monthly data) to keep file size manageable
        max_times = 360  # 30 years of monthly data
        num_times = min(max_times, len(times))
        start_index = len(times) - num_times  # Start from the most recent time steps
        
        # Downsample spatial resolution to reduce file size (take every 2nd point)
        spatial_step = 2
        lons = lons_full[::spatial_step]
        lats = lats_full[::spatial_step]
        
        years = num_times / 12
        print(f"Extracting {num_times} time steps (most recent {years:.1f} years)...")
        print(f"  Spatial resolution: {len(lons)}x{len(lats)} (downsampled from {len(lons_full)}x{len(lats_full)})")
        
        data = {
            'variable': var,
            'longitude': lons.tolist(),
            'latitude': lats.tolist(),
            'timeSteps': []
        }
        
        # Extract in reverse order: most recent first (index 0 = most recent)
        # Load all time steps at once for better performance
        print("  Loading data into memory...")
        da_subset = ds[var].isel(time=slice(start_index, start_index + num_times))
        da_subset = da_subset.load()
        
        # Iterate backwards through the time steps
        for i in range(num_times - 1, -1, -1):  # From num_times-1 down to 0
            t = start_index + i  # Actual time index in dataset
            time_idx = i  # Index in the subset array
            
            # Get values for this time step (already loaded)
            values = da_subset.isel(time=time_idx).values
            
            # Downsample spatial resolution (take every Nth point in both dimensions)
            values_downsampled = values[::spatial_step, ::spatial_step]
            
            # Vectorized processing: convert to list and handle NaN
            values_list = []
            for row in values_downsampled:
                # Use list comprehension which is faster than nested loops with conditionals
                values_list.append([None if np.isnan(v) or v == 0 else float(v) for v in row])
            
            # Get time as string
            time_str = str(times[t])
            
            # Store with most recent at index 0 (append as we go backwards)
            data['timeSteps'].append({
                'timeIndex': num_times - 1 - i,  # 0 for most recent, num_times-1 for oldest
                'time': time_str,
                'values': values_list
            })
            
            if (num_times - i) % 10 == 0:
                print(f"  Processed {num_times - i}/{num_times} time steps...")
        
        print(f"Saving data to snow_melt_data.json...")
        with open('snow_melt_data.json', 'w') as f:
            json.dump(data, f)
        
        print("Done!")
        return data
    except Exception as e:
        print(f"Error extracting snow melt data: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if ds is not None:
            ds.close()
        if gcs is not None:
            try:
                gcs.close()
            except:
                pass

if __name__ == "__main__":
    import sys
    import warnings
    
    # Suppress asyncio warnings during cleanup
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    try:
        print("=" * 50)
        print("Extracting CMIP6 Data for D3.js Visualization")
        print("=" * 50)
        
        # Extract all data types
        print("\n1. Extracting Global Atmospheric Temperature Data (ta, Emon)...")
        extract_temperature_data()
        
        print("\n2. Extracting Global Vegetation Carbon Data (cVeg, Emon)...")
        extract_vegetation_data()
        
        print("\n3. Extracting Ocean Temperature Data (bigthetao, Omon)...")
        extract_ocean_temperature_data()
        
        print("\n4. Extracting Global Surface Snow Melt Data (snm, LImon)...")
        extract_snow_melt_data()
        
        print("\n" + "=" * 50)
        print("Data extraction complete!")
        print("=" * 50)
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during extraction: {e}")
        sys.exit(1)

