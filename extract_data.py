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
        ds['time'] = pd.to_datetime(ds['time'].values)
        ds = ds.sel(time=slice('1985-01-01', '2014-12-31'))
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
        # --- Aggregate by year (reduce 360 months -> 30 years) ---
        print("  Aggregating to yearly means (1985–2014)...")
        da_subset = ds[var].groupby('time.year').mean(dim='time')
        times = da_subset['year'].values  # now only 1985–2014
        num_times = len(times)

        # Downsample spatially (keep good detail)
        spatial_step = 2
        lons = lons_full[::spatial_step]
        lats = lats_full[::spatial_step]

        print(f"Extracting {num_times} yearly steps ({times[0]}–{times[-1]})...")
        print(f"  Spatial resolution: {len(lons)}x{len(lats)} (downsampled from {len(lons_full)}x{len(lats_full)})")

        # Preload dataset
        da_subset = da_subset.load()

        # Iterate through each year
        for i in range(num_times):
            values = da_subset.isel(year=i).values
            values_downsampled = values[::spatial_step, ::spatial_step]
            values_list = [[None if np.isnan(v) or v <= 1e-6 else float(v) for v in row] for row in values_downsampled]
            time_str = str(times[i])
            data['timeSteps'].append({
                'timeIndex': i,
                'time': time_str,
                'values': values_list
            })
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{num_times} years...")

        print(f"Saving data to vegetation_data.json...")
        data['time_range'] = {
            "start": str(ds['time'].values[0])[:10],
            "end": str(ds['time'].values[-1])[:10],
            "num_steps": len(ds['time'].values)
        }   
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
        
        # --- Normalize longitudes and prepare sorting for map alignment ---
        if np.max(lons) > 180:
            print("  Converting longitude from 0–360 → –180–180 and sorting...")
            lons = np.where(lons > 180, lons - 360, lons)
            sort_idx = np.argsort(lons)
            lons = lons[sort_idx]
            needs_data_sort = True
        else:
            needs_data_sort = False
            sort_idx = np.arange(len(lons))

        print("Longitude count:", len(lons))
        print("Longitude range:", lons[0], "→", lons[-1])
        print("Longitude sorted ascending:", np.all(np.diff(lons) >= 0))

        
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
        ds['time'] = pd.to_datetime(ds['time'].values)
        # Filter to data from 1985 onwards
        ds = ds.sel(time=slice('1985-01-01', None))
        
        # Get surface level first, then aggregate by year
        print("  Loading data into memory...")
        if 'lev' in ds[var].dims:
            da = ds[var].isel(lev=0)
        elif 'olev' in ds[var].dims:
            da = ds[var].isel(olev=0)
        elif 'depth' in ds[var].dims:
            da = ds[var].isel(depth=0)
        else:
            da = ds[var]
        
        # Apply longitude sorting if needed before aggregation
        if needs_data_sort:
            # Reorder the data along the longitude dimension
            dims = list(da.dims)
            if 'lon' in dims:
                da = da.isel(lon=sort_idx)
            elif 'i' in dims:
                da = da.isel(i=sort_idx)
        
        # Aggregate monthly data to yearly means
        print("  Aggregating to yearly means (from most recent year to 1985)...")
        da_yearly = da.groupby('time.year').mean(dim='time')
        # Get the year dimension name (could be 'year' or the grouped coordinate name)
        year_dim = [d for d in da_yearly.dims if d != 'lat' and d != 'lon' and d != 'i' and d != 'j'][0]
        years = da_yearly[year_dim].values  # Years in ascending order (1985, 1986, ..., most recent)
        
        # Downsample spatial resolution to reduce file size
        spatial_step = 2
        lons_full = lons
        lats_full = lats
        lons = lons_full[::spatial_step] if hasattr(lons_full, '__getitem__') else lons_full
        lats = lats_full[::spatial_step] if hasattr(lats_full, '__getitem__') else lats_full
        
        num_years = len(years)
        print(f"Extracting {num_years} yearly steps ({years[0]}–{years[-1]})...")
        if hasattr(lons_full, '__len__') and hasattr(lats_full, '__len__'):
            print(f"  Spatial resolution: {len(lons)}x{len(lats)} (downsampled from {len(lons_full)}x{len(lats_full)})")
        
        data = {
            'variable': var,
            'longitude': lons.tolist() if hasattr(lons, 'tolist') else lons.flatten().tolist() if hasattr(lons, 'flatten') else list(lons),
            'latitude': lats.tolist() if hasattr(lats, 'tolist') else lats.flatten().tolist() if hasattr(lats, 'flatten') else list(lats),
            'timeSteps': []
        }
        
        # Preload dataset
        da_yearly = da_yearly.load()
        
        # Extract in reverse order: most recent year first (index 0 = most recent)
        # Iterate backwards through the years
        for i in range(num_years - 1, -1, -1):  # From most recent year down to 1985
            year = years[i]
            year_idx = i  # Index in the yearly array

            values = da_yearly.isel({year_dim: year_idx}).values
            
            # Debug visualization: check longitude alignment (only for first year)
            if i == num_years - 1:  # only plot once for the most recent year
                import matplotlib.pyplot as plt
                # Downsample before plotting
                values_preview = values[::spatial_step, ::spatial_step]

                lon_grid, lat_grid = np.meshgrid(lons, lats)
                plt.figure(figsize=(10, 4))
                plt.title("Sanity Check: Ocean Temperature Grid Alignment")

                plt.pcolormesh(lon_grid, lat_grid, values_preview, shading='auto', cmap='coolwarm')
                plt.xlabel("Longitude (°)")
                plt.ylabel("Latitude (°)")
                plt.colorbar(label="Temperature (°C)")
                plt.show()

            # Downsample spatial resolution
            values_downsampled = values[::spatial_step, ::spatial_step]

            # Convert to list and handle NaNs
            values_list = []
            for row in values_downsampled:
                values_list.append([None if np.isnan(v) else float(v) for v in row])

            # Save the time step (most recent year at index 0)
            time_str = str(year)
            data['timeSteps'].append({
                'timeIndex': num_years - 1 - i,  # 0 for most recent, increasing to oldest
                'time': time_str,
                'values': values_list
            })

            if (num_years - i) % 5 == 0:
                print(f"  Processed {num_years - i}/{num_years} years...")
        
        print(f"Saving data to ocean_temperature_data.json...")
        data['time_range'] = {
            "start": str(ds['time'].values[0])[:10],
            "end": str(ds['time'].values[-1])[:10],
            "num_steps": len(ds['time'].values)
        }
        with open('ocean_temperature_data.json', 'w') as f:
            print("Final longitude range before saving:", min(data['longitude']), "→", max(data['longitude']))
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
        # print("\n1. Extracting Global Atmospheric Temperature Data (ta, Emon)...")
        # extract_temperature_data()
        
        # print("\n2. Extracting Global Vegetation Carbon Data (cVeg, Emon)...")
        # extract_vegetation_data()
        
        print("\n1. Extracting Ocean Temperature Data (bigthetao, Omon)...")
        extract_ocean_temperature_data()
        
        # print("\n4. Extracting Global Surface Snow Melt Data (snm, LImon)...")
        # extract_snow_melt_data()
        
        print("\n" + "=" * 50)
        print("Data extraction complete!")
        print("=" * 50)
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during extraction: {e}")
        sys.exit(1)

