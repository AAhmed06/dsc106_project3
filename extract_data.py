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
        subset = df.query("activity_id=='CMIP' & variable_id == 'ta' & table_id == 'Amon' & experiment_id=='historical'")
        
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
        
        # Extract data for multiple time steps and one pressure level
        var = 'ta'
        plev_index = 0  # Surface level
        
        # Get coordinate arrays
        lons = ds['lon'].values
        lats = ds['lat'].values
        times = ds['time'].values
        
        # Limit to first 12 time steps for manageable file size
        num_times = min(12, len(times))
        
        print(f"Extracting {num_times} time steps...")
        
        data = {
            'variable': var,
            'longitude': lons.tolist(),
            'latitude': lats.tolist(),
            'timeSteps': []
        }
        
        for t in range(num_times):
            da2d = ds[var].isel(time=t, plev=plev_index)
            values = da2d.values
            
            # Convert to list, handling NaN
            values_list = []
            for row in values:
                values_list.append([float(v) if not np.isnan(v) else None for v in row])
            
            # Get time as string
            time_str = str(times[t])
            
            data['timeSteps'].append({
                'timeIndex': t,
                'time': time_str,
                'values': values_list
            })
            
            if (t + 1) % 3 == 0:
                print(f"  Processed {t + 1}/{num_times} time steps...")
        
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
        subset = df.query("activity_id=='CMIP' & variable_id == 'cVeg' & table_id == 'Lmon' & experiment_id=='historical'")
        
        if len(subset) == 0:
            print("No vegetation data found!")
            return None
        
        zstore = subset.zstore.values[-1]
        print(f"Using zstore: {zstore}")
        
        gcs = gcsfs.GCSFileSystem(token='anon')
        mapper = gcs.get_mapper(zstore)
        
        print("Opening dataset...")
        ds = xr.open_zarr(mapper, consolidated=True)
        
        var = 'cVeg'
        lons = ds['lon'].values
        lats = ds['lat'].values
        times = ds['time'].values
        
        num_times = min(12, len(times))
        
        print(f"Extracting {num_times} time steps...")
        
        data = {
            'variable': var,
            'longitude': lons.tolist(),
            'latitude': lats.tolist(),
            'timeSteps': []
        }
        
        for t in range(num_times):
            da = ds[var].isel(time=t)
            values = da.values
            
            values_list = []
            for row in values:
                values_list.append([float(v) if not np.isnan(v) and v > 1e-6 else None for v in row])
            
            time_str = str(times[t])
            
            data['timeSteps'].append({
                'timeIndex': t,
                'time': time_str,
                'values': values_list
            })
            
            if (t + 1) % 3 == 0:
                print(f"  Processed {t + 1}/{num_times} time steps...")
        
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

if __name__ == "__main__":
    import sys
    import warnings
    
    # Suppress asyncio warnings during cleanup
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    try:
        print("=" * 50)
        print("Extracting CMIP6 Data for D3.js Visualization")
        print("=" * 50)
        
        # Extract temperature data
        print("\n1. Extracting Temperature Data (ta)...")
        extract_temperature_data()
        
        # Extract vegetation data
        print("\n2. Extracting Vegetation Data (cVeg)...")
        extract_vegetation_data()
        
        print("\n" + "=" * 50)
        print("Data extraction complete!")
        print("=" * 50)
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during extraction: {e}")
        sys.exit(1)

