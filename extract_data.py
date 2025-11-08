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
        lons = ds['lon'].values
        lats = ds['lat'].values
        times = ds['time'].values
        
        # Limit to 60 time steps (5 years of monthly data) for manageable file size
        # Take the most recent 60 time steps (last 60)
        num_times = min(60, len(times))
        start_index = len(times) - num_times  # Start from the most recent 60
        
        print(f"Extracting {num_times} time steps (most recent 5 years)...")
        print(f"  Time range: {times[start_index]} to {times[-1]}")
        
        data = {
            'variable': var,
            'longitude': lons.tolist(),
            'latitude': lats.tolist(),
            'timeSteps': []
        }
        
        # Extract in reverse order: most recent first (index 0 = most recent)
        # Iterate backwards through the time steps
        for i in range(num_times - 1, -1, -1):  # From 59 down to 0
            t = start_index + i  # Actual time index in dataset (oldest to newest)
            
            if 'plev' in ds[var].dims:
                da2d = ds[var].isel(time=t, plev=plev_index)
            else:
                da2d = ds[var].isel(time=t)
            values = da2d.values
            
            # Convert to list, handling NaN and converting from Kelvin to Celsius
            values_list = []
            for row in values:
                values_list.append([float(v) - 273.15 if not np.isnan(v) else None for v in row])
            
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
        subset = df.query("activity_id=='CMIP' & variable_id == 'cVeg' & table_id == 'Emon' & experiment_id=='historical'")
        
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
        
        num_times = min(60, len(times))
        
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

def extract_ice_thickness_data():
    """Extract ice sheet thickness data (lithk) from CMIP6"""
    gcs = None
    ds = None
    try:
        print("Loading CMIP6 catalog...")
        df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
        
        print("Querying ice sheet thickness data...")
        subset = df.query("activity_id=='CMIP' & variable_id == 'lithk' & table_id == 'IfxGre' & experiment_id=='historical'")
        
        if len(subset) == 0:
            print("No ice sheet thickness data found!")
            return None
        
        zstore = subset.zstore.values[-1]
        print(f"Using zstore: {zstore}")
        
        gcs = gcsfs.GCSFileSystem(token='anon')
        mapper = gcs.get_mapper(zstore)
        
        print("Opening dataset...")
        ds = xr.open_zarr(mapper, consolidated=True)
        
        var = 'lithk'
        
        # Check for coordinate variables - lithk uses Greenland grid (ygre, xgre)
        # Try to get lon/lat coordinates in various formats
        lons = None
        lats = None
        
        # Check standard coordinate names
        if 'lon' in ds.coords and 'lat' in ds.coords:
            lons = ds['lon'].values
            lats = ds['lat'].values
        elif 'lon' in ds.data_vars and 'lat' in ds.data_vars:
            lons = ds['lon'].values
            lats = ds['lat'].values
        else:
            # Check for 2D coordinate arrays (common in Greenland grids)
            coord_vars = [v for v in ds.data_vars if 'lon' in v.lower() or 'lat' in v.lower()]
            if coord_vars:
                print(f"Found coordinate variables: {coord_vars}")
                for cv in coord_vars:
                    if 'lon' in cv.lower() and lons is None:
                        lon_data = ds[cv].values
                        # If 2D, take first row/column; if 1D, use directly
                        if len(lon_data.shape) == 2:
                            lons = lon_data[0, :] if lon_data.shape[0] > 0 else lon_data[:, 0]
                        else:
                            lons = lon_data
                    elif 'lat' in cv.lower() and lats is None:
                        lat_data = ds[cv].values
                        if len(lat_data.shape) == 2:
                            lats = lat_data[:, 0] if lat_data.shape[1] > 0 else lat_data[0, :]
                        else:
                            lats = lat_data
        
        # If still no coordinates, create synthetic based on grid dimensions
        if lons is None or lats is None:
            print("Warning: Standard lon/lat coordinates not found. Creating synthetic coordinates for Greenland...")
            dims = ds[var].dims
            if len(dims) >= 2:
                y_dim, x_dim = dims[0], dims[1]
                y_size, x_size = ds[var].sizes[y_dim], ds[var].sizes[x_dim]
                # Create approximate lat/lon grid for Greenland
                if lats is None:
                    lats = np.linspace(60, 85, y_size)  # Greenland latitude range
                if lons is None:
                    lons = np.linspace(-75, -10, x_size)  # Greenland longitude range
                print(f"Using synthetic coordinates: lat range {lats[0]:.1f} to {lats[-1]:.1f}, lon range {lons[0]:.1f} to {lons[-1]:.1f}")
            else:
                raise ValueError("Cannot determine coordinate system for ice sheet data")
        
        # Ice sheet thickness data is typically time-invariant (fixed field)
        # But we'll handle it as if it has time dimension for consistency
        if 'time' in ds[var].dims:
            times = ds['time'].values
            num_times = min(60, len(times))
        else:
            # If no time dimension, create a single time step
            times = np.array(['1850-01-01'])
            num_times = 1
        
        print(f"Extracting {num_times} time step(s)...")
        
        data = {
            'variable': var,
            'longitude': lons.tolist() if hasattr(lons, 'tolist') else lons.flatten().tolist() if hasattr(lons, 'flatten') else list(lons),
            'latitude': lats.tolist() if hasattr(lats, 'tolist') else lats.flatten().tolist() if hasattr(lats, 'flatten') else list(lats),
            'timeSteps': []
        }
        
        for t in range(num_times):
            if 'time' in ds[var].dims:
                da = ds[var].isel(time=t)
            else:
                da = ds[var]
            values = da.values
            
            # Handle 2D coordinate arrays - flatten if needed
            if len(values.shape) == 2:
                values_list = []
                for row in values:
                    values_list.append([float(v) if not np.isnan(v) and v > 0 else None for v in row])
            else:
                # If already flattened or different shape
                values_list = [[float(v) if not np.isnan(v) and v > 0 else None for v in values]]
            
            time_str = str(times[t]) if num_times > 1 else str(times[0])
            
            data['timeSteps'].append({
                'timeIndex': t,
                'time': time_str,
                'values': values_list
            })
            
            if (t + 1) % 3 == 0 or t == 0:
                print(f"  Processed {t + 1}/{num_times} time step(s)...")
        
        print(f"Saving data to ice_thickness_data.json...")
        with open('ice_thickness_data.json', 'w') as f:
            json.dump(data, f)
        
        print("Done!")
        return data
    except Exception as e:
        print(f"Error extracting ice sheet thickness data: {e}")
        import traceback
        traceback.print_exc()
        return None
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
        num_times = min(60, len(times))
        
        print(f"Extracting {num_times} time steps...")
        
        data = {
            'variable': var,
            'longitude': lons.tolist() if hasattr(lons, 'tolist') else lons.flatten().tolist() if hasattr(lons, 'flatten') else list(lons),
            'latitude': lats.tolist() if hasattr(lats, 'tolist') else lats.flatten().tolist() if hasattr(lats, 'flatten') else list(lats),
            'timeSteps': []
        }
        
        # Get surface level (lev=0 or depth=0)
        for t in range(num_times):
            if 'lev' in ds[var].dims:
                da = ds[var].isel(time=t, lev=0)
            elif 'olev' in ds[var].dims:
                da = ds[var].isel(time=t, olev=0)
            elif 'depth' in ds[var].dims:
                da = ds[var].isel(time=t, depth=0)
            else:
                da = ds[var].isel(time=t)
            values = da.values
            
            values_list = []
            for row in values:
                values_list.append([float(v) if not np.isnan(v) else None for v in row])
            
            time_str = str(times[t])
            
            data['timeSteps'].append({
                'timeIndex': t,
                'time': time_str,
                'values': values_list
            })
            
            if (t + 1) % 3 == 0:
                print(f"  Processed {t + 1}/{num_times} time steps...")
        
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
        lons = ds['lon'].values
        lats = ds['lat'].values
        times = ds['time'].values
        
        num_times = min(60, len(times))
        
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
                values_list.append([float(v) if not np.isnan(v) and v != 0 else None for v in row])
            
            time_str = str(times[t])
            
            data['timeSteps'].append({
                'timeIndex': t,
                'time': time_str,
                'values': values_list
            })
            
            if (t + 1) % 3 == 0:
                print(f"  Processed {t + 1}/{num_times} time steps...")
        
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
        
        # print("\n2. Extracting Global Vegetation Carbon Data (cVeg, Emon)...")
        # extract_vegetation_data()
        
        # print("\n3. Extracting Greenland Ice Sheet Thickness Data (lithk, IfxGre)...")
        # extract_ice_thickness_data()
        
        # print("\n4. Extracting Ocean Temperature Data (bigthetao, Omon)...")
        # extract_ocean_temperature_data()
        
        # print("\n5. Extracting Global Surface Snow Melt Data (snm, LImon)...")
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

