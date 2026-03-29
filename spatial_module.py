import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
import warnings
import urllib.request
import os


OUTLET_MASL = 1383.0  # Lake Chivero elevation — JICA 2018


def download_srtm(south: float, north: float, west: float, east: float,
                  out_path: str, api_key: str = "") -> None:
    """Download SRTM 30m DEM tile from OpenTopography for the given bounding box."""
    if os.path.exists(out_path):
        print(f"DEM already exists at {out_path} — skipping download.")
        return
    url = (
        f"https://portal.opentopography.org/API/globaldem"
        f"?demtype=SRTMGL1"
        f"&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )
    print(f"Downloading SRTM DEM...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved to {out_path}")


def inspect_dem(dem_path: str) -> dict:
    """Return key metadata from a DEM GeoTIFF. Always run this first."""
    with rasterio.open(dem_path) as src:
        data = src.read(1, masked=True)
        meta = {
            "crs":          src.crs.to_string(),
            "epsg":         src.crs.to_epsg(),
            "resolution_m": src.res,
            "bounds":       src.bounds,
            "nodata":       src.nodata,
            "dtype":        src.dtypes[0],
            "elev_min":     float(data.min()),
            "elev_max":     float(data.max()),
            "elev_mean":    float(data.mean()),
        }
    return meta


def clip_dem_to_aoi(dem_path: str, aoi_geojson: str,
                    out_path: str, buffer_m: float = 2000) -> str:
    """
    Clip DEM to the Area of Interest with a buffer.
    buffer_m: metres to expand around the footprint (default 2 km).
    Returns path to the clipped file.
    """
    aoi = gpd.read_file(aoi_geojson)
    aoi_proj = aoi.to_crs(epsg=32736)
    aoi_buffered = gpd.GeoDataFrame(
        geometry=aoi_proj.buffer(buffer_m), crs="EPSG:32736"
    ).to_crs(epsg=4326)

    geoms = [mapping(geom) for geom in aoi_buffered.geometry]

    with rasterio.open(dem_path) as src:
        clipped, transform = rasterio.mask.mask(
            src, geoms, crop=True, nodata=-9999
        )
        meta = src.meta.copy()
        meta.update({
            "height":    clipped.shape[1],
            "width":     clipped.shape[2],
            "transform": transform,
            "nodata":    -9999,
        })

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)

    print(f"Clipped DEM saved to {out_path}")
    return out_path


def sample_elevation_along_network(dem_path: str,
                                    network_geojson: str,
                                    n_points: int = 300,
                                    outlet_masl: float = OUTLET_MASL
                                    ) -> gpd.GeoDataFrame:
    """
    Sample DEM elevation at n evenly-spaced points along the sewer network.
    Computes hydraulic head H relative to Lake Chivero (outlet).

    Returns GeoDataFrame with columns:
        distance_m, elevation_masl, head_m, geometry
    """
    network = gpd.read_file(network_geojson)
    network_proj = network.to_crs(epsg=32736)

    total_length = network_proj.geometry.length.sum()
    interval = total_length / n_points

    points, distances = [], []
    cumulative = 0.0

    for geom in network_proj.geometry:
        d = 0.0
        while d <= geom.length:
            pt = geom.interpolate(d)
            points.append(pt)
            distances.append(cumulative + d)
            d += interval
        cumulative += geom.length

    pts_gdf = gpd.GeoDataFrame(
        {"distance_m": distances, "geometry": points},
        crs="EPSG:32736"
    )
    pts_wgs84 = pts_gdf.to_crs(epsg=4326)
    coords = [(pt.x, pt.y) for pt in pts_wgs84.geometry]

    with rasterio.open(dem_path) as dem:
        nodata = dem.nodata or -9999
        raw_elevations = [val[0] for val in dem.sample(coords)]

    elevations = np.array(raw_elevations, dtype=float)

    # Fill any nodata voids by linear interpolation
    void_mask = (elevations == nodata) | (elevations < -500)
    if void_mask.any():
        warnings.warn(f"{void_mask.sum()} nodata voids found — interpolating.")
        x = np.arange(len(elevations))
        elevations[void_mask] = np.interp(
            x[void_mask], x[~void_mask], elevations[~void_mask]
        )

    pts_gdf["elevation_masl"] = elevations
    pts_gdf["head_m"] = pts_gdf["elevation_masl"] - outlet_masl

    # Keep only gravity-fed segments (positive head)
    result = pts_gdf[pts_gdf["head_m"] > 0].copy()
    print(f"Sampled {len(result)} points with positive hydraulic head.")
    return result


def elevation_from_footprint(dem_path: str,
                              footprint_geojson: str,
                              n_points: int = 100,
                              outlet_masl: float = OUTLET_MASL
                              ) -> gpd.GeoDataFrame:
    """
    Fallback: sample DEM along the Firle footprint perimeter when
    the full pipe network GeoJSON is not yet available.
    """
    footprint = gpd.read_file(footprint_geojson).to_crs(epsg=32736)
    perimeter = footprint.geometry.boundary.iloc[0]

    distances = np.linspace(0, perimeter.length, n_points)
    pts = [perimeter.interpolate(d) for d in distances]

    pts_gdf = gpd.GeoDataFrame(
        {"distance_m": distances, "geometry": pts},
        crs="EPSG:32736"
    ).to_crs(epsg=4326)

    coords = [(pt.x, pt.y) for pt in pts_gdf.geometry]

    with rasterio.open(dem_path) as dem:
        elevations = [v[0] for v in dem.sample(coords)]

    pts_gdf = pts_gdf.to_crs(epsg=32736)
    pts_gdf["elevation_masl"] = elevations
    pts_gdf["head_m"] = pts_gdf["elevation_masl"] - outlet_masl

    return pts_gdf[pts_gdf["head_m"] > 0].copy()


def characterise_head(profile: gpd.GeoDataFrame) -> dict:
    """Compute summary statistics on the hydraulic head profile."""
    H = profile["head_m"]
    return {
        "H_mean":                  round(H.mean(), 2),
        "H_max":                   round(H.max(), 2),
        "H_min":                   round(H.min(), 2),
        "H_std":                   round(H.std(), 2),
        "H_range":                 round(H.max() - H.min(), 2),
        "n_gravity_segments":      len(H),
        "total_network_length_km": round(profile["distance_m"].max() / 1000, 2),
    }


def validate_dem_harare(profile: gpd.GeoDataFrame) -> None:
    """
    Sanity-check extracted elevations against known Harare reference values.
    Harare plateau: ~1470–1500 m MASL
    Lake Chivero:   ~1383 m MASL
    Expected head:  ~87–120 m
    """
    H_mean   = profile["head_m"].mean()
    elev_mean = profile["elevation_masl"].mean()

    assert 1400 < elev_mean < 1550, \
        f"Mean elevation {elev_mean:.0f} m outside expected range 1400–1550 m"
    assert 60 < H_mean < 150, \
        f"Mean head {H_mean:.1f} m outside expected range 60–150 m"

    print(f"Validation passed.")
    print(f"  Mean elevation : {elev_mean:.1f} m MASL")
    print(f"  Mean head H    : {H_mean:.1f} m")
    print(f"  Max head H     : {profile['head_m'].max():.1f} m")