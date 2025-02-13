import geopandas as gpd

# Load the shapefile
shapefile_path = "../geotiles/AHN_subunits_GeoTiles.shp"
gdf = gpd.read_file(shapefile_path)

print(gdf.head())

# Extract bounds for each geometry
gdf["bounds"] = gdf.geometry.apply(lambda geom: (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3])))

# Save to CSV
csv_path = "../geotiles/tile_lookup.csv"
gdf[["bounds", "GT_AHNSUB"]].to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")

