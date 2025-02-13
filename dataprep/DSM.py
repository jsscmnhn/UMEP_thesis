import requests
import rasterio
from rasterio.plot import show
from io import BytesIO


def fetch_ahn_wcs(bbox, output_file="dtm.tif", coverage="dtm_05m", resolution=0.5):
    width = int((bbox[2] - bbox[0]) / resolution)
    height = int((bbox[3] - bbox[1]) / resolution)

    print(width, height)
    #
    # width = 2700
    # height = 1000

    # WCS Service URL
    WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"

    # Construct query parameters
    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "FORMAT": "GEOTIFF",
        "COVERAGE": coverage,
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "CRS": "EPSG:28992",
        "RESPONSE_CRS": "EPSG:28992",
        "WIDTH": str(width),
        "HEIGHT": str(height)
    }

    # Send GET request
    response = requests.get(WCS_URL, params=params, headers={"User-Agent": "Mozilla/5.0"})

    if response.status_code == 200:
        # Save response as GeoTIFF
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"AHN data saved as {output_file}")

        # Open with rasterio
        dataset = rasterio.open(output_file)
        return dataset
    else:
        print(f"Failed to fetch AHN data: HTTP {response.status_code}")
        return None


bbox = (94000, 469000, 95000, 470000)  # (minx, miny, maxx, maxy)
dtm = fetch_ahn_wcs(bbox)
dsm = fetch_ahn_wcs(bbox, "dsm.tif", "dsm_05m")