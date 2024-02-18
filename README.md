# Clifftop Explorer

Clifftop Explorer is a Python script that helps users discover and visualize potential cliff locations within a specified radius of a given latitude and longitude coordinates. It utilizes elevation data and satellite imagery to identify and display the top cliff locations.

## Features

- Fetches elevation data from the Open Topography API within a specified radius.
- Processes elevation data to identify potential cliff locations based on gradient thresholds.
- Clusters and filters cliff locations to remove duplicates or closely located cliffs.
- Utilizes satellite imagery from Mapbox to visualize the top cliff locations.
- Provides details such as latitude, longitude, angle, height, and distance for each top cliff found.

## Requirements

- Python 3.x
- Required Python packages: `requests`, `numpy`, `rasterio`, `matplotlib`, `scikit-learn`, `geopy`, `PIL`

## Installation

1. Clone the repository:

2. `pip install requests numpy rasterio matplotlib scikit-learn geopy Pillow`

3. Obtain a Mapbox access token from Mapbox to use satellite imagery.

4. Obtain a OpenTopography acces token