import requests
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from math import cos, radians

from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import numpy as np
import os
from PIL import Image
import io

def cluster_and_filter_cliffs(cliffs, eps=0.1, min_samples=1):
    """
    Cluster and filter cliffs that are within `eps` meters of each other, keeping only the highest drop in each cluster.
    
    :param cliffs: A list of cliff dictionaries with 'lat', 'lon', and 'angle' keys.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: A filtered list of cliffs after clustering and filtering.
    """
    print(f"Clustering and filtering {len(cliffs)} cliffs...")
    # Convert (lat, lon) to a list of tuples for DBSCAN
    coords = [(cliff['lat'], cliff['lon']) for cliff in cliffs]
    
    # Use DBSCAN to cluster based on geographic distance
    kms_per_radian = 6371.0088
    eps_rad = eps / kms_per_radian
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    
    filtered_cliffs = []
    for cluster_num in range(num_clusters):
        if cluster_num == -1:
            # Skip noise points
            continue
        
        # Extract cliffs in this cluster
        cluster_cliffs = [cliffs[i] for i in range(len(cliffs)) if cluster_labels[i] == cluster_num]
        
        # Find the cliff with the highest drop in this cluster
        best_cliff = max(cluster_cliffs, key=lambda x: x['angle'])
        filtered_cliffs.append(best_cliff)
    print(f"Filtered {len(filtered_cliffs)} cliffs from {len(cliffs)} after clustering, from {num_clusters} clusters.")
    return filtered_cliffs

def calculate_bounding_box(latitude, longitude, radius_km):
    """
    Calculate the bounding box for a given center point and radius.
    
    :param latitude: Latitude of the center point.
    :param longitude: Longitude of the center point.
    :param radius_km: Radius in kilometers.
    :return: A tuple (south, north, west, east) representing the bounding box.
    """
    # Convert radius in kilometers to degrees
    radius_in_deg = radius_km / 111  # Rough approximation
    lat_adjust = radius_in_deg
    long_adjust = radius_in_deg / cos(radians(latitude))
    
    south = latitude - lat_adjust
    north = latitude + lat_adjust
    west = longitude - long_adjust
    east = longitude + long_adjust
    
    return south, north, west, east

def fetch_elevation_data(dem_type, latitude, longitude, radius_km, api_key, output_format="GTiff"):
    south, north, west, east, = calculate_bounding_box(latitude, longitude, radius_km)
    filename = f"elevation_data.{latitude}{longitude}.{radius_km}.{output_format.lower()}"
    if os.path.isfile(filename):
        print(f"Using existing elevation data from {filename}")
        return filename
    base_url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": output_format,
        "API_Key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Elevation data saved to {filename}")
        return filename
    else:
        print(f"Failed to fetch data: {response.status_code}, {response.text}")
        return None

def process_elevation_data(filename):
    """
    Process elevation data to find potential cliff locations based on a gradient threshold.
    
    :param filename: Path to the GeoTIFF file containing elevation data.
    :return: A list of dictionaries representing potential cliffs, each with average angle and total height.
    """
    with rasterio.open(filename) as dataset:
        # Read the first band of elevation data
        elevation_data = dataset.read(1)
        
        # Calculate gradients along both axes
        grad_x, grad_y = np.gradient(elevation_data, edge_order=2)
        
        # Calculate the magnitude of the gradient
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        cliffs = []
        for y, x in zip(*np.where(grad_magnitude > 0)):
            # Convert pixel coordinates back to geographic coordinates
            lon, lat = dataset.xy(y, x)
            
            # Calculate the gradient angle (0 to 90 degrees)
            if grad_x[y, x] != 0:
                gradient_angle = np.arctan(grad_y[y, x] / grad_x[y, x]) * (180 / np.pi)
            else:
                # Skip calculations if the gradient in the x-direction is zero
                continue
            
            # Check if the gradient angle is within the specified range (over 60 to 60 degrees)
            if gradient_angle > 60 or gradient_angle < -60:
                # Start calculating total height and angles within the cliff range
                total_height = grad_magnitude[y, x]
                num_points = 1
                for i in range(y+1, elevation_data.shape[0]):
                    # Calculate the gradient angle at each point
                    if grad_x[i, x] != 0:
                        angle = np.arctan(grad_y[i, x] / grad_x[i, x]) * (180 / np.pi)
                    else:
                        # Skip calculations if the gradient in the x-direction is zero
                        break
                    # Check if the angle is within the specified range
                    if angle <= 60 and angle >= -60:
                        # Add the height to the total height
                        total_height += grad_magnitude[i, x]
                        num_points += 1
                    else:
                        # Exit the loop if the angle is outside the specified range
                        break
                
                # Calculate the average angle and total height
                average_angle = gradient_angle
                if num_points > 1:
                    average_angle = (average_angle + gradient_angle) / 2
                    total_height /= num_points
                
                # Append the cliff information to the list of cliffs
                cliffs.append({'lat': lat, 'lon': lon, 'angle': average_angle, 'height': total_height})
            
        return cliffs

def find_best_cliffs(cliffs, top_n=200):
    """
    Find the best cliffs based on the greatest elevation drop, limited to top N cliffs.
    
    :param cliffs: A list of dictionaries, where each dictionary contains 'lat', 'lon', and 'drop' keys.
    :param top_n: Number of top cliffs to return.
    :return: A list of dictionaries representing the top N cliffs.
    """
    # Sort cliffs by drop and return the top N
    return sorted(cliffs, key=lambda x: x['angle'], reverse=True)[:top_n]

def plot_elevation_with_cliffs(elevation_file, cliffs):
    with rasterio.open(elevation_file) as raster:
        # Read the first band of the raster
        array = raster.read(1)
        transform = raster.transform
        
        # Plot the elevation data
        fig, ax = plt.subplots(figsize=(10, 10))
        show(array, ax=ax, transform=transform, cmap='terrain', title='Elevation with Cliffs')

        num_cliffs = len(cliffs)
        # Create a colormap
        cmap = plt.cm.get_cmap('viridis', num_cliffs)  
        # Plot each cliff with a color from the colormap
        for i, cliff in enumerate(cliffs):
            x, y = (cliff['lon'], cliff['lat'])
            color = cmap(i)  # Get color from colormap
            ax.plot(x, y, 'o', markersize=5, color=color, label=f"{i + 1}: {round(cliff['angle'],2)}º")

        if num_cliffs < 10:
            ax.legend()

        plt.show()

def fetch_satellite_image(latitude, longitude, radius_km, width=1280, height=1280, access_token=""):
    # Calculate bounding box coordinates using the provided function
    south, north, west, east = calculate_bounding_box(latitude, longitude, radius_km)
    
    # Construct the Mapbox Static Images API URL with the bounding box satellite-v9 or streets-v12
    mode="satellite-v9"
    base_url = f"https://api.mapbox.com/styles/v1/mapbox/{mode}/static/[{west},{south},{east},{north}]/{width}x{height}?access_token={access_token}"
    print(base_url)
    
    response = requests.get(base_url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to fetch satellite image: {response.status_code}, {response.text}")
        return None

def plot_cliffs_on_satellite_image(satellite_image, cliffs, lat, long, radius_km):
    # Calculate bounding box coordinates using the provided function
    south, north, west, east = calculate_bounding_box(lat, long, radius_km)
    img = Image.open(io.BytesIO(satellite_image))
    img_array = np.array(img)
    
    fig, ax = plt.subplots()
    ax.imshow(img_array, extent=[west, east, south, north])
    for cliff in cliffs:
        ax.plot(cliff['lon'], cliff['lat'], 'ro')  # Plot cliffs as red dots
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Cliffs on Satellite Image')
    plt.show()


def main():
    try:
        # Prompt user for input
        # latitude = float(input("Enter the latitude of the starting point: "))
        # longitude = float(input("Enter the longitude of the starting point: "))
        # radius_km = float(input("Enter the radius in kilometers: "))
        latitude = 10.013655343803803
        longitude = -84.11622265719353
        radius_km = 2
        
        # Input validation
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180 or radius_km <= 0:
            raise ValueError("Invalid input values. Latitude must be between -90 and 90, " +
                             "longitude between -180 and 180, and radius must be positive.")
        
        # Continue with the existing workflow
        api_key = ""  # Consider managing this securely
        dem_type = "SRTMGL1"
          # Fetch elevation data within the bounding box
        filename = fetch_elevation_data(dem_type, latitude, longitude, radius_km, api_key)
        if filename:
            cliffs = process_elevation_data(filename)
            # Cluster and filter cliffs that are too close to each other
            cliffs_filtered = cluster_and_filter_cliffs(cliffs)
            # Then find the top 5 cliffs from the filtered list
            top_cliffs = find_best_cliffs(cliffs_filtered)
            
            # Print details of the top 5 cliffs
            for i, cliff in enumerate(top_cliffs, start=1):
                distance = geodesic((latitude, longitude), (cliff['lat'], cliff['lon'])).kilometers
                print(f"Cliff {i}: Located at {cliff['lat']}, {cliff['lon']}, with an angle of {round(cliff['angle'],2)},  and {cliff['height']} m. Distance: {round(distance, 2)} km.")
            
            # Plot the top 5 cliffs along with the elevation data
            # plot_elevation_with_cliffs(filename, top_cliffs)
            satellite_image = fetch_satellite_image(latitude, longitude, radius_km)
            if satellite_image:
                plot_cliffs_on_satellite_image(satellite_image, top_cliffs, latitude, longitude, radius_km)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()