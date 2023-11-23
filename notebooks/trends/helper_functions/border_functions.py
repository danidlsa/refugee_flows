import pandas as pd
import country_converter as coco
from geopy.distance import geodesic

def is_list_of_scalars(lst):
    return isinstance(lst, list) and len(lst) == 2 and all(isinstance(elem, (int, float)) for elem in lst)

contig_df = pd.read_csv('../../data/clean/unhcr.csv', engine='pyarrow').drop_duplicates(['iso_o','iso_d'])[['Country_o','Country_d','contig','iso_o']]

def flatten_nested_lists(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list) and is_list_of_scalars(item):
            flattened_list.append(item)
        elif isinstance(item, list):
            flattened_list.extend(flatten_nested_lists(item))
    return flattened_list



def find_closest_cities(iso3_country1, iso3_country2, boundary_df, cities_df, k, full_names = False):
    if full_names:
        iso3_country1 = coco.convert(iso3_country1, to='iso3')
        iso3_country2 = coco.convert(iso3_country2, to='iso3')
    # Filter boundary DataFrame for the first country
    boundary_country1 = boundary_df[boundary_df['country_code'] == iso3_country1]
    
    # Filter cities DataFrame for the second country
    cities_country2 = cities_df[cities_df['country_code'] == iso3_country2].copy()
    
    if len(boundary_country1) == 0 or len(cities_country2) == 0:
        return None

    # Calculate the distances between the boundary and cities
    distances = []
    # if len(boundary_country1.iloc[0]['geo_shape']['geometry']['coordinates'][0][0]) == 2:
    #     boundary_coords_list = boundary_country1.iloc[0]['geo_shape']['geometry']['coordinates'][0]
    # else:
    #     boundary_coords_list = boundary_country1.iloc[0]['geo_shape']['geometry']['coordinates'][0][0]
    

    boundary_coords_list = flatten_nested_lists(boundary_country1.iloc[0]['geo_shape']['geometry']['coordinates'])

    for _, city_row in cities_country2.iterrows():
        min_distance = float('inf')
        
        for boundary_coords in boundary_coords_list:
            city_coords = city_row['Coordinates']
            # Convert boundary coordinates to (latitude, longitude) format
            boundary_coords = boundary_coords[::-1]
            boundary_distance = geodesic(boundary_coords, city_coords).kilometers
            min_distance = min(min_distance, boundary_distance)
                
        distances.append(min_distance)

    # Add the distances as a new column to the DataFrame
    cities_country2['distance to closest border'] = distances

    # Sort the distances and select the top k cities
    closest_cities = cities_country2.nsmallest(k, 'distance to closest border')

    return closest_cities

def contiguous_countries(country):
    return contig_df[(contig_df.Country_o == country) & (contig_df.contig == 1)].Country_d.tolist()


def process_country(country):
    contigs = contiguous_countries(country)
    bordering_cities = dict()
    for contig in contigs:
        closest_cities = find_closest_cities(country, contig, coordinates, cities, 2, full_names=True)
        if closest_cities is not None:
            bordering_cities.update({contig: closest_cities['ASCII Name'].to_list()})
    return country, bordering_cities


# Define a function to update the shared dictionary
def update_shared_dict(result):
    country, bordering_cities = result
    bordering_country_cities.update({country: bordering_cities})

