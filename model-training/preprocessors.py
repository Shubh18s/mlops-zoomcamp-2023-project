import pandas as pd
from math import radians, sin, cos, acos, pi, atan2, sqrt

def degree_to_radians(deg):
  return (deg * (pi/180))
  
def get_distance_in_km(lat1,lon1,lat2,lon2):
    R = 6371 # Radius of the earth in km 3963 for miles 6371 for kms
    dLat = degree_to_radians(lat2-lat1) # deg2rad below
    dLon = degree_to_radians(lon2-lon1) 
    a = sin(dLat/2) * sin(dLat/2) + cos(degree_to_radians(lat1)) * cos(degree_to_radians(lat2)) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    # d = haversine((lat1, lon1), (lat2, lon2), unit='km')
    return d