import pandas as pd

def deg2rad(deg):
  return (deg * (pi/180))
  
def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    R = 6371 # Radius of the earth in km 3963 for miles 6371 for kms
    dLat = deg2rad(lat2-lat1) # deg2rad below
    dLon = deg2rad(lon2-lon1) 
    a = sin(dLat/2) * sin(dLat/2) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    # d = haversine((lat1, lon1), (lat2, lon2), unit='km')
    return d


def preprocess_data():
    print("hello")