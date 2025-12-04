import ee

def polygon_from_coordinates(coords):
    return ee.Geometry.Polygon(coords)
