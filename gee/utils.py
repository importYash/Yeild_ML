import ee

def reduce_mean(image, geom, scale=250):
    return image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        bestEffort=True
    ).getInfo()
