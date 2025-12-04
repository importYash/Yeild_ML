import ee
import numpy as np

OILSEED_PARAMS = {
    "mustard":   {"RUE": 1.45, "HI": 0.28},
    "soybean":   {"RUE": 1.30, "HI": 0.40},
    "groundnut": {"RUE": 1.50, "HI": 0.35},
    "sunflower": {"RUE": 1.40, "HI": 0.28},
    "sesame":    {"RUE": 1.35, "HI": 0.27}
}

def compute_oilseed_yield(geom, start, end, crop):
    params = OILSEED_PARAMS[crop]

    ndvi_col = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .select("NDVI") \
        .filterBounds(geom).filterDate(start, end) \
        .map(lambda img: img.divide(10000))

    solar_col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .select("surface_solar_radiation_downwards_sum") \
        .filterBounds(geom).filterDate(start, end)

    temp_col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .select("temperature_2m").filterDate(start, end)

    rain_col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .select("precipitation").filterDate(start, end)

    def daily(img):
        date = img.date()

        ndvi  = ndvi_col.filterDate(date, date.advance(1, "day")).mean()
        solar = solar_col.filterDate(date, date.advance(1, "day")).mean()
        temp  = temp_col.filterDate(date, date.advance(1, "day")).mean()
        rain  = rain_col.filterDate(date, date.advance(1, "day")).mean()

        apar = ndvi.multiply(solar).multiply(0.5)

        t_c = temp.subtract(273.15)
        temp_stress = t_c.gt(18).And(t_c.lt(32)).multiply(1.0) \
                        .add(t_c.lte(18).Or(t_c.gte(32)).multiply(0.5))

        rain_stress = rain.divide(8).clamp(0.3, 1.0)

        biomass = apar.multiply(params["RUE"]) \
                       .multiply(temp_stress) \
                       .multiply(rain_stress)

        return biomass

    biomass_col = solar_col.map(daily)

    total_biomass = biomass_col.sum().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=500
    ).getInfo()

    biomass = list(total_biomass.values())[0]  # g/m²

    biomass_t_ha = biomass / 1000.0

    yield_t_ha = biomass_t_ha * params["HI"]

    area_ha = geom.area().getInfo() / 10000

    return {
        "crop": crop,
        "yield_t_ha": yield_t_ha,
        "area_ha": area_ha,
        "production_tonnes": yield_t_ha * area_ha
    }
