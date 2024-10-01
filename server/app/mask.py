# app/mask.py

import ee, numpy as np
from app.gee_image import GeeImage
from collections import defaultdict

class ImageMask(GeeImage):
    def __init__(self) -> None:
        super().__init__()
        self.features = {}
        self.features_geometries = None
        self.color_map = {None: [0, 0, 0], 0: [0, 0, 0]}
        self.model = ""
        self.feature_image = None
        self.pixels = defaultdict(list)
        self.mean = defaultdict(list)
        self.cov = defaultdict(list)
        self.threshold = None
        self.X_train = None
        self.y_train = None

    def setClassData(self, data):
        self.features_geometries = defaultdict(list)
        for i, element in enumerate(data['geojson'], 1):
            class_name = element['properties']['class']
            self.features_geometries[class_name].append(element['geometry']['coordinates'][0])
            self.features[element['properties']['class']] = i
            self.color_map[i] = self.hexToRgb(element['properties']['fill'])
        self.model = data['model']
        self.threshold = data['thresholds']
        self.mask()

    def mask(self):
        ee_geometry = defaultdict(list)
        for key, value in self.features_geometries.items():
            ee_geometry[key].append(ee.Geometry.Polygon([value][0]))

        all_geometries = []
        for value in ee_geometry.values():
            all_geometries += value

        combine_ee_geometry = all_geometries[0]
        for element in all_geometries[1:]:
            combine_ee_geometry = combine_ee_geometry.union(element)
        self.feature_image = self.sentinal_image.clip(combine_ee_geometry)

        training_pixels = []
        training_lables = []
        for key, value in self.features.items():
            pixels = []
            for element in ee_geometry[key]:
                pixel_value, class_value = self.sample_region(element, value)
                pixels.append(pixel_value)
                training_pixels.append(pixel_value)
                training_lables.append(class_value)
            self.pixels[key] = np.vstack(pixels)
            self.mean[key] = np.mean(self.pixels[key], axis=0)
            self.cov[key] = np.cov(self.pixels[key],  rowvar=False)
        self.X_train = np.vstack(training_pixels)
        self.y_train = np.hstack(training_lables)   

    

    def sample_region(self, region, class_label):
        sampled = self.feature_image.sample(region=region, scale=self.scale, numPixels=500)
        pixels = sampled.select(self.bands).getInfo()
        values = [x['properties'] for x in pixels['features']]
        return np.array([[x[b] for b in self.bands] for x in values]),  np.array([class_label] * len(values))
    
    @classmethod
    def hexToRgb(cls, hex):
        hex = hex.lstrip('#')
        r = int(hex[0:2], 16)
        g = int(hex[2:4], 16)
        b = int(hex[4:6], 16)
        return [r, g, b]

        