---
jupyter:
  kernelspec:
    display_name: env
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.4
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
# Sementic Segmentation of Satellite Imagery
:::

::: {.cell .markdown}
This software efficiently performs semantic segmentation on satellite
images to classify various features, utilizing both machine learning
(ML) and deep learning (DL) techniques.

It employs a two-layer architecture:

First Layer (Machine Learning): Users start by applying a machine
learning algorithm for semantic segmentation, choosing the most suitable
one based on factors such as class characteristics and pixel overlap.
The algorithm produces an initial segmented mask, which is returned to
the user.

Second Layer (Deep Learning - SAM-2): The segmented mask from the ML
model is then passed to a pre-trained deep learning model, Segment
Anything Model-2 (SAM-2), for fine-tuning according to user
requirements. SAM-2, being pre-trained, needs only a small dataset for
fine-tuning, allowing it to deliver accurate results with minimal
additional training. Once fine-tuned, users can perform segmentation
directly through the deep learning model without needing to provide
additional class samples.

This approach ensures efficient and precise segmentation, leveraging
SAM-2's pre-trained capabilities to achieve high accuracy with minimal
training data.
:::

::: {.cell .markdown}
![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/flowchart.png)
:::

::: {.cell .markdown vscode="{\"languageId\":\"plaintext\"}"}
Users can interact with an OGC-certified web-based mapping service
(Leaflet) to easily select a Region of Interest (ROI) on satellite
imagery. Additionally, users can define and choose sample points
representing specific features or classes (such as land, water, urban
areas, and vegetation) that will guide the segmentation and
classification process. These samples are then input into a variety of
machine learning models for segmentation, including:

1.  Random Forest Classifier
2.  Mahalanobis Distance Classifier
3.  Maximum Likelihood Classifier
4.  Parallelepiped Classifier

This interactive system empowers users to provide precise inputs, which
are then processed by the ML models to efficiently segment the satellite
imagery, producing accurate Land Use and Land Cover (LULC)
classifications.
:::

::: {.cell .markdown}
## 1. Region of Interest (ROI) Selection and Loading Satellite Imagery {#1-region-of-interest-roi-selection-and-loading-satellite-imagery}
:::

::: {.cell .markdown}
![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_1.png)
:::

::: {.cell .markdown}
The steps below show selection of region of interest, here Muzaffar
Nagar, and sampling of features
:::

::: {.cell .markdown}
1.  Selection of Region of Interest
    ![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_2.png)
    ![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_3.png)
:::

::: {.cell .markdown}
1.  In the advanced tab, users have the option to customize the
    selection of specific bands for more complex segmentation. For
    non-technical users, the default settings are pre-configured to
    bands B4 (Red), B3 (Green), and B2 (Blue), ensuring a standard
    true-color visualization. In this case, bands B7 (Shortwave Infrared
    2), B4 (Red), and B1 (Coastal Aerosol) are selected to enhance the
    segmentation of urban areas. This combination of bands helps to
    differentiate built-up structures from vegetation and water bodies,
    aiding in more accurate urban area identification.
    ![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_4.png)
:::

::: {.cell .markdown}
1.  The image is then retrieved from the Google Earth Engine\'s
    Sentinel-2 database, ensuring high-quality satellite data for
    analysis.
:::

::: {.cell .code execution_count="1"}
``` python
import ee
import json
import geemap
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import datetime
from PIL import Image
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

# ee.Authenticate() #Authenticated
ee.Initialize(project='fet-image-segmentation')
```
:::

::: {.cell .code execution_count="16"}
``` python
# app/gee_image.py

class GeeImage():

    def __init__(self) -> None:
        self.roi = []
        self.bands = []
        self.scale = 30
        self.img_array = []
        self.normalized_image = []
        self.sentinal_image = None
        self.start_date = '2023-03-01'
        self.end_date = '2023-03-31'

    def setRoiData(self, data):
        self.roi = data['geojson'][0]['geometry']['coordinates'][0]
        self.bands = [ band for band in data['bands'].values()]
        self.scale = 30
        if data.get('date', None):
            self.start_date = data['date']
            date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
            new_date = date_obj + relativedelta(months=1)
            self.end_date = new_date.strftime('%Y-%m-%d')

    
    def getImage(self):
        roi = ee.Geometry.Polygon([self.roi])
        self.sentinal_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(self.start_date, self.end_date) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first() \
        .select(self.bands)
        image_clipped = self.sentinal_image.clip(roi)
        self.img_array = geemap.ee_to_numpy(image_clipped, region=roi, bands=self.bands, scale=self.scale)
        self.normalized_image = (self.img_array - np.min(self.img_array)) / (np.max(self.img_array) - np.min(self.img_array))

    def getBands(self):
        return self.bands

    def getRawImage(self):
        return self.img_array
    
    def getNormalizedImage(self):
        return self.normalized_image

```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::
:::

::: {.cell .code execution_count="17"}
``` python


def get_area(image, scale):
    non_black_pixels = np.count_nonzero(np.any(image != [0, 0, 0], axis=-1))
    area_km2 = (non_black_pixels * scale**2) / 10**6
    return area_km2

def preprocess(image_array, is255):
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image_array must be an RGB image array with shape (height, width, 3).")
    
    if not is255:
        image_array_255 = (image_array * 255).astype(np.uint8)
    else:
        image_array_255 = image_array
    black_pixel_mask = np.all(image_array_255 == [0, 0, 0], axis=-1)

    rgba_image = np.zeros((image_array_255.shape[0], image_array_255.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image_array_255
    rgba_image[:, :, 3] = 255
    rgba_image[black_pixel_mask, 3] = 0
    img = Image.fromarray(rgba_image, 'RGBA')
    img = np.array(img)
    return img
```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::
:::

::: {.cell .code execution_count="5"}
``` python
data = {}
with open("./assets/roi.json") as f:
    data = json.load(f)
image = GeeImage()
image.setRoiData(data)
image.getImage()

normalized_image = image.getNormalizedImage()

image_png = preprocess(normalized_image, False)

plt.figure(figsize=(10, 10))
plt.imshow(normalized_image)
plt.title('Satellite Image')
plt.show()
```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::

::: {.output .display_data}
![](vertopal_6d4a1de205364c3d906e206ccd1ebb9f/85ffdf80a112fcfed05813557a27065e14e4b282.png)
:::
:::

::: {.cell .markdown}
![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_5.PNG)
:::

::: {.cell .markdown}
## 2. Feature Sampling for ML Models {#2-feature-sampling-for-ml-models}
:::

::: {.cell .markdown}
Features are sampled from each class to train the machine learning
models. Here, the following samples are taken.

1.  Urban
2.  Agriculture
3.  Water
4.  Barren
:::

::: {.cell .markdown}
![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_6.PNG)
:::

::: {.cell .markdown}
## 3. Machine Learning Model Selection and Thresholds Adjustment {#3-machine-learning-model-selection-and-thresholds-adjustment}
:::

::: {.cell .markdown}
This software offers four machine learning algorithms and one deep
learning model:

1.  Mahalanobis Distance Classifier
2.  Maximum Likelihood Classifier
3.  Random Forest Classifier
4.  Parallelepiped Classifier
5.  Segment Anything Model 2 by Meta (Deep Learning Model)

The selection of the best machine learning model depends on factors like
multiclass classification, the presence of background or undefined
classes, and overlapping pixel values among classes. The chosen model
must effectively address class imbalances, manage overlapping pixels,
and accurately identify unknown or background regions.

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/pixels.png)
:::

::: {.cell .markdown}
The most appropriate model can be selected from the Advanced tab, with
the Mahalanobis Distance Classifier set as the default. For advanced
operations and segmentation, adjusting the threshold can lead to more
refined and precise segmentation results.
:::

::: {.cell .markdown}
![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_7.PNG)
:::

::: {.cell .markdown}
## 3. Pixel Value Extraction Of Features/Classes {#3-pixel-value-extraction-of-featuresclasses}
:::

::: {.cell .markdown}
Pixel values for all the features or classes are extracted from the
image using the provided coordinates corresponding to each class or
feature, which are then used to train the machine learning model. This
sampled data serves as the training input to help the model classify or
segment the image based on the specified features.
:::

::: {.cell .code execution_count="18"}
``` python

class ImageMask():
    def __init__(self, bands, scale, img_array, start_date, end_date) -> None:
        self.features = {}
        self.bands = bands
        self.scale = scale
        self.img_array = img_array
        self.start_date = start_date
        self.end_date = end_date
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
            if class_name not in self.features:
                self.features[class_name] = i
                self.color_map[i] = self.hexToRgb(element['properties']['fill'])
        self.model = data['model']
        self.threshold = data['thresholds']
        self.mask()

    def mask(self):
        ee_geometry = defaultdict(list)
        for key, value in self.features_geometries.items():
            for element in value:
                geom = ee.Geometry.Polygon(element)
                ee_geometry[key].append(geom)
        all_geometries = []
        for value in ee_geometry.values():
            for element in value:
                all_geometries.append(element)

        combine_ee_geometry = all_geometries[0]
        for element in all_geometries[1:]:
            combine_ee_geometry = combine_ee_geometry.union(element)
        self.feature_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(combine_ee_geometry) \
        .filterDate(self.start_date, self.end_date) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first() \
        .select(self.bands)

        training_pixels = []
        training_lables = []
        for key, value in self.features.items():
            pixels = []
            for element in ee_geometry[key]:
                pixel_value, class_value = self.sample_region(element, value)
                pixels.extend(pixel_value)
                training_pixels.extend(pixel_value)
                training_lables.extend(class_value)
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
```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::
:::

::: {.cell .markdown}
## 5. Image Segmentation Using Selected Machine Learning Model {#5-image-segmentation-using-selected-machine-learning-model}
:::

::: {.cell .markdown}
This example uses Random Forest Classifier to segement the image.
:::

::: {.cell .code execution_count="19"}
``` python
# app/models.py


class Models(ImageMask):
    def __init__(self, bands, scale, img_array, start_date, end_date) -> None:
        super().__init__(bands, scale, img_array, start_date, end_date)
        reshaped_array = self.img_array.reshape((-1, len(self.bands)))
        self.non_zero_mask = (reshaped_array != 0).any(axis=1)
        self.non_zero_img_array = reshaped_array[self.non_zero_mask]
        self.output_pixels = np.zeros(self.non_zero_img_array.shape[0], dtype=np.int32)
        self.colored_mask = defaultdict()
        self.binary_masks = defaultdict()

    def getColoredMask(self):
        if self.model == 'Mahalanobis Distance Classifier':
            self.mahalanobis()
        elif self.model == 'Maximum Likelyhood Classifier':
            self.maximumLikelyHood()
        elif self.model == 'Random Forest Classifier':
            self.randomForest()
        else:
            self.parallelepiped()

        return self.colored_mask

    
    def mahalanobis(self): 
        for key, value in self.threshold.items():
            value = int(value)
            if value <= 0:
                value = 1
            elif value > 10:
                value = 9
            self.threshold[key] = value
        def classify_pixel(pixel, means, thresholds, inv_cov_key):
            distances = {}
            for key, i in self.features.items():
                distance = mahalanobis(pixel, means[key], inv_cov_key[key])
                if distance < thresholds[key]:
                    distances[i] = distance
            return min(distances, key=distances.get) if distances else 0
        
        inv_cov_key = {}
        for key, value in self.cov.items():
            inv_cov_key[key] = np.linalg.inv(value)

        for i, pixel in enumerate(self.non_zero_img_array):
            self.output_pixels[i] = classify_pixel(pixel, self.mean, self.threshold, inv_cov_key)
        self.colorMask()


    def maximumLikelyHood(self):
        
        if not self.threshold:
            self.threshold = 10
        self.threshold = int(self.threshold)
        if self.threshold < 5:
            self.threshold = 5
        elif self.threshold > 15:
            self.threshold = 15
        threshold = 10**(-int(self.threshold))
        for i, pixel in enumerate(self.non_zero_img_array):
            max_likelihood = -np.inf
            for key, k in self.features.items():
                likelihood = multivariate_normal(mean=self.mean[key], cov=self.cov[key]).pdf(pixel)
                if likelihood > max_likelihood and likelihood > threshold:
                    max_likelihood = likelihood
                    self.output_pixels[i] = k
        self.colorMask()

    def randomForest(self):
        random_forest =  RandomForestClassifier(n_estimators=100, random_state=25)
        random_forest.fit(self.X_train, self.y_train)
        self.output_pixels = random_forest.predict(self.non_zero_img_array)
        self.colorMask()

    def parallelepiped(self):
        parallelepiped_model = ParallelepipedClassifier()
        parallelepiped_model.fit(self.X_train, self.y_train)
        self.output_pixels = parallelepiped_model.classify(self.non_zero_img_array)
        self.colorMask()

    
    def colorMask(self):
        non_zero_mask = np.any(self.img_array != 0, axis=-1)
        for key, value in self.features.items():
            mask = np.zeros((self.img_array.shape[0], self.img_array.shape[1], 3), dtype=np.uint8)
            k = 0
            for i in range(self.img_array.shape[0]):
                for j in range(self.img_array.shape[1]):
                    if non_zero_mask[i, j]:
                        if self.output_pixels[k] == value:
                            mask[i][j] = self.color_map.get(value, self.color_map[None])
                        else:
                            mask[i][j] = self.color_map.get(None)
                        k += 1
                    else:
                        mask[i][j] = self.color_map.get(None)
            self.colored_mask[key] = mask



# Parallelepiped Model Class

class ParallelepipedClassifier:
    def __init__(self):
        self.thresholds = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for cls in classes:
            class_data = X[y == cls]
            min_values = np.min(class_data, axis=0)
            max_values = np.max(class_data, axis=0)
            self.thresholds[cls] = (min_values, max_values)

    def classify(self, X):
        labels = []
        for point in X:
            label = self._classify_point(point)
            labels.append(label)
        return labels

    def _classify_point(self, point):
        for label, (min_values, max_values) in self.thresholds.items():
            if np.all(point >= min_values) and np.all(point <= max_values):
                return label
        return None

```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::
:::

::: {.cell .code execution_count="20"}
``` python
class_data = {}
with open("./assets/class.json") as f:
    class_data = json.load(f)

mask = Models(image.bands, image.scale, image.img_array, image.start_date, image.end_date)
mask.setClassData(class_data)
colored_mask_pngs = mask.getColoredMask()
```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::
:::

::: {.cell .code execution_count="21"}
``` python
fig, axes = plt.subplots(1, len(colored_mask_pngs.keys()), figsize=(10, 5))

# Loop over the images and plot each one
for i, (key, img) in enumerate(colored_mask_pngs.items()):
    area = get_area(img, mask.scale)  # Calculate area
    png_mask = preprocess(img, True)
    axes[i].imshow(png_mask)  # Plot the image in grayscale
    axes[i].set_title(key, fontsize=12)  # Set title for each image
    axes[i].set_xlabel(f'Area: {area} Km sq', fontsize=10)  # Set subtitle for each image


plt.tight_layout()
plt.show()
```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::

::: {.output .display_data}
![](vertopal_6d4a1de205364c3d906e206ccd1ebb9f/c728d385449262f76a30898915ac1e3e3f3dbe4a.png)
:::
:::

::: {.cell .code execution_count="22"}
``` python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

original = GeeImage()
original.setRoiData(data)
original.bands = ['B4', 'B3', 'B2']
original.getImage()
org_img = original.getNormalizedImage()
org_png = preprocess(org_img, False)

axes[0].imshow(org_png)  # Display the original image
axes[0].axis('off')  # Hide axis for clarity
axes[0].set_title("Original Image", fontsize=12)

for key, img in colored_mask_pngs.items():
    png_mask = preprocess(img, True)
    axes[1].imshow(png_mask)  # Plot the image

# Hide the axis for clarity
axes[1].axis('off')  # Hide axis for the stacked image
axes[1].set_title("Mask", fontsize=12)

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
```{=html}

            <style>
                .geemap-dark {
                    --jp-widgets-color: white;
                    --jp-widgets-label-color: white;
                    --jp-ui-font-color1: white;
                    --jp-layout-color2: #454545;
                    background-color: #383838;
                }

                .geemap-dark .jupyter-button {
                    --jp-layout-color3: #383838;
                }

                .geemap-colab {
                    background-color: var(--colab-primary-surface-color, white);
                }

                .geemap-colab .jupyter-button {
                    --jp-layout-color3: var(--colab-primary-surface-color, white);
                }
            </style>
            
```
:::

::: {.output .display_data}
![](vertopal_6d4a1de205364c3d906e206ccd1ebb9f/ca0c05407d62af6143bde35cd33edd978c08a72d.png)
:::
:::

::: {.cell .markdown}
![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_8.PNG)
:::

::: {.cell .markdown}
## 5. Area and Opacity {#5-area-and-opacity}
:::

::: {.cell .markdown}
Users have the option to adjust the opacity of each segmented class,
enhancing the clarity of visualizing different features. This
flexibility allows for better differentiation between overlapping areas,
making it easier to assess the spatial distribution and area coverage of
each class. Additionally, users can retrieve the exact area for each
class.

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/opacity.PNG)
:::

::: {.cell .markdown}
## 6. GeoJson/KML Export {#6-geojsonkml-export}
:::

::: {.cell .markdown}
The application facilitates the export of user-selected features in
GeoJSON and KML formats, streamlining integration with other GIS tools
for further analysis and reporting.

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/geojson.PNG)
:::

::: {.cell .markdown}
## 7. Conclusion {#7-conclusion}
:::

::: {.cell .markdown}
The machine learning (ML) model generates a segmented mask by using
samples of classes provided by the user as training data. This segmented
mask not only serves as the output for the ML model but also plays a
crucial role in fine-tuning the Segment Anything Model 2 (SAM-2). The
mask is piped into SAM-2, which, being pre-trained for segmentation
tasks, uses it to adapt and fine-tune its performance based on the
specific user requirements. After sufficient fine-tuning, SAM-2 can
perform segmentation directly, without the need for user-provided class
samples.
:::
