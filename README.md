# Sementic Segmentation of Satellite Imagery

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

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/flowchart.png)

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

## 1. Region of Interest (ROI) Selection and Loading Satellite Imagery

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_1.png)

The steps below show selection of region of interest, here Muzaffar
Nagar, and sampling of features

1.  Selection of Region of Interest
    ![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_2.png)
    ![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_3.png)

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

1.  The image is then retrieved from the Google Earth Engine\'s
    Sentinel-2 database, ensuring high-quality satellite data for
    analysis.

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_5.PNG)

## 2. Feature Sampling for ML Models

Features are sampled from each class to train the machine learning
models. Here, the following samples are taken.

1.  Urban
2.  Agriculture
3.  Water
4.  Barren

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_6.PNG)

## 3. Machine Learning Model Selection and Thresholds Adjustment {#3-machine-learning-model-selection-and-thresholds-adjustment}

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

The most appropriate model can be selected from the Advanced tab, with
the Mahalanobis Distance Classifier set as the default. For advanced
operations and segmentation, adjusting the threshold can lead to more
refined and precise segmentation results.

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/ui_7.PNG)

## 3. Pixel Value Extraction Of Features/Classes

Pixel values for all the features or classes are extracted from the
image using the provided coordinates corresponding to each class or
feature, which are then used to train the machine learning model. This
sampled data serves as the training input to help the model classify or
segment the image based on the specified features.


## 5. Image Segmentation Using Selected Machine Learning Model

This example uses Random Forest Classifier to segement the image.

## 5. Area and Opacity

Users have the option to adjust the opacity of each segmented class,
enhancing the clarity of visualizing different features. This
flexibility allows for better differentiation between overlapping areas,
making it easier to assess the spatial distribution and area coverage of
each class. Additionally, users can retrieve the exact area for each
class.

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/opacity.PNG)

## 6. GeoJson/KML Export

The application facilitates the export of user-selected features in
GeoJSON and KML formats, streamlining integration with other GIS tools
for further analysis and reporting.

![](https://raw.githubusercontent.com/ammar26627/segmentation-client/refs/heads/main/images/geojson.PNG)

## 7. Conclusion

The machine learning (ML) model generates a segmented mask by using
samples of classes provided by the user as training data. This segmented
mask not only serves as the output for the ML model but also plays a
crucial role in fine-tuning the Segment Anything Model 2 (SAM-2). The
mask is piped into SAM-2, which, being pre-trained for segmentation
tasks, uses it to adapt and fine-tune its performance based on the
specific user requirements. After sufficient fine-tuning, SAM-2 can
perform segmentation directly, without the need for user-provided class
samples.
