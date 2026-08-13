# FoodVision Mini --- EfficientNetB2 vs Vision Transformer

A computer vision project for classifying **Pizza, Sushi, and Steak**
images using deep learning. The project compares **EfficientNetB2** and
a **Vision Transformer (ViT)** and evaluates their suitability for
deployment based on accuracy, inference speed, and model size.

## Project Overview

The main goal of this project is not only to find the model with the
highest accuracy, but also to understand the trade-off between:

-   **Classification performance**
-   **Inference speed**
-   **Model size**
-   **Deployment practicality**

Two models are compared:

-   **EfficientNetB2**
-   **Vision Transformer (ViT)**

The comparison is visualized using the following plot:

![FoodVision Mini Inference Speed vs
Performance](foodvision-mini-inference-speed-vs-performance.png)

## Dataset

The project uses a three-class food image dataset:

-   🍕 Pizza
-   🍣 Sushi
-   🥩 Steak

The dataset is included in the repository as a ZIP file.

> **Dataset licensing:** The dataset/images may have licensing or
> copyright terms separate from this project's source code. Check the
> original dataset source and its license before redistributing the
> images.

## Model Comparison

The final comparison gives the following results:

  Model              Test Accuracy   Prediction Time / Image   Model Size
  ---------------- --------------- ------------------------- ------------
  EfficientNetB2          \~96.88%                  \~0.10 s    \~29.8 MB
  ViT                     \~97.22%                  \~0.40 s   \~327.4 MB

### EfficientNetB2

**Advantages** - Faster inference. - Much smaller model. - High test
accuracy. - More practical when memory and latency are important.

**Disadvantage** - Slightly lower test accuracy than ViT in this
comparison.

### Vision Transformer

**Advantages** - Achieves the higher test accuracy in this comparison. -
Provides strong classification performance.

**Disadvantages** - Significantly slower inference. - Much larger model
size. - Requires substantially more storage than EfficientNetB2.

## Deployment Decision

The results demonstrate that the model with the highest accuracy is not
necessarily the best model for deployment.

### If accuracy is the priority

**ViT** is the better choice based on the measured test accuracy:

``` text
ViT:          ~97.22%
EfficientNetB2: ~96.88%
```

### If speed and model size are priorities

**EfficientNetB2** is the more practical choice:

``` text
EfficientNetB2: ~0.10 s/image, ~29.8 MB
ViT:            ~0.40 s/image, ~327.4 MB
```

Therefore, the final deployment choice should depend on the requirements
of the target application. EfficientNetB2 offers a strong balance of
accuracy, inference speed, and model size, while ViT offers the highest
measured accuracy at the cost of significantly greater inference time
and storage requirements.

## Repository Structure

``` text
ViT-Food101/
│
├── README.md
├── LICENSE
│
├── data/
│   └── pizza_sushi_steak.zip
│
├── notebooks/
│   ├── ViT_Pizza_Sushi_Steak_Training.ipynb
│   └── EfficientNetB2_vs_ViT_Comparison_and_Deployment.ipynb
│
├── data_setup.py
├── engine.py
├── model.py
├── predictions.py
├── train.py
├── utils.py
│
└── foodvision-mini-inference-speed-vs-performance.png
```

> The exact filenames and folders may differ depending on the final
> repository organization.

## Project Workflow

The project follows this general workflow:

1.  Prepare the Pizza, Sushi, and Steak dataset.
2.  Create training and testing datasets and data loaders.
3.  Apply the required image transformations.
4.  Train/evaluate the Vision Transformer.
5.  Train/evaluate EfficientNetB2.
6.  Measure test accuracy.
7.  Measure prediction time per image.
8.  Compare model sizes.
9.  Visualize the model comparison.
10. Select a model based on deployment requirements.

## Main Notebooks

### ViT Training

`ViT_Pizza_Sushi_Steak_Training.ipynb`

This notebook contains the Vision Transformer training workflow for the
Pizza, Sushi, and Steak classification task.

### Model Comparison and Deployment

`EfficientNetB2_vs_ViT_Comparison_and_Deployment.ipynb`

This notebook compares EfficientNetB2 and ViT using:

-   Test accuracy
-   Prediction time per image
-   Model size

The results are then used to evaluate which model is more appropriate
for deployment.

## Results

The measured results show a clear trade-off:

-   **ViT** achieves approximately **0.34 percentage points higher test
    accuracy**.
-   **EfficientNetB2** is approximately **4× faster per image** based on
    the displayed inference measurements.
-   **EfficientNetB2** is approximately **11× smaller** than ViT based
    on the displayed model sizes.

This demonstrates why deployment decisions should consider more than
classification accuracy.

## Running the Project

Clone the repository:

``` bash
git clone https://github.com/rahulkumarmeena29/ViT-Food101.git
cd ViT-Food101
```

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

The notebooks can then be opened and executed in Jupyter Notebook,
JupyterLab, or Google Colab.

## Google Colab

The project can also be run using Google Colab.

Add the link to your main Colab notebook here:

``` text
YOUR_COLAB_NOTEBOOK_LINK
```

For example, in Markdown:

``` markdown
[Open in Google Colab](YOUR_COLAB_NOTEBOOK_LINK)
```

## Inference Comparison

The comparison plot uses:

-   **X-axis:** Prediction time per image (seconds)
-   **Y-axis:** Test accuracy (%)
-   **Bubble size:** Model size (MB)

A desirable deployment model generally aims for:

-   High accuracy
-   Low prediction time
-   Small model size

The plot therefore provides a simple visual representation of the
trade-offs between the two models.

## Technologies Used

-   Python
-   PyTorch
-   TorchVision
-   NumPy
-   Matplotlib
-   Jupyter Notebook
-   Google Colab

## License

### Source Code

The source code in this repository is licensed under the **MIT
License**.

Copyright (c) 2026 Rahul Meena

You may use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the source code, subject to the terms of the MIT
License.

### Dataset

The MIT License for this repository's source code does **not
automatically apply to the Pizza, Sushi, and Steak images**.

The dataset and images may be subject to separate copyright and
licensing terms. Refer to the original dataset source for the applicable
license and attribution requirements.

## Disclaimer

The reported performance values are based on the experiments represented
in this repository. Actual performance may vary depending on hardware,
preprocessing, batch size, runtime environment, and other deployment
conditions.
