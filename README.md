# U-Net Image Segmentation with PyTorch

An educational image-segmentation experiment that explores a U-Net-style encoder-decoder workflow in PyTorch, from paired-image loading through training and qualitative visualization.

## Problem Statement

Image segmentation assigns a class or target value to every pixel rather than producing one label for an entire image. U-Net is well suited to this task because its contracting path learns contextual features while its expanding path restores spatial detail. Skip connections are intended to preserve fine-grained information, which is especially useful when training data is limited.

This repository is a notebook-based lab and portfolio project. It documents an experimental workflow rather than a production-ready or independently reproduced model.

## Project Highlights

- Custom PyTorch `Dataset` and `DataLoader` pipeline for paired inputs and targets
- U-Net-style encoder, bottleneck, and transposed-convolution decoder
- Center-cropped encoder features intended for decoder skip connections
- Tensor conversion and random horizontal-flip augmentation in the image pipeline
- Automatic CUDA selection when available, with CPU fallback
- Training and validation loops with loss-history collection
- Qualitative comparison of input images, predictions, and target masks

## Intended Architecture

The notebook defines a U-Net-style model with:

- Reusable convolution blocks
- Four encoder stages followed by max pooling
- A bottleneck between the encoder and decoder
- Four decoder stages using transposed convolutions
- Center cropping to align encoder and decoder feature-map dimensions
- Skip connections intended to transfer encoder detail into the decoder
- A three-channel output for image-shaped segmentation targets

The model code is educational and currently requires implementation cleanup before it should be treated as an end-to-end validated U-Net implementation.

## Repository Layout

```text
.
|-- task_C.ipynb             # Dataset, model, training, and visualization workflow
|-- CI_Lab_5_PyTorch.pdf     # Lab assignment and reference material
`-- README.md                # Project documentation
```

## Dataset Format

The dataset is not included in this repository. The notebook expects it in Google Drive at these notebook-relative locations:

```text
./drive/MyDrive/dataset1/
|-- train/
`-- val/
```

Each file must contain a paired sample in a single image:

```text
+----------------------+----------------------+
|     input image      |     target mask      |
+----------------------+----------------------+
```

`MyDataset.__getitem__` splits every file at half its width. The left half becomes the model input and the right half becomes the target. Samples should therefore have two equally sized halves with matching spatial content.

## Running the Notebook

1. Upload or open [`task_C.ipynb`](task_C.ipynb) in Google Colab.
2. Place the paired dataset files in `dataset1/train` and `dataset1/val` within Google Drive.
3. Run the Google Drive mounting cell and authorize Colab to access the dataset.
4. Verify that the notebook resolves `./drive/MyDrive/dataset1/train/*` and `./drive/MyDrive/dataset1/val/*`.
5. Run the remaining cells in order, reviewing and correcting the model implementation where necessary.

The notebook is configured with:

| Setting | Value |
| --- | --- |
| Training batch size | `4` |
| Validation batch size | `1` |
| Optimizer | Adam |
| Learning rate | `0.01` |
| Loss function | Mean squared error (`nn.MSELoss`) |
| Epochs | `30` |
| Device | CUDA when available, otherwise CPU |

No standalone training script or dependency manifest is provided; the workflow is designed around the imports and cells in the Colab notebook.

## Results

The notebook contains the qualitative visualization workflow for comparing source images, predicted outputs, and target masks. It also records and prints training and validation loss history during the epoch loop.

See the outputs embedded in [`task_C.ipynb`](task_C.ipynb) for the available experiment context. This repository does not provide verified benchmark metrics, a saved model checkpoint, or claims about segmentation accuracy.
