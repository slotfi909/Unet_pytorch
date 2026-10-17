# U-Net Image Segmentation with PyTorch

An educational Jupyter notebook exploring image segmentation with a U-Net-style model in PyTorch. It contains paired-image loading, encoder-decoder model definitions, training and validation loops, and plots comparing inputs, predictions, and targets. The accompanying [lab handout](CI_Lab_5_PyTorch.pdf) provides assignment context.

The model is an unfinished exercise: the forward pass contains an invalid `self.convblock()(x)` call, `self.out(None)`, and decoder channel mismatches. These need correction before training can run successfully.

## Usage

Use Google Colab, since the notebook imports `google.colab` and mounts Google Drive. Other imports include PyTorch, torchvision, NumPy, pandas, Matplotlib, scikit-learn, Pillow, and tqdm.

1. Open [`task_C.ipynb`](task_C.ipynb) in Google Colab.
2. Place the dataset in Google Drive under `MyDrive/dataset1/train/` and `MyDrive/dataset1/val/`, as described below.
3. Run the import and Drive-mounting cells, authorizing access when prompted. The dataset paths assume Colab's working directory is `/content`.
4. Run the data-loading cells, complete the model implementation, then run the remaining cells in order.

The training configuration uses Adam with a learning rate of `0.01`, mean squared error loss, and `30` epochs. Batch sizes are `4` for training and `1` for validation. CUDA is selected when available, otherwise CPU is used.

## Dataset Format

The dataset is not included. The notebook reads image files from:

```text
/content/drive/MyDrive/dataset1/
|-- train/
`-- val/
```

Each file must contain an input image on the left and its target mask on the right, in equally sized halves. `MyDataset` splits the file at half its width. Use three-channel images with consistent dimensions within each batch; the model accepts three input channels and defines a three-channel output.
