
# View Consistent Gaussian Splatting for Novel View Synthesis

This repository contains the implementation of the method described in the paper "View Consistent Gaussian Splatting from Blurry Images for Novel View Synthesis". The method aims to improve novel view synthesis from blurry images by leveraging view-consistent Gaussian splatting .
## Dataset

We used the deblur nerf dataset for training and evaluation. You can download the dataset from the following link:

[Deblur NeRF Dataset](https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_)
## Pretrained Model

You can find the pretrained model for the Real Motion Blur 2D Deblurring model at the following link:

[Real Motion Blur 2D Deblurring Pretrained Model](https://github.com/ZhendongWang6/Uformer)

## How to Run

To run the code, follow these steps:

1. Clone this repository:
```shell
git clone https://github.com/DTG777/View-Consistent-Gaussian-Splatting-from-Blurry-Images-for-Novel-View-Synthesis.git
```
2. Navigate to the project directory:
```shell
cd view-consistent-gaussian-splatting
```
4. Install dependencies (assuming you have Python and pip installed):
```shell
pip install -r requirements.txt
```
5. Run the main script:
```shell
sh script/deblur.sh
```

