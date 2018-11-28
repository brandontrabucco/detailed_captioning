# Detailed Captioning

Thanks for reading my code! This repository is my research workspace for detailed image captioning. I am actively updating this repo with my latest ideas and results, so please keep checking in Your contributions are welcome. Continue reading to get started training a model.


## Image Captioning

End-to-end approaches for image captioning achieve great success. These architectures use neural networks to encode an image into a latent featurized representation. These features are decoded using a second neural network into a sequence of word ids. [Show And Tell](https://arxiv.org/abs/1411.4555) is the among first generation of the monolithic CNN-encoder, and RNN-decoder framework for end-to-end image captioning. The likelihood of the ground truth captions given the image is maximized using gradient descent. More recent approaches focus on image-attention, object-detection, and styling captions.


## Motivation

The number of recent papers (since 2015) that claim to achieve state-of-the-art performance is increasing. In these papers, scientists  often implenent their own versions of the papers they benchmark against. However, in these implementations, the resulting metrics such as BLEU-4 always differ, and are sometimes lower then reported in the original papers. This misalignment causes a direct comparison between papers difficult. Therefore, we create a consistent framework using tensorflow to implement the aforementioned state-of-the-art models, to create new models for end-to-end image captioning, and to provide consistent performance evaluations.


## Installation

This repository depends on the TensorFlow automatic differentiation library, and a few other computation libraries. Install the following python packages using pip.

```
pip install Pillow
pip install numpy
pip install tensorflow
```

Additionally, this repository relies on a few external code bases from github. In particular, download the TensorFlow [object detection](https://github.com/tensorflow/models/tree/03612984e9f7565fed185977d251bbc23665396e/research/object_detection) API and run the setup script.

```
git clone https://github.com/tensorflow/models.git 
cd models/research/object_detection/
pip install -e .
```

For evaluation purposes on the COCO dataset, we use Tsung-Yi Lin's repository. Download and install the [coco-caption](https://github.com/tylin/coco-caption/tree/3a9afb2682141a03e1cdc02b0df6770d2c884f6f) repository and edit to suite your needs.

Finally, all of the models we implement use the [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors to initialize the word embeddings matrix. We provide a clean implementation of a cached data loader for glove, with a conveniet vocabulary format, and various useful utility functions. Download and install our [data loader](https://github.com/brandontrabucco/glove/tree/3d9cb98573119b0a3d1f3e6405881b9156ad9421) from github.

```
git clone https://gitub.com/brandontrabucco/glove.git
cd glove/
pip install -r requirements.txt
cd embeddings/
chmod +x download_and_extract.sh
download_and_extract.sh
cd ../
python tests.py
```

## Data Preprocessing

...

## Training

...

