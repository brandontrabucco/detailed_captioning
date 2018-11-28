# Detailed Captioning

Thanks for reading my code! This repository is my research workspace for detailed image captioning. I am actively updating this repo with my latest ideas and results, so please keep checking in Your contributions are welcome. Continue reading to get started training a model.


## Image Captioning

End-to-end approaches for image captioning achieve great success. These architectures use neural networks to encode an image into a latent featurized representation. These features are decoded using a second neural network into a sequence of word ids. [Show And Tell](https://arxiv.org/abs/1411.4555) is the among first generation of the monolithic CNN-encoder, and RNN-decoder framework for end-to-end image captioning. The likelihood of the ground truth captions given the image is maximized using gradient descent. More recent approaches focus on image-attention, object-detection, and styling captions.


## Motivation

The number of recent papers (since 2015) that claim to achieve state-of-the-art performance is increasing. In these papers, scientists  often implenent their own versions of the papers they benchmark against. However, in these implementations, the resulting metrics such as BLEU-4 always differ, and are sometimes lower then reported in the original papers. This misalignment causes a direct comparison between papers difficult. Therefore, we create a consistent framework using tensorflow to implement the aforementioned state-of-the-art models, to create new models for end-to-end image captioning, and to provide consistent performance evaluations.


## Installation




