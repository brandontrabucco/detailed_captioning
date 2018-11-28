# Detailed Captioning

Thanks for reading my code! This repository is my research workspace for detailed image captioning. I am actively updating this repo with my latest ideas and results, so please keep checking in. Continue reading to get started training a model.


## Image Captioning

End-to-end approaches for image captioning achieve great success. These architectures use neural networks to encode an image into a latent featurized representation. These features are decoded using a second neural network into a sequence of word ids.

[Show And Tell](https://github.com/brandontrabucco/detailed_captioning/edit/master/README.md) is the among first generation of the monolithic CNN-encoder, and RNN-decoder framework for end-to-end image captioning. The likelihood of the ground truth captions given the image is maximized using gradient descent.

