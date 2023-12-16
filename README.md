# PNDID
Printed Numerical Digits Image Dataset

Disclaimer: This dataset doesn't seems to be exhaustive enough (or weak deformations that lead to overfitting). I am not actively working on it (at the moment, lack of time and willingness) so it may stay like this for an undefined amount of time.

The set decomposes into 90k 28x28 B&W images (based on MNIST dataset format). The first 30k images can be generated from "Google Fonts"'s fonts and then those are deformed to generate the other 60k to form a total 90k set. The deformations are taken from 'Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis', presented during ICDAR 2003 (try "ICDAR 2003 simard" on google). The fun part is to find parameters that wont lead to overfitting while creating enough exhaustivity to let the Neural Network generalize.
