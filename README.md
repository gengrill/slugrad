Slugrad: Naive C++ Implementation of a Single-Precision Autograd Engine (with a Drag..)
---

<img src="slugrad.png" alt="SLUGRAD! Slow C++ Autograd Engine" width="300px">

This is an absolute minimal, barebones, no-frills C++ implementation of a single-precision autograd engine - akin to ["Andreij Karpathy's Micrograd"](https://github.com/karpathy/micrograd). Slugrad is sloooooooooooow as as a snail..

### WHY???
Purely for fun and educational purposes.

### HOW???
The main idea is to box standard floats in a wrapper struct with reference semantics and an additional gradient member.
Arithmetic operations on these boxed values build up the computational graph by storing references to operands in resulting nodes.
Backpropagation is then just a traversal of that graph in reverse topological order from a starting node, calculating partial derivatives for each operation with respect to their stored data members using the chain rule.
Just like in micrograd, this turns out to be enough to train a small Multi-Layer Perceptron to classify a non-linearly seperable dataset:

```
$ clang++ -std=c++20 slugrad.cpp -o slugrad
$ ./slugrad
Step 0 Loss=0.011609, Accuracy=0.27
Step 1 Loss=0.0101711, Accuracy=0.38
Step 2 Loss=0.0100195, Accuracy=0.57
Step 3 Loss=0.00980061, Accuracy=0.78
Step 4 Loss=0.0089826, Accuracy=0.86
Step 5 Loss=0.00701868, Accuracy=0.91
...
```

### COOL I WANNA USE THIS!!1
No, you probably really don't.. for actual training you'd want a vectorized implementation that handles layers of multiple neurons as weight matrices and calculates activations as well as partial derivatives using matrix multiplication.
(Also, you'd want something that allows checkpointing and saving models to disk, doesn't spend 80% of its time boxing and unboxing values to and from lambda captures, half or even quarter precision data types, and many more..)
