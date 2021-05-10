# TensorFlow MNIST Work

Example scripts to train TensorFlow 2 Neural Networks on fashion-mnist or simpler mnist datasets.

## Prerequisites

- Linux
- Python 3.8, tensorflow-cpu, scikit-learn

## Running

Run the simpler fashion_custom_seq.py, or the more intensive fashion_custom_cnn.py, which will download the fashion-mnist
dataset, and create models as well as print their evaluation metrics.
```
./fashion_custom_seq.py 

...
Epoch 00050: val_loss did not improve from 0.21901
Training:       0.17545602 loss / 0.93274999 acc
Validation:     0.21901140 loss / 0.91986114 acc
Running final test with model 0: 0.3304 loss / 0.8980 acc

Average loss / accuracy on testset: 0.3304 loss / 0.89800 acc
Standard deviation: (+-0.0000) loss / (+-0.0000) acc
```

You can run the more complex fashion_custom_cnn.py, but it takes much longer and will use more resources.
```
./fashion_custom_cnn.py
 
```


## Running local data
You can also download the fashion-mnist/mnist dataset to your folder and run it calling 
fashion_util.load_fashion_data(download=False)
- Sample modified tree output:
    
    ```
    ├── fashion_custom_cnn.py
    ├── fashion_custom_seq.py
    ├── fashion_util.py
    ├── data
    │ └── fashion
    │     ├── t10k-images-idx3-ubyte.gz
    │     ├── t10k-labels-idx1-ubyte.gz
    │     ├── train-images-idx3-ubyte.gz
    │     └── train-labels-idx1-ubyte.gz
    ```

## Results
The seq.py script uses several deep nn's to predict results. It's faster than the cnn script, but results are lower.
A best accuracy of 90.18% was achieved using 3 nn layers with 2 dropout layers (1 in between each), an image augment
value of 4, no horizontal image flipping, random image rotation up to 10 deg, 200 epochs and 5 iterations. 
```
Epoch 00200: val_loss did not improve from 0.11998
Running final test with model 0: 0.3778 loss / 0.9018 acc
```

The updated cnn.py script uses several cnn's and deep nn's to predict results, and achieved 93.25% accuracy.
```
Epoch 00200: val_loss did not improve from 0.07657
Running final test with model 0: 0.2062 loss / 0.9325 acc
```

## Built With

* [Python3.8](https://www.python.org/) - The scripting code used
* [TensorFlow](https://www.tensorflow.org/) - For creating neural networks
* [scikit-learn](https://scikit-learn.org/stable/) - For parsing test/validation sets

