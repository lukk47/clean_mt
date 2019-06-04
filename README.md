# Mean Teacher using TensorFlow

This is a modified version of https://github.com/CuriousAI/mean-teacher after removing all unrelated code and spliting the mean_teacher.py file into several independent functions.


You can install the dependencies and prepare the datasets with the following commands:

```
pip install tensorflow==1.2.1 numpy scipy pandas
./prepare_data.sh
```

or directly use the prebuilt docker image:
```
docker pull loklu/mt_tensorflow:tf1.2.1_py35_lib2
```

To train the model, run:

* `python train_svhn.py` to train on SVHN using 500 labels
* `python train_cifar10.py` to train on CIFAR-10 using 4000 labels

These runners converge fairly quickly and produce a fair accuracy.

To reproduce the experiments in the paper run: `python -m experiments.cifar10_final_eval` or similar.
They use different hyperparameters, and each of the runs takes roughly four times as long to converge as the example runners above.
See the experiments directory for the complete set of experiments.
