# DeepLearning_nmt
Project repo for deep learning, based on nmt of Tensorflow tutorial.

Data that used to train our model is available at: https://drive.google.com/open?id=1RIcqkbwe3tVxLFdQefZS9owtNxsRAtwv

These model folders in this repo can produce the translations using three models we have trained. Train logs and checkpoints are contained inside each model folder.

You will need to do the following to use the checkpoints and generate translations from English to Chinese:

1. Clone this git repo:
```
git clone https://github.com/Epica-Wang/DeepLearning_nmt.git
```

2. Run the Python code to load the checkpoints and make inference.

* To use naive nmt model, please run:
```
python predict_naive.py
```
Then type English sentence you want to translate. Parameters used for the model and translation will be printed on your screen.
