# A TRPO implementation

## Training

```zsh
python3 train.py /path/to/folder/
```

Will either create a new model or load a saved model from the given folder. Model will be trained and saved in the given folder.

## Viewing

```zsh
python3 view.py /path/to/folder/
```

Will render a new episode using the saved model from the given folder.

## Evaluating

```zsh
python3 eval.py /path/to/folder/
```

Will evaluate the saved model from the given folder.
