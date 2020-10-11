# A TRPO implementation

![CodeQL](https://github.com/FTC-8856/trpo/workflows/CodeQL/badge.svg?branch=main)

Based off of <https://github.com/pat-coady/trpo>

## Training

```zsh
python3 trpo/train.py /path/to/folder/
```

Will either create a new model or load a saved model from the given folder. Model will be trained and saved in the given folder.

## Viewing

```zsh
python3 trpo/view.py /path/to/folder/
```

Will render a new episode using the saved model from the given folder.

## Evaluating

```zsh
python3 trpo/eval.py /path/to/folder/
```

Will evaluate the saved model from the given folder.
