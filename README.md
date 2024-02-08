# MGAtt-LSTM

The implementation of the paper "MGAtt-LSTM: A multi-scale spatial correlation prediction model of PM2.5 concentration based on multi-graph attention"

It is important to note that due to **confidentiality agreements**, we are unable to provide the complete dataset. We have only shared a part of North China data, and to ensure the code runs smoothly, **we have made modifications to the model input and some of the model structure.** As a result, **the final results may differ from those presented in the paper**, and we apologize for any inconvenience this may cause.

## Data

Due to GitHub's limit on uploading files larger than 100MB, please unzip `/data/pollution/airpollution.csv.zip` into the `/data/pollution/` folder to access the original data.

## Requirements

```python
pip install -r requirements.txt
```


## Training

```python
python train.py
```

