### MVTec AD Dataset - Bottle

---

#### Training arguments

```bash
python .\train_mvtec.py --model simple_model --epochs 20 --batch_size 32 --num_workers 4 --dataset datasets/mvtecad/bottle --out models/basic/mvtec-bottle
```

#### Training progress

```bash
Epoch 1/20
6/6 [==============================] - 19s 3s/step - loss: 0.1148 - mse: 0.1128 - mae: 0.3169
Epoch 2/20
6/6 [==============================] - 20s 3s/step - loss: 0.0875 - mse: 0.0859 - mae: 0.2815
Epoch 3/20
6/6 [==============================] - 20s 3s/step - loss: 0.0606 - mse: 0.0582 - mae: 0.2172
Epoch 4/20
6/6 [==============================] - 21s 3s/step - loss: 0.0509 - mse: 0.0485 - mae: 0.1879
Epoch 5/20
6/6 [==============================] - 21s 3s/step - loss: 0.0456 - mse: 0.0433 - mae: 0.1711
Epoch 6/20
6/6 [==============================] - 21s 3s/step - loss: 0.0410 - mse: 0.0391 - mae: 0.1615
Epoch 7/20
6/6 [==============================] - 21s 3s/step - loss: 0.0356 - mse: 0.0337 - mae: 0.1458
Epoch 8/20
6/6 [==============================] - 19s 3s/step - loss: 0.0283 - mse: 0.0266 - mae: 0.1275
Epoch 9/20
6/6 [==============================] - 22s 4s/step - loss: 0.0188 - mse: 0.0169 - mae: 0.0985
Epoch 10/20
6/6 [==============================] - 19s 3s/step - loss: 0.0109 - mse: 0.0084 - mae: 0.0629
Epoch 11/20
6/6 [==============================] - 19s 3s/step - loss: 0.0094 - mse: 0.0077 - mae: 0.0598
Epoch 12/20
6/6 [==============================] - 19s 3s/step - loss: 0.0086 - mse: 0.0075 - mae: 0.0625
Epoch 13/20
6/6 [==============================] - 19s 3s/step - loss: 0.0079 - mse: 0.0068 - mae: 0.0568
Epoch 14/20
6/6 [==============================] - 19s 3s/step - loss: 0.0071 - mse: 0.0060 - mae: 0.0522
Epoch 15/20
6/6 [==============================] - 19s 3s/step - loss: 0.0061 - mse: 0.0050 - mae: 0.0462
Epoch 16/20
6/6 [==============================] - 19s 3s/step - loss: 0.0052 - mse: 0.0042 - mae: 0.0423
Epoch 17/20
6/6 [==============================] - 19s 3s/step - loss: 0.0048 - mse: 0.0037 - mae: 0.0379
Epoch 18/20
6/6 [==============================] - 19s 3s/step - loss: 0.0046 - mse: 0.0036 - mae: 0.0365
Epoch 19/20
6/6 [==============================] - 20s 3s/step - loss: 0.0044 - mse: 0.0034 - mae: 0.0355
Epoch 20/20
6/6 [==============================] - 20s 3s/step - loss: 0.0043 - mse: 0.0034 - mae: 0.0358
```

#### Training results (on test set)

```python
...

MODEL_PATH = 'models/basic/mvtec-bottle/model.h5'
MODEL_HISTORY_PATH = 'models/basic/mvtec-bottle/history.json'

...
```

```python
...

DATASET_PATH = 'datasets/mvtecad/bottle/'

...
```

```bash
1/1 [==============================] - 1s 1s/step - loss: 0.0046 - mse: 0.0038 - mae: 0.0357
2/2 [==============================] - 3s 2s/step - loss: 0.0046 - mse: 0.0038 - mae: 0.0356
```

-   1 is normal, 2 is anomaly

#### Kernel Density Estimation (KDE) & Reconstruction Error

```bash
Normal values:  (11310.150001088396, 0.00423112799739258, 0.004847864375302666, 0.0004086582784052215)
Anomaly values:  (11293.909557117842, 12.062287046707189, 0.0047827543645736674, 0.0007189690768629724)
```

-   (**KDE mean**, **KDE std**, **reconstruction error mean**, **reconstruction error std**)

#### Chosen thresholds for anomaly detection

```python
density_threshold = 11305
reconstruction_error_threshold = 0.005
```

-   I chose the thresholds based on the KDE mean and the reconstruction error mean
    -   The reconstruction error mean is a bit vague in this case, because the reconstruction error is around the same value for both normal and anomaly images. So, I chose the thresholds based on the KDE mean while keeping the reconstruction error threshold a bit higher than the mean (to avoid false positives).

#### AUROC

```bash
ROC AUC score:  0.9518072289156626
False positives:  0
```
