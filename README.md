# SensitiveMamba: Selective State Space Modeling for RobustTime Series Anomaly Detection


## SensitiveMamba
![images](./images/tsad_framework.png)

- We propose a novel TSAD framework, **SensitiveMamba**, that utilizes Mamba, also known as the S6 model.
- We leverage the long-sequence modeling capability of Mamba.
- We also leverage hiearchical contrastive learning and heteroscedastic uncertainty estimation to enhance sensitivity to robust patterns in the time series data.


## Get Started!
1. Install Python 3.10 and the required packages in `requirements.txt`.
2. Download data from here: [SWaT](https://github.com/yuesuoqingqiu/SensitiveHUE), [SMD](https://github.com/imperial-qore/TranAD/tree/main/data), [SMAP_MSL](https://github.com/imperial-qore/TranAD/tree/main/data).

### ✂️ Preprocess the data
```bash
# SWaT
python preprocess.py --config_path ./config/star.yaml --data_name SWaT
# SMD
python preprocess.py --config_path ./config/star.yaml --data_name SMD
# MSL
python preprocess.py --config_path ./config/star.yaml --data_name MSL
# SMAP
python preprocess.py --config_path ./config/star.yaml --data_name SMAP
```


### 🚀 Train and evaluate

```bash
# SWaT
python main.py --config_path ./config/star.yaml --data_name SWaT
# SMD
python main.py --config_path ./config/star.yaml --data_name SMD
# MSL
python main.py --config_path ./config/star.yaml --data_name MSL
# SMAP
python main.py --config_path ./config/star.yaml --data_name SMAP
```

### 🛠️ Implementation details
- SWaT: Mamba Encoder layer: 6, $d_{model}$: 128, $d_{hidden}$: 512
- SMD:  Mamba Encoder layer: 6, $d_{model}$: 128, $d_{hidden}$: 512
- MSL:  Mamba Encoder layer: 4, $d_{model}$: 64, $d_{hidden}$: 128
- SMAP: Mamba Encoder layer: 6, $d_{model}$: 64, $d_{hidden}$: 128

## Results
![images](./images/results.png)

- SensitiveMamba exceeds all baseline models for all datasets.
- Visualization of the reconstructed output can be found in `./sensitive_hue/model_states/{dataset_name}/reconstruction.png`.


## Acknowledgement 

We are grateful for the following awesome projects when implementing SensitiveMamba:

- [SensitiveMamba](https://github.com/yuesuoqingqiu/SensitiveHUE)
- [Mamba](https://github.com/state-spaces/mamba)
