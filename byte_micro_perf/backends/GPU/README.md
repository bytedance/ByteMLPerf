## Basic components
### [torch](https://pytorch.org/)
prefer torch==2.5.1 cuda==12.4
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

## Third party libraries
### [flash-attention](https://github.com/Dao-AILab/flash-attention)
```bash
# normal installation
python setup.py install

# for hopper gpu, could install flash_attention_3
cd hopper
python setup.py install
```

### [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
```bash
python setup.py install
```

### [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
```bash
python setup.py install
```