# ConvRNN in Pytorch
This codes are for `Spatial RNN`. Both `convolutional LSTM` and `convolutional GRU` are implemented in `Pytorch`. Compuation under `GPU/CPU` are both supported. 

## Usage
For `LSTM`,
```python
import ConvLSTM
layer1 = ConvLSTM.ConvLSTMCell(input_shape=(channel, height, width),
                               hidden_c=hidden_channel_of_LSTM,
                               kernel_shape=(kernel_h, kernel_w), 
                               isGPU=True)
```

For `GRU`, 
```python
import ConvGRU
layer1 = ConvGRU.ConvGRUCell(input_shape=(channel, height, width),
                             hidden_c=hidden_channel_of_GRU,
                             kernel_shape=(kernel_h, kernel_w), 
                             isGPU=True)
```

## Environment
This codes are tested under `Ubuntu 18.04`, `Python 3.5`, `Pytorch 0.4.1`. 
