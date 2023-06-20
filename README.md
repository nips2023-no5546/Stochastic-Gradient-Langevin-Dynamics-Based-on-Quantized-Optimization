READ ME
===

[toc] 

## URL

~~~
https://github.com/nips2023-no5546/Stochastic-Gradient-Langevin-Dynamics-Based-on-Quantized-Optimization
~~~


## Package Specificaion

### Fundamental Package 

| File Name         | Specification |
|-------------------|---------------|
| nips_quant.py     | Quantization Processing |
| Read_Data.py      | Data Read (MNIST, CIFAR10, CIFAR100, ImageNet:146GB) |
| torch_learning.py | Optimizer and Scheduler |
| torch_resnet.py   | ResNet-50               |
| torch_SmallNet.py | LeNet                   |
| torch_nn02.py     | Main Operation          |


### Service Package 
- Common Service Package

| File Name         | Specification |
|-------------------|---------------|
| my_debug.py       | For Debugging |

- Specific Service Packgae

| File Name         | Specification |
|-------------------|---------------|
| service_*****.py  | ***** Service |

- Argument File for multiple batch learning process

| File Name         | Specification |
|-------------------|---------------|
| argdata.dat  | multiple processing |

- YAML for setting service_process.py
No specific modifications are needed for config_service.yaml and config_learning.yaml files. Any hyperparameter-related changes should be made in the config_quantization.yaml file. For more details, please refer to the Supplementary Material.

| File Name         | Corresponding Python File |
|-------------------|---------------|
| config_service.yaml  | setting service_process.py |
| config_learning.yaml | torch.learning.py / torch_nn02.py|
| config_quantization.yaml  | nips_quant.py.py |

## Torch_nn02.py Usage

### Set python Environment
In Ubuntu, you can use the following steps. 
First, check the Python version, and make sure to select version 3.10. 
Note that the library packages may vary depending on the Python version. 
It is recommended to use a virtual environment to avoid conflicts between packages.
~~~
sudo update-alternatives --config python3
~~~

### Basic IO

#### Input Files 

| Data Set          | Parameter|  Notes |
|-------------------|----------|--------|
| MNIST Data Set    | MNIST    |        |
| CIFAR10 Data Set  | CIFAR10  |        |
| CIFAR100 Data Set | CIFAR100 |        |
| ImageNet          | ImageNet | Not Available but Implemented |

#### output Files 

| Spec | Format | Example|
|---|---|---|
| Neural Network File | torchnn02+ Model name + Algorithm name  + Epoch.pt | torch_nn02ResNetAdam.pt |
| Operation File  | operation + Model name + Algorithm name + Epoch.txt | operation_ResNetAdam100.txt |
| Error Trend File| error_ + Model name + Algorithm name + Epoch.pickle | error_ResNetAdam100.pickle |

### For Cifar-10 Data Set 

#### LeNet
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n LeNet -g 1
~~~

#### ResNet 
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n ResNet -g 1 
~~~

#### Note
- When you don't use the CUDA but CPU mode, you don't set 'g' option or '-g 0'.
- For ResNet, you should set '-g 1' option, because the size of the network is so large. 

## Quantization Algorithm

- QSGD
~~~
python torch_nn02.py -m QSGD -d CIFAR10 -e 100 -n ResNet -g 1
~~~

- QtAdam
~~~
python torch_nn02.py -m QtAdam -d CIFAR10 -e 100 -n ResNet -g 1
~~~

## Torch_testNN.py Usage
- The Data Set,  Network Spec (*.pt file), Network name should be specified 
- The test program plot an error trend when an error file exists

### Sinlge Processing 
~~~
python torch_testNN.py -d CIFAR10 -n ResNet -ng 1 -e error_ResNetAdam15.pickle -m torch_nn02ResNetAdam.pt 
~~~

### Batch Processing
- The test program takes the processing argument from the "argdata.dat" on the working directory
~~~
python torch_nn02.py -bt 1
~~~


### Scheduler Option '-sn'
- $t$ denotes the number of the epoch.

| Scheduler Name | Parameter  | Specification |
|----------------|------------|---------------|
| ConstantLR     | Default    |               |
| LambdaLR       | LambdaLR   | $0.95^{t}$    |
| ExponentialLR  | exp        |               |
| CyclicLR       | cyclicLR   |
| CosineAnnealingWarmRestarts | CAWR or CosineAnnealingWarmRestarts | $\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}}) \left( 1 + \cos \left( \frac{T_{\text{cur}}}{T_{\text{max}}} \pi \right) \right)$   |
| CustomCosineAnnealingWarmUpRestarts | CCAWR or CustomCosineAnnealingWarmRestarts |   |


## Appendix

#### Torch_nn02.py Help Message

~~~
(py3.10.0) sderoen@sderoen-System-Product-Name:~/Works/python_work_2023/nips2023_work$ python torch_nn02.py -h
usage: torch_nn02.py [-h] [-g DEVICE] [-l LEARNING_RATE] [-e TRAINING_EPOCHS] [-b BATCH_SIZE] [-f MODEL_FILE_NAME] [-m MODEL_NAME] [-n NET_NAME]
                     [-d DATA_SET] [-a AUTOPROC] [-pi PROC_INDEX_NAME] [-rl NUM_RESNET_LAYERS] [-qp QPARAM] [-rd RESULT_DIRECTORY]
                     [-sn SCHEDULER_NAME] [-ev EVALUATION] [-bt BATCHPROC] [-ag ARG_DATA_FILE] [-ntf NOTI_TARGET_FILE] [-lp LRNPARAM]

options:
  -h, --help            show this help message and exit
  -g DEVICE, --device DEVICE
                        Using [(0)]CPU or [1]GPU
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning_rate (default=0.001)
  -e TRAINING_EPOCHS, --training_epochs TRAINING_EPOCHS
                        training_epochs (default=15)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch_size (default=100)
  -f MODEL_FILE_NAME, --model_file_name MODEL_FILE_NAME
                        model file name (default='torch_nn02_CNN.pt')
  -m MODEL_NAME, --model_name MODEL_NAME
                        model name 'SGD', 'Adam', 'AdamW', 'ASGD', 'NAdam', 'RAdam' (default='Adam')
  -n NET_NAME, --net_name NET_NAME
                        Network name 'CNN', 'LeNet', 'ResNet' (default='LeNet')
  -d DATA_SET, --data_set DATA_SET
                        data set 'MNIST', 'CIFAR10' (default='MNIST')
  -a AUTOPROC, --autoproc AUTOPROC
                        Automatic Process without any plotting [(0)] plotting [1] Automatic-no plotting
  -pi PROC_INDEX_NAME, --proc_index_name PROC_INDEX_NAME
                        Process Index Name. It is generated automatically (default='')
  -rl NUM_RESNET_LAYERS, --num_resnet_layers NUM_RESNET_LAYERS
                        The number of layers in a block to ResNet (default=5)
  -qp QPARAM, --QParam QPARAM
                        Quantization Parameter, which is read from the config_quantization.yaml file (default=0)
  -rd RESULT_DIRECTORY, --result_directory RESULT_DIRECTORY
                        Directory for Result (Default: ./result)
  -sn SCHEDULER_NAME, --scheduler_name SCHEDULER_NAME
                        Learning rate scheduler (Default : Constant Learning)
  -ev EVALUATION, --evaluation EVALUATION
                        Only Inference or Evaluation with a learned model [(0)] Training and Inference [1] Inference Only
  -bt BATCHPROC, --batchproc BATCHPROC
                        Batch Processing with 'argdata.dat' file or Not [(0)] Single processing [1] Multiple Processing
  -ag ARG_DATA_FILE, --arg_data_file ARG_DATA_FILE
                        Argument data File for Batch Processing (default: 'argdata.dat')
  -ntf NOTI_TARGET_FILE, --noti_target_file NOTI_TARGET_FILE
                        Target file for Notification (default: 'work_win01.bat')
  -lp LRNPARAM, --LrnParam LRNPARAM
                        Learning Parameter, which is read from the config_learning.yaml file (default=0)
~~~

### Examples

- When the learning and the testing data in a specific directory, we set the arguments for the test program such that  

~~~
(python01)>python torch_testNN.py -d CIFAR10 -m ./Result_data/LeNet/torch_nn02LeNetAdam100.pt -n LeNet -e ./Result_data/LeNet/error_LeNetAdam100.pickle -p 100 -ng 0
~~~


## Service_process.py 

- Service process provides the function for gathering the learning and the testing of each algorithm.
- After learning, it plots and writes summary files from the result files generated by the test program.


### help 
~~~
(py3.10.0) D:\Work_2023\nips2023_work>python service_process_board.py -h
usage: test pytorch_inference [-h] [-rd RESULT_DIR] [-t TRAINING] [-pr PROCESSING] [-gp GRAPHIC] [-ea EQUAL_ALGORITHM]
                              [-el EQUAL_LEARNING_RATE] [-cf CONFIG]

====================================================
service_process_board.py :
                    Written by Jinwuk @ 2023-04-25
====================================================
Example : service_process_board.py

options:
  -h, --help            show this help message and exit
  -rd RESULT_DIR, --result_dir RESULT_DIR
                        Depict result directory (Default: result)
  -t TRAINING, --training TRAINING
                        [0] test [(1)] training
  -pr PROCESSING, --processing PROCESSING
                        [(0)] single file processing [1] multiple files processing
  -gp GRAPHIC, --graphic GRAPHIC
                        [0] no plot graph [(1)] active plot graph
  -ea EQUAL_ALGORITHM, --equal_algorithm EQUAL_ALGORITHM
                        [(0)] no equal_algorithm [1] equal_algorithm
  -el EQUAL_LEARNING_RATE, --equal_learning_rate EQUAL_LEARNING_RATE
                        [0] no equal_learning_rate [1] equal_learning_rate
  -cf CONFIG, --config CONFIG
                        Config file for Service Process (default) config_service.yaml
~~~

