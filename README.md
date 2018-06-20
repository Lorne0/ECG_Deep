## CVD Detection by Deep ECG Learning

[Presentation Slides](https://docs.google.com/presentation/d/1mtZBkdHWctmorvy7p3cUPE1tPw5hvd8mwKBZHAtBm8s/edit?usp=sharing)

### Step 0: Setup

    pip install wfdb
    pip install PyWavelets

If you haven't installed Keras and Tensorflow, please install them by yourself. Then run

    mkdir result data

### Step 1: Download Data
Open your python and run

    import wfdb
    wfdb.dl_database('mitdb', dl_dir='data')
            
Wait for a few minutes to download the data.

### Step 2: Experiment Part I

In Part I, we compare the performance of different data preprocessing methods, including denoise, normalize, and augmentation. The detailed information is in the presentation slide. **NLRAV** and **NSVFQ** are two different kinds of labels of the diseases. Also see the slides for detailed information.

    python preprocess.py <NLRAV/NSVFQ> <denoise/no> <normalize/no> <augment/no> <random_seed>
        
For example, if you want to try all the preprocessing methods for NLRAV setting, and use 1229 as the random seed for splitting train/valid/test data,

    python preprocess.py NLRAV denoise normalize augment 1229
                
Then you'll get **data_NLRAV.pk** under the directory. Then run
                    
    python train.py <NLRAV/NSVFQ> <save_file_name>
                            
You'll get **save_file_name** in /result. To easily run on different random seeds for multiple times, we save the result in the **save_file_name**, you can use

    python check_result.py save_file_name
                                    
to count the average performance in the **save_file_name**.

#### Result:

|NLRAV            |SE    |ACC   |AUC   |SP    |vSE   |vACC  |
|-----------------|------|------|------|------|------|------|
|default          |0.9790|0.9881|0.9865|0.9940|0.9774|0.9882|
|denoise          |0.9772|0.9882|0.9859|**0.9947**|0.9760|0.9882|
|normalize        |0.9810|0.9885|0.9874|0.9938|0.9807|0.9887|
|denoise+normalize|0.9823|**0.9891**|**0.9880**|0.9937|0.9817|0.9896|
|augment          |0.9806|0.9839|0.9843|0.9879|0.9804|0.9842|
|denoise+augment  |0.9802|0.9842|0.9846|0.9889|0.9796|0.9844|
|normalize+augment|0.9844|0.9863|0.9871|0.9898|0.9841|0.9869|
|all              |**0.9853**|0.9864|0.9876|0.9898|0.9846|0.9871|

|NSVFQ            |SE    |ACC   |AUC   |SP    |vSE   |vACC  |
|-----------------|------|------|------|------|------|------|
|default          |0.9481|0.9836|0.9707|0.9934|0.9544|0.9840|
|denoise          |0.9464|0.9834|0.9698|0.9932|0.9532|0.9841|
|normalize        |0.9450|**0.9843**|0.9698|**0.9945**|0.9535|0.9853|
|denoise+normalize|0.9457|0.9835|0.9696|0.9936|0.9543|0.9847|
|augment          |**0.9591**|0.9786|0.9722|0.9852|0.9631|0.9792|
|denoise+augment  |0.9565|0.9777|0.9706|0.9848|0.9637|0.9782|
|normalize+augment|0.9572|0.9801|0.9723|0.9874|0.9643|0.9806|
|all              |0.9572|0.9805|**0.9725**|0.9878|0.9639|0.9806|

### Step 3: Experiment Part II
In this part, we want to compare the performance of different deep learning models.
Like previous step, run

    python preprocess2.py <NLRAV/NSVFQ> <random_seed>
        
to get **data2_<NLRAV/NSVFQ>.pk**. Then run
            
    python train2.py <NLRAV/NSVFQ> <model_type> <save_file_name>
                    
model\_type: 1D, 1D-small, 1D-large, LSTM, BiLSTM, Dense

### Step 3-1: 2D model
Since the preprocessing of 2D is more complex, we split it from the previous step. Run

    python preprocess_2D.py <NLRAV/NSVFQ> <random_seed>

to get **data_2D_<NLRAV/NSVFQ>.pk** . It takes a few minutes to finish this preprocessing. Then run
        
    python train_2D.py <NLRAV/NSVFQ> <save_file_name>

#### Result:
|NLRAV   |SE    |ACC   |AUC   |SP    |vSE   |vACC  |
|--------|------|------|------|------|------|------|
|1D      |0.9869|0.9938|0.9922|0.9974|0.9900|0.9948|
|1D-small|0.9852|0.9929|0.9912|0.9972|0.9900|0.9939|
|1D-large|**0.9878**|0.9932|0.9922|0.9966|0.9908|0.9940|
|LSTM    |0.9875|0.9936|0.9923|0.9971|0.9911|0.9944|
|BiLSTM  |**0.9878**|**0.9943**|**0.9926**|0.9974|0.9919|0.9953|
|Dense   |0.9828|0.9917|0.9895|0.9962|0.9862|0.9916|
|2D      |0.9875|0.9938|0.9925|**0.9975**|0.9872|0.9938|

|NSVFQ   |SE    |ACC   |AUC   |SP    |vSE   |vACC  |
|--------|------|------|------|------|------|------|
|1D      |**0.9753**|0.9917|0.9853|0.9954|0.9765|0.9916|
|1D-small|0.9736|0.9914|0.9846|0.9955|0.9738|0.9913|
|1D-large|0.9726|0.9905|0.9837|0.9949|0.9741|0.9902|
|LSTM    |0.9677|0.9923|0.9824|**0.9970**|0.9742|0.9930|
|BiLSTM  |0.9748|**0.9930**|**0.9859**|**0.9970**|0.9767|0.9932|
|Dense   |0.9598|0.9908|0.9784|0.9969|0.9600|0.9907|
|2D      |0.9648|0.9905|0.9804|0.9960|0.9683|0.9908|

