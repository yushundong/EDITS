# EDITS
Open-source code for "EDITS: Modeling and Mitigating Data Bias for Graph Neural Networks".

## Citation

If you find it useful, please cite our paper. Thank you!

```
@inproceedings{dong2022edits,
  title={Edits: Modeling and mitigating data bias for graph neural networks},
  author={Dong, Yushun and Liu, Ninghao and Jalaian, Brian and Li, Jundong},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1259--1269},
  year={2022}
}
```

## Environment
Experiments are carried out on a Titan RTX with Cuda 10.1. 

Library details can be found in requirements.txt.

Notice: Cuda is enabled for default settings.

## Usage
Default dataset for node classification is bail. Pre-processed datasets (default for bail) are provided in *pre_processed*.
Use as
```
python train.py
```
for preprocessing and 
```
python classification.py
```
for downstream node classification task.

## Log example for node classification on German

(1) Directly do node classification task with the orginal attributed network (set args.preprocessed_using as 0):
```
python classification.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.17086205277713312
Average of all Wasserstein distance value across feature dimensions: 0.006328224176930857
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.28068607580843574
Average of all Wasserstein distance value across feature dimensions: 0.01039578058549762
****************************************************************************
100%|██████████| 1000/1000 [00:19<00:00, 51.55it/s]
Optimization Finished!
Total time elapsed: 19.3985s
Delta_{SP}: 0.25681942171303873
Delta_{EO}: 0.19327731092436973
F1: 0.8131868131868131
AUC: 0.7433142857142857
```
(2-1) Debiasing attributed network with EDITS:
```
python train.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.17086205277713312
Average of all Wasserstein distance value across feature dimensions: 0.006328224176930857
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.28068607580843574
Average of all Wasserstein distance value across feature dimensions: 0.01039578058549762
****************************************************************************
100%|██████████| 500/500 [00:18<00:00, 26.48it/s]
Preprocessed datasets saved.
```
(2-2) Carry out node classification task with the output of EDITS (set args.preprocessed_using as 1):
```
python classification.py
```
```
****************************After debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.05468372589251738
Average of all Wasserstein distance value across feature dimensions: 0.0023775532996746693
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.09379225766248622
Average of all Wasserstein distance value across feature dimensions: 0.004077924246195052
****************************************************************************
100%|██████████| 1000/1000 [00:16<00:00, 60.58it/s]
Optimization Finished!
Total time elapsed: 16.5090s
Delta_{SP}: 0.0008183306055645767
Delta_{EO}: 0.0010504201680672232
F1: 0.8116710875331565
AUC: 0.7140571428571428
```

## Log example for node classification on Credit

(1) Directly do node classification task with the orginal attributed network (set args.preprocessed_using as 0):
```
python classification.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.032023121509008844
Average of all Wasserstein distance value across feature dimensions: 0.0024633170391545264
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.05786050779584359
Average of all Wasserstein distance value across feature dimensions: 0.004450808291987968
****************************************************************************
100%|██████████| 1000/1000 [04:38<00:00,  3.59it/s]
Optimization Finished!
Total time elapsed: 278.3364s
Delta_{SP}: 0.13269706500135625
Delta_{EO}: 0.10837075709847321
F1: 0.8175731450059617
AUC: 0.734433504547214
```
(2-1) Debiasing attributed network with EDITS:
```
python train.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.032023121509008844
Average of all Wasserstein distance value across feature dimensions: 0.0024633170391545264
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.05786050779584359
Average of all Wasserstein distance value across feature dimensions: 0.004450808291987968
****************************************************************************
100%|██████████| 500/500 [05:37<00:00,  1.48it/s]
Preprocessed datasets saved.
```
(2-2) Carry out node classification task with the output of EDITS (set args.preprocessed_using as 1):
```
python classification.py
```
```
****************************After debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.02081495033963641
Average of all Wasserstein distance value across feature dimensions: 0.0023127722599596014
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.035380875298436004
Average of all Wasserstein distance value across feature dimensions: 0.003931208366492889
****************************************************************************
100%|██████████| 1000/1000 [04:49<00:00,  3.45it/s]
Optimization Finished!
Total time elapsed: 289.9188s
Delta_{SP}: 0.1008967181280005
Delta_{EO}: 0.0756249315646026
F1: 0.8186898763169949
AUC: 0.7305752326134218
```

## Log example for node classification on Bail

(1) Directly do node classification task with the orginal attributed network (set args.preprocessed_using as 0):
```
python classification.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.017159402967899848
Average of all Wasserstein distance value across feature dimensions: 0.0009533001648833249
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.01976526986139568
Average of all Wasserstein distance value across feature dimensions: 0.0010980705478553154
****************************************************************************
100%|██████████| 1000/1000 [01:27<00:00, 11.39it/s]
Optimization Finished!
Total time elapsed: 87.7692s
Delta_{SP}: 0.07623766006332078
Delta_{EO}: 0.05521238465644773
F1: 0.7910319057200345
AUC: 0.8713191402411585
```

(2-1) Debiasing attributed network with EDITS:
```
python train.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.017159402967899848
Average of all Wasserstein distance value across feature dimensions: 0.0009533001648833249
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.01976526986139568
Average of all Wasserstein distance value across feature dimensions: 0.0010980705478553154
****************************************************************************
100%|██████████| 100/100 [00:34<00:00,  2.93it/s]
Preprocessed datasets saved.
```
(2-2) Carry out node classification task with the output of EDITS (set args.preprocessed_using as 1):
```
python classification.py
```
```
****************************After debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.006414602640346333
Average of all Wasserstein distance value across feature dimensions: 0.000458185902881881
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.008129858181420972
Average of all Wasserstein distance value across feature dimensions: 0.0005807041558157836
****************************************************************************
100%|██████████| 1000/1000 [01:17<00:00, 12.92it/s]
Optimization Finished!
Total time elapsed: 77.3908s
Delta_{SP}: 0.05741161830379604
Delta_{EO}: 0.03746663616258683
F1: 0.7984610831606985
AUC: 0.8914640940634824
```
