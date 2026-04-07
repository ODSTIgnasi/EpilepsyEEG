# EpilepsyEEG
UAB RICSE 2025 Deep Learning Challenge

#Database
https://physionet.org/content/chbmit/1.0.0/

# MVP
Preprocessing:
Super windos: 
0. Justify not using transformers
1. Size of inner window(ex. 1 seg)
2. Normalize data(ex. data entry, number of channels)
3. Sizer of the superwindow (ex. 3 inner windows)
4. Adapt the code of the data augmentation for this new configuration(only apply data augmentation if there is at least one sizure inner window)
5. Compare LSTM and bi-LSTM

# MORE
Same pipeline, different strategy for data augmentation
Feature selection:
1. Preprocessing try Transforms(ex. DWT)
2. Cluster channels(ex., PCA) group by similarity
3. Channel reeduction by impact

To run in the cluster do a sbatch run.sh
