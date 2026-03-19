import Load_and_Preprocessing_Data as lp

CHB_MIT = 'chbmit'

dataset_choice = CHB_MIT
classification_type = "binary"


if dataset_choice == CHB_MIT and classification_type == 'binary':
    eeg_data, eeg_label = lp.load_chbmit_data()
