import Load_and_Preprocessing_Data as lp
import Model_Training as MT
import Visualization as Vs

CHB_MIT = 'chbmit'

dataset_choice = CHB_MIT
classification_type = "binary"

dataset_path = '../../dataset/'

if dataset_choice == CHB_MIT and classification_type == 'binary':
    eeg_data, eeg_label = lp.load_chbmit_data(dataset_path)
    history = MT.binary_model(eeg_data, eeg_label)
    Vs.Visualization_plots(history)
