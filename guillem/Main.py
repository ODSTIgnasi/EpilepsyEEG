import Load_and_Preprocessing_Data as lp
import Model_Training as MT
import Visualization as Vs
import ML_Classifiers_Training_Validation as mlc
import Baseline_models as Bm
import Time_Frequency_Domain_Analysis as TFD

CHB_MIT = 'chbmit'

dataset_choice = CHB_MIT
classification_type = "binary"

dataset_path = '../../dataset/'

if dataset_choice == CHB_MIT and classification_type == 'binary':
    eeg_data, eeg_label = lp.load_chbmit_data(dataset_path)
    history = MT.binary_model(eeg_data, eeg_label, dataset_choice)
    Vs.Visualization_plots(history)
    mlc.binary_ML_Classifiers(eeg_data, eeg_label,dataset_choice)
    Bm.baseline_methods(eeg_data, eeg_label)
    TFD.TF_Analysis(frequency_domain=256)
