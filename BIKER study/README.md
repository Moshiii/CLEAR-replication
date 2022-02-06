# BIKER study
---

This folder contains the modified BIKER script for easier data loading and ablation study.

Note that the modification is only limited to data representation and preinting. the algorithm such as metrics calculation is not changes.


test_new_data_loader.py loads CSV file instead of hdf5 file.
test_new_data_loader_test_class.py is the test in class level.
test_new_data_loader_test_first_stage.py is the test for only the first stage of BIKER.
test_new_data_loader_test_first_stage_class.py is the test for only the first stage of BIKER at class level.

please overwrite the algorithm folder to original BIKER before using the script.