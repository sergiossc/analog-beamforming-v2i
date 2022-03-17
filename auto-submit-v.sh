executable = run_lloyd.py
log = run_lloyd.$(Cluster).$(Process).out
error = run_lloyd.$(Cluster).$(Process).err
should_transfer_files = Yes
transfer_input_files = utils.py, s000-test_set_4x4.npy, s000-test_set_4x4.npy
when_to_transfer_output = ON_EXIT
arguments = s000-test_set_4x4.npy random_from_samples 2 mse 1000 /home/snow/analog-beamforming-v2i/results/s000-test_set_4x4 3e67b4cb-2255-4b1a-9f10-aa7a4e9c8d71 58314 
queue
arguments = s000-test_set_4x4.npy random_from_samples 4 mse 1000 /home/snow/analog-beamforming-v2i/results/s000-test_set_4x4 e6a5ed2b-04bc-4725-9415-9978cbb0438c 71522 
queue