# oe_npx_select_channels
Connect to Open Ephys and select channels on a Neuropixels probe.

Currently only implemented for Npx 2.0. See notebook for example usage.

## Visualize your channel layouts:
<img width="1114" alt="image" src="https://github.com/user-attachments/assets/0729f003-e933-42e2-8b45-59304f46f430" />

## Automatically loop through various layouts:
<img width="1034" alt="image" src="https://github.com/user-attachments/assets/dc88879a-243f-41fc-918a-bf9228f390f4" />

## Example usage
```python
import time
from oe_npx_selector.npx_selector import Npx2_Channel_Selector
npx = Npx2_Channel_Selector()
npx.oe_connect()  # this will raise an error unless the Open Ephys GUI is already open with an Npx2 probe attached
npx.show_available_configs()

# Example: loop through each of the 12 banks on the probe and record each for 60 seconds
rec_time_s = 60
for shank in range(4):
    for bank in range(3):
        npx.gui.set_start_new_dir()  # npx.gui is an instance of OpenEphysHTTPServer
        npx.set_electrode_config('linear_single_shank', shank=shank, bank=bank)  # this changes the stored config in the Python object
        npx.oe_select_current_channels()  # this actually sends the channel ids to OE
        time.sleep(1)
        npx.gui.record(rec_time_s)  # blocks until recording is done
```
