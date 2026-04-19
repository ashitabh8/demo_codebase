# WiFi Arduino Python Basic Test README

## Overview
For this test:

- Python sends `[1, 2, 3]`
- Arduino adds `10` to each value
- Arduino sends back `[11, 12, 13]`

---

## What to change

There are only a few things you need to change.

### In the python file:
Just update the IP

### In the Arduino code

Update your WiFi credentials here:

```cpp
const char* SSID = "Ashitabh";
const char* PASSWORD = "ashitabh";


