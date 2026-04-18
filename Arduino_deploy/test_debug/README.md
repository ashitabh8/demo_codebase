# WiFi Arduino Python Basic Test README

## Overview

This is a basic WiFi communication test between a Python client and an Arduino.

The goal is to verify that:

- Python can connect to the Arduino over WiFi
- Python can send a small dummy array of numbers
- Arduino can receive the numbers
- Arduino can modify the numbers
- Arduino can send the modified numbers back
- Python can print the returned result

For this test:

- Python sends `[1, 2, 3]`
- Arduino adds `10` to each value
- Arduino sends back `[11, 12, 13]`

---

## What to change

There are only a few things you need to change.

### In the Arduino code

Update your WiFi credentials here:

```cpp
const char* SSID = "Ashitabh";
const char* PASSWORD = "ashitabh";

In the python file:
Just update the IP