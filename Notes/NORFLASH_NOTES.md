# Memory Management on the Giga R1: Internal vs. NOR Flash

Task: Benchmark NOR Flash (each weight matrix individually can fit into SRAM)
1. write PC to NORFlash
2. write NORFlash to SDRAM
3. write NORFlash to SRAM

I put the task before the explanation incase you miss that last part :p

## Why your old approach worked (internal flash)
When you put data in a header file as a **C array**, the compiler bakes that data directly into your program binary. When you upload the program, it all lands in the MCU's **internal flash** — which is where the processor naturally looks for code and `const` data. 

So it "just works" because the data lives where the CPU already knows how to find it.

---

## Why that doesn't work for NOR flash
The NOR flash on the Giga R1 is a **separate chip** — it's an external memory device sitting on a **QSPI bus** (a fast serial interface). The processor doesn't automatically know your data is there, and it doesn't automatically look there when your program runs. 

> **Analogy:** Think of internal flash like a book the CPU is already holding, while the NOR flash is a book on a shelf across the room. You have to explicitly tell the CPU: *"Go to that shelf, open that book, read page 47."*

---

## So what do you actually need to do?
There are essentially two steps:

1.  **Write your data to the NOR flash:** This isn't like uploading your sketch — the normal Arduino upload process only touches internal flash. You need code that talks to the QSPI NOR flash chip, erases a region, and then writes your data bytes into it. 
2.  **Read that data back:** When your main program runs, you need to read that data back from the NOR flash. Your code sends commands over the QSPI bus saying *"give me the bytes starting at address X, length Y"* and copies them into RAM where you can use them.

### The "erase-before-write" thing is important
This is the single biggest conceptual difference from just stuffing data in a header. Flash memory (both NOR and NAND) has a physical quirk: it can only flip bits from **1 to 0**. To flip them back to **1** (i.e., to rewrite), you must **erase**, which resets a whole sector to all 1s. 

* **The cycle:** Erase sector $\rightarrow$ Write data $\rightarrow$ Read data later.
* **The Constraint:** You can't just overwrite one byte — you erase a whole sector (typically **4KB** at a time), then write. Think of it like a whiteboard you must wipe clean before writing again.

---

## What library/driver handles this?
On the Giga R1 (which runs on Mbed OS underneath), there's typically a QSPI flash driver that handles the low-level bus communication. The `QSPIFBlockDevice` class in Mbed gives you a block device interface with simple methods:

* `read()`
* `program()`
* `erase()`

Arduino's ecosystem for the Giga wraps some of this for easier access.

---

## The mental model, in summary:

* **Data Source:** Your data exists somewhere (a C array, a file, etc.).
* **The Writer:** You run a "writer" sketch that takes that data and programs it into the NOR flash chip at a **known address**.
* **The Reader:** Your "main" sketch, when it needs the data, reads from that known address on the NOR flash into RAM, and then uses it normally.
* **Convention:** The known address is just a convention you pick — *"I'll always store my lookup table starting at address 0x00000000"* — and both the writer and reader agree on that.

**Note:** You might write the data once (like a one-time setup step), and then your main program just reads it every time it boots. Either way, the read path and write path are **explicit operations** in your code, not something the compiler/linker handles for you.


# Code Instructions

## Code for PC -> NORFlash (via Serial Monitor)
`flash_writer.ino`
Main idea is that the Arduino and python code on PC "talk" to each other.
1) Arduino tells Python script its ready to recieve data
2) Python says a Command first : Erase, Write, or Verify. You send starting address and the length you want to operate on so for 4KB you will have length 4096

## Read from NOR Flash write to SRAM
`flash_reader_example.ino`
1) You start from a specific address in NOR flash (e.g., 0x00000000) and read the data into a RAM buffer using the library QSPIFBlockDevice.h -> this has a function read() which works something like read(destination, source, numBytes)
2) The data is now in SRAM and can be accessed like any regular C array.

For verification you can sum the vaues and see whether it matches your orignal array - or some random values in the array.

