# Ethos-U core software

## Building

The core software is built with CMake. It is recommended to build out of tree like illustrated below.

```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=<toolchain> -DCMAKE_SYSTEM_PROCESSOR=cortex-m<nr><features>
$ make
```

Available build options can be listed with `cmake -LH ..`.

Supported CPU targets are any of the Cortex-M processors with any of the supported features, for example cortex-m33+nodsp+nofp. A toolchain file is required to cross compile the software.
