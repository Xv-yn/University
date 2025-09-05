#!/bin/bash

# Compile the C file
gcc main.c -o main -pthread

# If compilation was successful, run it
if [ $? -eq 0 ]; then
    ./main
    # Delete the binary after running
    rm main
else
    echo "Compilation failed!"
fi

