#!/bin/bash

# List of filenames to be removed
filenames=("1914" "2558" "1794" "1814" "2557" "2965" "2522" "2130" "2260" "2129"
    "2284" "2681" "2112" "2145" "1022" "3073" "1826" "1365" "2538" "1889"
    "2238" "2949" "2125" "2113" "2315" "2682" "1002" "1793" "1938" "2335"
    "2701" "2128" "977" "2209" "1896" "2659" "1849" "1522" "2123" "1382"
    "2280" "2151" "2527" "1044" "1832" "3136" "2743" "2134" "1910" "2300"
    "2210" "2111" "1361" "2281" "1518")

# Directory containing the files
directory="./sl_data"

# Loop through each filename and remove it
for filename in "${filenames[@]}"; do
    rm "${directory}/${filename}.npy"
done

