#!/bin/bash

# Read the file and extract the value of 'location'
drugname=$(grep 'Drug_Name ' ./input_deck2.txt | awk -F'= ' '{print $2}')
concentration=$(grep 'Concentrations ' ./input_deck2.txt | awk -F'= ' '{print $2}')
formatted_concentration=$(printf "%.2f" "$concentration")

# echo "recompile"
# cd ..
# make clean all
# cd bin

echo "run in-silico"
"./drug_sim" -input_deck ./input_deck2.txt -hill_file ./drugs/$drugname/IC50_samples.csv -herg_file ./herg/$drugname.csv

# echo "---------------------"
# echo "run post-processing"

# new_value="1"
# sed -i.old "s/^\(Is_Post_Processing = \).*/\1$new_value/" "./input_deck2.txt"

# "./drug_sim" -input_deck ./input_deck2.txt -hill_file ./drugs/$drugname/IC50_samples10.csv -init_file ./result/$formatted_concentration.csv -herg_file ./herg/$drugname.csv
# new_value="0"
# sed -i.old "s/^\(Is_Post_Processing = \).*/\1$new_value/" "./input_deck2.txt"
