import csv

# === File paths ===
input_file = 'Yotel job_items variants.txt'   # Tab-delimited .txt file
output_file = 'Yotel job_items variants.csv' # Output CSV file

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile)

    for row in reader:
        writer.writerow(row)

print(f"Converted '{input_file}' to '{output_file}' (no headers assumed).")
