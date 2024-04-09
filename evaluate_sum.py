import pandas as pd
from hough_approach import detect_coins_without_display, calculate_amount

import re

# Read the CSV file
df = pd.read_csv("annotations.csv")

# Define a function to extract numeric value from the string
def extract_numeric_value(text):
    # Match the pattern for euros and centimes
    match = re.search(r'(\d+) euros? (\d+) centimes?', text)
    if match:
        euros = int(match.group(1))
        centimes = int(match.group(2))
        return euros + centimes / 100
    else:
        # If no match found, return NaN
        return float('nan')

# Apply the function to the 'value' column
df['value_numeric'] = df['value'].apply(extract_numeric_value)

# Print the resulting dataframe
print(df)



images = df['image']
counts = df['count']
sums = df['value_numeric']

threshold = 0.2

correct_count = 0
mean_absolute_error = 0

for index, image in enumerate(images):
    #Run our method
    result = calculate_amount('data/'+image)
    #If the result of our method matches the label, increment the correct_count.
    if result[4] >= sums[index] - threshold and result[4] <= sums[index] + threshold:
        correct_count += 1
    mean_absolute_error += abs(counts[index] - result[4])

accuracy = float(correct_count) / len(images)
mean_absolute_error /= len(images)
print(f"Accuracy : {accuracy*100}%")
print(f"Mean absolute error : {mean_absolute_error}")
