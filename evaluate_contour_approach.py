import pandas as pd
from contour_approach import detect_without_display
# Load the CSV file into a DataFrame
df = pd.read_csv('annotations.csv')

images = df['image']
counts = df['count']

correct_count = 0
mean_absolute_error = 0

for index, image in enumerate(images):
    #Run our method
    result = detect_without_display('data/'+image)
    #If the result of our method matches the label, increment the correct_count.
    if result == counts[index]:
        correct_count += 1
    mean_absolute_error += abs(counts[index] - result)

mean_absolute_error /= len(images)

accuracy = float(correct_count) / len(images)
print(f"Accuracy : {accuracy*100}%")
print(f"Mean absolute error : {mean_absolute_error}")