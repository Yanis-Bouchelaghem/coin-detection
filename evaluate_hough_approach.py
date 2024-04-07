import pandas as pd
from hough_approach import detect_coins_without_display
# Load the CSV file into a DataFrame
df = pd.read_csv('annotations.csv')

images = df['image']
counts = df['count']

correct_count = 0

for index, image in enumerate(images):
    #Run our method
    result = detect_coins_without_display('data/'+image)
    #If the result of our method matches the label, increment the correct_count.
    if result == counts[index]:
        correct_count += 1


accuracy = float(correct_count) / len(images)
print(f"Accuracy : {accuracy*100}%")