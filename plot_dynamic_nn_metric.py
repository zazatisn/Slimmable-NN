import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

syntax = "python3 plot_dynamic_py <csv_filename>"

if len(sys.argv) != 2:
  print("ERROR: Incorrect number of arguments!")
  print(f"Rerun: {syntax}")
  sys.exit(-1)
else:
  csv_filename = sys.argv[1]
  if not os.path.exists(csv_filename):
    print(f"ERROR: File {csv_filename} does not exist!")
    print(f"Rerun: {syntax}")
    sys.exit(-2)

df = pd.read_csv(csv_filename, header = 0)
print(df)

threshold_values = df['Threshold'].values
increment_values = df['Average Increment'].values
accuracy_values = df['Test accuracy'].values

plt.figure(0)
plt.plot(threshold_values, increment_values)
plt.xlabel("Threshold")
plt.ylabel("Average Increments")
plt.figure(1)
plt.plot(threshold_values, accuracy_values)
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.show()