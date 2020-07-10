import sys

# Get the last line, everything after ':', then remove whitespace
time_str = sys.stdin.readlines()[-1].split(":")[-1].strip()

# time = float(time_str.split(" ")[0])
print(time_str)
