import sys
import time

if __name__ == '__main__':
    i = 1
    for i in range(12):
        sys.stdout.write(str(i) + " lines of data\n")
        time.sleep(1)