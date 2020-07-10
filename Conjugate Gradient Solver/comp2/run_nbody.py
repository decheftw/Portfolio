import os
import sys

PATH = "./nbody"

N = 10 if len(sys.argv) == 1 else sys.argv[1]

for i in range(N):
	os.system(PATH + "| python parse_gpu_output.py")

