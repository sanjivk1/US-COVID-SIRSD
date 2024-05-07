import concurrent.futures
import sys
import traceback

import numpy as np
import pandas as pd
import datetime
import math
import os
import matplotlib.pyplot as plt
from SIRfunctions import SIRG_combined


def crash_fun(i):
	# print((4-i)/i)
	print(i / (4 - i))
	return i


def exception_catching():
	with concurrent.futures.ProcessPoolExecutor() as executor:
		try:
			results = [executor.submit(crash_fun, i) for i in range(5)]
		except Exception as e:
			print('summit', e)
		try:
			for f in concurrent.futures.as_completed(results):
				n = f.result()
				print(n)
		except:
			traceback.print_exception(*sys.exc_info())


def catching_wrap():
	with concurrent.futures.ProcessPoolExecutor() as executor:
		[executor.submit(exception_catching) for _ in range(5)]


def main():
	# catching_wrap()
	exception_catching()
	return


if __name__ == "__main__":
	main()
