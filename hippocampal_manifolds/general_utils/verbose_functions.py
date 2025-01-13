"""
Created on Tue Nov 26 2024

@author: JulioEI
"""

import timeit
from datetime import datetime


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
            

def print_time_verbose(local_starttime, global_starttime= None):
    if isinstance(global_starttime, type(None)):
        print(f'\t{datetime.now():%Y-%m-%d %H:%M}', end=' ')
        print(f'- Running Time: {timeit.default_timer()-local_starttime:.2f}s')
    else:
        print(f'\t{datetime.now():%Y-%m-%d %H:%M}', end=' ')
        print(f'- Running Time: {timeit.default_timer()-local_starttime:.2f}s', end= ' ')
        print(f'({timeit.default_timer()-global_starttime:.2f}s)')
    
    