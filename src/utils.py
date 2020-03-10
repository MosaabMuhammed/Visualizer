from termcolor import colored
import operator as op
import os
from functools import reduce

#####################################
#       Color the background        #
#####################################
def bg(value, type='num', color='blue'):
    value = str('{:,}'.format(value)) if type == 'num' else str(value)
    return colored(' ' + value + ' ', color, attrs=['reverse', 'blink'])


#####################################
#          Show Annotations         #
#####################################
def show_annotation(dist, n=5, size=14, total=None):
    sizes = []  # Get highest value in y
    for p in dist.patches:
        height = p.get_height()
        sizes.append(height)

        dist.text(p.get_x() + p.get_width() / 2.,          # At the center of each bar. (x-axis)
                  height + n,                            # Set the (y-axis)
                  '{:1.2f}%'.format(height * 100 / total) if total else '{}'.format(height),  # Set the text to be written
                  ha='center', fontsize=size)
    dist.set_ylim(0, max(sizes) * 1.15)  # set y limit based on highest heights


#####################################
#          Show Annotations         #
#####################################
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

#####################################
#            Create Folder          #
#####################################
def create_folder(folder_name, verbose=True):
    # Create visulizer Directory if don't exist
    if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
        os.makedirs(os.path.join(os.getcwd(), folder_name))
        if verbose: print("Directory " , folder_name ,  " Created ")
    else:    
        if verbose: print("Directory " , folder_name ,  " already exists")