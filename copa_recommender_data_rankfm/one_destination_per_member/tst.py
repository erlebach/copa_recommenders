"""
Folder structure: 
.
./src/
./src/cool/
./src/cool/gordon.py

There are no __init__.py files at all.
"""

import sys

# Works
#from src.cool.gordon import *  # works
#prt()

# Works
#from src.cool import gordon  # works
#gordon.prt()

# Does not work
from src.cool import *  
#print(sys.path)
gordon.prt()  # did not work
prt()         # did not work


