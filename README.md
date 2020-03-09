# Visualizer:
A Python package that automate the process of visualization in the data science pipeline.

## Installation:
```python
pip install visualizer
```

## Usage:
```
import pandas as pd
from visualizer import Visualizer

# Set the paramters
df         = pd.read_csv("path/to/your/csv/file")
target_col = df['target_col']
path       = "path/where/you/want/to/save/the/images"

# Set the visualizer's parameters.
vis = Visualizer(df=df, path=path, target_col=target_col)

# Let the visualizer do all the work for you.
vis.visualizer_all()
```

## Further Ideas:

1. **plt.spy()** for the following:
   1. Nan values.
   2. Large numerical values.
   3. Sparse values.
2. Handling Time-Series and text columns.
3. Add **ignore_cols** to discard specific columns from visualizations.
4. make the individual plotting methods **static**.