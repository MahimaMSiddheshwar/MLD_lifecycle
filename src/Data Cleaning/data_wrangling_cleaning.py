import pandera as pa
import pandas as pd
import missingno as msno
import pyjanitor as jan

# create / manipulate DataFrames
df = pd.DataFrame({"age": [29, None, 52], "salary": [91_000, 110_000, np.nan]})

# handle missing data
df = df.fillna({"age": df.age.median(), "salary": df.salary.mean()})

# descriptive stats
profile = df.describe(include="all").T

# slicers
younger = df.loc[df.age < 40, ["age", "salary"]]

# visualize NA pattern
msno.matrix(df)

# schema enforcement
schema = pa.DataFrameSchema({"age": pa.Column(pa.Float, pa.Check.ge(0))})
schema.validate(df)

# janitor helpers
df = df.clean_names().remove_empty()
