from numbers_parser import Document
import pandas as pd

doc = Document("district_geocoded.numbers")
sheets = doc.sheets
tables = sheets[0].tables
data = tables[0].rows(values_only=True)
df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv("district_geocoded.csv", index=False)
print("Converted district_geocoded.numbers to district_geocoded.csv")
