import pandas as pd
import matplotlib.pyplot as plt
fed_files = [
    r"C:\Users\Admin\Desktop\Web Development Projects\Python\House Price Prediction\MORTGAGE30US.csv",
    r"C:\Users\Admin\Desktop\Web Development Projects\Python\House Price Prediction\RRVRUSQ156N.csv",
    r"C:\Users\Admin\Desktop\Web Development Projects\Python\House Price Prediction\CPIAUCSL.csv"
]

dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]

# print(dfs[0].head())

fed_data = pd.concat(dfs, axis=1)
# print(f"FED DATA : \n{fed_data}")

fed_data = fed_data.ffill().dropna()
# print(f"FED DATA WITHOUT NAN :\n {fed_data}")

zillow_files = [
    r"C:\Users\Admin\Desktop\Web Development Projects\Python\House Price Prediction\Metro_median_sale_price_uc_sfrcondo_week.csv",
    r"C:\Users\Admin\Desktop\Web Development Projects\Python\House Price Prediction\Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"
]
dfs = [pd.read_csv(f) for f in zillow_files]

# print(dfs[0])

dfs = [pd.DataFrame(df.iloc[0,5:]) for df in dfs]
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")


# print(dfs[0])

price_data = dfs[0].merge(dfs[1], on="month")
price_data.index = dfs[0].index
# print(price_data)

del price_data["month"]
price_data.columns = ["price", "value"]
# print(price_data)

from datetime import timedelta

fed_data.index = fed_data.index + timedelta(days=2)

price_data = fed_data.merge(price_data, left_index=True, right_index=True)

price_data.columns = ["interest", "vacancy", "cpi", "price", "value"]
# print (price_data)


price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

price_data.plot.line(y="price", use_index=True)
# plt.show()

price_data["next_quarter"] = price_data["adj_price"].shift(-13)

price_data.dropna(inplace=True)
print(price_data)