import pandas as pd

def bakery_data_process(file_name, items):
    df = pd.read_csv(file_name)
    dates = df["date"].values.tolist()
    dates = sorted(list(set(dates)))
    # print(dates[0:10])

    dfs = []
    for item in items:
        dfs.append(get_one_item(df, dates, item))
    
    return pd.concat(dfs, axis=0, ignore_index=True)

def get_one_item(df, dates, item):
    new_dates, new_item, new_quantity, new_price = [], [], [], []
    for date in dates:
        day_info = df[df["date"]==date]
        item_info = day_info[day_info["article"]==item]
        item_info["new_unit_price"] = item_info["unit_price"].apply(lambda x: float(x.replace(",", ".").replace(" €", "")))
        new_dates.append(date)
        new_item.append(item)
        new_quantity.append(sum(item_info["Quantity"]))
        new_price.append(sum(item_info["new_unit_price"])/item_info.shape[0])
    
    new_df = pd.DataFrame({"date": new_dates, "item": new_item, "quantity": new_quantity, "price": new_price})
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df["day_of_week"] = new_df["date"].apply(lambda x: x.weekday())
    new_df["sales"] = new_df["quantity"] * new_df["price"]
    return new_df


item_df = bakery_data_process("data/BakerySales.csv", ["TRADITIONAL BAGUETTE", "COUPE", "BAGUETTE"])
# print(item_df.head(10))
item_df.to_csv("data/bakery_train_mul.csv")