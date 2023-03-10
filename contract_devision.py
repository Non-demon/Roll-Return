import pandas as pd
import numpy as np

def divide(path, Future_Id_lst, type="-DOMI"):
    df_ALL = pd.read_csv(path)
    new_contract_code = [i.upper() for i in list(df_ALL["contract_code"])]
    df_ALL["new_contract_code"] = new_contract_code
    df_contract_code_and_exchange_id = pd.DataFrame([], columns=["exchange_id", "contract_id"])

    for future in Future_Id_lst:
        df_future = df_ALL.loc[df_ALL["new_contract_code"] == future + type]
        df_future = df_future.sort_values(by="trading_date")
        df_future["exchange_id"] = [df_future["exchange_id"].values[-1]] * len(df_future)

        list_contract = list(set(list(df_future["contract_code_ref"])))
        list_exchange = []

        for contract in list_contract:
            exchange = df_future.loc[df_future["contract_code_ref"] == contract]["exchange_id"].values[0]
            list_exchange.append(exchange)

        df_contract_code_and_exchange_id = pd.concat([df_contract_code_and_exchange_id, pd.DataFrame(np.array([list_exchange, list_contract]).T, columns=["exchange_id", "contract_id"])])

        df_future.to_csv(r"./Data/" + type.replace("-", "") + "/{}.csv".format(future))

    df_contract_code_and_exchange_id.to_csv(r"./Data/" + type.replace("-", "") + "/contract_code_and_exchange_id.csv")

    return 0

if __name__ == "__main__":
    path = r"./oringin/future_candle_202207270909.csv"
    Future_Id_lst = ['AG', 'AL', 'AU', 'A', 'BU', 'CF', 'CS', 'CU', 'C', 'FG', 'HC', 'I',
                  'JD', 'JM', 'J', 'L','MA', 'M', 'NI', 'OI', 'PP', 'P', 'RB', 'RM', 'RU',
                  'SF', 'SM', 'SR', 'TA', 'V', 'Y', 'ZC', 'ZN']

    divide(path, Future_Id_lst, type="-DOMI")

    print("Done!")