import datetime
import operator
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import warnings


def cal_date_span():
    path_1 = r".\Data\DOMI"
    path_2 = r".\Data\SEC"
    date_MIN = []
    date_MAX = []

    for csv_1 in os.listdir(path_1):
        df_future = pd.read_csv(os.path.join(path_1, csv_1))
        date_min = np.min(
            [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in df_future["datetime"].values])
        date_max = np.max(
            [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in df_future["datetime"].values])
        date_MIN.append(date_min)
        date_MAX.append(date_max)

    for csv_2 in os.listdir(path_2):
        df_future = pd.read_csv(os.path.join(path_2, csv_2))
        date_min = np.min(
            [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in df_future["datetime"].values])
        date_max = np.max(
            [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in df_future["datetime"].values])
        date_MIN.append(date_min)
        date_MAX.append(date_max)

    return np.max(date_MIN), np.min(date_MAX)


def cut_date(Strat_Date, End_Date, Future_Id_lst, sort_span = 60, holding_span = 50):
    dict_future_cut_date = {}

    for future in Future_Id_lst:
        path_domi = ".\Data\DOMI\\" + future + ".csv"
        path_sec = ".\Data\SEC\\" + future + ".csv"
        future_domi = pd.read_csv(path_domi)
        future_sec = pd.read_csv(path_sec)
        future_domi["datetime"] = [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in
                                   future_domi["datetime"].values]
        future_sec["datetime"] = [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in
                                  future_sec["datetime"].values]
        future_domi = future_domi.loc[future_domi["datetime"] >= Strat_Date]
        future_sec = future_sec.loc[future_sec["datetime"] >= Strat_Date]
        future_domi = future_domi.loc[future_domi["datetime"] <= End_Date]
        future_sec = future_sec.loc[future_sec["datetime"] <= End_Date]

        # 已经用程序确认两个df的长度和对应日期都一模一样
        date = list(future_domi["datetime"])

        dict_future = {}
        for i in range(int((len(date) - sort_span) / holding_span)):
            start_date_i = date[i * holding_span]
            calculate_date_i = date[sort_span + i * holding_span - 1]  # 计算交易信号的那一天（展期收益因子计算的平均窗口期）
            trading_date_i = date[sort_span + i * holding_span]  # 调仓的那一天
            end_date_i = date[sort_span + (i + 1) * holding_span]  # 下一个周期的调仓日

            dict_future[str(trading_date_i).split(" ")[0]] = [start_date_i, calculate_date_i, trading_date_i,
                                                              end_date_i]

        dict_future_cut_date[future] = dict_future

    return dict_future_cut_date["AU"]


# 这个函数的目的在于想要读取future_id在信号计算日期上的展期收益率
def calc_roll_return(future_id, date_Info):
    path_domi = ".\Data\DOMI\\" + future_id + ".csv"
    path_sec = ".\Data\SEC\\" + future_id + ".csv"

    Strat_Date = date_Info[0]  # 这个周期的排序开始日期
    Calcluate_Date = date_Info[1]  # 这个周期的排序结束日期，也是信号计算日期
    Trade_Date = date_Info[2]  # 这个周期的调仓日
    End_Date = date_Info[3]  # 下一个周期的第一天start-date

    future_domi = pd.read_csv(path_domi)  # 读取这个品种的主力合约信息
    future_sec = pd.read_csv(path_sec)  # 读取这个品种的次主力合约信息

    # 将dataframe中的日期列改为datetime格式
    future_domi["datetime"] = [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in
                               future_domi["datetime"].values]
    future_sec["datetime"] = [datetime.datetime.strptime(date_i.split(" ")[0], "%Y%m%d") for date_i in
                              future_sec["datetime"].values]

    # 获取大于这一周期排序期第一日（Strat_Date）的所有数据，因为每一个交易日每个品种都存在主力合约和次主力合约，所以两个dataframe的主键datetime一定是完全一样的
    future_domi = future_domi.loc[future_domi["datetime"] >= Strat_Date]
    future_sec = future_sec.loc[future_sec["datetime"] >= Strat_Date]
    # 获取小于等于下一个周期排序期第一日（Strat_next_Date）的所有数据
    future_domi = future_domi.loc[future_domi["datetime"] <= End_Date][
        ["datetime", "contract_code_ref", "open", "close"]]\
        .rename(columns = {"contract_code_ref": future_id + "_domi_contract_code", "open": future_id + "_domi_open",
                           "close": future_id + "_domi_close"})
    future_sec = future_sec.loc[future_sec["datetime"] <= End_Date][["contract_code_ref", "open", "close"]]\
        .rename(columns = {"contract_code_ref": future_id + "_sec_contract_code", "open": future_id + "_sec_open",
                           "close": future_id + "_sec_close"})

    df_sum = pd.concat([future_domi, future_sec], axis = 1)

    # 获取每个主力合约和次主力合约的到期日
    df_domi_contract_exchange = pd.read_csv("Data/domi_contract_code_and_exchange_id.csv", index_col = 0).to_dict()
    df_sec_contract_exchange = pd.read_csv("Data/sec_contract_code_and_exchange_id.csv", index_col = 0).to_dict()
    domi_date = [df_domi_contract_exchange["date"][contract_code_i] for contract_code_i in
                 list(df_sum[future_id + "_domi_contract_code"])]
    sec_date = [df_sec_contract_exchange["date"][contract_code_i] for contract_code_i in
                list(df_sum[future_id + "_sec_contract_code"])]

    df_sum[future_id + "_domi_contract_date"] = domi_date
    df_sum[future_id + "_sec_contract_date"] = sec_date

    # 获取主力合约和次主力合约的到期交易日之差，以及远期合约是主力合约还是次主力合约，作为label9
    Delta_Date = []
    label = []
    for i in range(len(domi_date)):
        date_domi = datetime.datetime.strptime(domi_date[i], "%Y/%m/%d")
        date_sec = datetime.datetime.strptime(sec_date[i], "%Y/%m/%d")
        Delta_Date.append(abs(int((date_domi - date_sec).days)))
        label.append("DOMI" if date_domi < date_sec else "SEC")  # 如果主力合约是近月合约那么label为DOMI， 否则为SEC

    df_sum[future_id + "_delta_date"] = Delta_Date
    df_sum[future_id + "_label"] = label

    RR = []
    for index, row in df_sum.iterrows():
        label = row[future_id + '_label']
        domi_close = row[future_id + "_domi_close"]
        sec_close = row[future_id + "_sec_close"]
        delta_date = row[future_id + "_delta_date"]

        if label == "DOMI":
            rr_i = ((domi_close - sec_close) / sec_close) * (365 / delta_date)
        elif label == "SEC":
            rr_i = ((sec_close - domi_close) / domi_close) * (365 / delta_date)

        RR.append(rr_i)
    df_sum[future_id + "_Roll_Return"] = RR

    res = np.mean(df_sum.loc[df_sum["datetime"] <= Calcluate_Date][future_id + "_Roll_Return"])
    Trade_info = df_sum.loc[df_sum["datetime"] >= Trade_Date]
    Trade_info = Trade_info.reset_index(drop = True)

    return res, Trade_info


def calc_spec_trading_date(Future_Id_lst, Date, dict_date_divide, cash = 1000000, trade_num = 0):
    # 返回holding期间的cash序列，以及下一个周期调仓之后的cash
    Rr_lst = []
    Info_future_lst = []

    for future_id in Future_Id_lst:
        Rr_future_id, Info_future_id = calc_roll_return(future_id, dict_date_divide[Date])
        Rr_lst.append(Rr_future_id)
        Info_future_lst.append(Info_future_id)

    # 将这个时间区间内的所有品种的数据全部拼接在一起，去除了重复列，再加上两列
    df_ALL = pd.concat(Info_future_lst, axis = 1)
    datetime = list(df_ALL.iloc[:, 0])
    year = [date_i.year for date_i in datetime]
    df_ALL.drop(["datetime"], axis = 1)
    df_ALL["datetime"] = datetime
    df_ALL["year"] = year

    # 按照展期收益率升序排列
    df_sort = pd.DataFrame(np.array([Future_Id_lst, Rr_lst]).T, columns = ["Future", "Roll_return"])
    df_sort["Roll_return"] = [float(ror_i) for ror_i in df_sort["Roll_return"]]
    df_sort = df_sort.sort_values(by = "Roll_return")

    # 取前百分之20的做空品种和做多品种
    short_name_lst = list(df_sort["Future"])[:int(0.2 * len(Future_Id_lst))]
    long_name_lst = list(df_sort["Future"])[- int(0.2 * len(Future_Id_lst)):]

    cash_sum = np.zeros(len(df_ALL))

    for short in short_name_lst:
        cash_i = cash / (2 * len(short_name_lst))  # 每个做空品种在周期开始可以分到的资产(多/空的各个品种平均分配）
        cash_i_lst = []  # 用来记录这一个周期中某一个做空品种的资产变化情况

        if df_ALL[short + "_label"].values[0] == "DOMI":  # 如果该品种的近月合约是主力合约
            for day in range(len(df_ALL)):  # df_ALL的长度一般比持仓长度多1，因为最后一个日期为下一个周期的第一个start_date
                if day == 0:  # 如果是第一天，那么以开仓价做空，收盘价记录净资产变化，交易手续费为一次
                    open_price = list(df_ALL[short + "_sec_open"])[day]  # 第一天的开盘价
                    close_price = list(df_ALL[short + "_sec_close"])[day]  # 第一天的收盘价
                    Real_ror = (1 - 0.0003) * (open_price / close_price)  # 交易一次
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                elif day == len(df_ALL) - 1:  # 如果是最后一天，那么以上一天收盘价做空，当天的开盘价记录净资产变化，交易手续费为一次
                    last_contract_code = list(df_ALL[short + "_sec_contract_code"])[day - 1]
                    now_contract_code = list(df_ALL[short + "_sec_contract_code"])[day]
                    last_close_price = list(df_ALL[short + "_sec_close"])[day - 1]
                    open_price = list(df_ALL[short + "_sec_open"])[day]

                    if last_contract_code != now_contract_code:  # 如果出现合约变换
                        Real_ror = (last_close_price / open_price) * (1 - 0.0003)  # 因为要合约交换，所以应该在当天开盘的时候就清仓
                    else:  # 如果没有出现合约变换
                        Real_ror = (last_close_price / open_price) * (1 - 0.0003)  # 交易一次
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                else:
                    last_contract_code = list(df_ALL[short + "_sec_contract_code"])[day - 1]
                    now_contract_code = list(df_ALL[short + "_sec_contract_code"])[day]
                    last_close_price = list(df_ALL[short + "_sec_close"])[day - 1]
                    close_price = list(df_ALL[short + "_sec_close"])[day]
                    open_price = list(df_ALL[short + "_sec_open"])[day]

                    if last_contract_code != now_contract_code:
                        Real_ror = (last_close_price / open_price) * (1 - 0.0003) * (1 - 0.0003) * (
                                    open_price / close_price)
                        trade_num = trade_num + 2
                    else:
                        Real_ror = (last_close_price / close_price)
                    cash_i = cash_i * Real_ror

                cash_i_lst.append(cash_i)

        elif df_ALL[short + "_label"].values[0] == "SEC":
            for day in range(len(df_ALL)):  # 因为df_ALL的最后一行是下一个周期的第一天
                if day == 0:
                    open_price = list(df_ALL[short + "_domi_open"])[day]
                    close_price = list(df_ALL[short + "_domi_close"])[day]
                    Real_ror = (1 - 0.0003) * (open_price / close_price)
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                elif day == len(df_ALL) - 1:
                    last_close_price = list(df_ALL[short + "_domi_close"])[day - 1]
                    open_price = list(df_ALL[short + "_domi_open"])[day]
                    Real_ror = (last_close_price / open_price) * (1 - 0.0003)
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                else:
                    last_contract_code = list(df_ALL[short + "_domi_contract_code"])[day - 1]
                    now_contract_code = list(df_ALL[short + "_domi_contract_code"])[day]
                    last_close_price = list(df_ALL[short + "_domi_close"])[day - 1]
                    close_price = list(df_ALL[short + "_domi_close"])[day]
                    open_price = list(df_ALL[short + "_domi_open"])[day]

                    if last_contract_code != now_contract_code:
                        Real_ror = (last_close_price / open_price) * (1 - 0.0003) * (1 - 0.0003) * (
                                open_price / close_price)
                        trade_num = trade_num + 2
                    else:
                        Real_ror = (last_close_price / close_price)
                    cash_i = cash_i * Real_ror

                cash_i_lst.append(cash_i)
        cash_sum = cash_sum + np.array(cash_i_lst)

    for long in long_name_lst:
        cash_i = cash / (2 * len(long_name_lst))
        cash_i_lst = []
        if df_ALL[long + "_label"].values[0] == "DOMI":
            for day in range(len(df_ALL)):  # 因为df_ALL的最后一行是下一个周期的第一天
                if day == 0:
                    open_price = list(df_ALL[long + "_sec_open"])[day]
                    close_price = list(df_ALL[long + "_sec_close"])[day]
                    Real_ror = (1 - 0.0003) * (close_price / open_price)
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                elif day == len(df_ALL) - 1:
                    last_close_price = list(df_ALL[long + "_sec_close"])[day - 1]
                    open_price = list(df_ALL[long + "_sec_open"])[day]
                    Real_ror = (open_price / last_close_price) * (1 - 0.0003)
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                else:
                    last_contract_code = list(df_ALL[long + "_sec_contract_code"])[day - 1]
                    now_contract_code = list(df_ALL[long + "_sec_contract_code"])[day]
                    last_close_price = list(df_ALL[long + "_sec_close"])[day - 1]
                    close_price = list(df_ALL[long + "_sec_close"])[day]
                    open_price = list(df_ALL[long + "_sec_open"])[day]

                    if last_contract_code != now_contract_code:
                        Real_ror = (open_price / last_close_price) * (1 - 0.0003) * (1 - 0.0003) * (
                                    close_price / open_price)
                        trade_num = trade_num + 1
                    else:
                        Real_ror = (close_price / last_close_price)
                    cash_i = cash_i * Real_ror

                cash_i_lst.append(cash_i)

        elif df_ALL[long + "_label"].values[0] == "SEC":
            for day in range(len(df_ALL)):  # 因为df_ALL的最后一行是下一个周期的第一天
                if day == 0:
                    open_price = list(df_ALL[long + "_domi_open"])[day]
                    close_price = list(df_ALL[long + "_domi_close"])[day]
                    Real_ror = (1 - 0.0003) * (close_price / open_price)
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                elif day == len(df_ALL) - 1:
                    last_close_price = list(df_ALL[long + "_domi_close"])[day - 1]
                    open_price = list(df_ALL[long + "_domi_open"])[day]
                    Real_ror = (open_price / last_close_price) * (1 - 0.0003)
                    trade_num = trade_num + 1
                    cash_i = cash_i * Real_ror

                else:
                    last_contract_code = list(df_ALL[long + "_domi_contract_code"])[day - 1]
                    now_contract_code = list(df_ALL[long + "_domi_contract_code"])[day]
                    last_close_price = list(df_ALL[long + "_domi_close"])[day - 1]
                    close_price = list(df_ALL[long + "_domi_close"])[day]
                    open_price = list(df_ALL[long + "_domi_open"])[day]

                    if last_contract_code != now_contract_code:
                        Real_ror = (open_price / last_close_price) * (1 - 0.0003) * (1 - 0.0003) * (
                                close_price / open_price)
                        trade_num = trade_num + 2
                    else:
                        Real_ror = (close_price / last_close_price)
                    cash_i = cash_i * Real_ror

                cash_i_lst.append(cash_i)
        cash_sum = cash_sum + np.array(cash_i_lst)
    df_final = pd.concat([df_ALL.iloc[:-1, :], pd.DataFrame(np.array(cash_sum[:-1]).T, columns = ["Cash"])], axis = 1)
    # 每个cash_num都比持仓周期长一个单位，所以应该出去最后一天，最后一天的cash作为调仓后手里的净资产return

    df_len = len(df_final)

    for future_id in Future_Id_lst:
        if future_id in short_name_lst:
            df_final[future_id + "_Signal"] = [-1] * df_len
            df_final[future_id + "Sort_Average_RR"] = [df_sort.loc[df_sort["Future"] == future_id][
                                                           "Roll_return"].values[0]] * df_len

        elif future_id in long_name_lst:
            df_final[future_id + "_Signal"] = [1] * df_len
            df_final[future_id + "Sort_Average_RR"] = [df_sort.loc[df_sort["Future"] == future_id][
                                                           "Roll_return"].values[0]] * df_len

        else:
            df_final[future_id + "_Signal"] = ["空仓"] * df_len
            df_final[future_id + "Sort_Average_RR"] = [df_sort.loc[df_sort["Future"] == future_id][
                                                           "Roll_return"].values[0]] * df_len

    # print("Done!")

    return df_final, cash_sum[-1], trade_num


def calc_spec_hold_and_sort(Future_Id_lst, dict_date_divide, cash = 1000000, trade_num = 0):
    df_final_all = []
    for key_date in dict_date_divide.keys():
        # key: trading_date
        # value: [start_date_i, calculate_date_i, trading_date_i, end_date_i]
        # 计算交易信号的那一天
        # print("{}已完成".format(str(key_date)))
        df_final, cash, trade_num = calc_spec_trading_date(Future_Id_lst, key_date, dict_date_divide, cash, trade_num)
        df_final_all.append(df_final)

    df_res = pd.concat(df_final_all, axis = 0)

    return df_res, trade_num / 12


def plot(df_res):
    asset = np.array(list(df_res["Cash"]))
    asset = asset / asset[0]

    plt.figure(figsize = (14, 7))
    plt.plot(range(len(asset)), asset, label = 'Revenue curve')
    plt.legend()  # 让图例生效

    plt.xlabel(u"time(s)")  # X轴标签
    plt.ylabel("Asset")  # Y轴标签
    plt.title("Revenue Cruve")  # 标题

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))

    plt.show()


def measure(df_res):
    unique_year = list(set(list(df_res["year"])))
    Revenue_Ratio = []

    for year_i in unique_year:
        df_year = df_res.loc[df_res["year"] == year_i]
        cash_i = list(df_year["Cash"])
        date_num_i = len(df_year)
        ror_i = float(cash_i[-1]) / float(cash_i[0])
        revenue_ratio = (ror_i ** (1 / date_num_i)) ** 252 - 1

        Revenue_Ratio.append(revenue_ratio)

    asset = df_res["Cash"].values
    ################################年化收益率##############################
    revenue_ratio = np.mean(np.array(Revenue_Ratio))

    ################################夏普比率##############################
    Std = np.std(np.array(Revenue_Ratio))
    sharp_ratio = (revenue_ratio - 0.035) / Std  # 国债利率作为无风险收益率

    ################################最大回撤率###############################
    Max_drop = 0
    # print("asset序列长度为：" + str(len(asset)))
    for i in range(len(asset)):
        for j in range(i + 1, len(asset)):
            cache = (asset[i] - asset[j]) / asset[i]
            if cache > Max_drop:
                Max_drop = cache
            else:
                continue

    print("年化收益率为：{}%".format(str(revenue_ratio * 100)))
    print("夏普比率为：{}".format(str(sharp_ratio)))
    print("收益波动率为：{}%".format(str(Std * 100)))
    print("最大回撤率为：{}%".format(str(Max_drop * 100)))

    return str(revenue_ratio), str(sharp_ratio), str(Std), str(Max_drop)


if __name__ == "__main__":
    warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
    Future_Id_lst = ['AG', 'AL', 'AU', 'A', 'BU', 'CF', 'CS', 'CU', 'C', 'FG', 'HC', 'I',
                     'JD', 'JM', 'J', 'L', 'MA', 'M', 'NI', 'OI', 'PP', 'P', 'RB', 'RM', 'RU',
                     'SF', 'SM', 'SR', 'TA', 'V', 'Y', 'ZC', 'ZN']
    sort_lst = [10, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    hold_lst = [10, 20, 30, 40, 50, 60, 70, 80]
    revenue_ratio_res = []
    sharp_ratio_res = []
    Max_drop_res = []
    trade_num_res = []

    # 获取所有品种存在主力合约和次主力合约的时间节点，为：2015-06-09 00:00:00和2022-07-26 00:00:00
    Strat_Date, End_Date = cal_date_span()
    for hold_i in hold_lst:
        revenue_ratio_hold_i = []
        sharp_ratio_hold_i = []
        Max_drop_hold_i = []
        trade_num_hold_i = []
        for sort_i in sort_lst:
            print("###################排序期长度为{}，持仓期长度为{}#####################".format(sort_i, hold_i))
            # 用来获得每个截面计算所用到的时间点，以字典的方式呈现
            # [start_date_i, calculate_date_i, trading_date_i, end_date_i]
            dict_date_divide = cut_date(Strat_Date, End_Date, Future_Id_lst, sort_span = sort_i, holding_span = hold_i)
            # 用于获取在某个截面(中间那个日期，trading_date)上某个品种的平均展期收益率
            # res, c = calc_roll_return("PP", dict_date_divide["2015-08-05"] )
            # df = calc_spec_trading_date(Future_Id_lst, "2015-08-05", dict_date_divide)
            df_res, trade_num = calc_spec_hold_and_sort(Future_Id_lst, dict_date_divide, cash = 1000000, trade_num = 0)
            df_res.to_csv("example.csv")
            revenue_ratio, sharp_ratio, Std, Max_drop = measure(df_res)
            plot(df_res)

            revenue_ratio_hold_i.append(revenue_ratio)
            sharp_ratio_hold_i.append(sharp_ratio)
            Max_drop_hold_i.append(Max_drop)
            trade_num_hold_i.append(trade_num)

        revenue_ratio_res.append(revenue_ratio_hold_i)
        sharp_ratio_res.append(sharp_ratio_hold_i)
        Max_drop_res.append(Max_drop_hold_i)
        trade_num_res.append(trade_num_hold_i)

    pd.DataFrame(revenue_ratio_res, index = hold_lst, columns = sort_lst).to_csv("Revenue_Ratio.csv")
    pd.DataFrame(sharp_ratio_res, index = hold_lst, columns = sort_lst).to_csv("Sharp_Ratio.csv")
    pd.DataFrame(Max_drop_res, index = hold_lst, columns = sort_lst).to_csv("Max_Drop.csv")
    pd.DataFrame(trade_num_res, index = hold_lst, columns = sort_lst).to_csv("Trade_Num.csv")

    print("Done!")
