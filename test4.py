from futu import *
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

ret, data = quote_ctx.get_capital_distribution("HK.00700")
if ret == RET_OK:
    print(data)
    print(data['capital_in_big'][0])    # 取第一条的流入资金额度，大单
    print(data['capital_in_big'].values.tolist())   # 转为 list
else:
    print('error:', data)
quote_ctx.close()