from huobi.client.trade import TradeClient
from huobi.utils import LogInfo

trade_client = TradeClient(api_key='mk0lklo0de-bab860fb-e28d3608-eab36', secret_key='3448139b-848dc9aa-9424d4e1-ccfa8')
list_obj = trade_client.get_feerate(symbols="htusdt,btcusdt,eosusdt")
LogInfo.output_list(list_obj)

limit_objs = trade_client.get_etp_holding_limit(currencies="btc3s")
LogInfo.output_list(limit_objs)
