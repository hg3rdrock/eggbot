from huobi.client.trade import TradeClient
from huobi.utils import LogInfo

trade_client = TradeClient(api_key='new-api-key', secret_key='new-api-secret')
list_obj = trade_client.get_feerate(symbols="htusdt,btcusdt,eosusdt")
LogInfo.output_list(list_obj)

limit_objs = trade_client.get_etp_holding_limit(currencies="btc3s")
LogInfo.output_list(limit_objs)
