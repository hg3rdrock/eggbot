

class HoldingLimit:
    """
    The holding limit for etp products, such as BTC3S, ETH3L, etc.

    :member
        symbol: The symbol, like "btcusdt".
        maker_fee: maker fee rate
        taker_fee: taker fee rate

    """

    def __init__(self):
        self.remainingAmount = ""
        self.currency = ""
        self.maxHoldings = ""

    def print_object(self, format_data=""):
        from huobi.utils.print_mix_object import PrintBasic
        PrintBasic.print_basic(self.remainingAmount, format_data + "Remaining Amount")
        PrintBasic.print_basic(self.currency, format_data + "Symbol")
        PrintBasic.print_basic(self.maxHoldings, format_data + "Max Holdings")