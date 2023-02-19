from arch import arch_model

volatility_model_config = [
    {
        "type": "volatility",
        "model_name": arch_model(),
        "vol": "Garch",
        "p": 1,
        "o": 0,
        "q": 1
    },

    {
        "type": "volatility",
        "library": 'pandas',
        "model_name":'ewm()',
        "span": 30,
        "min_periods": 30,
        "adjust": False
    },

    {
        "type": "volatility",
        "library": 'pandas',
        "model_name": 'rolling()',
        "window": 30
    }
]

assumption_config = [
    {
        "name": "volait",
        "cols": []
    },

    {
        "name": "auto_colinearity",
        "cols": []
    },

    {

    }
]

timeSeries_model_config = [
    {
        
    },

    {

    }
]