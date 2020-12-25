DATA_SAVE_DIR = f"datasets"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = ['macd', 'rsi_30', 'cci_30', 'dx_30']

## Model Parameters
A2C_PARAMS = {'n_steps': 5,
              'ent_coef': 0.01,
              'learning_rate': 0.0007,
              'verbose': 0,
              'timesteps': 20000}
PPO_PARAMS = {'n_steps': 2048,
              'ent_coef': 0.01,
              'learning_rate': 0.00025,
              'batch_size': 64,
              'verbose': 0,
              'timesteps': 20000}
DDPG_PARAMS = {'batch_size': 128,
               'buffer_size': 50000,
               'learning_rate': 0.001,
               'verbose': 0,
               'timesteps': 20000}
TD3_PARAMS = {'batch_size': 100,
              'buffer_size': 1000000,
              'learning_rate': 0.001,
              'verbose': 0,
              'timesteps': 30000}
SAC_PARAMS = {'batch_size': 64,
              'buffer_size': 100000,
              'learning_rate': 0.0001,
              'learning_starts': 100,
              'batch_size': 64,
              'ent_coef': 'auto_0.1',
              'timesteps': 50000,
              'verbose': 0}
