import logging

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from envlive import HuobiLiveEnv

# Set up logging
logging.basicConfig(
    filename="/root/goldegg.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)

# Load the model
model = A2C.load('./trained_models/CryptoPfoA2C_20210106-17h03.zip')

# Create live environment
val_env = DummyVecEnv([lambda: HuobiLiveEnv()])
obs = val_env.reset()
done = False

# Start runloop
logging.info("Start trading...")

while not done:
    action, _ = model.predict(obs)
    obs, _, done, _ = val_env.step(action)

logging.info("End trading.")
