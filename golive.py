import argparse
import logging

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from envlive import HuobiLiveEnv

parser = argparse.ArgumentParser()
parser.add_argument("--simulator", help="execute in simulation mode", action='store_true')
parser.add_argument("--logto", help="log file")
args = parser.parse_args()
if args.simulator:
    print('Running in simulation mode.')

if args.logto:
    logfile_name = args.logto
else:
    logfile_name = "./goldegg.log"

# Set up logging
logging.basicConfig(
    filename=logfile_name,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)

# Load the model
model = A2C.load('./trained_models/CryptoPfoA2C_20210106-17h03.zip')

# Create live environment
val_env = DummyVecEnv([lambda: HuobiLiveEnv(0.05, 280, simulate_mode=args.simulator, max_steps=24 * 30)])
obs = val_env.reset()
done = False

# Start runloop
logging.info("Start trading...")

while not done:
    action, _ = model.predict(obs)
    obs, _, done, _ = val_env.step(action)

logging.info("End trading.")
