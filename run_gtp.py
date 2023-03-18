import os
from pathlib import Path
from dlgo.gtp import GTPFrontend
from keras.models import load_model
from dlgo.agent.predict import load_prediction_agent
from dlgo.agent import termination
import h5py


class GtpRunner:
    def __init__(self):
        self.model_dir = 'checkpoints'
        self.model_name = 'copy_simple_medium_model_epoch_3.h5'
        self.model_path = self.get_model_path()

    def get_model_path(self):
        path = Path(__file__)
        project_lvl_path = path.parent
        model_dir_full_path = project_lvl_path.joinpath(self.model_dir)
        model_path = str(model_dir_full_path.joinpath(self.model_name))
        if not os.path.exists(model_path):
            raise FileNotFoundError
        return model_path

    def get_agent(self):
        model_file = None
        try:
            model_file = open(self.model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(self.model_path, "r") as model_file:
            agent = load_prediction_agent(model_file)
        return agent

    def run(self):
        agent = self.get_agent()
        strategy = termination.get("opponent_passes")
        termination_agent = termination.TerminationAgent(agent, strategy)
        frontend = GTPFrontend(termination_agent)
        frontend.run()


if __name__ == '__main__':
    runner = GtpRunner()
    runner.run()
