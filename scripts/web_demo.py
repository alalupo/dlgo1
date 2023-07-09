import argparse

from dlgo import httpfrontend
from dlgo.agent.loader import Loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True) # agent type

    args = parser.parse_args()

    acceptable_agents = ['predict', 'pg', 'q', 'ac', 'mcts']

    if args.agent not in acceptable_agents:
        raise ValueError('The name of the agent is wrong. Use one of these: predict, pg, q, ac, mcts')

    loader = Loader(args.agent)
    agent = loader.create_bot()
    bot = {args.agent: agent}

    web_app = httpfrontend.get_web_app(bot)
    web_app.run(host='127.0.0.1', port=5000, threaded=False)


if __name__ == '__main__':
    main()
