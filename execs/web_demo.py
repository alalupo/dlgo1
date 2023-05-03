import argparse

from dlgo import httpfrontend
from dlgo.agent.loader import Loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind-address', default='127.0.0.1')
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--pg-agent')
    parser.add_argument('--predict-agent')
    parser.add_argument('--q-agent')
    parser.add_argument('--ac-agent')

    args = parser.parse_args()

    loader = Loader('mcts')
    agent = loader.create_bot()
    bots = {'mcts': agent}
    if args.pg_agent:
        loader = Loader('pg')
        bots['pg'] = loader.create_bot()
    if args.predict_agent:
        loader = Loader('predict')
        bots['predict'] = loader.create_bot()
    if args.q_agent:
        loader = Loader('q')
        bots['q'] = loader.create_bot()
    if args.ac_agent:
        loader = Loader('ac')
        bots['ac'] = loader.create_bot()

    web_app = httpfrontend.get_web_app(bots)
    web_app.run(host=args.bind_address, port=args.port, threaded=False)


if __name__ == '__main__':
    main()
