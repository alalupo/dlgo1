from dlgo import httpfrontend
from dlgo.agent.loader import Loader

BOARD_SIZE = 5


def main():
    loader = Loader('mcts')
    bot = loader.create_bot()
    web_app = httpfrontend.get_web_app(bot)
    web_app.run()


if __name__ == '__main__':
    main()
