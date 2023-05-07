from dlgo import httpfrontend
from dlgo.agent.loader import Loader


def main():
    loader = Loader('zero')
    agent = loader.create_bot()
    bot = {'zero': agent}
    web_app = httpfrontend.get_web_app(bot)
    web_app.run(port=5000, threaded=False)


if __name__ == '__main__':
    main()
