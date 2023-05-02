from pathlib import Path

from dlgo import rl
from dlgo import scoring
from dlgo import goboard_fast as goboard
from dlgo.gotypes import Player
from dlgo.exp.experience import EpisodeExperienceCollector

from collections import namedtuple


def cleaning(file):
    if Path(file).is_file():
        Path.unlink(file)


def main():
    pass


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


class Simulator:
    def __init__(self, board_size, num_planes, num_games):
        self.board_size = board_size
        self.num_planes = num_planes
        self.num_games = num_games
        self.exp_name = f'exp_rl_bs{self.board_size}_{num_games}.h5'
        self.exp_path = str(Path.cwd() / 'exp' / self.exp_name)
        cleaning(self.exp_path)

    def simulate_game(self, black_player, white_player):
        moves = []
        game = goboard.GameState.new_game(self.board_size)
        agents = {
            Player.black: black_player,
            Player.white: white_player,
        }
        while not game.is_over():
            next_move = agents[game.next_player].select_move(game)
            moves.append(next_move)
            game = game.apply_move(next_move)

        # print_board(game.board)
        game_result = scoring.compute_game_result(game)
        print(f'GAME RESULT: {game_result}')

        return GameRecord(
            moves=moves,
            winner=game_result.winner,
            margin=game_result.winning_margin,
        )

    def experience_simulation(self, agent1, agent2):
        collector1 = EpisodeExperienceCollector(self.exp_path, self.board_size, self.num_planes)
        collector2 = EpisodeExperienceCollector(self.exp_path, self.board_size, self.num_planes)

        color1 = Player.black
        for i in range(self.num_games):
            print(f'Simulating game {i + 1}/{self.num_games}...')
            collector1.begin_episode()
            agent1.set_collector(collector1)
            collector2.begin_episode()
            agent2.set_collector(collector2)

            if color1 == Player.black:
                black_player, white_player = agent1, agent2
            else:
                white_player, black_player = agent2, agent1
            game_record = self.simulate_game(black_player, white_player)
            print(f'Game {i + 1} is over. Saving the episode...')
            if game_record.winner == color1:
                collector1.complete_episode(reward=1)
                collector2.complete_episode(reward=-1)
            else:
                collector2.complete_episode(reward=1)
                collector1.complete_episode(reward=-1)
            color1 = color1.other

        print(f'>>> {self.num_games} games completed.')
        return self.exp_path


if __name__ == '__main__':
    main()
