from pathlib import Path
import glob
import numpy as np

from keras.utils import to_categorical

from dlgo.data.generator import DataGenerator
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import GameState, Move
from dlgo.gosgf import Sgf_game
from dlgo.gotypes import Point


def create_npy():
    encoder = get_encoder_by_name('simple', 19)
    shape = encoder.shape_for_others()
    counter = 0
    sgf_content = b'(;GM[1]\nFF[4]\nSZ[19]\nPW[baduk123]\nWR[6d]\nPB[hazuki]\nBR[6d]\nDT[2008-07-01]\nPC[The KGS Go Server at http://www.gokgs.com/]\nKM[6.50]\nRE[B+Time]\nRU[Japanese]CA[UTF-8]ST[2]AP[CGoban:3]TM[0]OT[1x10 byo-yomi]\n;B[qd]\n;W[dp]\n;B[cd]\n;W[pp]\n;B[oc]\n;W[ed]\n;B[gc]\n;W[cc]\n;B[bc]\n;W[dc]\n;B[cg]\n;W[fe]\n;B[id]\n;W[be]\n;B[bd]\n;W[ce]\n;B[de]\n;W[df]\n;B[cf]\n;W[dd]\n;B[ae]\n;W[ee]\n;B[bf]\n;W[ci]\n;B[dg]\n;W[eg]\n;B[di]\n;W[dj]\n;B[eh]\n;W[fg]\n;B[cj]\n;W[ck]\n;B[bj]\n;W[bk]\n;B[ej]\n;W[dk]\n;B[bi]\n;W[co]\n;B[qj]\n;W[ql]\n;B[jq]\n;W[hq]\n;B[mq]\n;W[oq]\n;B[jo]\n;W[ol]\n;B[oj]\n;W[kd]\n;B[mc]\n;W[bb]\n;B[if]\n;W[kf]\n;B[ih]\n;W[kh]\n;B[og]\n;W[fi]\n;B[ei]\n;W[ic]\n;B[hd]\n;W[jc]\n;B[ij]\n;W[kj]\n;B[fq]\n;W[eq]\n;B[fp]\n;W[fr]\n;B[gr]\n;W[er]\n;B[gq]\n;W[in]\n;B[io]\n;W[jk]\n;B[gi]\n;W[fh]\n;B[fj]\n;W[hg]\n;B[ig]\n;W[gj]\n;B[gk]\n;W[hr]\n;B[gs]\n;W[hj]\n;B[hk]\n;W[hi]\n;B[hh]\n;W[gh]\n;B[ii]\n;W[nf]\n;B[of]\n;W[rk]\n;B[rj]\n;W[nr]\n;B[hn]\n;W[ne]\n;B[on]\n;W[pn]\n;B[ng]\n;W[mk]\n;B[lg]\n;W[kg]\n;B[lf]\n;W[le]\n;B[mr]\n;W[lc]\n;B[md]\n;W[sj]\n;B[si]\n;W[sk]\n;B[rh]\n;W[ms]\n;B[ls]\n;W[ns]\n;B[kr]\n;W[nn]\n;B[no]\n;W[oo]\n;B[mn]\n;W[nm]\n;B[me]\n;W[lb]\n;B[dn]\n;W[cn]\n;B[dm]\n;W[cm]\n;B[ma]\n;W[eo]\n;B[im]\n;W[fm]\n;B[el]\n;W[fo]\n;B[go]\n;W[gn]\n;B[gm]\n;W[fn]\n;B[ho]\n;W[gl]\n;B[fl]\n;W[hm]\n;B[hl]\n;W[fb]\n;B[ab]\n;W[gb]\n;B[la]\n;W[mb]\n;B[nb]\n;W[ka]\n;B[na]\n;W[kb]\n;B[nq]\n;W[or]\n;B[qk]\n;W[rl]\n;B[mm]\n;W[gi]\n;B[ik]\n;W[km]\n;B[kn]\n;W[ba]\n;B[nl]\n;W[om]\n;B[ak]\n;W[al]\n;B[aj]\n;W[bl]\n;B[fs]\n)\n'
    sgf = Sgf_game.from_string(sgf_content)
    feature_shape = tuple(np.insert(shape, 0, np.asarray([165])))
    features = np.zeros(feature_shape)
    labels = np.zeros((165,))
    game_state, first_move_done = GameState.new_game(19), True

    for item in sgf.main_sequence_iter():
        color, move_tuple = item.get_move()
        point = None
        if color is not None:
            if move_tuple is not None:
                row, col = move_tuple
                point = Point(row + 1, col + 1)
                move = Move.play(point)
            else:
                move = Move.pass_turn()
            if first_move_done and point is not None:
                features[counter] = encoder.encode(game_state)
                labels[counter] = encoder.encode_point(point)
                counter += 1
            game_state = game_state.apply_move(move)
            first_move_done = True

    data_file_name = 'KGS-2008-19-14002-train'
    path = Path.cwd()
    dir = path / 'data'
    feature_file = dir.joinpath(f'{data_file_name}_features_1_1')
    label_file = dir.joinpath(f'{data_file_name}_labels_1_1')
    np.save(str(feature_file), features)
    np.save(str(label_file), labels)
    # base = dir.joinpath(data_file_name + '_features_*.npy')
    # for feature_file in glob.glob(str(base)):
    #     label_file = feature_file.replace('features', 'labels')
    x = np.load(str(feature_file) + '.npy')
    y = np.load(str(label_file) + '.npy')
    print(x.shape, y.shape)
    x2 = x.astype('float32')
    y2 = to_categorical(y.astype(int), 361)
    print(x2.shape, y2.shape)


def test_generator():
    path = Path.cwd()
    dir = path / 'data'
    samples = [('KGS-2008-19-14002-.tar.gz', 9791), ('KGS-2012-19-13665-.tar.gz', 4261)]
    generator = DataGenerator(dir, samples, 19, 'train')
    print(f'num samples = {generator.get_num_samples()}')
    for item in generator.generate(128):
        x, y = item
        print(x.shape, y.shape)


def main():
    create_npy()


if __name__ == '__main__':
    main()
