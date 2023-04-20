from pathlib import Path

import numpy as np
from keras.utils import to_categorical

from dlgo.data.generator import DataGenerator
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import GameState, Move
from dlgo.gosgf import Sgf_game
from dlgo.gotypes import Point
from dlgo.tools.board_decoder import BoardDecoder
from dlgo.goboard_fast import Board
from dlgo.gotypes import Player, Point


def create_npy():
    encoder = get_encoder_by_name('simple', 19)
    shape = encoder.shape()
    counter = 0
    # games 2008-07-01-4.sgf:
    # sgf_content = b'(;GM[1]\nFF[4]\nSZ[19]\nPW[baduk123]\nWR[6d]\nPB[hazuki]\nBR[6d]\nDT[2008-07-01]\nPC[The KGS Go Server at http://www.gokgs.com/]\nKM[6.50]\nRE[B+Time]\nRU[Japanese]CA[UTF-8]ST[2]AP[CGoban:3]TM[0]OT[1x10 byo-yomi]\n;B[qd]\n;W[dp]\n;B[cd]\n;W[pp]\n;B[oc]\n;W[ed]\n;B[gc]\n;W[cc]\n;B[bc]\n;W[dc]\n;B[cg]\n;W[fe]\n;B[id]\n;W[be]\n;B[bd]\n;W[ce]\n;B[de]\n;W[df]\n;B[cf]\n;W[dd]\n;B[ae]\n;W[ee]\n;B[bf]\n;W[ci]\n;B[dg]\n;W[eg]\n;B[di]\n;W[dj]\n;B[eh]\n;W[fg]\n;B[cj]\n;W[ck]\n;B[bj]\n;W[bk]\n;B[ej]\n;W[dk]\n;B[bi]\n;W[co]\n;B[qj]\n;W[ql]\n;B[jq]\n;W[hq]\n;B[mq]\n;W[oq]\n;B[jo]\n;W[ol]\n;B[oj]\n;W[kd]\n;B[mc]\n;W[bb]\n;B[if]\n;W[kf]\n;B[ih]\n;W[kh]\n;B[og]\n;W[fi]\n;B[ei]\n;W[ic]\n;B[hd]\n;W[jc]\n;B[ij]\n;W[kj]\n;B[fq]\n;W[eq]\n;B[fp]\n;W[fr]\n;B[gr]\n;W[er]\n;B[gq]\n;W[in]\n;B[io]\n;W[jk]\n;B[gi]\n;W[fh]\n;B[fj]\n;W[hg]\n;B[ig]\n;W[gj]\n;B[gk]\n;W[hr]\n;B[gs]\n;W[hj]\n;B[hk]\n;W[hi]\n;B[hh]\n;W[gh]\n;B[ii]\n;W[nf]\n;B[of]\n;W[rk]\n;B[rj]\n;W[nr]\n;B[hn]\n;W[ne]\n;B[on]\n;W[pn]\n;B[ng]\n;W[mk]\n;B[lg]\n;W[kg]\n;B[lf]\n;W[le]\n;B[mr]\n;W[lc]\n;B[md]\n;W[sj]\n;B[si]\n;W[sk]\n;B[rh]\n;W[ms]\n;B[ls]\n;W[ns]\n;B[kr]\n;W[nn]\n;B[no]\n;W[oo]\n;B[mn]\n;W[nm]\n;B[me]\n;W[lb]\n;B[dn]\n;W[cn]\n;B[dm]\n;W[cm]\n;B[ma]\n;W[eo]\n;B[im]\n;W[fm]\n;B[el]\n;W[fo]\n;B[go]\n;W[gn]\n;B[gm]\n;W[fn]\n;B[ho]\n;W[gl]\n;B[fl]\n;W[hm]\n;B[hl]\n;W[fb]\n;B[ab]\n;W[gb]\n;B[la]\n;W[mb]\n;B[nb]\n;W[ka]\n;B[na]\n;W[kb]\n;B[nq]\n;W[or]\n;B[qk]\n;W[rl]\n;B[mm]\n;W[gi]\n;B[ik]\n;W[km]\n;B[kn]\n;W[ba]\n;B[nl]\n;W[om]\n;B[ak]\n;W[al]\n;B[aj]\n;W[bl]\n;B[fs]\n)\n'
    # game 2007-07-28-25.sgf:
    sgf_content = b'(;GM[1]\nFF[4]\nSZ[19]\nPW[Thekgs]\nWR[7d]\nPB[ico]\nBR[2d]\nDT[2007-07-28]\nPC[The KGS Go Server at http://www.gokgs.com/]\nKM[0.50]\nRE[B+8.50]\nRU[Japanese]OT[5x30 byo-yomi]CA[UTF-8]ST[2]AP[CGoban:3]TM[900]HA[5]AB[dd][pd][jj][dp][pp]\n\n;W[qf]\n;B[pi]\n;W[of]\n;B[nd]\n;W[rd]\n;B[qc]\n;W[ri]\n;B[pg]\n;W[pf]\n;B[rh]\n;W[qh]\n;B[qi]\n;W[rg]\n;B[rj]\n;W[sh]\n;B[qk]\n;W[lc]\n;B[ld]\n;W[kd]\n;B[le]\n;W[mc]\n;B[nc]\n;W[hc]\n;B[rc]\n;W[cf]\n;B[df]\n;W[dg]\n;B[ce]\n;W[ef]\n;B[de]\n;W[bf]\n;B[eg]\n;W[be]\n;B[cc]\n;W[dh]\n;B[ff]\n;W[eh]\n;B[fg]\n;W[cl]\n;B[jd]\n;W[jc]\n;B[ke]\n;W[kc]\n;B[cn]\n;W[nq]\n;B[np]\n;W[mp]\n;B[no]\n;W[or]\n;B[pr]\n;W[pq]\n;B[qq]\n;W[oq]\n;B[qr]\n;W[mo]\n;B[po]\n;W[jq]\n;B[eq]\n;W[gq]\n;B[mn]\n;W[ph]\n;B[oi]\n;W[oh]\n;B[ni]\n;W[ln]\n;B[lm]\n;W[kn]\n;B[km]\n;W[jn]\n;B[hp]\n;W[gp]\n;B[hn]\n;W[go]\n;B[ho]\n;W[mm]\n;B[nn]\n;W[jm]\n;B[hq]\n;W[gn]\n;B[hm]\n;W[en]\n;B[dm]\n;W[fl]\n;B[hk]\n;W[il]\n;B[gm]\n;W[fm]\n;B[gk]\n;W[dl]\n;B[ml]\n;W[bp]\n;B[cq]\n;W[bq]\n;B[br]\n;W[co]\n;B[do]\n;W[bn]\n;B[dr]\n;W[kl]\n;B[ll]\n;W[kk]\n;B[li]\n;W[kj]\n;B[ki]\n;W[mk]\n;B[nk]\n;W[mj]\n;B[mi]\n;W[nj]\n;B[ok]\n;W[fr]\n;B[cp]\n;W[bo]\n;B[ar]\n;W[hr]\n;B[ej]\n;W[dj]\n;B[id]\n;W[hd]\n;B[hf]\n;W[fd]\n;B[ee]\n;W[mg]\n;B[lg]\n;W[fj]\n;B[ng]\n;W[nf]\n;B[nh]\n;W[og]\n;B[re]\n;W[qe]\n;B[qd]\n;W[fh]\n;B[hh]\n;W[mb]\n;B[fb]\n;W[gb]\n;B[fc]\n;W[nb]\n;B[ob]\n;W[fa]\n;B[ea]\n;W[ga]\n;B[ec]\n;W[bd]\n;B[bc]\n;W[dn]\n;B[gc]\n;W[hb]\n;B[he]\n;W[gd]\n;B[fk]\n;W[ek]\n;B[gj]\n;W[oa]\n;B[pb]\n;W[lf]\n;B[kf]\n;W[mf]\n;B[iq]\n;W[ir]\n;B[jp]\n;W[kp]\n;B[os]\n;W[ns]\n;B[ps]\n;W[mr]\n;B[ad]\n;W[ae]\n;B[ac]\n;W[ed]\n;B[dc]\n;W[fi]\n;B[ik]\n;W[oj]\n;B[pj]\n;W[lk]\n;B[nl]\n;W[rf]\n;B[sd]\n;W[jo]\n;B[sj]\n;W[si]\n;B[ic]\n;W[ib]\n;B[mh]\n;W[gi]\n;B[hi]\n;W[eo]\n;B[pa]\n;W[na]\n;B[aq]\n;W[er]\n;B[ep]\n;W[gh]\n;B[gg]\n;W[op]\n;B[oo]\n;W[ds]\n;B[cs]\n;W[es]\n;B[jk]\n;W[jl]\n;B[ap]\n;W[ao]\n;B[ip]\n;W[sf]\n;B[se]\n;W[oe]\n;B[od]\n;W[md]\n;B[ge]\n;W[fp]\n;B[kg]\n;W[kq]\n;B[]\n;W[]\n)\n'
    sgf = Sgf_game.from_string(sgf_content)
    # feature_shape = tuple(np.insert(shape, 0, np.asarray([165])))
    feature_shape = tuple(np.insert(shape, 0, np.asarray([217])))
    features = np.zeros(feature_shape)
    # labels = np.zeros((165,))
    labels = np.zeros((217,))

    go_board = Board(19, 19)
    first_move_done = False
    move = None
    game_state = GameState.new_game(19)
    if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
        for setup in sgf.get_root().get_setup_stones():
            for move in setup:
                row, col = move
                go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
        first_move_done = True
        game_state = GameState(go_board, Player.white, None, move)

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

    # data_file_name = 'KGS-2008-19-14002-train'
    data_file_name = 'KGS-2007-19-11644-train'
    path = Path.cwd()
    dir = path / 'chosen_data'
    feature_file = dir.joinpath(f'{data_file_name}_features_1_1')
    label_file = dir.joinpath(f'{data_file_name}_labels_1_1')
    np.save(str(feature_file), features)
    np.save(str(label_file), labels)
    x = np.load(str(feature_file) + '.npy')
    y = np.load(str(label_file) + '.npy')
    print(x.shape, y.shape)
    x2 = x.astype('float32')
    y2 = to_categorical(y.astype(int), 361)
    print(x2.shape, y2.shape)


def test_generator():
    path = Path.cwd()
    dir = path / 'chosen_data'
    samples = [('KGS-2008-19-14002-.tar.gz', 9791), ('KGS-2007-19-11644-.tar.gz', 2756)]
    # samples = [('KGS-2007-19-11644-.tar.gz', 2756), ('KGS-2008-19-14002-.tar.gz', 9791)]
    # games: 2008-07-01-4.sgf, 2007-07-28-25.sgf
    generator = DataGenerator(dir, samples, 19, 'train')
    gen = generator.generate(32)
    x, y = next(gen)
    pos = x[11]
    decoder = BoardDecoder(pos)
    decoder.print()
    move = np.argmax(y[11], axis=None, out=None)
    encoder = get_encoder_by_name('simple', 19)
    point = encoder.decode_point_index(move)
    print(move)
    print(point)


def main():
    test_generator()


if __name__ == '__main__':
    main()
