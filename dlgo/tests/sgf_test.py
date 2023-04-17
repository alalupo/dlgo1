import unittest

from dlgo.gosgf.sgf_properties import interpret_go_point, serialise_go_point


#    A B C D E F G H I J K L M N O P Q R S
# 18 . . . . . . . . . . . . . . . . . . . A
# 17 . . . . . . . . . . . . . . . . . . . B
# 16 . . . . . . . . . . . . . . . . . . . C
# 15 . . . . . . . . . . . . . . . . x . . D
# 14 . . . . . . . . . . . . . . . . . . . E
# 13 . . . . . . . . . . . . . . . . . . . F
# 12 . . . . . . . . . . . . . . . . . . . G
# 11 . . . . . . . . . . . . . . . . . . . H
# 10 . . . . . . . . . . . . . . . . . . . I
#  9 . . . . . . . . . . . . . . . . . . . J
#  8 . . . . . . . . . . . . . . . . . . . K
#  7 . . . . . . . . . . . . . . . . . . . L
#  6 . . . . . . . . . . . . . . . . . . . M
#  5 . . . . . . . . . . . . . . . . . . . N
#  4 . . . . . . . . . . . . . . . . . . . O
#  3 . . . . . . . . . . . . . . . . . . . P
#  2 . . . . . . . . . . . . . . o . . . . Q
#  1 . . . . . . . . . . . . . . . . . . . R
#  0 . . . . . . . . . . . . . . . . . . . S

# B[qd] = row D, col Q
# W[oq] = row Q, col O

# It seems all online editors interpret a goboard this way
# despite the fact that SGF specification says there is no
# letter 'i' among the coordinates.


class SgfTest(unittest.TestCase):

    def test_sgf_move_coding(self):
        coords = 'oq'
        eight_bit_string = coords.encode('ascii')
        row, col = interpret_go_point(eight_bit_string, 19)
        print(f'row={row}, col={col}')
        self.assertEqual(2, row)
        self.assertEqual(14, col)

        coords = 'aa'
        eight_bit_string = coords.encode('ascii')
        row, col = interpret_go_point(eight_bit_string, 19)
        print(f'row={row}, col={col}')
        self.assertEqual(18, row)
        self.assertEqual(0, col)

        coords = 'ab'
        eight_bit_string = coords.encode('ascii')
        row, col = interpret_go_point(eight_bit_string, 19)
        print(f'row={row}, col={col}')
        self.assertEqual(17, row)
        self.assertEqual(0, col)

        coords = 'an'
        eight_bit_string = coords.encode('ascii')
        row, col = interpret_go_point(eight_bit_string, 19)
        print(f'row={row}, col={col}')
        self.assertEqual(5, row)
        self.assertEqual(0, col)

        coords = 'as'
        eight_bit_string = coords.encode('ascii')
        row, col = interpret_go_point(eight_bit_string, 19)
        print(f'row={row}, col={col}')
        self.assertEqual(0, row)
        self.assertEqual(0, col)

        coords = 'ss'
        eight_bit_string = coords.encode('ascii')
        row, col = interpret_go_point(eight_bit_string, 19)
        print(f'row={row}, col={col}')
        self.assertEqual(0, row)
        self.assertEqual(18, col)

        coords = 'ii'
        eight_bit_string = coords.encode('ascii')
        row, col = interpret_go_point(eight_bit_string, 19)
        print(f'row={row}, col={col}')
        self.assertEqual(10, row)
        self.assertEqual(8, col)

        point = (0, 0)
        outcome = serialise_go_point(point, 19)
        print(f'outcome for point: {point} is {outcome}')
        self.assertEqual(b'as', outcome)

        point = (0, 18)
        outcome = serialise_go_point(point, 19)
        print(f'outcome for point: {point} is {outcome}')
        self.assertEqual(b'ss', outcome)

        point = (18, 18)
        outcome = serialise_go_point(point, 19)
        print(f'outcome for point: {point} is {outcome}')
        self.assertEqual(b'sa', outcome)

        point = (10, 10)
        outcome = serialise_go_point(point, 19)
        print(f'outcome for point: {point} is {outcome}')
        self.assertEqual(b'ki', outcome)

        point = (2, 18)
        outcome = serialise_go_point(point, 19)
        print(f'outcome for point: {point} is {outcome}')
        self.assertEqual(b'sq', outcome)
