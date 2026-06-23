"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] is EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if board[i][j] is not EMPTY:
        raise ValueError("Invalid action: cell already occupied.")

    new_board = [row[:] for row in board]
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = (
        [board[r] for r in range(3)],
        [[board[r][c] for r in range(3)] for c in range(3)],
        [[board[i][i] for i in range(3)]],
        [[board[i][2 - i] for i in range(3)]],
    )

    for group in lines:
        for line in group:
            if line[0] is not EMPTY and line[0] == line[1] == line[2]:
                return line[0]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    return all(board[i][j] is not EMPTY for i in range(3) for j in range(3))


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    w = winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current = player(board)

    if current == X:
        _, action = max_value(board, -math.inf, math.inf)
    else:
        _, action = min_value(board, -math.inf, math.inf)

    return action


def max_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    v = -math.inf
    best_action = None

    for action in actions(board):
        score, _ = min_value(result(board, action), alpha, beta)
        if score > v:
            v = score
            best_action = action
        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return v, best_action


def min_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    v = math.inf
    best_action = None

    for action in actions(board):
        score, _ = max_value(result(board, action), alpha, beta)
        if score < v:
            v = score
            best_action = action
        beta = min(beta, v)
        if beta <= alpha:
            break

    return v, best_action