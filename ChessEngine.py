

class Gamestate():
    def __init__(self):

        #board is an 8x8 2D-list. each element has two char
        # first chat - color piece
        #2nd char - type of piece
        self.board = [
            ["bR","bN","bB","bQ","bK","bB","bN","bR"],
            ["bP","bP","bP","bP","bP","bP","bP","bP"],
            ["--","--","--","--","--","--","--","--"],
            ["--", "--", "--", "--", "--", "--", "--","--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "bP", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        ]

        self.whiteToMove = True
        self.moveLog= []


    def makeMove(self,move):
        self.board[move.startRow][move.startCol] = "--"
        self.board[move.endRow][move.endCol] = move.pieceMoved
        self.moveLog.append(move)
        self.whiteToMove = not self.whiteToMove # swap player

    def undoMove(self):
        if len(self.moveLog) != 0 :
            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove

    def getValidMoves(self):
        return self.getAllPossibleMoves()
    #----------------------------





    #-------------------------------
    def getAllPossibleMoves(self):
        moves = [] #Move((6,4),(4,4),self.board)
        for r in range(len(self.board)) :
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]  # fetching first letter
                if(turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    if piece  == 'P' :
                        # print("here line 47")
                        self.getPawnMoves(r,c,moves)


                        print(moves)
                    elif piece == 'R':
                        self.getRookMoves(r, c, moves)
                    elif piece == 'N':
                        self.getKnightMoves(r, c, moves)
                    elif piece == 'B':
                        self.getBishopMoves(r, c, moves)
                    elif piece == 'Q':
                        self.getQueenMoves(r, c, moves)
                    elif piece == 'K':
                        self.getKingMoves(r, c, moves)
        return moves

    def getPawnMoves(self, r, c, moves):
        """
        Get all pawn moves for the pawn located at row r, column c
        and add the moves to the list
        """
        if self.whiteToMove:  # white pawn moves
            # Normal forward move
            if r - 1 >= 0 and self.board[r - 1][c] == "--":
                moves.append(Move((r, c), (r - 1, c), self.board))
                # Initial two-square move
                if r == 6 and self.board[r - 2][c] == "--":
                    moves.append(Move((r, c), (r - 2, c), self.board))

            # Captures to the left
            if r - 1 >= 0 and c - 1 >= 0:  # checking bounds
                if self.board[r - 1][c - 1][0] == 'b':  # enemy piece to capture
                    moves.append(Move((r, c), (r - 1, c - 1), self.board))

            # Captures to the right
            if r - 1 >= 0 and c + 1 <= 7:  # checking bounds
                if self.board[r - 1][c + 1][0] == 'b':  # enemy piece to capture
                    moves.append(Move((r, c), (r - 1, c + 1), self.board))

        else:  # black pawn moves
            # Normal forward move
            if r + 1 <= 7 and self.board[r + 1][c] == "--":
                moves.append(Move((r, c), (r + 1, c), self.board))
                # Initial two-square move
                if r == 1 and self.board[r + 2][c] == "--":
                    moves.append(Move((r, c), (r + 2, c), self.board))

            # Captures to the left
            if r + 1 <= 7 and c - 1 >= 0:  # checking bounds
                if self.board[r + 1][c - 1][0] == 'w':  # enemy piece to capture
                    moves.append(Move((r, c), (r + 1, c - 1), self.board))

            # Captures to the right
            if r + 1 <= 7 and c + 1 <= 7:  # checking bounds
                if self.board[r + 1][c + 1][0] == 'w':  # enemy piece to capture
                    moves.append(Move((r, c), (r + 1, c + 1), self.board))

    def getRookMoves(self, r, c, moves):
        """Get all rook moves for the rook at position r, c and add to moves list"""
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # up, left, down, right
        enemyColor = "b" if self.whiteToMove else "w"

        for direction in directions:
            for i in range(1, 8):
                endRow = r + direction[0] * i
                endCol = c + direction[1] * i
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:  # check if on board
                    endPiece = self.board[endRow][endCol]
                    if endPiece == "--":  # empty space is valid
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor:  # enemy piece can be captured
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else:  # friendly piece
                        break
                else:  # off board
                    break

    def getBishopMoves(self, r, c, moves):
        """Get all bishop moves for the bishop at position r, c and add to moves list"""
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonals
        enemyColor = "b" if self.whiteToMove else "w"

        for direction in directions:
            for i in range(1, 8):
                endRow = r + direction[0] * i
                endCol = c + direction[1] * i
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                    endPiece = self.board[endRow][endCol]
                    if endPiece == "--":
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break

    def getKnightMoves(self, r, c, moves):
        """Get all knight moves for the knight at position r, c and add to moves list"""
        knightMoves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]  # all possible L-shapes
        allyColor = "w" if self.whiteToMove else "b"

        for move in knightMoves:
            endRow = r + move[0]
            endCol = c + move[1]
            if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor:  # not an ally piece (empty or enemy)
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    def getQueenMoves(self, r, c, moves):
        """Get all queen moves for the queen at position r, c and add to moves list"""
        # Queen moves are combination of rook and bishop moves
        self.getRookMoves(r, c, moves)
        self.getBishopMoves(r, c, moves)

    def getKingMoves(self, r, c, moves):
        """Get all king moves for the king at position r, c and add to moves list"""
        kingMoves = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]  # all 8 squares around king
        allyColor = "w" if self.whiteToMove else "b"

        for move in kingMoves:
            endRow = r + move[0]
            endCol = c + move[1]
            if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor:  # not an ally piece
                    moves.append(Move((r, c), (endRow, endCol), self.board))

class Move():

    #maps keys to value
    rankToRows = {"1":7,'2':6 , "3" : 5 , "4" : 4 , "5" : 3 , "6" : 2 , "7" : 1 , "8" : 0 }
    rowsToRank = {v: k for k, v in rankToRows.items()}
    filesToCols = {"a" : 0, "b" : 1, "c" : 2, "d" : 3 ,"e" : 4,"f" : 5, "g" : 6, "h" : 7 }
    colsToFile = { v : k for k , v in filesToCols.items()}



    def __init__(self,startSq,endSq,board):
        self.startRow  = startSq[0]
        self.startCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]
        self.moveID = self.startRow * 1000 + self.startCol*100 + self.endRow * 10 + self.endCol
        print(self.moveID)
    '''
    over riding the equals method
    '''

    def __eq__(self,other):
        if isinstance(other,Move):return self.moveID == other.moveID
        return False


    def getChessNotation(self):
        return self.getRankFile(self.startRow,self.startCol) + self.getRankFile(self.endRow,self.endCol)

    def getRankFile(self,r,c):
        return self.colsToFile[c] + self.rowsToRank[r]


