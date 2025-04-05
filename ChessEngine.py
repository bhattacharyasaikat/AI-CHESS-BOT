# ChessEngine.py
import math
import copy
import time # Optional: for debugging timings

class GameState():
    # --- Method Definitions FIRST ---

    # makes the move
    def makeMove(self, move):
        self.board[move.startRow][move.startCol] = '--'
        self.board[move.endRow][move.endCol] = move.pieceMoved
        self.moveLog.append(move)
        self.whiteToMove = not self.whiteToMove

        # Update King Location FIRST
        if move.pieceMoved == 'wK':
            self.wKingLoc = (move.endRow, move.endCol)
        elif move.pieceMoved == 'bK':
            self.bKingLoc = (move.endRow, move.endCol)

        # Handle Pawn Promotion
        if move.isPawnPromotion:
            self.board[move.endRow][move.endCol] = move.pieceMoved[0] + 'Q'

        # Handle En Passant Capture
        if move.isEnpassantMove:
            self.board[move.startRow][move.endCol] = '--' # Clear captured pawn

        # Update En Passant Possible State for next turn
        if move.pieceMoved[1] == 'P' and abs(move.startRow - move.endRow) == 2:
            self.enpassantPossible = ((move.startRow + move.endRow) // 2, move.startCol)
        else:
            self.enpassantPossible = ()

        # --- Handle Castle Rook Move ---
        if move.isCastleMove:
            # Sanity Check: Verify it looks like a real castle move based on coords/piece
            expected_end_col_diff = abs(move.endCol - move.startCol)
            is_king_move_correct = (expected_end_col_diff == 2) and \
                                   (move.pieceMoved[1] == 'K') and \
                                   (move.startRow == move.endRow) and \
                                   (move.startRow == 0 or move.startRow == 7)

            if not is_king_move_correct:
                 # Log error but DON'T raise exception, just skip the rook move part
                 print(f"ERROR: Corrupt castle move object detected in makeMove! Skipping rook move.")
                 print(f"Move: {move}, Start: ({move.startRow},{move.startCol}), End: ({move.endRow},{move.endCol}), Piece: {move.pieceMoved}")
            else:
                 # --- Move Rook Logically ---
                 if move.endCol - move.startCol == 2: # Kingside
                     rook_start_col = 7
                     rook_end_col = 5
                     if 0 <= rook_start_col < 8 and 0 <= rook_end_col < 8: # Bounds check
                         self.board[move.endRow][rook_end_col] = self.board[move.endRow][rook_start_col]
                         self.board[move.endRow][rook_start_col] = '--'
                     else: # Should not happen if is_king_move_correct passed
                          print(f"WARNING: Invalid rook cols calculated KS make: {rook_start_col}->{rook_end_col}")
                 else: # Queenside
                     rook_start_col = 0
                     rook_end_col = 3
                     if 0 <= rook_start_col < 8 and 0 <= rook_end_col < 8: # Bounds check
                         self.board[move.endRow][rook_end_col] = self.board[move.endRow][rook_start_col]
                         self.board[move.endRow][rook_start_col] = '--'
                     else: # Should not happen
                          print(f"WARNING: Invalid rook cols calculated QS make: {rook_start_col}->{rook_end_col}")

        # --- Update castling rights ---
        # Store previous rights for the log BEFORE updating
        # prev_castle_rights = copy.deepcopy(self.currentCastlingRight) # No, log stores state BEFORE move
        self.updateCastleRights(move)
        # Append the NEW state AFTER the move to the log
        self.castleRightsLog.append(copy.deepcopy(self.currentCastlingRight))

        # Reset checkmate/stalemate flags - they are properties of the *next* position
        self.checkMate = False
        self.stalemate = False


    def undoMove(self):
        if not self.moveLog: # Check if log is empty
             print("Warning: UndoMove called on empty movelog.")
             return

        move = self.moveLog.pop()
        self.board[move.startRow][move.startCol] = move.pieceMoved
        self.board[move.endRow][move.endCol] = move.pieceCaptured # Restore captured piece
        self.whiteToMove = not self.whiteToMove

        # Restore King Location FIRST
        if move.pieceMoved == 'wK':
            self.wKingLoc = (move.startRow, move.startCol)
        elif move.pieceMoved == 'bK':
            self.bKingLoc = (move.startRow, move.startCol)

        # --- Restore Previous En Passant Possible State ---
        # The en passant state depends on the move BEFORE the one just undone.
        if len(self.moveLog) > 0:
             prev_move = self.moveLog[-1]
             if prev_move.pieceMoved[1] == 'P' and abs(prev_move.startRow - prev_move.endRow) == 2:
                  self.enpassantPossible = ((prev_move.startRow + prev_move.endRow) // 2, prev_move.startCol)
             else:
                  self.enpassantPossible = ()
        else: # If we undid the first move
             self.enpassantPossible = ()


        # --- Restore Previous Castling Rights ---
        self.castleRightsLog.pop() # Remove rights state resulting from the move being undone
        if self.castleRightsLog: # Check if log is now empty
            self.currentCastlingRight = copy.deepcopy(self.castleRightsLog[-1]) # Restore previous state
        else: # If we undid the first move, restore initial rights
             self.currentCastlingRight = castleRights(True, True, True, True)


        # --- Undo castle ROOK move ---
        if move.isCastleMove:
             # Sanity Check: Verify it looked like a real castle move based on coords/piece
            expected_end_col_diff = abs(move.endCol - move.startCol)
            is_king_move_correct = (expected_end_col_diff == 2) and \
                                   (move.pieceMoved[1] == 'K') and \
                                   (move.startRow == move.endRow) and \
                                   (move.startRow == 0 or move.startRow == 7)
            if not is_king_move_correct:
                # Log error but DON'T raise exception, just skip the rook move part
                 print(f"ERROR: Corrupt castle move object detected in undoMove! Skipping rook move.")
                 print(f"Move: {move}, Start: ({move.startRow},{move.startCol}), End: ({move.endRow},{move.endCol}), Piece: {move.pieceMoved}")
            else:
                # --- Undo Rook Move Logically ---
                if move.endCol - move.startCol == 2: # Kingside (Undo F->H)
                    rook_original_col = 7 # H
                    rook_moved_to_col = 5 # F
                    if 0 <= rook_original_col < 8 and 0 <= rook_moved_to_col < 8: # Bounds check
                        self.board[move.endRow][rook_original_col] = self.board[move.endRow][rook_moved_to_col]
                        self.board[move.endRow][rook_moved_to_col] = '--'
                    else:
                         print(f"WARNING: Invalid rook cols calculated KS undo: {rook_original_col}, {rook_moved_to_col}")

                else: # Queenside (Undo D->A)
                    rook_original_col = 0 # A
                    rook_moved_to_col = 3 # D
                    if 0 <= rook_original_col < 8 and 0 <= rook_moved_to_col < 8: # Bounds check
                        self.board[move.endRow][rook_original_col] = self.board[move.endRow][rook_moved_to_col]
                        self.board[move.endRow][rook_moved_to_col] = '--'
                    else:
                         print(f"WARNING: Invalid rook cols calculated QS undo: {rook_original_col}, {rook_moved_to_col}")

        # --- Undo En Passant Capture ---
        if move.isEnpassantMove:
            # If we undid an en passant move, the captured square is now empty ('--')
            # The captured pawn needs to be restored at startRow, endCol
            captured_pawn_color = 'b' if move.pieceMoved[0] == 'w' else 'w'
            self.board[move.startRow][move.endCol] = captured_pawn_color + 'P'


        # Reset checkmate/stalemate flags - they are properties of the position *before* this move
        self.checkMate = False
        self.stalemate = False



    def updateCastleRights(self, move):
        piece_moved = move.pieceMoved
        start_row, start_col = move.startRow, move.startCol
        end_row, end_col = move.endRow, move.endCol # Use end_row/col for captures

        # --- King Moves ---
        if piece_moved == 'wK':
            self.currentCastlingRight.wks = False
            self.currentCastlingRight.wqs = False
        elif piece_moved == 'bK':
            self.currentCastlingRight.bks = False
            self.currentCastlingRight.bqs = False

        # --- Rook Moves ---
        elif piece_moved == 'wR':
            if start_row == 7: # White's back rank
                if start_col == 0: # A1 rook moved
                    self.currentCastlingRight.wqs = False
                elif start_col == 7: # H1 rook moved
                    self.currentCastlingRight.wks = False
        elif piece_moved == 'bR':
            if start_row == 0: # Black's back rank
                if start_col == 0: # A8 rook moved
                    self.currentCastlingRight.bqs = False
                elif start_col == 7: # H8 rook moved
                    self.currentCastlingRight.bks = False

        # --- Rook Captures ---
        # Need to check the square the captured piece was on
        piece_captured = move.pieceCaptured
        if piece_captured == 'wR':
             if end_row == 7: # If a white rook was captured on white's back rank
                  if end_col == 0: # A1 captured
                     self.currentCastlingRight.wqs = False
                  elif end_col == 7: # H1 captured
                     self.currentCastlingRight.wks = False
        elif piece_captured == 'bR':
             if end_row == 0: # If a black rook was captured on black's back rank
                  if end_col == 0: # A8 captured
                      self.currentCastlingRight.bqs = False
                  elif end_col == 7: # H8 captured
                      self.currentCastlingRight.bks = False


    # =====================================================
    # ===               MOVE VALIDATION                 ===
    # =====================================================
    def getValidMoves(self):
        # Save state
        tempEnpassantPossible = self.enpassantPossible
        tempCastleRights = copy.deepcopy(self.currentCastlingRight)

        # 1. Generate all pseudo-legal NON-CASTLING moves first
        pseudo_legal_moves = self.getAllMoves() # This gets pawn, rook, knight, bishop, queen, REGULAR king moves

        # 2. Generate pseudo-legal CASTLING moves separately
        castle_moves = []
        king_row, king_col = self.wKingLoc if self.whiteToMove else self.bKingLoc
        # Make sure the piece at king_loc is actually the king BEFORE attempting castle generation
        if 0 <= king_row < 8 and 0 <= king_col < 8 and \
           self.board[king_row][king_col] == ('wK' if self.whiteToMove else 'bK'):
            self.getCastleMoves(king_row, king_col, castle_moves) # Appends moves with isCastleMove=True

        # Combine the lists
        all_pseudo_legal = pseudo_legal_moves + castle_moves

        # 3. Filter ALL generated moves for legality
        legal_moves = []
        # *** Important: Reset flags before filtering loop ***
        self.checkMate = False
        self.stalemate = False

        for move in all_pseudo_legal: # Iterate through the combined list
            # --- Try making the move ---
            # Need to handle potential errors during makeMove if state is deeply corrupted
            try:
                self.makeMove(move) # Player A makes move, turn switches to B
            except Exception as e:
                print(f"ERROR during makeMove in getValidMoves filter: {e}")
                print(f"Problematic move: {move.getChessNotation()}")
                continue # Skip this move if makeMove fails

            # --- CORRECTED LEGALITY CHECK ---
            self.whiteToMove = not self.whiteToMove # Back to Player A's turn
            is_legal = not self.inCheck() # Check if Player A's king is safe AFTER the move
            self.whiteToMove = not self.whiteToMove # Back to Player B's turn
            # --- END CORRECTION ---

            if is_legal:
                legal_moves.append(move) # Add the fully validated legal move

            # --- Undo the move ---
            try:
                 self.undoMove() # Undo move, turn switches back to A
            except Exception as e:
                 print(f"ERROR during undoMove in getValidMoves filter: {e}")
                 print(f"Problematic move: {move.getChessNotation()}")
                 # If undo fails, the state is likely corrupt, maybe stop or try to recover?
                 # For now, just print and continue the loop, hoping later moves are okay.
                 # However, this indicates a deeper issue needs fixing.
                 pass


        # 4. Check for checkmate/stalemate based on legal move count *in original position*
        if len(legal_moves) == 0:
            # Must check if the king is *currently* under attack IN THE ORIGINAL POSITION
            if self.inCheck():
                self.checkMate = True
            else:
                self.stalemate = True
        # else: # Flags should have been reset before the loop
        #      self.checkMate = False
        #      self.stalemate = False


        # Restore state just to be safe, though make/undo should handle it
        self.enpassantPossible = tempEnpassantPossible
        self.currentCastlingRight = tempCastleRights # Restore from saved object

        return legal_moves # Return the list containing only strictly legal moves

    # --- Check/Attack functions ---
    def inCheck(self):
        """Checks if the current player's king is under attack."""
        king_row, king_col = self.wKingLoc if self.whiteToMove else self.bKingLoc
        # Add check for valid king location before calling squareUnderAttack
        if not (0 <= king_row < 8 and 0 <= king_col < 8):
             # This indicates a serious state corruption if king location is invalid
             print(f"ERROR: Invalid king location ({king_row},{king_col}) for {'White' if self.whiteToMove else 'Black'}")
             return True # Assume check if king location is invalid
        return self.squareUnderAttack(king_row, king_col)

    def squareUnderAttack(self, r, c):
        """Checks if the opponent can attack square (r, c)."""
        self.whiteToMove = not self.whiteToMove # Switch to opponent's perspective
        oppMoves = self.getAllMoves() # Get opponent's pseudo-legal moves
        self.whiteToMove = not self.whiteToMove # Switch back immediately
        for move in oppMoves:
            if move.endRow == r and move.endCol == c:
                return True # Found an attacking move
        return False


    # =====================================================
    # ===         PSEUDO-LEGAL MOVE GENERATION          ===
    # =====================================================
    def getAllMoves(self):
        """Generates all pseudo-legal moves (doesn't check for checks)."""
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if (turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    # Make sure piece is valid before calling function
                    if piece in self.moveFunctions:
                        try:
                             self.moveFunctions[piece](r, c, moves) # Calls specific piece move func
                        except Exception as e:
                             print(f"ERROR in move generation for {piece} at ({r},{c}): {e}")
                    # else: print(f"Warning: Unknown piece type '{piece}' at ({r},{c})") # Debug
        return moves

    def getPawnMoves(self, r, c, moves):
        piece_color = self.board[r][c][0]
        direction = -1 if piece_color == 'w' else 1
        opp_color = 'b' if piece_color == 'w' else 'w'
        start_row = 6 if piece_color == 'w' else 1

        # 1. One square forward
        if 0 <= r + direction < 8 and self.board[r + direction][c] == '--':
            moves.append(Move((r, c), (r + direction, c), self.board))
            # 2. Two squares forward (only if one square is possible and on start row)
            if r == start_row and 0 <= r + 2 * direction < 8 and self.board[r + 2 * direction][c] == '--': # Check bounds for 2 squares
                moves.append(Move((r, c), (r + 2 * direction, c), self.board))

        # 3. Captures
        for dc in [-1, 1]: # Check left and right columns
            if 0 <= c + dc < 8:
                # Diagonal capture
                if 0 <= r + direction < 8 and self.board[r + direction][c + dc][0] == opp_color:
                    moves.append(Move((r, c), (r + direction, c + dc), self.board))
                # En passant capture
                if (r + direction, c + dc) == self.enpassantPossible:
                     moves.append(Move((r, c), (r + direction, c + dc), self.board, isEnpassantMove = True))

    def getRookMoves(self, r, c, moves):
        directions = ((-1,0), (0,-1), (1,0), (0,1))
        ally_color = self.board[r][c][0]
        for d in directions:
            for i in range(1,8):
                endRow, endCol = r + d[0] * i, c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow][endCol]
                    if endPiece == '--':
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    else: # Hit a piece
                        if endPiece[0] != ally_color: # Capture opponent
                            moves.append(Move((r, c), (endRow, endCol), self.board))
                        break # Stop in this direction (hit own piece or captured opponent)
                else: # Off board
                    break

    def getBishopMoves(self, r, c, moves):
        directions = ((-1,-1), (1,-1), (1,1), (-1,1))
        ally_color = self.board[r][c][0]
        for d in directions:
            for i in range(1,8):
                endRow, endCol = r + d[0] * i, c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow][endCol]
                    if endPiece == '--':
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    else: # Hit a piece
                        if endPiece[0] != ally_color: # Capture opponent
                            moves.append(Move((r, c), (endRow, endCol), self.board))
                        break # Stop in this direction
                else: # Off board
                    break

    def getQueenMoves(self, r, c, moves):
        self.getRookMoves(r, c, moves)
        self.getBishopMoves(r, c, moves)

    def getKnightMoves(self, r, c, moves):
        knightMoves = ((-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1))
        ally_color = self.board[r][c][0]
        for m in knightMoves:
            endRow, endCol = r + m[0], c + m[1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow][endCol]
                if endPiece == '--' or endPiece[0] != ally_color: # Empty or capture
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    def getKingMoves(self, r, c, moves):
        directions = ((-1,-1), (1,-1), (1,1), (-1,1), (-1,0), (0,-1), (1,0), (0,1))
        ally_color = self.board[r][c][0]
        for d in directions:
            endRow, endCol = r + d[0], c + d[1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow][endCol]
                if endPiece == '--' or endPiece[0] != ally_color:
                    moves.append(Move((r, c), (endRow, endCol), self.board))
        # Note: Castle moves are added separately in getValidMoves after calling this

    # --- Castle Move Generation (called from getValidMoves) ---
    # def getCastleMoves(self, r, c, moves):
    #     """ Adds pseudo-legal castle moves to the list. """
    #     # *** Explicitly check if king is on the correct rank ***
    #     correct_rank = (self.whiteToMove and r == 7) or (not self.whiteToMove and r == 0)
    #     if not correct_rank:
    #          # print(f"DEBUG: getCastleMoves called for wrong rank {r} for {'W' if self.whiteToMove else 'B'}")
    #          return # Don't generate castles if king isn't on back rank

    #     # Cannot castle if currently in check
    #     if self.squareUnderAttack(r, c):
    #         return

    #     # Check Kingside
    #     if (self.whiteToMove and self.currentCastlingRight.wks) or \
    #        (not self.whiteToMove and self.currentCastlingRight.bks):
    #         if c + 2 < 8: # Boundary check
    #             if self.board[r][c+1] == '--' and self.board[r][c+2] == '--':
    #                 if not self.squareUnderAttack(r, c+1) and not self.squareUnderAttack(r, c+2):
    #                     moves.append(Move((r, c), (r, c+2), self.board, isCastleMove=True))

    #     # Check Queenside
    #     if (self.whiteToMove and self.currentCastlingRight.wqs) or \
    #        (not self.whiteToMove and self.currentCastlingRight.bqs):
    #         if c - 3 >= 0: # Boundary check
    #             if self.board[r][c-1] == '--' and self.board[r][c-2] == '--' and self.board[r][c-3] == '--':
    #                 if not self.squareUnderAttack(r, c-1) and not self.squareUnderAttack(r, c-2):
    #                     moves.append(Move((r, c), (r, c-2), self.board, isCastleMove=True))

    def getCastleMoves(self, r, c, moves):
        """ Adds pseudo-legal castle moves to the list. """
        if self.squareUnderAttack(r, c):
            return

        # Kingside castling
        if (self.whiteToMove and self.currentCastlingRight.wks) or \
           (not self.whiteToMove and self.currentCastlingRight.bks):
            if c + 2 < 8:  # Ensure g and h files are accessible
                if self.board[r][c+1] == '--' and self.board[r][c+2] == '--':
                    if not self.squareUnderAttack(r, c+1) and not self.squareUnderAttack(r, c+2):
                        moves.append(Move((r, c), (r, c+2), self.board, isCastleMove=True))

        # Queenside castling - CORRECTED SECTION
        if (self.whiteToMove and self.currentCastlingRight.wqs) or \
           (not self.whiteToMove and self.currentCastlingRight.bqs):
            if c - 2 >= 0:  # Corrected boundary check (was c - 3)
                if self.board[r][c-1] == '--' and self.board[r][c-2] == '--':  # Check correct squares
                    if not self.squareUnderAttack(r, c-1) and not self.squareUnderAttack(r, c-2):
                        moves.append(Move((r, c), (r, c-2), self.board, isCastleMove=True))

    # =====================================================
    # ===             MINIMAX SEARCH ENGINE             ===
    # =====================================================
    def minimax(self, depth, alpha, beta, isMaximiser):
        """ Recursive minimax function with alpha-beta pruning. """
        # Generate valid moves for the current state to check for terminal nodes
        possibleMoves = self.getValidMoves() # Updates self.checkMate, self.stalemate flags

        # Check for terminal state (checkmate/stalemate) or depth limit
        if self.checkMate:
            mate_score = math.inf + depth if not isMaximiser else -math.inf - depth
            return mate_score
        elif self.stalemate:
            return 0 # Draw
        elif depth == 0:
            # Reached depth limit, evaluate statically
            return self.evaluate_static_position()

        # --- Recursive Search ---
        if isMaximiser: # White (usually)
            maxEval = -math.inf
            for move in possibleMoves:
                self.makeMove(move)
                evaluation = self.minimax(depth - 1, alpha, beta, False) # Opponent plays next (minimizer)
                self.undoMove()
                if evaluation > maxEval:
                   maxEval = evaluation
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break # Alpha-beta pruning
            return maxEval
        else: # Minimiser (Black usually)
            minEval = math.inf
            for move in possibleMoves:
                self.makeMove(move)
                evaluation = self.minimax(depth - 1, alpha, beta, True) # Opponent plays next (maximizer)
                self.undoMove()
                if evaluation < minEval:
                    minEval = evaluation
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break # Alpha-beta pruning
            return minEval

    # --- Get Best Move (Top Level Search Function) ---
    def getBestMove(self, depth, isMaximiser):
        """ Finds the best move using minimax search. """
        start_time = time.time() # Optional timing

        # --- Save state before searching ---
        original_checkmate = self.checkMate
        original_stalemate = self.stalemate
        original_castle_rights = copy.deepcopy(self.currentCastlingRight)
        original_enpassant = self.enpassantPossible

        possibleMoves = self.getValidMoves() # Get legal moves for the current position

        # --- Handle terminal states immediately ---
        if self.checkMate or self.stalemate:
             self.checkMate = original_checkmate # Restore state before returning
             self.stalemate = original_stalemate
             self.currentCastlingRight = original_castle_rights
             self.enpassantPossible = original_enpassant
             return None # No move to return

        bestMove = None
        bestValue = -math.inf if isMaximiser else math.inf

        # --- Iterate through legal moves at the root ---
        for move in possibleMoves:
            self.makeMove(move)
            value = self.minimax(depth - 1, -math.inf, math.inf, not isMaximiser)
            self.undoMove()

            # --- Update Best Move based on evaluation ---
            if isMaximiser:
                if bestMove is None or value > bestValue:
                    bestValue = value
                    bestMove = copy.deepcopy(move)
                if bestValue >= math.inf: # Check >= infinity for mate scores
                     break # Found mate
            else: # Minimiser
                if bestMove is None or value < bestValue:
                    bestValue = value
                    bestMove = copy.deepcopy(move)
                if bestValue <= -math.inf: # Check <= -infinity for mate scores
                    break # Found mate

        # --- Restore state after searching ---
        self.checkMate = original_checkmate
        self.stalemate = original_stalemate
        self.currentCastlingRight = original_castle_rights
        self.enpassantPossible = original_enpassant

        end_time = time.time()
        # --- Optional Debug Print ---
        # if bestMove:
        #      print(f"GetBestMove found: {bestMove.getChessNotation()} (Value: {bestValue:.2f}) Time: {end_time - start_time:.3f}s")
        # elif not (self.checkMate or self.stalemate):
        #      print(f"GetBestMove found no move, but not terminal? Time: {end_time - start_time:.3f}s")


        # Failsafe: If loop finishes and bestMove is still None (and not terminal), pick first possible.
        if bestMove is None and not (self.checkMate or self.stalemate) and possibleMoves:
            print(f"WARNING: No best move selected despite {len(possibleMoves)} legal moves. Returning first.")
            bestMove = copy.deepcopy(possibleMoves[0])

        return bestMove


    # =====================================================
    # ===            STATIC BOARD EVALUATION            ===
    # =====================================================
    def evaluate_static_position(self):
        """ Calculates static score (material + position) without move generation. """

        # --- Piece Square Tables ---
        #Pawns
        wPpieceSquare = [[ 0,  0,  0,  0,  0,  0,  0,  0],
                         [50, 50, 50, 50, 50, 50, 50, 50],
                         [10, 10, 20, 30, 30, 20, 10, 10],
                         [ 5,  5, 10, 25, 25, 10,  5,  5],
                         [ 0,  0,  0, 20, 20,  0,  0,  0],
                         [ 5, -5,-10,  0,  0,-10, -5,  5],
                         [ 5, 10, 10,-20,-20, 10, 10,  5],
                         [ 0,  0,  0,  0,  0,  0,  0,  0]]

        #Knights
        wNpieceSquare = [[-50,-40,-30,-30,-30,-30,-40,-50],
                         [-40,-20,  0,  0,  0,  0,-20,-40],
                         [-30,  0, 10, 15, 15, 10,  0,-30],
                         [-30,  5, 15, 20, 20, 15,  5,-30],
                         [-30,  0, 15, 20, 20, 15,  0,-30],
                         [-30,  5, 10, 15, 15, 10,  5,-30],
                         [-40,-20,  0,  5,  5,  0,-20,-40],
                         [-50,-40,-30,-30,-30,-30,-40,-50]]

        #Bishops
        wBpieceSquare = [[-20,-10,-10,-10,-10,-10,-10,-20],
                         [-10,  0,  0,  0,  0,  0,  0,-10],
                         [-10,  0,  5, 10, 10,  5,  0,-10],
                         [-10,  5,  5, 10, 10,  5,  5,-10],
                         [-10,  0, 10, 10, 10, 10,  0,-10],
                         [-10, 10, 10, 10, 10, 10, 10,-10],
                         [-10,  5,  0,  0,  0,  0,  5,-10],
                         [-20,-10,-10,-10,-10,-10,-10,-20]]

        #Rooks
        wRpieceSquare = [[ 0,  0,  0,  0,  0,  0,  0,  0],
                         [ 5, 10, 10, 10, 10, 10, 10,  5],
                         [-5,  0,  0,  0,  0,  0,  0, -5],
                         [-5,  0,  0,  0,  0,  0,  0, -5],
                         [-5,  0,  0,  0,  0,  0,  0, -5],
                         [-5,  0,  0,  0,  0,  0,  0, -5],
                         [-5,  0,  0,  0,  0,  0,  0, -5],
                         [ 0,  0,  0,  5,  5,  0,  0,  0]]

        #Queens
        wQpieceSquare = [[-20,-10,-10, -5, -5,-10,-10,-20],
                         [-10,  0,  0,  0,  0,  0,  0,-10],
                         [-10,  0,  5,  5,  5,  5,  0,-10],
                         [ -5,  0,  5,  5,  5,  5,  0, -5],
                         [  0,  0,  5,  5,  5,  5,  0, -5],
                         [-10,  5,  5,  5,  5,  5,  0,-10],
                         [-10,  0,  5,  0,  0,  0,  0,-10],
                         [-20,-10,-10, -5, -5,-10,-10,-20]]

        #King - Midgame (adjust for endgame later if needed)
        wKpieceSquare = [[-30,-40,-40,-50,-50,-40,-40,-30],
                         [-30,-40,-40,-50,-50,-40,-40,-30],
                         [-30,-40,-40,-50,-50,-40,-40,-30],
                         [-30,-40,-40,-50,-50,-40,-40,-30],
                         [-20,-30,-30,-40,-40,-30,-30,-20],
                         [-10,-20,-20,-20,-20,-20,-20,-10],
                         [ 20, 20,  0,  0,  0,  0, 20, 20],
                         [ 20, 30, 10,  0,  0, 10, 30, 20]]
        # --- End Piece Square Tables ---

        bPpieceSquare = self.reverse_and_negate(wPpieceSquare)
        bNpieceSquare = self.reverse_and_negate(wNpieceSquare)
        bBpieceSquare = self.reverse_and_negate(wBpieceSquare)
        bRpieceSquare = self.reverse_and_negate(wRpieceSquare)
        bQpieceSquare = self.reverse_and_negate(wQpieceSquare)
        bKpieceSquare = self.reverse_and_negate(wKpieceSquare)

        piece_values = {'P': 100, 'R':500, 'N':320, 'B':330, 'Q':900, 'K':20000}
        pst_map = {
            'wP': wPpieceSquare, 'wN': wNpieceSquare, 'wB': wBpieceSquare, 'wR': wRpieceSquare, 'wQ': wQpieceSquare, 'wK': wKpieceSquare,
            'bP': bPpieceSquare, 'bN': bNpieceSquare, 'bB': bBpieceSquare, 'bR': bRpieceSquare, 'bQ': bQpieceSquare, 'bK': bKpieceSquare,
        }

        score = 0
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece != '--':
                    piece_type = piece[1]
                    piece_color = piece[0]
                    # Material score
                    material_value = piece_values.get(piece_type, 0) # Use .get for safety
                    score += material_value if piece_color == 'w' else -material_value
                    # Positional score from PST
                    try:
                        score += pst_map[piece][r][c]
                    except (KeyError, IndexError) as e:
                        print(f"Warning: PST lookup error for '{piece}' at ({r},{c}): {e}")

        return score

    # Helper function for piece-square tables
    def reverse_and_negate(self, table):
        """ Reverses rows and negates values for black pieces. """
        reversed_table = [row[:] for row in table[::-1]] # Deep copy
        for r in range(8):
            for c in range(8):
                reversed_table[r][c] *= -1
        return reversed_table


    # --- MOVED __init__ method to the END of the class definition ---
    def __init__(self):
        self.board = [
            ["bR","bN","bB","bQ","bK","bB","bN","bR"],
            ["bP","bP","bP","bP","bP","bP","bP","bP"],
            ["--","--","--","--","--","--","--","--"],
            ["--","--","--","--","--","--","--","--"],
            ["--","--","--","--","--","--","--","--"],
            ["--","--","--","--","--","--","--","--"],
            ["wP","wP","wP","wP","wP","wP","wP","wP"],
            ["wR","wN","wB","wQ","wK","wB","wN","wR"]]

        # Define move functions dictionary AFTER methods are defined
        self.moveFunctions = {'P': self.getPawnMoves, 'R': self.getRookMoves, 'N': self.getKnightMoves,
                              'B': self.getBishopMoves, 'Q': self.getQueenMoves, 'K': self.getKingMoves}

        self.whiteToMove = True
        self.moveLog = []
        self.wKingLoc = (7,4)
        self.bKingLoc = (0,4)
        self.checkMate = False
        self.stalemate = False
        self.enpassantPossible = ()
        self.currentCastlingRight = castleRights(True, True, True, True)
        # Log initial state correctly
        self.castleRightsLog = [copy.deepcopy(self.currentCastlingRight)]


# =====================================================
# ===              SUPPORTING CLASSES               ===
# =====================================================
class castleRights():
    def __init__(self, wks, bks, wqs, bqs):
        self.wks = wks # White King Side
        self.bks = bks # Black King Side
        self.wqs = wqs # White Queen Side
        self.bqs = bqs # Black Queen Side

    def __deepcopy__(self, memodict={}):
        return castleRights(self.wks, self.bks, self.wqs, self.bqs)


class Move():
    ranksToRows = {"1":7,"2":6,"3":5,"4":4, "5":3,"6":2,"7":1,"8":0}
    rowsToRanks = {v: k for k, v in ranksToRows.items()}
    filesToCols = {"a":0,"b":1,"c":2,"d":3, "e":4,"f":5,"g":6,"h":7}
    colsToFiles = {v: k for k, v in filesToCols.items()}

    def __init__(self, startSq, endSq, board, isEnpassantMove = False, isCastleMove = False):
        self.startRow, self.startCol = startSq
        self.endRow, self.endCol = endSq

        # Add boundary checks before accessing board
        self.pieceMoved = "??"
        if 0 <= self.startRow < 8 and 0 <= self.startCol < 8:
            self.pieceMoved = board[self.startRow][self.startCol]

        self.pieceCaptured = "??"
        if 0 <= self.endRow < 8 and 0 <= self.endCol < 8:
             self.pieceCaptured = board[self.endRow][self.endCol]
        else: # Handle case where end square is off-board (shouldn't happen with valid moves)
             self.pieceCaptured = "??" # Or handle more gracefully

        # Ensure pieceMoved is valid before checking type
        if isinstance(self.pieceMoved, str) and len(self.pieceMoved) == 2:
            self.isPawnPromotion = (self.pieceMoved == 'wP' and self.endRow == 0) or \
                                   (self.pieceMoved == 'bP' and self.endRow == 7)
        else:
            self.isPawnPromotion = False


        self.isEnpassantMove = isEnpassantMove
        if self.isEnpassantMove:
             # Ensure pieceMoved is valid before determining captured pawn color
            if isinstance(self.pieceMoved, str) and len(self.pieceMoved) == 2:
                self.pieceCaptured = 'wP' if self.pieceMoved[0] == 'b' else 'bP'
            else:
                self.pieceCaptured = "?P" # Indicate error


        self.isCastleMove = isCastleMove
        self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    def getChessNotation(self):
        if not (0 <= self.startRow < 8 and 0 <= self.startCol < 8 and \
                0 <= self.endRow < 8 and 0 <= self.endCol < 8):
             return f"Invalid({self.startRow},{self.startCol})->({self.endRow},{self.endCol})"
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)

    def getRankFile(self, r, c):
        if not (0 <= r < 8 and 0 <= c < 8):
            return "??"
        try:
            return self.colsToFiles[c] + self.rowsToRanks[r]
        except KeyError:
            return "KeyErr"

    def __str__(self):
        return self.getChessNotation()

    def __hash__(self):
        return hash(self.moveID)
# import math
# import copy


# class GameState():
#     def __init__(self):
#         # board is 8x8 2D List, each element of the list has 2 characters
#         # initial character == colour (b,w)
#         # second character == piece
#         # R == rook, N == knight, B == bishop, Q == Queen, K == king, P == pawn
#         # -- == empty space
#         self.board = [
#             ["bR","bN","bB","bQ","bK","bB","bN","bR"],
#             ["bP","bP","bP","bP","bP","bP","bP","bP"],
#             ["--","--","--","--","--","--","--","--"],
#             ["--","--","--","--","--","--","--","--"],
#             ["--","--","--","--","--","--","--","--"],
#             ["--","--","--","--","--","--","--","--"],
#             ["wP","wP","wP","wP","wP","wP","wP","wP"],
#             ["wR","wN","wB","wQ","wK","wB","wN","wR"]]
#         self.moveFunctions = {'P': self.getPawnMoves, 'R': self.getRookMoves, 'N': self.getKnightMoves,
#                               'B': self.getBishopMoves, 'Q': self.getQueenMoves, 'K': self.getKingMoves}
        
        
#         self.whiteToMove = True
#         self.AIturn = False
#         self.moveLog = []
#         # used in identifying checks and invalid moves
#         self.wKingLoc = (7,4)
#         self.bKingLoc = (0,4)
#         self.allmoveLog = []
#         self.checkMate = False
#         self.stalemate = False
        
#         self.enpassantPossible = ()
        
#         self.currentCastlingRight = castleRights(True, True, True, True)
#         self.castleRightsLog = [castleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
#                                              self.currentCastlingRight.wqs, self.currentCastlingRight.bqs)]
        
        
#     # makes the move
#     def makeMove(self, move):
#         # making the startSq empty
#         self.board[move.startRow][move.startCol] = '--'
#         # moving the piece
#         self.board[move.endRow][move.endCol] = move.pieceMoved
#         # appending move to the movelog
#         self.moveLog.append(move)
#         self.allmoveLog.append(move.getChessNotation())
#         # changes turn
#         self.whiteToMove = not self.whiteToMove
#         # checking if the kings were moved to update there locations
#         if move.pieceMoved == 'wK':
#             self.wKingLoc = (move.endRow, move.endCol)
#         elif move.pieceMoved == 'bK':
#             self.bKingLoc = (move.endRow, move.endCol)

#         # promotion
#         if move.isPawnPromotion:
#             self.board[move.endRow][move.endCol] = move.pieceMoved[0] + 'Q'

#         # enpassant
#         if move.isEnpassantMove:
#             self.board[move.startRow][move.endCol] = '--'
#         if move.pieceMoved[1] == 'P' and abs(move.startRow - move.endRow) == 2:
#             self.enpassantPossible = ((move.startRow + move.endRow) // 2, move.startCol)
#         else:
#             self.enpassantPossible = ()

#         if move.isCastleMove:
#             if move.endCol - move.startCol == 2:
#                 self.board[move.endRow][move.endCol-1] = self.board[move.endRow][move.endCol+1]
#                 self.board[move.endRow][move.endCol+1] = '--'
#             else:
#                 self.board[move.endRow][move.endCol+1] = self.board[move.endRow][move.endCol-2]
#                 self.board[move.endRow][move.endCol-2] = '--'

#         # update castling rights - whenever king or rook move is played
#         self.updateCastleRights(move)
#         self.castleRightsLog.append(castleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
#                                              self.currentCastlingRight.wqs, self.currentCastlingRight.bqs))

        
                

#     def undoMove(self):
#         if len(self.moveLog) != 0:
#             # removing move form log
#             move = self.moveLog.pop()
#             # reversing make move
#             self.board[move.startRow][move.startCol] = move.pieceMoved
#             self.board[move.endRow][move.endCol] = move.pieceCaptured
#             # changing turn
#             self.whiteToMove = not self.whiteToMove
#             # checking if the kings were moved to update there locations
#             if move.pieceMoved == 'wK':
#                 self.wKingLoc = (move.startRow, move.startCol)
#             elif move.pieceMoved == 'bK':
#                 self.bKingLoc = (move.startRow, move.startCol)

#             # undoing enpassant move
#             if move.isEnpassantMove:
#                 self.board[move.endRow][move.endCol] = '--'
#                 self.board[move.startRow][move.endCol] = move.pieceCaptured
#                 self.enpassantPossible = (move.endRow, move.endCol)
#             # undo a 2 square pawn advance
#             if move.pieceMoved[1] == 'P' and abs(move.startRow - move.endRow) == 2:
#                 self.enpassantPossible = ()
                
#             # undoing castle move
#             if move.isCastleMove:
#                 if move.endCol - move.startCol == 2:
#                     self.board[move.endRow][move.endCol+1] = self.board[move.endRow][move.endCol-1]
#                     self.board[move.endRow][move.endCol-1] = '--'
#                 else:
#                     self.board[move.endRow][move.endCol-2] = self.board[move.endRow][move.endCol+1]
#                     self.board[move.endRow][move.endCol+1] = '--'
            
#             # undoing castling rights
#             self.castleRightsLog.pop()
#             self.currentCastlingRight = self.castleRightsLog[-1]
            


#     def updateCastleRights(self, move):
#         if move.pieceMoved == 'wK':
#             self.currentCastlingRight.wks = False
#             self.currentCastlingRight.wqs = False
#         elif move.pieceMoved == 'bK':
#             self.currentCastlingRight.bks = False
#             self.currentCastlingRight.bqs = False
#         elif move.pieceMoved == 'wR':
#             if move.startRow == 7:
#                 if move.startCol == 0:
#                     self.currentCastlingRight.wqs = False
#                 elif move.startCol == 7:
#                     self.currentCastlingRight.wks = False

#         elif move.pieceMoved == 'bR':
#             if move.startRow == 0:
#                 if move.startCol == 0:
#                     self.currentCastlingRight.bqs = False
#                 elif move.startCol == 7:
#                     self.currentCastlingRight.bks = False
                

                

#     def getValidMoves(self):
#         tempEnpassantPossible = self.enpassantPossible
#         tempCastleRights = castleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
#                                         self.currentCastlingRight.wqs, self.currentCastlingRight.bqs)
#         # gets all move
#         moves = self.getAllMoves()
#         if self.whiteToMove:
#             self.getCastleMoves(self.wKingLoc[0], self.wKingLoc[1], moves)
#         else:
#             self.getCastleMoves(self.bKingLoc[0], self.bKingLoc[1], moves)
#         # goes backwords through the list
#         for i in range(len(moves)-1,-1,-1):
#             # makes the move and changes turn
#             self.makeMove(moves[i])
#             self.whiteToMove = not self.whiteToMove
#             if self.inCheck():
#                 # sees if that previous move puts the player in check
#                 moves.remove(moves[i])
#             # changes turn back and undoes the move
#             self.whiteToMove = not self.whiteToMove
#             self.undoMove()
#         # checks if there are no valid moves (either: stalemate or checkmate)
#         if len(moves) == 0:
#             # sees if in check or stalemate
#             if self.inCheck():
#                 self.checkMate = True
#             else:
#                 self.stalemate = True
        
#         # all the valid moves
#         self.enpassantPossible = tempEnpassantPossible
#         self.currentCastlingRight = tempCastleRights
#         return moves

#     def inCheck(self):
#         # checks which turn
#         if self.whiteToMove:
#             # returns a bool and checks if the whiite king is under attack
#             return self.squareUnderAttack(self.wKingLoc[0], self.wKingLoc[1])
#         else:
#             # then checks black king
#             return self.squareUnderAttack(self.bKingLoc[0], self.bKingLoc[1])
#     def isWhiteCheckmate(self):
#         if self.checkMate and not self.whiteToMove:
#             return True
#         return False
#     def isBlackCheckmate(self):
#         if self.checkMate and self.whiteToMove:
#             return True
#         return False    
    
#     def squareUnderAttack(self, r, c):
#         # sees opponent moves by changing turn gets all there moves and changes back turn
#         self.whiteToMove = not self.whiteToMove
#         oppMoves = self.getAllMoves()
#         self.whiteToMove = not self.whiteToMove
#         # checks all moves and sees if the end square is the square entered in the function
#         for move in oppMoves:
#             if move.endRow == r and move.endCol == c:
#                 return True
#         return False 


#     def getAllMoves(self):
#         # initialising the move list
#         moves = []
#         # going through each element in the list
#         for r in range(len(self.board)):
#             for c in range(len(self.board[r])):
#                 # checking piece colour
#                 turn = self.board[r][c][0]
#                 if (turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
#                     # using dictionary to reduce if statements
#                     piece = self.board[r][c][1]
#                     self.moveFunctions[piece](r, c, moves)
#         return moves




                     
#     def getPawnMoves(self, r, c, moves):
#         # white pawn moves
#         if self.whiteToMove:
#             # checking if square above is empty
#             if self.board[r-1][c] == '--':
#                 # if it is we append that as a valid move
#                 moves.append(Move((r, c), (r-1, c), self.board))
#                 # checks if the piece hasn't been moved so it can do a double move
#                 if r == 6 and self.board[r-2][c] == '--':
#                     moves.append(Move((r, c), (r-2, c), self.board))
#             # captures to the left        
#             if c - 1 >= 0: 
#                 if self.board[r-1][c-1][0] == 'b':
#                     moves.append(Move((r, c), (r-1, c-1), self.board))
#                 elif (r-1, c-1) == self.enpassantPossible:
#                     moves.append(Move((r, c), (r-1, c-1), self.board, isEnpassantMove = True))
#             # captures to the right
#             if c + 1 <= 7: 
#                 if self.board[r-1][c+1][0] == 'b':
#                     moves.append(Move((r, c), (r-1, c+1), self.board))
#                 elif (r-1, c+1) == self.enpassantPossible:
#                     moves.append(Move((r, c), (r-1, c+1), self.board, isEnpassantMove = True))

#         # black pawn moves
#         else:
#             if self.board[r + 1][c] == '--':
#                 # checking if square below is empty
#                 moves.append(Move((r, c), (r + 1, c), self.board))
#                 # checks if the piece hasn't been moved so it can do a double move
#                 if r == 1 and self.board[r + 2][c] == '--':
#                     moves.append(Move((r, c), (r+2, c), self.board))
#             # captures to the left
#             if c - 1 >= 0:
#                 if self.board[r + 1][c - 1][0] == 'w':
#                     moves.append(Move((r, c), (r + 1, c-1), self.board))
#                 elif (r + 1, c - 1) == self.enpassantPossible:
#                     moves.append(Move((r, c), (r + 1, c - 1), self.board, isEnpassantMove = True))
#             # captures to the right
#             if c + 1 <= 7:
#                 if self.board[r + 1][c + 1][0] == 'w':
#                     moves.append(Move((r, c), (r+1, c+1), self.board))
#                 elif (r + 1, c + 1) == self.enpassantPossible:
#                     moves.append(Move((r, c), (r + 1, c + 1), self.board, isEnpassantMove = True))
        
#     def getRookMoves(self, r, c, moves):
#         # directions up, down, left and right(not in that order)
#         directions = ((-1,0), (0,-1), (1,0), (0,1))
#         # conditional expression in order to get opponents colour
#         oppColour = 'b' if self.whiteToMove else 'w'
#         #goes through each direction
#         for d in directions:
#             # loops 8 times length/width of the board as a rook can move 8 squares
#             for i in range(1,8):
#                 # adds a factor of the direction
#                 endRow = r + d[0] * i
#                 endCol = c + d[1] * i
#                 #checks if the final position is off the board
#                 if 0 <= endRow < 8 and 0 <= endCol < 8:
#                     # checks if it can move to that end square by checking if it is empty
#                     endPiece = self.board[endRow][endCol]
#                     if endPiece == '--':
#                         moves.append(Move((r, c), (endRow, endCol), self.board))
#                     elif endPiece[0] == oppColour:
#                         moves.append(Move((r, c), (endRow, endCol), self.board))
#                         break
#                     else:
#                         break
#                 # breaks if it of the board
#                 else:
#                     break
            
#     def getBishopMoves(self, r, c, moves):
#         # directions the bishop ccan move in (diaganols)
#         directions = ((-1,-1), (1,-1), (1,1), (-1,1))
#         # gets opponents colour
#         oppColour = 'b' if self.whiteToMove else 'w'
#         # goes through the directions
#         for d in directions:
#             # iterates 8 times
#             for i in range(1,8):
#                 # multiplies the end position by i
#                 endRow = r + d[0] * i
#                 endCol = c + d[1] * i
#                 # checks if the endSq is off the board
#                 if 0 <= endRow < 8 and 0 <= endCol < 8:
#                     endPiece = self.board[endRow][endCol]
#                     # if the endSq is empty it is a valid move
#                     if endPiece == '--':
#                         moves.append(Move((r, c), (endRow, endCol), self.board))
#                     # if there is an opponents piece we can take it then breaks out of that direction
#                     elif endPiece[0] == oppColour:
#                         moves.append(Move((r, c), (endRow, endCol), self.board))
#                         break
#                     # if it is a friendly piece then you can no longer go in that direction so we be break
#                     else:
#                         break
#                 # breaks if it off the board
#                 else:
#                     break
                
#     def getQueenMoves(self, r, c, moves):
#         # can move in all directions so we use the rook and bishop valid move checks
#         self.getRookMoves(r, c, moves)
#         self.getBishopMoves(r, c, moves)

#     def getKnightMoves(self, r, c, moves):
#         # knight moves
#         knightMoves = ((-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1))
#         # gets ally colour
#         allyColour = 'w' if self.whiteToMove else 'b'
#         for m in knightMoves:
#             endRow = r + m[0]
#             endCol = c + m[1]
#             if 0 <= endRow < 8 and 0 <= endCol < 8:
#                 endPiece = self.board[endRow][endCol]
#                 if endPiece[0] != allyColour:
#                     moves.append(Move((r, c), (endRow, endCol), self.board))

#     def getKingMoves(self, r, c, moves):
#         # king can only move 1 square but any direction
#         directions = ((-1,-1), (1,-1), (1,1), (-1,1), (-1,0), (0,-1), (1,0), (0,1))
#         # easier to check if the 
#         allyColour = 'w' if self.whiteToMove else 'b'
#         for d in directions:
#             endRow = r + d[0]
#             endCol = c + d[1]
#             # checks if off the board
#             if 0 <= endRow < 8 and 0 <= endCol < 8:
#                 endPiece = self.board[endRow][endCol]
#                 if endPiece[0] != allyColour:
#                     moves.append(Move((r, c), (endRow, endCol), self.board))

    

#     def getCastleMoves(self, r, c, moves):
#         # cant castle if a aquare a is under attack
#         if self.squareUnderAttack(r, c):
#             return
#         # can move there
#         if (self.whiteToMove and self.currentCastlingRight.wks) or (not self.whiteToMove and self.currentCastlingRight.bks):
#             self.getKingsideCastleMoves(r, c, moves)
#         if (self.whiteToMove and self.currentCastlingRight.wqs) or (not self.whiteToMove and self.currentCastlingRight.bqs):
#             self.getQueensideCastleMoves(r, c, moves)

#     def getKingsideCastleMoves(self, r, c, moves):
#         if self.board[r][c + 1] == '--' and self.board[r][c + 2] == '--':
#             if not self.squareUnderAttack(r, c + 1) and not self.squareUnderAttack(r, c + 2):
#                 moves.append(Move((r, c), (r, c + 2), self.board, isCastleMove = True))

#     def getQueensideCastleMoves(self, r, c, moves):
#         if self.board[r][c - 1] == '--' and self.board[r][c - 2] == '--' and self.board[r][c - 3] == '--':
#             if not self.squareUnderAttack(r, c - 1) and not self.squareUnderAttack(r, c - 2):
#                 moves.append(Move((r, c), (r, c - 2), self.board, isCastleMove = True))


#     def minimax(self, depth, alpha, beta, isMaximiser):
#         possibleMoves = self.getValidMoves()
#         value = self.boardEval()
#         if depth == 0 or value == -math.inf or value == math.inf:
#                 return value
#         if isMaximiser:
#             maxEval = -math.inf
#             for move in possibleMoves:
#                 self.makeMove(move)
#                 evaluation = self.minimax(depth - 1, alpha, beta, False)
#                 self.undoMove()
#                 if (evaluation > maxEval):
#                     maxEval = evaluation
#                 alpha = max(alpha, evaluation)
#                 if beta <= alpha:
#                     break
#             return maxEval
#         else:
#             minEval = math.inf
#             for move in possibleMoves:
#                 self.makeMove(move)
#                 evaluation = self.minimax(depth - 1, alpha, beta, True)
#                 self.undoMove()
#                 minEval = min(evaluation, minEval)
#                 beta = min(evaluation, beta)
#                 if beta <= alpha:
#                     break
#             return minEval
                         
#     def getBestMove(self, depth, isMaximiser):
#         # function to get and play best move for AI
#         tempCheckmate = copy.deepcopy(self.checkMate)
#         tempStalemate = copy.deepcopy(self.stalemate)
#         tempCastle = copy.deepcopy((self.currentCastlingRight.wks, self.currentCastlingRight.bks,
#                                              self.currentCastlingRight.wqs, self.currentCastlingRight.bqs))

#         self.AIturn = True
#         moves = self.getValidMoves()
#         bestValue = -math.inf if isMaximiser else math.inf
       
#         self.all_valus= []
#         for move in moves:
#             self.makeMove(move)
#             value = self.minimax(depth - 1, -math.inf, math.inf, not isMaximiser)
#             self.undoMove()
#             if value == -math.inf and not isMaximiser:
#                 return move
#             elif value == math.inf and isMaximiser:
#                 return move
#             else:
#                 if isMaximiser:
#                     try:
#                         if value > bestValue:
#                             bestValue = value
#                             bestMove = copy.deepcopy(move)
#                     except:
#                         pass
#                 else:
#                     try:
#                         if value < bestValue:
#                             bestValue = value
#                             bestMove = copy.deepcopy(move)
#                     except:
#                         pass
#         self.AIturn = False
#         self.checkMate = copy.deepcopy(tempCheckmate)
#         self.stalemate = copy.deepcopy(tempStalemate)
#         self.currentCastlingRight = castleRights(tempCastle[0],tempCastle[1],tempCastle[2],tempCastle[3])

#         return bestMove
     


#     def printAllMoveID(self, l):
#         # testing
#         for move in l:
#             print(move.moveID)
#             v = self.boardEval()
#             print(v)
        

#     def boardEval(self):
#         moves = self.getValidMoves()
#         if not self.whiteToMove and self.checkMate:
#             return math.inf
#         elif self.whiteToMove and self.checkMate:
#             return -math.inf
#         elif self.stalemate:
#             return 0
#         #Pawns
#         wPpieceSquare = [[ 0,  0,  0,  0,  0,  0,  0,  0],
#                          [50, 50, 50, 50, 50, 50, 50, 50],
#                          [10, 10, 20, 30, 30, 20, 10, 10],
#                          [5,  5, 10, 25, 25, 10,  5, 5],
#                          [0,  0,  0, 20, 20,  0,  0, 0],
#                          [5, -5,-10,  0,  0,-10, -5,  5],
#                          [5, 10, 10,-20,-20, 10, 10,  5],
#                          [0,  0,  0,  0,  0,  0,  0,  0]]

#         #Knights
#         wNpieceSquare = [[-50,-40,-30,-30,-30,-30,-40,-50],
#                          [-40,-20,  0,  0,  0,  0,-20,-40],
#                          [-30,  0, 10, 15, 15, 10,  0,-30],
#                          [-30,  5, 15, 20, 20, 15,  5,-30],
#                          [-30,  0, 15, 20, 20, 15,  0,-30],
#                          [-30,  5, 10, 15, 15, 10,  5,-30],
#                          [-40,-20,  0,  5,  5,  0,-20,-40],
#                          [-50,-40,-30,-30,-30,-30,-40,-50]]

#         #Bishops
#         wBpieceSquare = [[-20,-10,-10,-10,-10,-10,-10,-20],
#                          [-10,  0,  0,  0,  0,  0,  0,-10],
#                          [-10,  0,  5, 10, 10,  5,  0,-10],
#                          [-10,  5,  5, 10, 10,  5,  5,-10],
#                          [-10,  0, 10, 10, 10, 10,  0,-10],
#                          [-10, 10, 10, 10, 10, 10, 10,-10],
#                          [-10,  5,  0,  0,  0,  0,  5,-10],
#                          [-20,-10,-10,-10,-10,-10,-10,-20]]

#         #Rooks
#         wRpieceSquare = [[0,  0,  0,  0,  0,  0,  0,  0],
#                         [ 5, 10, 10, 10, 10, 10, 10,  5],
#                         [-5,  0,  0,  0,  0,  0,  0, -5],
#                         [-5,  0,  0,  0,  0,  0,  0, -5],
#                         [-5,  0,  0,  0,  0,  0,  0, -5],
#                         [-5,  0,  0,  0,  0,  0,  0, -5],
#                         [-5,  0,  0,  0,  0,  0,  0, -5],
#                         [ 0,  0,  0,  5,  5,  0,  0,  0]]

#         #Queens
#         wQpieceSquare = [[-20,-10,-10, -5, -5,-10,-10,-20],
#                          [-10,  0,  0,  0,  0,  0,  0,-10],
#                          [-10,  0,  5,  5,  5,  5,  0,-10],
#                          [ -5,  0,  5,  5,  5,  5,  0, -5],
#                          [  0,  0,  5,  5,  5,  5,  0, -5],
#                          [-10,  5,  5,  5,  5,  5,  0,-10],
#                          [-10,  0,  5,  0,  0,  0,  0,-10],
#                          [-20,-10,-10, -5, -5,-10,-10,-20]]
        
#         #King
#         wKpieceSquare = [[-30,-40,-40,-50,-50,-40,-40,-30],
#                          [-30,-40,-40,-50,-50,-40,-40,-30],
#                          [-30,-40,-40,-50,-50,-40,-40,-30],
#                          [-30,-40,-40,-50,-50,-40,-40,-30],
#                          [-20,-30,-30,-40,-40,-30,-30,-20],
#                          [-10,-20,-20,-20,-20,-20,-20,-10],
#                          [ 20, 20,  0,  0,  0,  0, 20, 20],
#                          [ 20, 30, 10,  0,  0, 10, 30, 20]]
        
#         bPpieceSquare = self.reversePiece(wPpieceSquare)
#         bNpieceSquare = self.reversePiece(wNpieceSquare)
#         bBpieceSquare = self.reversePiece(wBpieceSquare)
#         bRpieceSquare = self.reversePiece(wRpieceSquare)
#         bQpieceSquare = self.reversePiece(wQpieceSquare)
#         bKpieceSquare = self.reversePiece(wKpieceSquare)
        
#         values = {'wP': 100, 'wR':500, 'wN':300, 'wB':300, 'wQ':900, 'wK':20000,
#                   'bP': -100, 'bR':-500, 'bN':-300, 'bB':-300, 'bQ':-900, 'bK':-20000}
#         score = 0
#         for r in range(len(self.board)):
#             for c in range(len(self.board[r])):
#                 piece = self.board[r][c]
#                 if piece != '--':
#                     score += values[self.board[r][c]]
#                     #Pawns
#                     if piece == 'wP':
#                         score += wPpieceSquare[r][c]
#                     elif piece == 'bP':
#                         score += bPpieceSquare[r][c]
#                     #Knights
#                     elif piece == 'wN':
#                         score += wNpieceSquare[r][c]
#                     elif piece == 'bN':
#                         score += bNpieceSquare[r][c]
#                     #Bishops
#                     elif piece == 'wB':
#                         score += wBpieceSquare[r][c]
#                     elif piece == 'bB':
#                         score += bBpieceSquare[r][c]
#                     #Rooks
#                     elif piece == 'wR':
#                         score += wRpieceSquare[r][c]
#                     elif piece == 'bR':
#                         score += bRpieceSquare[r][c]
#                     #Queens
#                     elif piece == 'wQ':
#                         score += wQpieceSquare[r][c]
#                     elif piece == 'bQ':
#                         score += bQpieceSquare[r][c]
#                     #Kings
#                     elif piece == 'wK':
#                         score += wKpieceSquare[r][c]
#                     elif piece == 'bK':
#                         score += bKpieceSquare[r][c]
                
#         return score


#     def reversePiece(self, l):
#         newList = []
#         for row in l:
#             newList.insert(0,row)
#         for r in range(8):
#             for c in range(8):
#                 newList[r][c] = newList[r][c] * -1
                
#         return newList
                   


# class castleRights():

#     def __init__(self, wks, bks, wqs, bqs):
#         self.wks = wks
#         self.bks = bks
#         self.wqs = wqs
#         self.bqs = bqs
    

# class Move():
#     # able to allow chess notation to python array location
#     ranksToRows = {"1":7,"2":6,"3":5,"4":4,
#                    "5":3,"6":2,"7":1,"8":0}
#     rowsToRanks = {v: k for k, v in ranksToRows.items()}

#     filesToCols = {"a":0,"b":1,"c":2,"d":3,
#                    "e":4,"f":5,"g":6,"h":7}
#     colsToFiles = {v: k for k, v in filesToCols.items()}

    
#     def __init__(self, startSq, endSq, board, isEnpassantMove = False, isCastleMove = False):
#         # start location/square
#         self.startRow = startSq[0]
#         self.startCol = startSq[1]
#         # end location/square
#         self.endRow = endSq[0]
#         self.endCol = endSq[1]
#         # piece moved/captured
#         self.pieceMoved = board[self.startRow][self.startCol]
#         self.pieceCaptured = board[self.endRow][self.endCol]
#         # bool to see if either black or white pawn has been moved to the end row
#         self.isPawnPromotion = ((self.pieceMoved == 'wP' and self.endRow == 0) or (self.pieceMoved == 'bP' and self.endRow == 7))
        
#         self.isEnpassantMove = isEnpassantMove
        
#         if self.isEnpassantMove:
#             self.pieceCaptured = 'wP' if self.pieceMoved == 'bP' else 'bP'

#         self.isCastleMove = isCastleMove

#         # to compare the moves
#         self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
        
#     # move class Object Equality
#     def __eq__(self, other):
#         if isinstance(other, Move):
#             return self.moveID == other.moveID
    
#     #with use of the fileToRank dictionaries we can print out the move in chess notation (e2e4)
#     def getChessNotation(self):
#         return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)
#     def getRankFile(self, r, c):
#         return self.colsToFiles[c] + self.rowsToRanks[r]


