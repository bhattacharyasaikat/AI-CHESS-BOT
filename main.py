# main.py
import chess
import chess.engine
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# Make sure ChessEngine.py is in the same directory or accessible via PYTHONPATH
from ChessEngine import GameState, Move, castleRights

class ChessBotBattle:
    def __init__(self, engine1_path, engine2_path, engine1_python=False, engine2_python=False,
                 time_limit=0.1, games=2, ai_depth=3):
        # Store the *initial* configuration
        self.initial_config = {
            "p1_path": engine1_path, "p1_python": engine1_python,
            "p2_path": engine2_path, "p2_python": engine2_python,
        }
        self.time_limit = time_limit
        self.games = games
        self.ai_depth = ai_depth
        self.results = []

        # These will be set per game in run_battle
        self.engine1_path = None
        self.engine2_path = None
        self.engine1_python = None
        self.engine2_python = None
        self.engine1 = None # Holds the UCI engine object for player 1 if applicable
        self.engine2 = None # Holds the UCI engine object for player 2 if applicable

        # These hold the actual engine process objects if needed
        self.engine1_obj = None
        self.engine2_obj = None


    def convert_to_gamestate(self, chess_board):
        """Convert chess.Board to your GameState format"""
        gs = GameState()
        gs.board = []

        # Convert board layout (chess uses different rank order)
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                piece = chess_board.piece_at(chess.square(file, rank))
                if piece:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    row.append(f"{color}{piece.symbol().upper()}")
                else:
                    row.append("--")
            gs.board.append(row)

        # Set game state properties
        gs.whiteToMove = chess_board.turn == chess.WHITE
        gs.enpassantPossible = self.get_enpassant(chess_board)
        gs.currentCastlingRight = self.get_castle_rights(chess_board)
        # Ensure find_kings is robust or handle potential errors if kings aren't found (shouldn't happen in valid games)
        try:
            gs.wKingLoc, gs.bKingLoc = self.find_kings(gs.board)
            if gs.wKingLoc == (-1, -1) or gs.bKingLoc == (-1, -1):
                 # This indicates a problem either in find_kings or an invalid board state
                 print(f"Warning: King not found in board state:\n{gs.board}")
                 # Consider raising an error or handling this state appropriately
        except Exception as e:
             print(f"Error finding kings: {e}")
             # Handle error state, maybe by setting king locations to an invalid default or raising

        return gs

    def get_enpassant(self, board):
        if board.ep_square is None:
            return ()
        # Your GameState expects (row, col) where row 0 is rank 8
        ep_rank_chess = chess.square_rank(board.ep_square) # 0-7
        ep_file_chess = chess.square_file(board.ep_square) # 0-7
        # Convert chess rank (0-7) to your row index (0-7, where 0 is rank 8)
        ep_row_gs = 7 - ep_rank_chess
        return (ep_row_gs, ep_file_chess)

    def get_castle_rights(self, board):
        # Assumes castleRights class handles boolean inputs correctly
        return castleRights(
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.BLACK)
        )

    def find_kings(self, board_list):
        """Find king locations from your GameState board list"""
        w_king = (-1, -1)
        b_king = (-1, -1)
        for r in range(8):
            for c in range(8):
                piece = board_list[r][c]
                if piece == 'wK':
                    w_king = (r, c)
                elif piece == 'bK':
                    b_king = (r, c)
        return w_king, b_king

    def get_ai_move(self, board):
        """Get move from your Python chess engine"""
        try:
            game_state = self.convert_to_gamestate(board)
            is_maximizer = game_state.whiteToMove # True if White to move
            # Assuming getBestMove is a method of GameState
            # Make sure GameState has access to its required methods/data
            best_move_obj = GameState.getBestMove(game_state, depth=self.ai_depth, isMaximiser=is_maximizer)

            if best_move_obj is None:
                print("Warning: Python AI returned None move. Possible checkmate/stalemate missed or error?")
                # Handle this case - perhaps return None and let play_game handle it?
                return None

            # Convert your Move object to UCI format
            start_row, start_col = best_move_obj.startRow, best_move_obj.startCol
            end_row, end_col = best_move_obj.endRow, best_move_obj.endCol

            # Check if coordinates are valid before converting
            if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
                print(f"Error: Invalid move coordinates from AI: ({start_row},{start_col}) -> ({end_row},{end_col})")
                return None # Indicate error

            uci_move_str = f"{self.col_to_file(start_col)}{self.row_to_rank(start_row)}" \
                           f"{self.col_to_file(end_col)}{self.row_to_rank(end_row)}"

            # Handle promotions (assuming 'Q' promotion for simplicity, adjust if needed)
            if best_move_obj.isPawnPromotion:
                # Determine promotion piece (e.g., based on move notation or always Queen)
                promotion_piece = 'q' # Default to Queen
                # You might need more logic in your ChessEngine to specify the piece
                uci_move_str += promotion_piece

            # Validate the move using python-chess before returning
            move = chess.Move.from_uci(uci_move_str)
            if move not in board.legal_moves:
                print(f"Error: Python AI generated illegal move: {uci_move_str}")
                print("Legal moves:", [m.uci() for m in board.legal_moves])
                # Decide how to handle this - maybe return a random legal move?
                # For now, return None to indicate failure
                return None
            return move

        except Exception as e:
            print(f"Error in get_ai_move: {e}")
            import traceback
            traceback.print_exc()
            return None # Indicate error

    def row_to_rank(self, row):
        # Converts your GameState row (0-7, where 0 is rank 8) to chess rank ('1'-'8')
        return str(8 - row)

    def col_to_file(self, col):
        # Converts your GameState col (0-7) to chess file ('a'-'h')
        return chr(ord('a') + col)

    def play_game(self, game_num):
        """Plays a single game with the currently assigned engines/roles."""
        board = chess.Board()
        game_moves = []

        # Determine engine names based on roles for *this* game
        white_is_uci = not self.engine1_python
        black_is_uci = not self.engine2_python
        white_name = self.engine1_path if white_is_uci else "Your AI"
        black_name = self.engine2_path if black_is_uci else "Your AI"

        print(f"\nStarting Game {game_num + 1} (White: {white_name}, Black: {black_name})")

        while not board.is_game_over(claim_draw=True): # Check for draws too
            start_time = time.time()
            move = None
            current_player_name = ""

            try:
                if board.turn == chess.WHITE:
                    current_player_name = white_name
                    if self.engine1_python: # White is Python AI
                        move = self.get_ai_move(board)
                    else: # White is UCI Engine
                        if self.engine1 is None: # Check if engine object exists
                            raise RuntimeError(f"Engine 1 (UCI) object not available for game {game_num + 1}.")
                        result = self.engine1.play(
                            board,
                            chess.engine.Limit(time=self.time_limit)
                        )
                        # Check for mate/stalemate possibility from engine result
                        if result.move is None:
                            print(f"Warning: UCI Engine {current_player_name} returned no move.")
                        move = result.move
                else: # Black's turn
                    current_player_name = black_name
                    if self.engine2_python: # Black is Python AI
                        move = self.get_ai_move(board)
                    else: # Black is UCI Engine
                        if self.engine2 is None: # Check if engine object exists
                            raise RuntimeError(f"Engine 2 (UCI) object not available for game {game_num + 1}.")
                        result = self.engine2.play(
                            board,
                            chess.engine.Limit(time=self.time_limit)
                        )
                         # Check for mate/stalemate possibility from engine result
                        if result.move is None:
                             print(f"Warning: UCI Engine {current_player_name} returned no move.")
                        move = result.move

                # Check if a move was successfully generated
                if move is None:
                    # This could happen if AI returns None or engine indicates no move (mate/stalemate)
                    # The board.is_game_over() check should ideally catch mate/stalemate before this
                    print(f"No move generated for {current_player_name}. Board state:\n{board}")
                    print(f"FEN: {board.fen()}")
                    print(f"Game over status: {board.is_game_over(claim_draw=True)}")
                    outcome_obj = board.outcome(claim_draw=True)
                    if outcome_obj:
                        print(f"Outcome: {outcome_obj}")
                    else:
                         print("Outcome object is None, but no move generated. Breaking game loop.")
                    break # Exit the game loop if no move is possible or generated

                # Final validation just in case
                if move not in board.legal_moves:
                    print(f"FATAL ERROR: Attempting illegal move {move.uci()} by {current_player_name}")
                    print(f"FEN: {board.fen()}")
                    print("Legal moves:", [m.uci() for m in board.legal_moves])
                    # Record as an error?
                    raise ValueError(f"Illegal move {move.uci()} attempted by {current_player_name}")


                board.push(move)
                move_time = time.time() - start_time
                game_moves.append(move.uci())

                # Optional: Print board state less frequently
                # print(f"{current_player_name} moved: {move.uci()} ({move_time:.2f}s)")
                # if len(game_moves) < 10 or len(game_moves) % 20 == 0: # Print less often
                #    print(f"\nMove {board.fullmove_number}{'.' if board.turn == chess.BLACK else '...'} ({current_player_name})")
                #    print(board)
                #    print("-" * 10)
                print(f"{current_player_name} moved: {move.uci()} ({move_time:.2f}s)")
                print(board)  # Always print the board
                print()  # Add empty line for spacing

            except Exception as e:
                 print(f"\nError during game {game_num + 1} on {current_player_name}'s turn: {e}")
                 import traceback
                 traceback.print_exc()
                 # Record game as error and stop this game
                 self.results.append({
                     "game": game_num + 1,
                     "white_player": white_name,
                     "black_player": black_name,
                     "moves": board.fullmove_number,
                     "result": "Error",
                     "winner": "Error",
                     "moves_list": " ".join(game_moves),
                     "termination": f"Runtime Error: {e}"
                 })
                 return "Error" # Signal error occurred

        # Game finished normally
        outcome_obj = board.outcome(claim_draw=True)
        result_code = outcome_obj.result() if outcome_obj else "* " # Get result string like "1-0", "0-1", "1/2-1/2" or "*" if unknown
        termination_reason = outcome_obj.termination.name if outcome_obj else "Unknown"

        # Determine winner name based on the roles in *this specific game*
        if result_code == "1-0":
            winner_name = white_name
        elif result_code == "0-1":
            winner_name = black_name
        elif result_code == "1/2-1/2":
            winner_name = "Draw"
        else: # Game ended without standard result (maybe interrupted, or error)
             winner_name = "Unknown/Error" # Or reflect the termination reason


        self.results.append({
            "game": game_num + 1,
            "white_player": white_name,
            "black_player": black_name,
            "moves": board.fullmove_number, # Number of full moves completed
            "result": result_code,
            "winner": winner_name,
            "moves_list": " ".join(game_moves),
            "termination": termination_reason.replace("_", " ").title() # Nicer formatting
        })

        return result_code


    def run_battle(self):
        """Run all games, managing engine initialization and role swapping."""

        # Initialize potential UCI engines ONCE based on initial config
        try:
             if not self.initial_config["p1_python"]:
                 print(f"Initializing Engine 1 (UCI): {self.initial_config['p1_path']}")
                 self.engine1_obj = chess.engine.SimpleEngine.popen_uci(self.initial_config["p1_path"])
             else:
                 print("Engine 1 is Python AI.")
                 self.engine1_obj = None

             if not self.initial_config["p2_python"]:
                  print(f"Initializing Engine 2 (UCI): {self.initial_config['p2_path']}")
                  # Only initialize if different from engine 1 or if engine 1 is Python
                  if self.initial_config["p1_python"] or self.initial_config["p1_path"] != self.initial_config["p2_path"]:
                      self.engine2_obj = chess.engine.SimpleEngine.popen_uci(self.initial_config["p2_path"])
                  else: # Both are the same UCI engine path
                      print("Engine 2 uses the same UCI path as Engine 1, reusing process.")
                      self.engine2_obj = self.engine1_obj # Reuse the same engine object/process
             else:
                 print("Engine 2 is Python AI.")
                 self.engine2_obj = None

        except chess.engine.EngineTerminatedError as e:
             print(f"\nFATAL ERROR: Engine process terminated unexpectedly during initialization: {e}")
             print("Check engine paths and permissions.")
             return # Stop if engines can't be initialized
        except Exception as e:
            print(f"\nFATAL ERROR: Could not initialize UCI engines: {e}")
            print("Check engine paths.")
            import traceback
            traceback.print_exc()
            return # Stop if engines can't be initialized

        print("-" * 20)
        print(f"Starting {self.games} game(s)...")
        print("-" * 20)


        for i in tqdm(range(self.games), desc="Playing games"):
            # Determine config for this game (swap roles on even game indices: 0, 2, 4...)
            # Game 1 (index 0): P1=White, P2=Black
            # Game 2 (index 1): P2=White, P1=Black
            is_swapped = i % 2 != 0

            self.engine1_python = self.initial_config["p2_python"] if is_swapped else self.initial_config["p1_python"]
            self.engine1_path = self.initial_config["p2_path"] if is_swapped else self.initial_config["p1_path"]
            self.engine2_python = self.initial_config["p1_python"] if is_swapped else self.initial_config["p2_python"]
            self.engine2_path = self.initial_config["p1_path"] if is_swapped else self.initial_config["p2_path"]

            # Assign the correct *pre-initialized* engine object based on the current role
            # Player 1 (White in this game) uses engine object 2 if swapped AND P1 is UCI, OR engine object 1 if not swapped AND P1 is UCI
            self.engine1 = self.engine2_obj if is_swapped and not self.engine1_python else \
                           self.engine1_obj if not is_swapped and not self.engine1_python else None
            # Player 2 (Black in this game) uses engine object 1 if swapped AND P2 is UCI, OR engine object 2 if not swapped AND P2 is UCI
            self.engine2 = self.engine1_obj if is_swapped and not self.engine2_python else \
                           self.engine2_obj if not is_swapped and not self.engine2_python else None

            # Play the game with the correctly assigned self.engine1/engine2 objects and flags
            outcome = self.play_game(i) # play_game now uses self.engine1/2 correctly
            print(f"Game {i+1} result: {outcome}")
            # Optional: Add a small delay between games if needed
            # time.sleep(1)

        # Clean up the engine process(es) after all games are finished
        print("\nQuitting engine processes...")
        if self.engine1_obj and (self.engine1_obj is not self.engine2_obj): # Quit engine 1 if it exists and is unique
            try:
                self.engine1_obj.quit()
                print("Engine 1 process quit.")
            except chess.engine.EngineTerminatedError:
                 print("Engine 1 process already terminated.")
            except Exception as e:
                 print(f"Error quitting engine 1 process: {e}")

        if self.engine2_obj: # Quit engine 2 if it exists (might be the same as engine 1)
            try:
                self.engine2_obj.quit()
                print("Engine 2 process quit.")
            except chess.engine.EngineTerminatedError:
                 print("Engine 2 process already terminated.")
            except Exception as e:
                 print(f"Error quitting engine 2 process: {e}")


        # Generate and display statistics
        self.generate_stats()


    def generate_stats(self):
        """Generate and display statistics from all games"""
        if not self.results:
             print("\nNo games were played or recorded.")
             return

        df = pd.DataFrame(self.results)

        print("\n" + "="*15 + " Final Results " + "="*15)
        # Display relevant columns, handling potential errors
        cols_to_show = ["game", "white_player", "black_player", "moves", "result", "winner", "termination"]
        # Filter df to only include existing columns to prevent KeyErrors
        existing_cols = [col for col in cols_to_show if col in df.columns]
        print(df[existing_cols].to_string(index=False)) # Use to_string for better formatting

        # --- Summary statistics (filter out errors) ---
        valid_games = df[df['result'].isin(["1-0", "0-1", "1/2-1/2"])].copy() # Ensure we work on a copy

        if not valid_games.empty:
            print("\n" + "="*10 + " Win/Loss/Draw Summary (Valid Games) " + "="*10)

            # Determine the names of the two competitors from the initial config
            p1_name = self.initial_config["p1_path"] if not self.initial_config["p1_python"] else "Your AI"
            p2_name = self.initial_config["p2_path"] if not self.initial_config["p2_python"] else "Your AI"
            if p1_name == p2_name: # Handle self-play or identical engines
                p1_display_name = f"{p1_name} (P1)"
                p2_display_name = f"{p2_name} (P2)"
            else:
                p1_display_name = p1_name
                p2_display_name = p2_name

            # Count wins for each player based on their initial designation (P1/P2)
            p1_wins = len(valid_games[((valid_games['white_player'] == p1_name) & (valid_games['result'] == '1-0')) |
                                      ((valid_games['black_player'] == p1_name) & (valid_games['result'] == '0-1'))])
            p2_wins = len(valid_games[((valid_games['white_player'] == p2_name) & (valid_games['result'] == '1-0')) |
                                      ((valid_games['black_player'] == p2_name) & (valid_games['result'] == '0-1'))])
            draws = len(valid_games[valid_games['result'] == '1/2-1/2'])

            summary_dict = {
                p1_display_name: p1_wins,
                p2_display_name: p2_wins,
                "Draw": draws
            }
            summary_series = pd.Series(summary_dict)
            print(summary_series)

            # Calculate performance score (e.g., P1 score = P1 wins + 0.5 * draws)
            total_valid_games = len(valid_games)
            if total_valid_games > 0:
                p1_score = p1_wins + 0.5 * draws
                p2_score = p2_wins + 0.5 * draws
                print(f"\nScore: {p1_display_name} {p1_score}/{total_valid_games}  |  {p2_display_name} {p2_score}/{total_valid_games}")


            # Average moves per game
            avg_moves = valid_games['moves'].mean()
            print(f"\nAverage moves per valid game: {avg_moves:.1f}")

            # Termination reasons
            print("\n" + "="*10 + " Game Termination Reasons (Valid Games) " + "="*10)
            termination_counts = valid_games['termination'].value_counts()
            print(termination_counts)

            # --- Visualization ---
            try:
                plt.figure(figsize=(13, 6))

                # Bar chart for Wins/Draws
                plt.subplot(1, 2, 1)
                summary_series.plot(kind='bar', color=['lightcoral', 'lightblue', 'lightgrey'], edgecolor='black')
                plt.title(f"Game Outcomes ({total_valid_games} valid games)")
                plt.ylabel("Number of Games")
                plt.xlabel("Player / Result")
                plt.xticks(rotation=0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                # Pie chart for Termination Reasons
                plt.subplot(1, 2, 2)
                # Only plot if there are termination reasons to show
                if not termination_counts.empty:
                     # Explode the largest slice slightly for emphasis if desired
                     explode = [0.05] * len(termination_counts) # Small explode for all, or calculate largest
                     termination_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                                             pctdistance=0.85, explode=explode,
                                             colors=plt.cm.Paired(range(len(termination_counts))))
                     plt.title("Termination Reasons")
                     plt.ylabel('') # Hide the default ylabel
                else:
                     plt.text(0.5, 0.5, 'No termination data available', horizontalalignment='center', verticalalignment='center')
                     plt.axis('off') # Hide axes if no data


                plt.suptitle(f"Chess Battle Results: {p1_display_name} vs {p2_display_name}", fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                plt.savefig("chess_battle_stats.png")
                print("\nStatistics plot saved to chess_battle_stats.png")

            except ImportError:
                print("\nWarning: Matplotlib not found. Skipping plot generation.")
                print("Install it using: pip install matplotlib")
            except Exception as plot_e:
                print(f"\nCould not generate plot: {plot_e}")

        else:
            print("\nNo valid games completed to generate statistics.")
            # Check if there were errors
            error_games = df[df['result'] == 'Error']
            if not error_games.empty:
                print(f"\n{len(error_games)} game(s) resulted in errors:")
                print(error_games[['game', 'termination']].to_string(index=False))


        # --- Save all game data to CSV ---
        try:
            df.to_csv("chess_battle_results.csv", index=False)
            print("Detailed results saved to chess_battle_results.csv")
        except Exception as csv_e:
            print(f"\nCould not save CSV results: {csv_e}")


# ===============================
# Main execution block
# ===============================
if __name__ == "__main__":
    # --- Configuration ---
    # Option 1: Stockfish vs Your AI
    stockfish_path = "/stockfish/stockfish-windows-x86-64-avx2.exe" # <<< IMPORTANT: SET YOUR PATH TO STOCKFISH
    your_ai_path = None # Path not needed for Python engine

    # Option 2: Your AI vs Your AI (Self-play)
    # engine1_is_python = True
    # engine1_uci_path = None
    # engine2_is_python = True
    # engine2_uci_path = None

    # Option 3: Stockfish vs Stockfish
    # engine1_is_python = False
    # engine1_uci_path = stockfish_path
    # engine2_is_python = False
    # engine2_uci_path = stockfish_path


    # --- Battle Setup ---
    battle = ChessBotBattle(
        # Player 1 (Initially White in Game 1)
        engine1_path=stockfish_path,   # Path to engine OR None if Python
        engine1_python=False,          # True if Player 1 is your Python AI

        # Player 2 (Initially Black in Game 1)
        engine2_path=your_ai_path,     # Path to engine OR None if Python
        engine2_python=True,           # True if Player 2 is your Python AI

        # Battle Parameters
        time_limit=0.1,               # Time per move in seconds (adjust based on engine strength)
        games=5,                      # Total number of games (will alternate colors)
        ai_depth=5                # Depth for your Python AI's search (if used)
    )

    # --- Run the Competition ---
    battle.run_battle()

    print("\nBattle finished.")

