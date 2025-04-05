import chess
import chess.engine
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class ChessBotBattle:
    def __init__(self, engine1_path, engine2_path, time_limit=0.1, games=2):
        """
        Initialize the bot battle arena.
        
        Args:
            engine1_path (str): Path to first engine executable
            engine2_path (str): Path to second engine executable
            time_limit (float): Time per move in seconds
            games (int): Number of games to play
        """
        self.engine1_path = engine1_path
        self.engine2_path = engine2_path
        self.time_limit = time_limit
        self.games = games
        self.results = []
        
    def setup_engines(self):
        """Initialize both chess engines"""
        self.engine1 = chess.engine.SimpleEngine.popen_uci(self.engine1_path)
        self.engine2 = chess.engine.SimpleEngine.popen_uci(self.engine2_path)
        
    def close_engines(self):
        """Clean up engine processes"""
        self.engine1.quit()
        self.engine2.quit()
        
    def play_game(self, game_num):
        """Play a single game between the two engines"""
        board = chess.Board()
        game_moves = []
        
        while not board.is_game_over():
            start_time = time.time()
            
            if board.turn == chess.WHITE:
                result = self.engine1.play(board, chess.engine.Limit(time=self.time_limit))
            else:
                result = self.engine2.play(board, chess.engine.Limit(time=self.time_limit))
            
            move_time = time.time() - start_time
            move = result.move
            game_moves.append(move.uci())
            board.push(move)
            
            # Optional: print the board for visualization
            if len(game_moves) < 10:  # Only show first few moves
                print(f"\nMove {len(game_moves)}:")
                print(board)
        
        # Record game result
        result = board.result()
        outcome = "1-0" if result == "1-0" else "0-1" if result == "0-1" else "1/2-1/2"
        
        self.results.append({
            "game": game_num,
            "moves": len(game_moves),
            "result": outcome,
            "winner": "White" if result == "1-0" else "Black" if result == "0-1" else "Draw",
            "moves_list": " ".join(game_moves),
            "termination": board.result(claim_draw=True)
        })
        
        return outcome
    
    def run_battle(self):
        """Run all games and collect statistics"""
        self.setup_engines()
        
        print(f"Starting {self.games} game(s) between {self.engine1_path} (White) and {self.engine2_path} (Black)")
        
        for i in tqdm(range(self.games), desc="Playing games"):
            outcome = self.play_game(i+1)
            print(f"\nGame {i+1} completed: {outcome}")
            
            # Alternate colors for fairness
            self.engine1_path, self.engine2_path = self.engine2_path, self.engine1_path
            self.setup_engines()  # Reinitialize with swapped colors
            
        self.close_engines()
        self.generate_stats()
    
    def generate_stats(self):
        """Generate and display statistics from all games"""
        df = pd.DataFrame(self.results)
        
        print("\n=== Final Results ===")
        print(df[['game', 'moves', 'result', 'termination']])
        
        # Summary statistics
        summary = df['winner'].value_counts()
        print("\n=== Win/Loss Summary ===")
        print(summary)
        
        # Average moves per game
        avg_moves = df['moves'].mean()
        print(f"\nAverage moves per game: {avg_moves:.1f}")
        
        # Termination reasons
        termination_counts = df['termination'].value_counts()
        print("\n=== Game Termination Reasons ===")
        print(termination_counts)
        
        # Visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        summary.plot(kind='bar', color=['green', 'red', 'gray'])
        plt.title("Game Outcomes")
        plt.ylabel("Number of Games")
        
        plt.subplot(1, 2, 2)
        termination_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title("Termination Reasons")
        
        plt.tight_layout()
        plt.savefig("chess_battle_stats.png")
        print("\nStatistics saved to chess_battle_stats.png")
        
        # Save all game data to CSV
        df.to_csv("chess_battle_results.csv", index=False)
        print("Detailed results saved to chess_battle_results.csv")


if __name__ == "__main__":
    # Example using Stockfish (you'll need to download it first)
    # Download from https://stockfishchess.org/download/
    
    # Windows example paths:
    ENGINE1 = "/stockfish/stockfish-windows-x86-64-avx2.exe"
    ENGINE2 = "/stockfish/stockfish-windows-x86-64-avx2.exe"
    
    # Linux/Mac example paths:
    # ENGINE1 = "./stockfish"  # After making it executable
    # ENGINE2 = "./stockfish"  # Same engine vs itself
    
    # Create and run the battle
    battle = ChessBotBattle(
        engine1_path=ENGINE1,
        engine2_path=ENGINE2,
        time_limit=0.1,  # 100ms per move
        games=2         # Number of games to play
    )
    battle.run_battle()