"""
================================================================================
SCRIPT 02: FEATURE ENGINEERING & PREPROCESSING
================================================================================

MY GOAL:
--------
The raw PGN data (text) I collected is great for humans, but useless for a
machine learning model. The model needs numbers.

In this script, I am converting my "Chess History" into a "Math Problem."
I am simulating every game up to Move 15 to take a snapshot of the board.
From that snapshot, I extract specific features to predict the winner.

KEY DECISIONS I MADE:
---------------------
1.  **Consistent Scope:** I am applying the same filters as my EDA (Summer 2025,
    Rapid 10min) to ensure I am training the model on relevant data.
2.  **The "Move 15" Snapshot:** I chose move 15 because the opening theory
    usually ends here, and the middle-game strategy begins.
3.  **Turn Fairness:** To calculate "Mobility" (how many moves I have), I ensure
    the board snapshot is always taken when it is *MY* turn. If the simulation
    stops on the opponent's turn, I undo the last move.
4.  **Handling Correlations:** Instead of feeding the model "My Rating" and
    "Opponent Rating" (which correlate), I calculate "Rating Difference."
5.  **Opening Encoding:** To prevent "The Curse of Dimensionality" (too many
    columns), I only keep my Top 7 most frequent openings and group the
    rest as "Other."

"""

import pandas as pd
import numpy as np
import chess.pgn
import io
import warnings
from datetime import datetime

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
INPUT_FILE = "data/chess_games_raw.csv"
OUTPUT_FILE = "data/chess_ml_ready.csv"
MY_USERNAME = "currystan"
MOVE_CUTOFF = 15
SUMMER_START_DATE = pd.Timestamp('2025-06-01')

def get_board_state_features(pgn_text):
    """
    This function replays a single game up to the cutoff point and extracts
    the board features (Material, Mobility, King Safety).
    """
    try:
        # 1. Parse the PGN string into a game object
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            return None

        # Determine my color for this specific game
        headers = game.headers
        am_i_white = headers.get("White", "").lower() == MY_USERNAME.lower()
        my_color = chess.WHITE if am_i_white else chess.BLACK

        board = game.board()
        
        # 2. Replay the game up to Move 15
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            # Stop if we hit the move count
            if board.fullmove_number > MOVE_CUTOFF:
                break
                
        # CRITICAL CHECK: If the game ended early (e.g., fast checkmate),
        # we can't use it for prediction as it's an outlier.
        if board.fullmove_number < MOVE_CUTOFF:
            return None

        # 3. Ensure "Turn Fairness"
        # If the simulation stopped and it's the opponent's turn, the library
        # will report 0 legal moves for me. To fix this, I check the turn.
        # If it's not my turn, I undo (pop) the last move to measure the state
        # when *I* had to make a decision.
        if board.turn != my_color:
            try:
                board.pop()
            except IndexError:
                return None

        # --- FEATURE EXTRACTION ---
        
        # Define standard piece values
        piece_values = {
            chess.PAWN: 1, 
            chess.KNIGHT: 3, 
            chess.BISHOP: 3, 
            chess.ROOK: 5, 
            chess.QUEEN: 9
        }

        # Calculate Material (Total army value)
        my_material = 0
        opp_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == my_color:
                    my_material += value
                else:
                    opp_material += value

        # Calculate Mobility (My available legal moves)
        mobility_count = board.legal_moves.count()
        
        # King Safety (Has my king moved from its starting square?)
        my_king_sq = board.king(my_color)
        start_sq = chess.E1 if am_i_white else chess.E8
        has_moved_king = 1 if my_king_sq != start_sq else 0

        # Return the extracted data row
        return {
            "material_diff": my_material - opp_material, # Difference avoids correlation
            "mobility_count": mobility_count,
            "king_moved": has_moved_king,
            "is_white": 1 if am_i_white else 0
        }
    except Exception:
        # If any PGN parsing fails, skip this game
        return None

def process_data():
    print("--- STARTING PHASE 4: FEATURE ENGINEERING ---")
    
    # 1. Load Data
    print(f"Loading raw data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run the collection script first.")
        return

    raw_count = len(df)
    
    # 2. Apply Scope Filters (Matching my EDA)
    print("Applying filters (Summer 2025, Rapid 10min, No Draws)...")
    
    # Convert timestamp for filtering
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Filter: Only Summer 2025
    df = df[df['date'] >= SUMMER_START_DATE].copy()
    
    # Filter: Only 10 Minute Rapid games (Time Control '600')
    df = df[df['time_control'].astype(str) == '600'].copy()
    
    # Filter: Remove Draws (Binary Classification is cleaner)
    df = df[df['outcome'] != 'draw'].copy()
    
    print(f"Data reduced from {raw_count} to {len(df)} games for analysis.")

    # 3. Apply Chess Logic
    # This runs the board simulation on every game.
    print(f"Simulating games to Move {MOVE_CUTOFF}...")
    features = df['pgn'].apply(get_board_state_features)
    
    # Convert the list of results into a DataFrame
    features_df = pd.json_normalize(features)
    
    # Combine original metadata with new chess features
    df_ml = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    # Drop rows where feature extraction failed (short games)
    df_ml = df_ml.dropna(subset=['material_diff'])

    # 4. Create Rating Features
    # I calculate the difference because raw ratings correlate too heavily.
    df_ml['rating_diff'] = df_ml['my_rating'] - df_ml['opponent_rating']

    # 5. Encode Categorical Data (Openings)
    # I use a "Top N" strategy: Keep the 7 most common openings, map the rest to "Other".
    top_openings = df_ml['opening'].value_counts().head(7).index
    df_ml['opening_simplified'] = df_ml['opening'].apply(lambda x: x if x in top_openings else 'Other')
    
    # One-Hot Encoding
    # I use drop_first=True to avoid multicollinearity (crucial for Logistic Regression)
    dummy_openings = pd.get_dummies(df_ml['opening_simplified'], prefix='open', drop_first=True)
    
    # 6. Final Assembly
    df_ml = pd.concat([df_ml, dummy_openings], axis=1)
    
    # Map Target: Win = 1, Loss = 0
    df_ml['target'] = df_ml['outcome'].apply(lambda x: 1 if x == 'win' else 0)

    # 7. Select Final Columns
    # I keep 'rating_diff' for now, but I might drop it during training to test
    # the "pure" board state predictive power.
    keep_cols = ['target', 'is_white', 'material_diff', 'rating_diff', 'mobility_count', 'king_moved'] + list(dummy_openings.columns)
    
    final_dataset = df_ml[keep_cols]
    
    # 8. Save
    final_dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Machine Learning dataset saved to {OUTPUT_FILE} ({len(final_dataset)} rows)")
    print("Features ready for training:")
    print(final_dataset.columns.tolist())

if __name__ == "__main__":
    process_data()