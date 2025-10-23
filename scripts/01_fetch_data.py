"""
================================================================================
SCRIPT 01: FETCH CHESS.COM DATA
================================================================================

WHY WE NEED THIS SCRIPT:
------------------------
Every data science project starts with data. This script is our first and most
crucial step: getting the raw material for our analysis. Its job is to connect
to the chess.com public API, ask for all of my past games, and download them.

WHAT IT DOES:
-------------
1. It finds the URLs for all my monthly game archives on chess.com.
2. It goes to each of those URLs and downloads every single game.
3. For each game, it carefully pulls apart the raw data (called PGN) to
   extract the important details we care about, like who won, what the
   ratings were, the opening played, etc.
4. Finally, it organizes all this information neatly into a single CSV file,
   which will be the foundation for all our future analysis in the project.

This script only needs to be run once to get the data.
"""

# We'll need these libraries to make our script work.
import requests  # The standard for making web requests in Python.
import pandas as pd  # The best library for working with data tables.
import chess.pgn # The library that understands the language of chess games (PGN).
import io  # A tool to treat a string of text like it's a file.
import os  # Helps us work with file paths and directories.
from tqdm import tqdm  # For creating those satisfying progress bars.
import time # To let us pause the script for a moment.
from datetime import datetime # Needed to correctly handle date and time information.


# --- Configuration ---
# Fill this in with your details.
# This is the only part of the script you should need to change.
USERNAME = "currystan" # Your chess.com username
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chess_games_raw.csv")


def get_archive_urls(username):
    """
    First, we need to find out where all the monthly game archives are.
    This function gets the list of URLs where chess.com stores them.
    """
    print(f"Finding game archives for user: {username}...")
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    
    # BEST PRACTICE: We add a User-Agent header to identify our script.
    # Many APIs, including chess.com's, will block requests without one
    # to prevent anonymous bots from spamming their service.
    headers = {
        "User-Agent": f"My Chess Analysis Project"
    }

    try:
        # We pass our custom headers with the request.
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("archives", [])
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not fetch archive URLs. Reason: {e}")
        return []


def process_game(game_pgn, username):
    """
    This is the core function where we dissect each individual game.
    We take the raw PGN text and pull out all the details we care about.
    """
    try:
        pgn_io = io.StringIO(game_pgn)
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            return None

        game_data = {}
        headers = game.headers

        # The API gives us date and time as separate text fields. We need to
        # combine them to create a proper, sortable timestamp.
        date_str = headers.get("UTCDate")
        time_str = headers.get("UTCTime")
        full_datetime_str = f"{date_str} {time_str}"
        datetime_obj = datetime.strptime(full_datetime_str, '%Y.%m.%d %H:%M:%S')
        game_data["timestamp"] = int(datetime_obj.timestamp())

        game_data["url"] = headers.get("Link")
        game_data["date"] = headers.get("UTCDate")
        
        # Some games might not have ratings (e.g., vs. martin (bot)). We'll default to 0.
        # i play alot of games against bots when i am offline :)
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
        
        if headers.get("White", "").lower() == username.lower():
            game_data["my_color"] = "white"
            game_data["my_rating"] = white_elo
            game_data["opponent_username"] = headers.get("Black")
            game_data["opponent_rating"] = black_elo
            my_result_code = headers.get("Result")
        else:
            game_data["my_color"] = "black"
            game_data["my_rating"] = black_elo
            game_data["opponent_username"] = headers.get("White")
            game_data["opponent_rating"] = white_elo
            result_code = headers.get("Result")
            if result_code == "1-0":
                my_result_code = "0-1"
            elif result_code == "0-1":
                my_result_code = "1-0"
            else:
                my_result_code = "1/2-1/2"

        if my_result_code == "1-0":
            game_data["outcome"] = "win"
        elif my_result_code == "0-1":
            game_data["outcome"] = "loss"
        else:
            game_data["outcome"] = "draw"

        game_data["time_control"] = headers.get("TimeControl")
        game_data["time_class"] = headers.get("TimeClass")
        opening_url = headers.get("ECOUrl", "")
        game_data["opening"] = opening_url.split("/")[-1].replace("-", " ")
        game_data["number_of_moves"] = game.end().board().fullmove_number
        game_data["pgn"] = game_pgn

        return game_data
    except Exception:
        # Now that we've fixed the main bugs, we can go back to failing
        # silently. If any other weird game data shows up, we'll just skip it.
        return None


def main():
    """
    This is the main function that orchestrates the whole process.
    It calls the other functions in the right order to get the job done.
    """

    print(f"--- Starting Data Collection for {USERNAME} ---")
    archive_urls = get_archive_urls(USERNAME)
    if not archive_urls:
        print("Couldn't find any game archives. Is the username correct? Exiting.")
        return

    # We also need our header for the requests inside the loop.
    headers = {
        "User-Agent": f"My Chess Analysis Project"
    }
    
    all_games = []
    print(f"Found {len(archive_urls)} monthly archives. This might take a few minutes...")

    for url in tqdm(archive_urls, desc="Fetching Archives"):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            games_in_month = response.json().get("games", [])

            for game_info in games_in_month:
                pgn = game_info.get("pgn")
                if pgn:
                    game_details = process_game(pgn, USERNAME)
                    if game_details:
                        all_games.append(game_details)
            
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"\nWarning: Could not fetch data from {url}. Error: {e}. Skipping this month.")

    print("\n--- Data collection complete! ---")

    if not all_games:
        print("It seems no games were processed. The archives might be empty.")
        return

    print(f"Successfully processed {len(all_games)} games.")
    print("Creating a DataFrame and saving it to a CSV file...")
    df = pd.DataFrame(all_games)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"--- All done! Your data has been saved to '{OUTPUT_FILE}' ---")


if __name__ == "__main__":
    main()