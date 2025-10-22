# Checkmate with Data: Analyzing My Summer Chess Journey


## 1. Project Motivation and Goals

This past summer, I spent a lot of my free time playing chess on `chess.com`. While I felt like I was improving, I also hit frustrating plateaus and couldn't always pinpoint why I was losing games. This got me thinking: are there hidden patterns in my games? Could I use the data from my own matches to truly understand my habits, find my weaknesses, and maybe even predict how a game will turn out?

This project is my chance to answer those questions using the skills we're learning in DSA 210. It's the perfect opportunity to apply the entire data science pipeline, from fetching data with an API to building a machine learning model, to a topic I'm genuinely passionate about. My goal isn't just to analyze the data, but to tell the story of my summer chess journey and see what insights I can uncover to become a better player.

## 2. Detailed Research Questions

The project is structured around three key analytical themes, each with specific, measurable questions:

**Theme A: Exploratory and Temporal Analysis**

1.  **Performance Evolution (Time Series Analysis):**
    *   What is the temporal trend of my ELO rating over the summer? Can we identify periods of rapid improvement, stagnation, or decline?
    *   Is there a discernible weekly or daily pattern in my playing activity? Do I play more on weekends, and does my win rate correlate with the time of day or day of the week?

2.  **Playing Style and Opening Repertoire:**
    *   What is the frequency distribution of the chess openings I employ when playing as White versus as Black?
    *   Which openings yield my highest and lowest win rates? Does my performance in a specific opening justify its frequent use?

**Theme B: Statistical Inference and Hypothesis Testing**

3.  **The Advantage of the First Move:**
    *   Is my win rate when playing with the White pieces statistically significantly higher than my win rate with the Black pieces? This will be tested formally.

4.  **Game Dynamics and Outcomes:**
    *   Is there a statistically significant difference in the average length (number of moves) of my winning games compared to my losing games?

**Theme C: Predictive Modeling (Supervised Machine learning)**

5.  **Outcome Prediction:**
    *   Can a supervised classification model accurately predict the final outcome of a game (Win/Loss) based on the state of the game after the first 15 moves? This simulates the ability to assess an advantage early in the game.

## 3. Data Provenance and Collection

*   **Data Source:** The project will exclusively use my personal game history from the `chess.com` platform. This data is publicly available via their API.
*   **Collection Methodology:**
    1.  A Python script will be developed to programmatically access the `chess.com` Public Data API. The script will use the `requests` library to make GET requests to the endpoint: `https://api.chess.com/pub/player/{username}/games/archives`.
    2.  This endpoint returns a list of URLs, each pointing to a monthly archive of games in JSON format. The script will iterate through these URLs to download all game data for the specified period.
    3.  Each game's data, which includes metadata and the full move list in Portable Game Notation (PGN) format, will be parsed using the `python-chess` library.
*   **Final Dataset:** The raw data will be processed and structured into a single Pandas DataFrame. Each row will represent one chess game, and columns will consist of extracted features such as `Date`, `MyRating`, `OpponentRating`, `MyColor`, `GameOutcome`, `OpeningName`, `NumberOfMoves`, etc. This cleaned dataset will be saved as a CSV file for analysis.

## 4. Methodology and Analysis Plan

The project will systematically apply the following methods:

**4.1. Data Cleaning and Feature Engineering:**
*   Handle incomplete or aborted games.
*   Convert date/time strings to datetime objects for time series analysis.
*   For the machine learning phase, engineer features from the game state after 15 moves, including:
    *   `MaterialAdvantage`: The difference in the value of pieces on the board.
    *   `DevelopedPieces`: Count of minor and major pieces that have moved from their starting squares.
    *   `RatingDifference`: The ELO rating gap between me and my opponent.

**4.2. Exploratory Data Analysis (EDA) & Visualization:**
*   **Time Series Plot:** Visualize my ELO rating over time using `Matplotlib` to answer RQ1.
*   **Bar Charts:** Use `Seaborn` to display win/loss/draw counts by piece color (RQ3) and to show the frequency and success rates of my top 10 openings (RQ2).
*   **Heatmap:** Visualize my win rate by day of the week and hour of the day to identify patterns (RQ1).
*   **Histograms & Boxplots:** Analyze the distribution of game lengths for wins vs. losses (RQ4).

**4.3. Hypothesis Testing:**
*   For RQ3 (Win rate White vs. Black), a **two-proportion z-test** will be conducted to determine if the difference in win proportions is statistically significant (p < 0.05).
*   For RQ4 (Game length in wins vs. losses), an **independent samples t-test** will be used to compare the means of the two groups. The assumptions of the t-test (e.g., normality, equal variances) will be checked.

**4.4. Supervised Machine Learning:**
*   **Problem Framing:** This is a binary classification task. The target variable is `GameOutcome`, simplified to 'Win' or 'Loss' (draws will be excluded for this model).
*   **Model Selection:** I will implement and compare several models using `scikit-learn`:
    1.  **Logistic Regression:** As a baseline for performance.
    2.  **Random Forest Classifier:** An ensemble method to capture more complex interactions between features.
*   **Training & Evaluation:** The data will be split into an 80% training set and a 20% testing set. Model performance will be evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and a **Confusion Matrix**.

## 5. Instructions for Reproducibility

To ensure the project is fully reproducible, the final GitHub repository will be structured as follows:

*   `README.md`: This project proposal and final report.
*   `requirements.txt`: A file listing all Python dependencies. The environment can be replicated using `pip install -r requirements.txt`. Key libraries will include `pandas`, `requests`, `python-chess`, `scikit-learn`, `matplotlib`, and `seaborn`.
*   `/scripts`: A directory containing standalone Python scripts (e.g., `01_fetch_data.py` for data collection).
*   `/data`: A directory for the raw and cleaned datasets (the `.csv` file).
*   `analysis.ipynb`: A Jupyter Notebook containing the full analysis, from data loading to visualization, hypothesis testing, and machine learning, with clear Markdown explanations.

**To reproduce the analysis, a user will follow these steps:**
1.  Clone the repository: `git clone [repository-url]`
2.  Create a Python virtual environment and activate it.
3.  Install dependencies: `pip install -r requirements.txt`
4.  (Optional) Run the data collection script: `python scripts/01_fetch_data.py --username [your-chess.com-username]`
5.  Open and run the `analysis.ipynb` notebook to see the complete analysis.

## 7. Ethical Considerations

This project uses my own publicly available data, minimizing privacy concerns. However, I will acknowledge the limitations of the analysis: the findings are specific to a single, amateur player and should not be generalized. The predictive model will be interpreted as a tool for self-assessment and learning, not as a definitive predictor of success. The use of the `chess.com` API will be done responsibly, respecting any rate limits to avoid disrupting their service.
