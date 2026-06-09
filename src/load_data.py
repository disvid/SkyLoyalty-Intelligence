"""
src/load_data.py
----------------
Loads raw CSV files into pandas DataFrames.

Actual dataset files (confirmed from screenshots):
  1. Customer Loyalty History.csv  — 1 row per customer, 16 columns
  2. Customer Flight Activity.csv  — monthly activity, 8 columns
  3. Calendar.csv                  — date dimension: Date, Start of Year,
                                     Start of Quarter, Start of Month
  4. Airline Loyalty Data Dictionary.csv — metadata only
"""

import os
import pandas as pd

from config import DATA_DIR, LOYALTY_FILE, FLIGHT_FILE, CALENDAR_FILE, DICT_FILE
from src.utils import get_logger, timer

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and performs minimal structural validation of raw data files.

    Usage
    -----
    loader = DataLoader()
    loyalty, flight, calendar, dictionary = loader.load_all()
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a single CSV with error handling.
        Tries the exact filename first, then a lowercase fallback.
        """
        path = os.path.join(self.data_dir, filename)

        # Try exact name first
        if not os.path.exists(path):
            # Try lowercase version
            path_lower = os.path.join(self.data_dir, filename.lower())
            if os.path.exists(path_lower):
                path = path_lower
            else:
                raise FileNotFoundError(
                    f"Data file not found: {path}\n"
                    f"Please place '{filename}' in the '{self.data_dir}/' directory.\n"
                    f"Files in data dir: {os.listdir(self.data_dir) if os.path.exists(self.data_dir) else 'DIR NOT FOUND'}"
                )
        try:
            df = pd.read_csv(path, **kwargs)
            logger.info(f"Loaded '{filename}': {df.shape[0]:,} rows × {df.shape[1]} cols")
            return df
        except Exception as e:
            logger.error(f"Failed to load '{filename}': {e}")
            raise

    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise column names:
          - Strip leading/trailing whitespace
          - Lowercase
          - Replace spaces with underscores
          - Remove non-word characters (except underscore)

        Examples:
          'Loyalty Number'            → 'loyalty_number'
          'Dollar Cost Points Redeemed' → 'dollar_cost_points_redeemed'
          'CLV'                       → 'clv'
          'Start of Year'             → 'start_of_year'
        """
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^\w]", "", regex=True)
        )
        return df

    # ── Public API ─────────────────────────────────────────────────────────────

    @timer
    def load_loyalty(self) -> pd.DataFrame:
        """
        Load Customer Loyalty History.csv

        Confirmed columns (from screenshot):
          Loyalty Number, Country, Province, City, Postal Code,
          Gender, Education, Salary, Marital Status, Loyalty Card,
          CLV, Enrollment Type, Enrollment Year, Enrollment Month,
          Cancellation Year, Cancellation Month

        After standardisation:
          loyalty_number, country, province, city, postal_code,
          gender, education, salary, marital_status, loyalty_card,
          clv, enrollment_type, enrollment_year, enrollment_month,
          cancellation_year, cancellation_month

        Returns
        -------
        pd.DataFrame
        """
        df = self._load_csv(LOYALTY_FILE)
        df = self._standardise_columns(df)

        # Verify key columns exist after standardisation
        required = ["loyalty_number", "loyalty_card", "clv",
                    "enrollment_year", "enrollment_month"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Loyalty table missing expected columns: {missing}")
            logger.info(f"Actual columns: {list(df.columns)}")

        logger.info(f"Loyalty columns: {list(df.columns)}")
        return df

    @timer
    def load_flight_activity(self) -> pd.DataFrame:
        """
        Load Customer Flight Activity.csv

        Confirmed columns (from screenshot):
          Loyalty Number, Year, Month, Total Flights,
          Distance, Points Accumulated, Points Redeemed,
          Dollar Cost Points Redeemed

        NOTE: There is NO 'Flights Booked' or 'Flights With Companions'
        column in this dataset — only 'Total Flights'.

        After standardisation:
          loyalty_number, year, month, total_flights,
          distance, points_accumulated, points_redeemed,
          dollar_cost_points_redeemed

        Returns
        -------
        pd.DataFrame
        """
        df = self._load_csv(FLIGHT_FILE)
        df = self._standardise_columns(df)

        # Verify key columns
        required = ["loyalty_number", "year", "month", "total_flights",
                    "distance", "points_accumulated", "points_redeemed"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Flight table missing expected columns: {missing}")
            logger.info(f"Actual columns: {list(df.columns)}")

        logger.info(f"Flight activity columns: {list(df.columns)}")
        return df

    @timer
    def load_calendar(self) -> pd.DataFrame:
        """
        Load Calendar.csv

        Confirmed columns (from screenshot):
          Date, Start of Year, Start of Quarter, Start of Month

        After standardisation:
          date, start_of_year, start_of_quarter, start_of_month

        The calendar covers daily dates from 2012-01-01 onwards.
        Used for enriching flight activity with quarter/season info.

        Returns
        -------
        pd.DataFrame
        """
        df = self._load_csv(CALENDAR_FILE)
        df = self._standardise_columns(df)

        # Parse date columns
        for col in ["date", "start_of_year", "start_of_quarter", "start_of_month"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info(f"Calendar columns: {list(df.columns)}")
        logger.info(
            f"Calendar range: {df['date'].min().date()} → {df['date'].max().date()}"
            if "date" in df.columns else "Date column not found"
        )
        return df

    @timer
    def load_dictionary(self) -> pd.DataFrame:
        """
        Load Airline Loyalty Data Dictionary.csv

        Confirmed columns (from screenshot):
          Table, Field, Description

        Used for reference only — not included in modelling.

        Returns
        -------
        pd.DataFrame
        """
        df = self._load_csv(DICT_FILE)
        logger.info(f"Data Dictionary: {df.shape[0]} entries")
        return df

    @timer
    def load_all(self) -> tuple:
        """
        Load all four datasets.

        Returns
        -------
        tuple : (loyalty_df, flight_df, calendar_df, dictionary_df)
        """
        loyalty    = self.load_loyalty()
        flight     = self.load_flight_activity()
        calendar   = self.load_calendar()
        dictionary = self.load_dictionary()
        return loyalty, flight, calendar, dictionary