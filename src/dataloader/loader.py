import hydra
from omegaconf import DictConfig, OmegaConf
import yfinance as yf
import pandas as pd
import os
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def load_data(cfg: DictConfig):
    """
    Downloads historical stock data for configured tickers using yfinance
    and saves them as raw CSV files.
    """
    logger.info(f"Loading configuration:\n{OmegaConf.to_yaml(cfg)}")

    tickers = list(cfg.data.tickers)
    start_date = cfg.data.start_date
    end_date = cfg.data.end_date
    raw_dir = cfg.data.raw_dir

    # Create raw data directory if it doesn't exist
    os.makedirs(raw_dir, exist_ok=True)

    logger.info(f"Downloading data for {tickers} from {start_date} to {end_date}")

    # Download data using yfinance
    # auto_adjust=True: returns Open, High, Low, Close, Volume (adjusted for splits/dividends)
    # group_by='ticker': ensures MultiIndex (Ticker, Price)
    try:
        data = yf.download(
            tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True
        )
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return

    if data.empty:
        logger.error(
            "No data downloaded. Check your internet connection or ticker symbols."
        )
        return

    # Handle case where only one ticker is requested (yfinance might not return MultiIndex in some versions)
    if len(tickers) == 1:
        ticker = tickers[0]
        # If columns are not MultiIndex, we just use the dataframe as is
        if isinstance(data.columns, pd.MultiIndex):
            # This theoretically shouldn't happen with single ticker + group_by, but safe to check
            if ticker in data.columns.levels[0]:
                df = data[ticker].copy()
            else:
                # Fallback
                df = data.copy()
        else:
            df = data.copy()

        # Reset index to make Date a column
        df.reset_index(inplace=True)
        output_path = os.path.join(raw_dir, f"{ticker}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {ticker} data to {output_path}")
        return

    # Save each ticker to a separate CSV file
    for ticker in tickers:
        try:
            if ticker not in data.columns:
                logger.warning(f"Ticker {ticker} not found in downloaded data columns.")
                continue

            # Extract data for the specific ticker
            df = data[ticker].copy()

            # Check if dataframe is empty or all NaN
            if df.empty or df.isna().all().all():
                logger.warning(f"Data for {ticker} is empty or all NaN.")
                continue

            # Reset index to make Date a column
            df.reset_index(inplace=True)

            # Save to CSV
            output_path = os.path.join(raw_dir, f"{ticker}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {ticker} data to {output_path}")

        except KeyError as e:
            logger.error(f"Failed to process ticker {ticker}: {e}")
        except Exception as e:
            logger.error(f"An error occurred while saving {ticker}: {e}")


if __name__ == "__main__":
    load_data()
