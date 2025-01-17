import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

# Set wide layout for the app
st.set_page_config(layout="wide")

# Cache data fetching to avoid re-downloading on each interaction
@st.cache_data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, interval='1d')
    return data.ffill().dropna()

# Sidebar input function
def sidebar_inputs():
    with st.sidebar.expander("Strategy Parameters", expanded=True):
        rolling_window = st.slider("Rolling Window Size", min_value=10, max_value=200, value=70, step=2)
        z_score_entry = st.slider("Z-Score Threshold for Entry", min_value=0.5, max_value=3.0, value=1.623, step=0.01)
        stop_loss_upper = st.slider("Stop-Loss Upper Threshold", min_value=2.0, max_value=5.0, value=3.0, step=0.01)
        stop_loss_lower = st.slider("Stop-Loss Lower Threshold", min_value=-5.0, max_value=-2.0, value=-3.0, step=0.01)

    return rolling_window, z_score_entry, stop_loss_upper, stop_loss_lower

# Function to generate dynamic color for Sharpe Ratio
def get_sharpe_ratio_color(sharpe_ratio):
    if sharpe_ratio > 0:
        return "#4CAF50"  # Green for positive
    else:
        return "#F44336"  # Red for negative

# Welcome Page
def welcome_page():
    st.title("Pairs Trading Strategy : Cointegration Test")
    st.write("""
        Pairs trading is a statistical arbitrage strategy that operates on the principle of identifying two financial assets that exhibit a long-term correlation. 
        These assets are typically expected to maintain a stable relationship over time due to economic or market factors. The strategy involves monitoring the 
        relative price movements of these assets and taking trading positions when their prices deviate significantly from their historical equilibrium. 
        The expectation is that the prices will revert to their mean relationship, allowing a trader to profit from this reversion.
    """)
    
    st.subheader("Cointegration and the Engle-Granger Test")
    st.write("""
        A core concept in pairs trading is **cointegration**, which is used to determine whether two non-stationary time series move together over time. 
        Even if individual series exhibit trends or random walks, cointegration ensures that the difference (or linear combination) between the two series 
        remains stationary, meaning it fluctuates around a constant mean. This is a critical requirement for pairs trading because it ensures a predictable 
        relationship between the assets, which can be exploited when deviations occur.
    """)
    
    st.markdown(
        r"""
        The Engle-Granger test begins by performing a linear regression between the two time series:
        $$ Y_t = \alpha + \beta X_t + \epsilon_t $$
        In this equation:
        - \( Y_t \) and \( X_t \) represent the prices of the two assets.
        - \( \beta \) is the slope, indicating the relationship between the two series.
        - \( \epsilon_t \) represents the residual, which captures the spread or the difference between the two assets after accounting for their linear relationship.
        
        The residuals $$ \( \epsilon_t \) are then tested for stationarity using a statistical test such as the Augmented Dickey-Fuller (ADF) test. 
        If the residuals are stationary, it indicates that the pair of assets is cointegrated, confirming that their price relationship reverts to the mean over time. 
        This stationarity of the residuals is the foundation of the pairs trading strategy, as it provides the mean-reverting characteristic necessary for trading.
        """
    )
    
    st.subheader("Spread and Z-Score")
    st.write("""
        Once a pair of assets is confirmed to be cointegrated, the next step is to calculate the **spread**, which represents the difference in their prices. 
        The spread is defined as:
    """)
    
    st.markdown(
        r"""
        $$ Spread_t = Y_t - \beta \cdot X_t $$
        """
    )
    
    st.write("""
        Here, \( \beta \) is derived from the regression analysis, and it represents the scaling factor that balances the relationship between the two assets.
        
        To evaluate the spread, rolling statistics such as the mean and standard deviation are computed over a defined time window. These measures allow for 
        tracking deviations in the spread and determining whether the spread has moved significantly away from its equilibrium. The degree of this deviation 
        is captured by the **Z-Score**, calculated as:
    """)
    
    st.markdown(
        r"""
        $$ Z\text{-}Score_t = \frac{Spread_t - \text{Mean}(Spread)}{\text{StdDev}(Spread)} $$
        """
    )
    
    st.write("""
        The Z-Score provides a standardized metric to assess how far the spread has diverged from its average value. This becomes the basis for generating trading signals.
    """)
    
    st.subheader("Trading Signals and Strategy Execution")
    st.write("""
        Trading signals are derived from the Z-Score. When the Z-Score crosses predefined thresholds, it indicates that the spread has deviated significantly 
        from its mean and is likely to revert. For example, a **buy signal** is triggered when the Z-Score falls below a negative threshold, such as \( -1.5 \), 
        indicating that the spread is undervalued. Conversely, a **sell signal** occurs when the Z-Score exceeds a positive threshold, such as \( 1.5 \), 
        suggesting the spread is overvalued. Positions are closed when the Z-Score returns to a neutral range, such as between \( -0.5 \) and \( 0.5 \).
        
        To manage risk, stop-loss thresholds are also defined. If the Z-Score breaches extreme values, such as \( -3.0 \) or \( 3.0 \), positions are closed to 
        limit potential losses. This ensures that the strategy does not rely solely on mean reversion and incorporates risk management principles.
    """)



# Strategy Page
def strategy_page():
    st.title("Pairs Trading Strategy")
    st.write("Analyze and trade based on pairs trading strategy.")

    # Input fields for stock tickers
    ticker1 = st.text_input("Enter the first stock ticker:", "KO")
    ticker2 = st.text_input("Enter the second stock ticker:", "PEP")

    # Input for date range
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2022-12-31"))

    # Get sidebar inputs
    rolling_window, z_score_entry, stop_loss_upper, stop_loss_lower = sidebar_inputs()

    # Button to run the strategy
    if st.button("Run Strategy"):
        try:
            # Fetch data
            data = fetch_data([ticker1, ticker2], start_date, end_date)

            if data.empty:
                st.error("No data fetched. Check tickers or internet connection.")
                return

            # Extract 'Close' prices
            data_close = data['Close']
            data_close.columns = [ticker1, ticker2]

            col1, col2 = st.columns(2)

            with col1:
                try:
                    logo_url1 = yf.Ticker(ticker1).info.get("logo_url", "")
                    st.image(logo_url1, caption=f"{ticker1} Logo", use_container_width=True)
                except Exception:
                    st.write(f"Logo for {ticker1} not available.")

            with col2:
                try:
                    logo_url2 = yf.Ticker(ticker2).info.get("logo_url", "")
                    st.image(logo_url2, caption=f"{ticker2} Logo", use_container_width=True)
                except Exception:
                    st.write(f"Logo for {ticker2} not available.")

            # Perform cointegration test
            score, p_value, _ = coint(data_close[ticker1], data_close[ticker2])
            cointegration_status = "Passed" if p_value < 0.05 else "Failed"

            # Linear regression for the spread
            y = data_close[ticker1]
            x = sm.add_constant(data_close[ticker2])
            model = sm.OLS(y, x).fit()
            beta = model.params[ticker2]

            # Calculate the spread
            data['Spread'] = data_close[ticker1] - beta * data_close[ticker2]

            # Define rolling statistics for Z-score
            data['Spread_Mean'] = data['Spread'].rolling(window=rolling_window).mean()
            data['Spread_StdDev'] = data['Spread'].rolling(window=rolling_window).std()
            data['Z-Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_StdDev']

            # Generate trading signals
            data['Long_Signal'] = data['Z-Score'] < -z_score_entry
            data['Short_Signal'] = data['Z-Score'] > z_score_entry
            data['Exit_Signal'] = (data['Z-Score'] > -0.5) & (data['Z-Score'] < 0.5)

            # Stop-loss signals
            data['Stop_Loss_Signal'] = (data['Z-Score'] > stop_loss_upper) | (data['Z-Score'] < stop_loss_lower)

            # Simulate positions
            data['Position'] = 0
            data.loc[data['Long_Signal'], 'Position'] = 1
            data.loc[data['Short_Signal'], 'Position'] = -1
            data.loc[data['Exit_Signal'], 'Position'] = 0
            data.loc[data['Stop_Loss_Signal'], 'Position'] = 0
            data['Position'] = data['Position'].ffill()

            # Calculate returns
            data[f'{ticker1}_Return'] = data_close[ticker1].pct_change()
            data[f'{ticker2}_Return'] = data_close[ticker2].pct_change()
            data['Strategy_Return'] = data['Position'].shift(1) * (
                data[f'{ticker1}_Return'] - beta * data[f'{ticker2}_Return']
            )
            data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

             # Display side-by-side charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Closing Prices of Both Stocks")
                st.line_chart(data_close)

            with col2:
                st.subheader("Cumulative Returns")
                st.line_chart(data['Cumulative_Return'])

            # Performance metrics
            sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std()
            sharpe_ratio_annualized = sharpe_ratio * (252 ** 0.5)
            cumulative = data['Cumulative_Return']
            drawdown = (cumulative / cumulative.cummax()) - 1
            max_drawdown = drawdown.min()
            stop_loss_exits = data['Stop_Loss_Signal'].sum()

            # Generate dynamic color for Sharpe Ratio
            sharpe_color = get_sharpe_ratio_color(sharpe_ratio_annualized)

            # Create 2x2 Metrics Grid
            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px;">
                    <div style="background-color: #87CEEB; padding: 5px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h4 style="margin: 5px 0;">Cointegration</h4>
                        <p>Status: <b>{cointegration_status}</b></p>
                        <p>P-Value: {p_value:.4f}</p>
                    </div>
                    <div style="background-color: {sharpe_color}; padding: 5px; border-radius: 5px; text-align: center; color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h4 style="margin: 5px 0;">Sharpe Ratio</h4>
                        <p style="font-size: 24px; margin: 5px 0;"><b>{sharpe_ratio_annualized:.4f}</b></p>
                    </div>
                    <div style="background-color: #87CEEB; padding: 5px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h4 style="margin: 5px 0;">Maximum Drawdown</h4>
                        <p>{max_drawdown:.2%}</p>
                    </div>
                    <div style="background-color: #87CEEB; padding: 5px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h4 style="margin: 5px 0;">Trades Exited due to Stop Loss</h4>
                        <p>{stop_loss_exits}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("## Additional Visualizations")
            st.markdown("""
                Here are some additional visualizations to help you analyze the pairs trading strategy:
                1. Spread Analysis Over Time
                2. Z-Score Distribution
            """)

            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Spread Over Time")
                st.line_chart(data['Spread'])
                

            with col4:
                st.subheader("Z-Score of the Spread")
                fig, ax = plt.subplots(figsize=(12, 6))  # Create a matplotlib figure
                data['Z-Score'].plot(ax=ax, title="Z-Score of the Spread")
                ax.axhline(2, color='red', linestyle='--', label='Upper Threshold (Z = 2)')
                ax.axhline(-2, color='green', linestyle='--', label='Lower Threshold (Z = -2)')
                ax.axhline(0, color='black', linestyle='-', label='Mean (Z = 0)')
                ax.set_xlabel("Date")
                ax.set_ylabel("Z-Score")
                ax.legend()
                ax.grid()
                st.pyplot(fig)


        except Exception as e:
            st.error(f"An error occurred: {e}")

# Main App with Tabs
def main():
    tabs = st.tabs(["Welcome Page", "Strategy Page"])

    with tabs[0]:
        welcome_page()

    with tabs[1]:
        strategy_page()

if __name__ == "__main__":
    main()

