import yfinance as yf
import pandas as pd

# 下载苹果公司的股票数据
ticker = "AAPL"  # 您可以替换为其他股票代码，例如 "GOOG", "MSFT", "TSLA" 等
start_date = "2023-01-01"  # 您可以更改开始日期
end_date = "2023-12-31"  # 您可以更改结束日期

# 下载数据
data = yf.download(ticker, start=start_date, end=end_date)

# 打印下载数据，确保数据获取正确
print(data.head())

# 将数据保存为 CSV 文件
output_path = "/Users/pingxia/python/beginners/Time-LLM/dataset/financial_data.csv"
data.to_csv(output_path)

print(f"CSV 文件已保存到 {output_path}")
