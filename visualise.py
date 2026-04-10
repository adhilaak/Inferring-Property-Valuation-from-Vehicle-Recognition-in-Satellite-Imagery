import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import numpy as np
from scipy.stats import linregress
import matplotlib.ticker as ticker


os.makedirs("plots", exist_ok=True)


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


df = pd.read_csv("data/final_combined_data.csv")


df_clean = df.dropna(subset=["avg_car_value", "avg_house_price"])

df_grouped = df_clean.groupby("postcode").agg({
    "avg_car_value": "median",
    "avg_house_price": "median",
    "location": "count"
}).rename(columns={"location": "num_cars"}).reset_index()


df_grouped = df_grouped[df_grouped["avg_house_price"] < 2_000_000]


df_grouped['log_car_value'] = np.log1p(df_grouped['avg_car_value'])
df_grouped['log_house_price'] = np.log1p(df_grouped['avg_house_price'])


try:
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title('Test Plot: If you see this, plotting works')
    plt.show()
except Exception as e:
    print(f"Error in test plot: {e}")


try:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_grouped['avg_car_value'], kde=True, color="#0072B2")
    plt.title('Distribution of Avg Car Value')
    plt.xlabel('Average Car Value (£)')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    sns.histplot(df_grouped['avg_house_price'], kde=True, color="#D55E00")
    plt.title('Distribution of Avg House Price')
    plt.xlabel('Average House Price (£)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("plots/fig1_car_and_house_price_distributions.png", dpi=300)
    print("Saved: plots/fig1_car_and_house_price_distributions.png")
    plt.show()  # DEBUG
    plt.close()
except Exception as e:
    print(f"Error in Distribution Plots (Original): {e}")

try:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_grouped['log_car_value'], kde=True, color="#0072B2")
    plt.title('Log Distribution of Avg Car Value')
    plt.xlabel('Log Average Car Value')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    sns.histplot(df_grouped['log_house_price'], kde=True, color="#D55E00")
    plt.title('Log Distribution of Avg House Price')
    plt.xlabel('Log Average House Price')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("plots/fig2_log_car_and_house_price_distributions.png", dpi=300)
    print("Saved: plots/fig2_log_car_and_house_price_distributions.png")
    plt.show()  # DEBUG
    plt.close()
except Exception as e:
    print(f"Error in Distribution Plots (Log): {e}")


try:
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    scatter = sns.scatterplot(
        data=df_grouped,
        x="log_car_value",
        y="log_house_price",
        size="num_cars",
        sizes=(20, 100),
        alpha=0.5,
        color="#0072B2",
        legend=False
    )
    reg = sns.regplot(
        data=df_grouped,
        x="log_car_value",
        y="log_house_price",
        scatter=False,
        color="#D55E00",
        line_kws={"linewidth": 3, "label": "Linear Fit"}
    )
    slope, intercept, r_value, p_value, std_err = linregress(df_grouped['log_car_value'], df_grouped['log_house_price'])
    plt.text(0.05, 0.95, f"$R^2$ = {r_value**2:.2f}\np = {p_value:.3g}", transform=plt.gca().transAxes, fontsize=13, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    top_n = df_grouped.nlargest(5, "num_cars")
    for i, row in top_n.iterrows():
        plt.text(row["log_car_value"], row["log_house_price"], row["postcode"], fontsize=10, color="black")
    plt.title("Figure 3. Log of Avg Car Value vs Log of Avg House Price per Postcode\nwith Linear Regression")
    plt.xlabel("Log of Average Car Value (£)")
    plt.ylabel("Log of Average House Price (£)")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    if any(labels):
        plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fig3_log_car_vs_house_scatter_linear_labeled.png", dpi=300)
    print("Saved: plots/fig3_log_car_vs_house_scatter_linear_labeled.png")
    plt.show()  # DEBUG
    plt.close()
except Exception as e:
    print(f"Error in Scatter Plot with Regression: {e}")


try:
    correlation = df_grouped[["log_car_value", "log_house_price"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, annot_kws={"size": 14})
    plt.title("Figure 4. Correlation Matrix (Log Transformed Values)")
    plt.tight_layout()
    plt.savefig("plots/fig4_log_car_vs_house_corr_heatmap_median.png", dpi=300)
    print("Saved: plots/fig4_log_car_vs_house_corr_heatmap_median.png")
    plt.show()  # DEBUG
    plt.close()
except Exception as e:
    print(f"Error in Correlation Matrix Heatmap: {e}")


try:
    df_grouped['house_price_decile'] = pd.qcut(df_grouped['avg_house_price'], 10, labels=[f"D{i+1}" for i in range(10)])
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df_grouped,
        x='house_price_decile',
        y='avg_car_value'
    )
    plt.title("Figure 5. Distribution of Avg Car Value by House Price Decile")
    plt.xlabel("House Price Decile (D1 = lowest, D10 = highest)")
    plt.ylabel("Average Car Value (£)")
    plt.tight_layout()
    plt.savefig("plots/fig5_car_value_by_house_price_decile.png", dpi=300)
    print("Saved: plots/fig5_car_value_by_house_price_decile.png")
    plt.show()  # DEBUG
    plt.close()
except Exception as e:
    print(f"Error in Boxplot by House Price Decile: {e}")


try:
    plt.figure(figsize=(12, 8))
    

    scatter = sns.scatterplot(
        data=df_grouped,
        x="log_car_value",
        y="log_house_price",
        size="num_cars",
        sizes=(20, 200),
        alpha=0.6,
        color="#0072B2",
        legend=False
    )
    
   
    reg = sns.regplot(
        data=df_grouped,
        x="log_car_value",
        y="log_house_price",
        scatter=False,
        color="#D55E00",
        line_kws={"linewidth": 2, "label": "Linear Fit"},
        ci=95
    )
    
   
    slope, intercept, r_value, p_value, std_err = linregress(df_grouped['log_car_value'], df_grouped['log_house_price'])
    
   
    stats_text = (
        f"$R^2$ = {r_value**2:.3f}\n"
        f"p-value = {p_value:.2e}\n"
        f"Slope = {slope:.3f}\n"
        f"n = {len(df_grouped)}"
    )
    
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white",
                      alpha=0.8,
                      edgecolor="gray"))
    

    top_n = df_grouped.nlargest(5, "num_cars")
    texts = []
    for i, row in top_n.iterrows():
        texts.append(plt.text(row["log_car_value"],
                            row["log_house_price"],
                            row["postcode"],
                            fontsize=10,
                            color="black",
                            bbox=dict(facecolor='white',
                                    alpha=0.7,
                                    edgecolor='none',
                                    pad=1)))
    

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.title("Figure 3. Relationship Between Car Value and House Price\nby Postcode (Log Scale)",
              fontsize=14, pad=20)
    plt.xlabel("Log of Average Car Value (£)", fontsize=12)
    plt.ylabel("Log of Average House Price (£)", fontsize=12)
    
  
    def log_to_actual(x, pos):
        return f"£{np.expm1(x):,.0f}"
    
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(log_to_actual))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_to_actual))
    
    plt.tight_layout()
    plt.savefig("plots/fig3_enhanced_scatter.png", dpi=300, bbox_inches='tight')
    print("Saved: plots/fig3_enhanced_scatter.png")
    plt.close()
except Exception as e:
    print(f"Error in Enhanced Scatter Plot: {e}")


try:
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=df_grouped,
        x="avg_car_value",
        y="avg_house_price",
        size="num_cars",
        sizes=(20, 200),
        alpha=0.6,
        color="#0072B2",
        legend=False
    )
    
    reg = sns.regplot(
        data=df_grouped,
        x="avg_car_value",
        y="avg_house_price",
        scatter=False,
        color="#D55E00",
        line_kws={"linewidth": 2, "label": "Linear Fit"},
        ci=95
    )

    slope, intercept, r_value, p_value, std_err = linregress(df_grouped['avg_car_value'], df_grouped['avg_house_price'])
    
    stats_text = (
        f"$R^2$ = {r_value**2:.3f}\n"
        f"p-value = {p_value:.2e}\n"
        f"Slope = {slope:.3f}\n"
        f"n = {len(df_grouped)}"
    )
    
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white",
                      alpha=0.8,
                      edgecolor="gray"))
    
    plt.title("Figure 4. Relationship Between Car Value and House Price\nby Postcode (Actual Values)",
              fontsize=14, pad=20)
    plt.xlabel("Average Car Value (£)", fontsize=12)
    plt.ylabel("Average House Price (£)", fontsize=12)

    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('£{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('£{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig("plots/fig4_actual_values_scatter.png", dpi=300, bbox_inches='tight')
    print("Saved: plots/fig4_actual_values_scatter.png")
    plt.close()
except Exception as e:
    print(f"Error in Actual Value Scatter Plot: {e}")


try:
    df_grouped['house_price_decile'] = pd.qcut(df_grouped['avg_house_price'], 10, labels=[f"D{i+1}" for i in range(10)])
    
    plt.figure(figsize=(14, 8))
    
   
    box = sns.boxplot(
        data=df_grouped,
        x='house_price_decile',
        y='avg_car_value',
        palette="viridis",
        width=0.7,
        showfliers=True
    )
    

    sns.stripplot(
        data=df_grouped,
        x='house_price_decile',
        y='avg_car_value',
        color='black',
        alpha=0.3,
        size=4,
        jitter=True
    )
    

    means = df_grouped.groupby('house_price_decile')['avg_car_value'].mean()
    plt.plot(range(len(means)), means, 'r--', label='Mean', linewidth=2)
    
    plt.title("Figure 5. Distribution of Car Values by House Price Decile",
              fontsize=14, pad=20)
    plt.xlabel("House Price Decile (D1 = lowest, D10 = highest)", fontsize=12)
    plt.ylabel("Average Car Value (£)", fontsize=12)
    

    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('£{x:,.0f}'))
    
  
    plt.legend(['Mean Value'])
    
 
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("plots/fig5_enhanced_boxplot.png", dpi=300, bbox_inches='tight')
    print("Saved: plots/fig5_enhanced_boxplot.png")
    plt.close()
except Exception as e:
    print(f"Error in Enhanced Boxplot: {e}")


try:
    correlation = df_grouped[["avg_car_value", "avg_house_price", "num_cars"]].corr()
    
    plt.figure(figsize=(8, 6))
    

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        cbar=True,
        annot_kws={"size": 12},
        square=True,
        vmin=-1,
        vmax=1
    )
    
    plt.title("Figure 6. Correlation Matrix of Key Variables",
              fontsize=14, pad=20)
    

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig("plots/fig6_enhanced_correlation.png", dpi=300, bbox_inches='tight')
    print("Saved: plots/fig6_enhanced_correlation.png")
    plt.close()
except Exception as e:
    print(f"Error in Enhanced Correlation Heatmap: {e}")
