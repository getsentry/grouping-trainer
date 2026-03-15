import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

sns.set_theme(style="darkgrid")

PATH = "eval/similarities/2026-02-26-16-25-36-val-and-test/similarities.csv"
df = pl.read_csv(PATH)

model_names = ["prod", "gte-finetuned"]
cos_sim_cols = [f"cos_sim_{model_name}" for model_name in model_names]

fig, axes = plt.subplots(len(model_names), 1, figsize=(10, 4 * len(model_names)), sharex=True)
axes: list[plt.Axes] = list(axes)

print(df.select(cos_sim_cols).describe())

for ax, model_name, col in zip(axes, model_names, cos_sim_cols, strict=True):
    sns.histplot(df[col].to_numpy(), bins=50, ax=ax)
    ax.set_title(model_name)
    ax.set_xlabel("Cosine Similarity")

fig.tight_layout()
plt.savefig("similarity_distributions.png", dpi=150)
# plt.show()
