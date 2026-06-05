# ckpt-15k (dim=768) vs large-no-prefix (dim=768), dataset: test_full3

Command to repro:

```bash
python -m eval.compare \
    --name_model1 ckpt-15k \
    --gcs_model1 gs://$GROUPING_TRAINER_BUCKET/runs/2026-05-21-09-13-40-ckpt-15k/similarities/test_full3 \
    --threshold_model1 0.90 \
    --name_model2 large-no-prefix \
    --gcs_model2 gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full3 \
    --threshold_model2 0.90 \
    --overwrite
```

### Column definitions

- **model**: The name of the model being evaluated.
- **pred_GROUP_rate**: The fraction of pairs this model groups together—lower means more separate issues are created. It's smaller than prod b/c the test dataset contains far more borderline cases; it's missing pairs that are very close.  This bias also means precision_GROUP is lower than what it'd be in prod.
- **precision_GROUP**: When the model groups a pair, how often is it correct? Higher = less over-grouping.
- **precision_SEPARATE**: When the model separates a pair, how often is it correct?
- **recall_GROUP**: Of all pairs that should be grouped, what fraction does the model correctly group? Higher = less under-grouping.
- **recall_SEPARATE**: Of all pairs that should be separate, what fraction does the model correctly separate?

## Aggregate results


### Overall metrics

| model           | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------------|-----------------|-----------------|--------------------|--------------|-----------------|
| ckpt-15k        | 0.41            | 0.97            | 0.67               | 0.67         | 0.97            |
| large-no-prefix | 0.38            | 0.97            | 0.65               | 0.63         | 0.97            |

### Project-averaged metrics (152 projects)

| model           | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------------|-----------------|-----------------|--------------------|--------------|-----------------|
| ckpt-15k        | 0.28            | 0.94            | 0.64               | 0.46         | 0.95            |
| large-no-prefix | 0.29            | 0.95            | 0.66               | 0.49         | 0.96            |

### Conditional probabilities

P(large-no-prefix GROUP | ckpt-15k GROUP)    = 0.8462

P(large-no-prefix GROUP | ckpt-15k SEPARATE) = 0.0656

P(large-no-prefix GROUP | ckpt-15k GROUP, distance < 0.005) = 0.9779  (n=8295)

### Thresholds

```json
{
  "ckpt-15k": 0.9,
  "large-no-prefix": 0.9
}
```

### Distance distribution

| statistic  | value    |
|------------|----------|
| count      | 150303.0 |
| null_count | 0.0      |
| mean       | 0.042189 |
| std        | 0.032109 |
| min        | 0.000514 |
| 25%        | 0.016303 |
| 50%        | 0.035027 |
| 75%        | 0.063571 |
| max        | 0.245969 |

GROUP rate: 58.75%

### Platform stats

| platform   | n_pairs | n_projects | label_GROUP_rate | proportion |
|------------|---------|------------|------------------|------------|
| cocoa      | 30321   | 58         | 0.38             | 0.2        |
| csharp     | 21936   | 21         | 0.62             | 0.15       |
| go         | 17160   | 8          | 0.95             | 0.11       |
| java       | 17043   | 58         | 0.32             | 0.11       |
| javascript | 24492   | 53         | 0.73             | 0.16       |
| native     | 6560    | 41         | 0.31             | 0.04       |
| node       | 3242    | 12         | 0.88             | 0.02       |
| php        | 9592    | 8          | 0.75             | 0.06       |
| python     | 13428   | 8          | 0.57             | 0.09       |
| ruby       | 6529    | 8          | 0.59             | 0.04       |

### Short stacktraces (query_tokens <= p10 = 32 tokens, 15296 pairs)

label GROUP rate: 83.18%
| model           | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------------|-----------------|-----------------|--------------------|--------------|-----------------|
| ckpt-15k        | 0.79            | 0.97            | 0.69               | 0.92         | 0.84            |
| large-no-prefix | 0.49            | 0.99            | 0.32               | 0.58         | 0.98            |

### Long stacktraces (query_tokens >= p90 = 764 tokens, 15036 pairs)

label GROUP rate: 59.37%
| model           | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------------|-----------------|-----------------|--------------------|--------------|-----------------|
| ckpt-15k        | 0.39            | 0.97            | 0.64               | 0.63         | 0.97            |
| large-no-prefix | 0.41            | 0.96            | 0.66               | 0.66         | 0.96            |

## Threshold sweep


### Threshold sweep for ckpt-15k

| threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| 0.8       | 0.54            | 0.9             | 0.78               | 0.83         | 0.86            |
| 0.85      | 0.48            | 0.93            | 0.73               | 0.76         | 0.92            |
| 0.87      | 0.45            | 0.95            | 0.71               | 0.73         | 0.94            |
| 0.9       | 0.41            | 0.97            | 0.67               | 0.67         | 0.97            |

### Threshold sweep for large-no-prefix

| threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| 0.8       | 0.55            | 0.9             | 0.79               | 0.84         | 0.87            |
| 0.85      | 0.47            | 0.94            | 0.73               | 0.76         | 0.93            |
| 0.87      | 0.44            | 0.95            | 0.7                | 0.71         | 0.95            |
| 0.9       | 0.38            | 0.97            | 0.65               | 0.63         | 0.97            |

## Platform-level results


### Metrics by platform, avg over projects (ckpt-15k, threshold=0.9)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 30321   | 58         | 0.38             | 0.9           | 0.155           | 0.939           | 0.651              | 0.296        | 0.99            |
| csharp     | 21936   | 21         | 0.617            | 0.9           | 0.337           | 0.927           | 0.653              | 0.538        | 0.906           |
| go         | 17160   | 8          | 0.953            | 0.9           | 0.668           | 0.994           | 0.331              | 0.775        | 0.965           |
| java       | 17043   | 58         | 0.322            | 0.9           | 0.149           | 0.953           | 0.72               | 0.341        | 0.991           |
| javascript | 24492   | 53         | 0.729            | 0.9           | 0.458           | 0.95            | 0.491              | 0.6          | 0.917           |
| native     | 6560    | 41         | 0.31             | 0.9           | 0.186           | 0.868           | 0.803              | 0.454        | 0.975           |
| node       | 3242    | 12         | 0.884            | 0.9           | 0.552           | 0.94            | 0.516              | 0.621        | 0.916           |
| php        | 9592    | 8          | 0.75             | 0.9           | 0.338           | 0.961           | 0.686              | 0.56         | 0.954           |
| python     | 13428   | 8          | 0.568            | 0.9           | 0.304           | 0.955           | 0.587              | 0.496        | 0.968           |
| ruby       | 6529    | 8          | 0.59             | 0.9           | 0.296           | 0.931           | 0.531              | 0.429        | 0.948           |

### Metrics by platform, avg over projects (large-no-prefix, threshold=0.9)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 30321   | 58         | 0.38             | 0.9           | 0.177           | 0.92            | 0.661              | 0.312        | 0.973           |
| csharp     | 21936   | 21         | 0.617            | 0.9           | 0.368           | 0.93            | 0.683              | 0.608        | 0.94            |
| go         | 17160   | 8          | 0.953            | 0.9           | 0.614           | 0.995           | 0.392              | 0.722        | 0.974           |
| java       | 17043   | 58         | 0.322            | 0.9           | 0.15            | 0.959           | 0.72               | 0.33         | 0.991           |
| javascript | 24492   | 53         | 0.729            | 0.9           | 0.489           | 0.93            | 0.511              | 0.628        | 0.909           |
| native     | 6560    | 41         | 0.31             | 0.9           | 0.172           | 0.869           | 0.79               | 0.461        | 0.962           |
| node       | 3242    | 12         | 0.884            | 0.9           | 0.404           | 0.982           | 0.496              | 0.487        | 0.982           |
| php        | 9592    | 8          | 0.75             | 0.9           | 0.372           | 0.957           | 0.744              | 0.637        | 0.962           |
| python     | 13428   | 8          | 0.568            | 0.9           | 0.329           | 0.951           | 0.604              | 0.535        | 0.96            |
| ruby       | 6529    | 8          | 0.59             | 0.9           | 0.287           | 0.945           | 0.488              | 0.436        | 0.96            |

### Min threshold for >= 95% avg project precision_GROUP by platform (ckpt-15k)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 30321   | 58         | 0.38             | 0.91          | 0.144           | 0.954           | 0.645              | 0.273        | 0.993           |
| csharp     | 21936   | 21         | 0.617            | 0.92          | 0.304           | 0.95            | 0.612              | 0.478        | 0.926           |
| go         | 17160   | 8          | 0.953            | 0.8           | 0.747           | 0.962           | 0.446              | 0.853        | 0.812           |
| java       | 17043   | 58         | 0.322            | 0.9           | 0.149           | 0.953           | 0.72               | 0.341        | 0.991           |
| javascript | 24492   | 53         | 0.729            | 0.9           | 0.458           | 0.95            | 0.491              | 0.6          | 0.917           |
| native     | 6560    | 41         | 0.31             | 0.95          | 0.09            | 0.982           | 0.734              | 0.204        | 1.0             |
| node       | 3242    | 12         | 0.884            | 0.93          | 0.454           | 0.961           | 0.471              | 0.507        | 0.954           |
| php        | 9592    | 8          | 0.75             | 0.9           | 0.338           | 0.961           | 0.686              | 0.56         | 0.954           |
| python     | 13428   | 8          | 0.568            | 0.9           | 0.304           | 0.955           | 0.587              | 0.496        | 0.968           |
| ruby       | 6529    | 8          | 0.59             | 0.92          | 0.248           | 0.953           | 0.484              | 0.358        | 0.967           |

### Min threshold for >= 95% avg project precision_GROUP by platform (large-no-prefix)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 30321   | 58         | 0.38             | 0.94          | 0.136           | 0.964           | 0.637              | 0.216        | 0.985           |
| csharp     | 21936   | 21         | 0.617            | 0.93          | 0.316           | 0.956           | 0.619              | 0.523        | 0.956           |
| go         | 17160   | 8          | 0.953            | 0.8           | 0.771           | 0.959           | 0.474              | 0.879        | 0.731           |
| java       | 17043   | 58         | 0.322            | 0.89          | 0.159           | 0.952           | 0.725              | 0.353        | 0.987           |
| javascript | 24492   | 53         | 0.729            | 0.93          | 0.408           | 0.953           | 0.457              | 0.516        | 0.944           |
| native     | 6560    | 41         | 0.31             | 0.96          | 0.108           | 0.954           | 0.751              | 0.272        | 0.997           |
| node       | 3242    | 12         | 0.884            | 0.86          | 0.548           | 0.957           | 0.583              | 0.641        | 0.962           |
| php        | 9592    | 8          | 0.75             | 0.89          | 0.386           | 0.952           | 0.762              | 0.669        | 0.956           |
| python     | 13428   | 8          | 0.568            | 0.9           | 0.329           | 0.951           | 0.604              | 0.535        | 0.96            |
| ruby       | 6529    | 8          | 0.59             | 0.91          | 0.26            | 0.955           | 0.476              | 0.399        | 0.969           |

<details>
<summary>Metrics by platform</summary>

![Metrics by platform](metrics_by_platform.png)
</details>


<details>
<summary>Similarity distribution (ckpt-15k)</summary>

![Similarity distribution (ckpt-15k)](similarity_distribution_ckpt-15k.png)
</details>


<details>
<summary>Similarity distribution (large-no-prefix)</summary>

![Similarity distribution (large-no-prefix)](similarity_distribution_large-no-prefix.png)
</details>


## Project-level results

**Project win rate for large-no-prefix**: 43/152 (28%) projects where both precision_GROUP and recall_GROUP are higher


<details>
<summary>Dumbbell by project</summary>

![Dumbbell by project](dumbbell_by_project.png)
</details>


### >= 15% group rate increase (stratified by platform)

| org_id | project_id | platform | n_pairs | label_GROUP_rate | ckpt-15k_GROUP_rate | ckpt-15k_prec | ckpt-15k_rec | large-no-prefix_GROUP_rate | large-no-prefix_prec | large-no-prefix_rec | group_rate_increase |
|--------|------------|----------|---------|------------------|---------------------|---------------|--------------|----------------------------|----------------------|---------------------|---------------------|
| id_87  | id_259     | python   | 88      | 0.64             | 0.22                | 1.0           | 0.34         | 0.43                       | 0.92                 | 0.62                | 0.22                |
| id_66  | id_199     | csharp   | 998     | 0.69             | 0.21                | 0.98          | 0.29         | 0.36                       | 0.98                 | 0.51                | 0.15                |

### >= 10% group rate decrease

| org_id | project_id | platform   | n_pairs | label_GROUP_rate | ckpt-15k_GROUP_rate | ckpt-15k_prec | ckpt-15k_rec | large-no-prefix_GROUP_rate | large-no-prefix_prec | large-no-prefix_rec | group_rate_decrease |
|--------|------------|------------|---------|------------------|---------------------|---------------|--------------|----------------------------|----------------------|---------------------|---------------------|
| id_55  | id_222     | go         | 9407    | 1.0              | 0.99                | 1.0           | 0.99         | 0.53                       | 1.0                  | 0.53                | 0.46                |
| id_10  | id_205     | node       | 30      | 0.57             | 0.6                 | 0.5           | 0.53         | 0.2                        | 1.0                  | 0.35                | 0.4                 |
| id_46  | id_159     | javascript | 2487    | 0.43             | 0.47                | 0.69          | 0.75         | 0.29                       | 0.93                 | 0.62                | 0.18                |
| id_44  | id_154     | ruby       | 1026    | 0.86             | 0.38                | 0.92          | 0.41         | 0.25                       | 0.94                 | 0.27                | 0.13                |


---

_Real report with original org/project IDs at_ `gs://$GROUPING_TRAINER_BUCKET/eval/comparisons/test_full3/ckpt-15k_dim768_vs_large-no-prefix_dim768/`
