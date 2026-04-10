# v2 (dim=64) vs large-con (dim=64), dataset: test_full2

Command to repro:

```bash
python eval/compare.py \
    --name_model1 v2 \
    --name_model2 large-con \
    --gcs_model1 gs://grouping-data/runs/issue_grouping_v2/similarities/test_full2 \
    --gcs_model2 gs://grouping-data/runs/2026-04-07-11-56-28-large-con/similarities/test_full2 \
    --threshold_model1 0.92 \
    --threshold_model2 0.90 \
    --dim_model1 64 \
    --dim_model2 64 \
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

| model     | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| v2        | 0.28            | 0.98            | 0.51               | 0.43         | 0.98            |
| large-con | 0.41            | 0.97            | 0.62               | 0.64         | 0.97            |

### Project-averaged metrics (210 projects)

| model     | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| v2        | 0.19            | 0.93            | 0.54               | 0.28         | 0.97            |
| large-con | 0.32            | 0.95            | 0.62               | 0.49         | 0.96            |

### Conditional probabilities

P(large-con GROUP | v2 GROUP)    = 0.8420

P(large-con GROUP | v2 SEPARATE) = 0.2464

P(large-con GROUP | v2 GROUP, distance < 0.005) = 0.9868  (n=11628)

### Thresholds

```json
{
  "v2": {
    "default": 0.92,
    "cocoa": 0.8,
    "csharp": 0.75,
    "go": 0.8,
    "node": 0.9
  },
  "large-con": 0.9
}
```

### Distance distribution

| statistic  | value    |
|------------|----------|
| count      | 235298.0 |
| null_count | 0.0      |
| mean       | 0.040396 |
| std        | 0.030303 |
| min        | 0.000514 |
| 25%        | 0.015989 |
| 50%        | 0.033667 |
| 75%        | 0.060711 |
| max        | 0.245969 |

GROUP rate: 62.56%

### Platform stats

| platform   | n_pairs | n_projects | label_GROUP_rate | proportion |
|------------|---------|------------|------------------|------------|
| cocoa      | 35279   | 66         | 0.36             | 0.15       |
| csharp     | 25468   | 27         | 0.66             | 0.11       |
| go         | 23302   | 14         | 0.91             | 0.1        |
| java       | 25776   | 70         | 0.37             | 0.11       |
| javascript | 62545   | 82         | 0.78             | 0.27       |
| native     | 8834    | 49         | 0.27             | 0.04       |
| node       | 11573   | 20         | 0.82             | 0.05       |
| php        | 12019   | 14         | 0.72             | 0.05       |
| python     | 17559   | 15         | 0.54             | 0.07       |
| ruby       | 12943   | 15         | 0.61             | 0.06       |

### Short stacktraces (query_tokens <= p10 = 34 tokens, 24059 pairs)

label GROUP rate: 81.67%
| model     | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| v2        | 0.6             | 0.98            | 0.43               | 0.72         | 0.94            |
| large-con | 0.45            | 0.98            | 0.31               | 0.54         | 0.94            |

### Long stacktraces (query_tokens >= p90 = 931 tokens, 23577 pairs)

label GROUP rate: 70.76%
| model     | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| v2        | 0.3             | 0.99            | 0.41               | 0.41         | 0.99            |
| large-con | 0.52            | 0.98            | 0.58               | 0.72         | 0.97            |

## Threshold sweep


### Threshold sweep for large-con

| threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| 0.8       | 0.57            | 0.91            | 0.76               | 0.84         | 0.87            |
| 0.85      | 0.5             | 0.95            | 0.69               | 0.75         | 0.93            |
| 0.87      | 0.46            | 0.96            | 0.66               | 0.71         | 0.95            |
| 0.9       | 0.41            | 0.97            | 0.62               | 0.64         | 0.97            |

## Platform-level results


### Metrics by platform, avg over projects (v2, platform-specific)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | 0.8           | 0.133           | 0.862           | 0.649              | 0.218        | 0.988           |
| csharp     | 25468   | 27         | 0.662            | 0.75          | 0.316           | 0.908           | 0.542              | 0.473        | 0.939           |
| go         | 23302   | 14         | 0.91             | 0.8           | 0.548           | 0.988           | 0.299              | 0.642        | 0.912           |
| java       | 25776   | 70         | 0.368            | 0.92          | 0.042           | 0.958           | 0.641              | 0.097        | 0.997           |
| javascript | 62545   | 82         | 0.781            | 0.92          | 0.325           | 0.927           | 0.348              | 0.396        | 0.925           |
| native     | 8834    | 49         | 0.272            | 0.92          | 0.091           | 0.807           | 0.726              | 0.148        | 0.974           |
| node       | 11573   | 20         | 0.825            | 0.9           | 0.309           | 0.915           | 0.301              | 0.332        | 0.917           |
| php        | 12019   | 14         | 0.722            | 0.92          | 0.077           | 0.931           | 0.447              | 0.115        | 0.961           |
| python     | 17559   | 15         | 0.535            | 0.92          | 0.099           | 0.968           | 0.492              | 0.164        | 0.989           |
| ruby       | 12943   | 15         | 0.61             | 0.92          | 0.076           | 0.948           | 0.434              | 0.109        | 0.995           |

### Metrics by platform, avg over projects (large-con, threshold=0.9)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | 0.9           | 0.163           | 0.944           | 0.661              | 0.29         | 0.977           |
| csharp     | 25468   | 27         | 0.662            | 0.9           | 0.408           | 0.939           | 0.618              | 0.615        | 0.927           |
| go         | 23302   | 14         | 0.91             | 0.9           | 0.61            | 0.993           | 0.4                | 0.723        | 0.95            |
| java       | 25776   | 70         | 0.368            | 0.9           | 0.141           | 0.951           | 0.71               | 0.313        | 0.987           |
| javascript | 62545   | 82         | 0.781            | 0.9           | 0.534           | 0.939           | 0.453              | 0.662        | 0.88            |
| native     | 8834    | 49         | 0.272            | 0.9           | 0.158           | 0.908           | 0.801              | 0.399        | 0.983           |
| node       | 11573   | 20         | 0.825            | 0.9           | 0.456           | 0.968           | 0.404              | 0.517        | 0.946           |
| php        | 12019   | 14         | 0.722            | 0.9           | 0.382           | 0.964           | 0.622              | 0.58         | 0.937           |
| python     | 17559   | 15         | 0.535            | 0.9           | 0.288           | 0.938           | 0.604              | 0.478        | 0.959           |
| ruby       | 12943   | 15         | 0.61             | 0.9           | 0.259           | 0.912           | 0.519              | 0.396        | 0.957           |

### Min threshold for >= 95% avg project precision_GROUP by platform (v2)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | null          | null            | null            | null               | null         | null            |
| csharp     | 25468   | 27         | 0.662            | 0.94          | 0.067           | 0.955           | 0.437              | 0.101        | 0.997           |
| go         | 23302   | 14         | 0.91             | 0.59          | 0.719           | 0.95            | 0.406              | 0.819        | 0.671           |
| java       | 25776   | 70         | 0.368            | 0.86          | 0.07            | 0.951           | 0.653              | 0.159        | 0.994           |
| javascript | 62545   | 82         | 0.781            | 0.96          | 0.191           | 0.962           | 0.316              | 0.237        | 0.967           |
| native     | 8834    | 49         | 0.272            | 0.99          | 0.001           | 1.0             | 0.677              | 0.002        | 1.0             |
| node       | 11573   | 20         | 0.825            | 0.95          | 0.193           | 0.999           | 0.278              | 0.21         | 0.994           |
| php        | 12019   | 14         | 0.722            | null          | null            | null            | null               | null         | null            |
| python     | 17559   | 15         | 0.535            | 0.9           | 0.126           | 0.959           | 0.505              | 0.208        | 0.983           |
| ruby       | 12943   | 15         | 0.61             | 0.94          | 0.051           | 0.953           | 0.426              | 0.074        | 0.998           |

### Min threshold for >= 95% avg project precision_GROUP by platform (large-con)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | 0.92          | 0.142           | 0.957           | 0.651              | 0.245        | 0.982           |
| csharp     | 25468   | 27         | 0.662            | 0.92          | 0.38            | 0.952           | 0.6                | 0.572        | 0.955           |
| go         | 23302   | 14         | 0.91             | 0.78          | 0.779           | 0.952           | 0.559              | 0.891        | 0.675           |
| java       | 25776   | 70         | 0.368            | 0.9           | 0.141           | 0.951           | 0.71               | 0.313        | 0.987           |
| javascript | 62545   | 82         | 0.781            | 0.95          | 0.377           | 0.957           | 0.388              | 0.47         | 0.969           |
| native     | 8834    | 49         | 0.272            | 0.97          | 0.063           | 1.0             | 0.731              | 0.169        | 1.0             |
| node       | 11573   | 20         | 0.825            | 0.86          | 0.611           | 0.951           | 0.499              | 0.673        | 0.88            |
| php        | 12019   | 14         | 0.722            | 0.89          | 0.4             | 0.957           | 0.637              | 0.615        | 0.932           |
| python     | 17559   | 15         | 0.535            | 0.92          | 0.242           | 0.951           | 0.575              | 0.408        | 0.974           |
| ruby       | 12943   | 15         | 0.61             | 0.94          | 0.171           | 0.956           | 0.482              | 0.272        | 0.991           |

<details>
<summary>Metrics by platform</summary>

![Metrics by platform](metrics_by_platform.png)
</details>


<details>
<summary>Similarity distribution (v2)</summary>

![Similarity distribution (v2)](similarity_distribution_v2.png)
</details>


<details>
<summary>Similarity distribution (large-con)</summary>

![Similarity distribution (large-con)](similarity_distribution_large-con.png)
</details>


## Project-level results


<details>
<summary>Dumbbell by project</summary>

![Dumbbell by project](dumbbell_by_project.png)
</details>


### >= 15% group rate increase (stratified by platform)

| org_id           | project_id       | platform   | n_pairs | label_GROUP_rate | v2_GROUP_rate | v2_prec | v2_rec | large-con_GROUP_rate | large-con_prec | large-con_rec | group_rate_increase |
|------------------|------------------|------------|---------|------------------|---------------|---------|--------|----------------------|----------------|---------------|---------------------|
| 1123562          | 4504657826480128 | javascript | 36      | 1.0              | 0.14          | 1.0     | 0.14   | 0.97                 | 1.0            | 0.97          | 0.83                |
| 18005            | 4508876123734016 | php        | 5687    | 0.96             | 0.15          | 1.0     | 0.15   | 0.92                 | 1.0            | 0.95          | 0.77                |
| 494745           | 4508122107412480 | csharp     | 2923    | 0.99             | 0.19          | 1.0     | 0.2    | 0.96                 | 1.0            | 0.97          | 0.77                |
| 36448            | 81737            | go         | 1248    | 1.0              | 0.39          | 1.0     | 0.39   | 0.99                 | 1.0            | 0.99          | 0.6                 |
| 1005940          | 4506037809184768 | php        | 289     | 0.9              | 0.02          | 1.0     | 0.03   | 0.52                 | 1.0            | 0.58          | 0.49                |
| 463977           | 4508760825462784 | javascript | 4758    | 0.98             | 0.44          | 1.0     | 0.45   | 0.92                 | 1.0            | 0.94          | 0.48                |
| 474806           | 4509124176052224 | node       | 1083    | 0.89             | 0.32          | 0.99    | 0.36   | 0.74                 | 0.99           | 0.82          | 0.42                |
| 157624           | 1824476          | node       | 1494    | 0.96             | 0.52          | 1.0     | 0.54   | 0.92                 | 1.0            | 0.96          | 0.4                 |
| 71339            | 2818170          | javascript | 3647    | 0.97             | 0.42          | 0.99    | 0.43   | 0.81                 | 0.99           | 0.83          | 0.39                |
| 27134            | 270058           | javascript | 1513    | 0.83             | 0.46          | 0.91    | 0.51   | 0.85                 | 0.89           | 0.91          | 0.38                |
| 131610           | 290653           | php        | 971     | 0.59             | 0.03          | 0.93    | 0.05   | 0.4                  | 0.93           | 0.63          | 0.37                |
| 183536           | 4509513485123584 | php        | 95      | 0.58             | 0.09          | 1.0     | 0.16   | 0.45                 | 0.98           | 0.76          | 0.36                |
| 335354           | 6271291          | csharp     | 5539    | 1.0              | 0.65          | 1.0     | 0.65   | 1.0                  | 1.0            | 1.0           | 0.35                |
| 433797           | 4504451302621184 | ruby       | 2078    | 0.71             | 0.11          | 0.98    | 0.15   | 0.45                 | 0.99           | 0.63          | 0.34                |
| 141073           | 5741739          | python     | 2380    | 0.74             | 0.16          | 0.99    | 0.22   | 0.5                  | 0.99           | 0.66          | 0.34                |
| 942219           | 4505920070877184 | ruby       | 1909    | 0.59             | 0.06          | 0.88    | 0.09   | 0.36                 | 0.87           | 0.52          | 0.29                |
| 248451           | 1511685          | python     | 1990    | 0.65             | 0.13          | 0.95    | 0.19   | 0.41                 | 0.95           | 0.59          | 0.27                |
| 512760           | 5974150          | java       | 458     | 0.68             | 0.15          | 1.0     | 0.22   | 0.42                 | 0.96           | 0.6           | 0.27                |
| 4505624712839168 | 4505958273974272 | ruby       | 1831    | 0.62             | 0.07          | 0.99    | 0.11   | 0.34                 | 0.96           | 0.52          | 0.27                |
| 482609           | 4508210482905088 | python     | 88      | 0.64             | 0.15          | 0.92    | 0.21   | 0.4                  | 0.89           | 0.55          | 0.25                |
| 354398           | 2633852          | node       | 2578    | 0.88             | 0.18          | 0.99    | 0.2    | 0.43                 | 1.0            | 0.49          | 0.25                |
| 350427           | 6553847          | java       | 2982    | 0.62             | 0.06          | 0.98    | 0.1    | 0.31                 | 0.98           | 0.49          | 0.25                |
| 33646            | 4506597175394304 | csharp     | 1135    | 0.5              | 0.1           | 0.93    | 0.18   | 0.34                 | 0.97           | 0.66          | 0.24                |
| 194313           | 4506110352293888 | cocoa      | 202     | 0.65             | 0.11          | 0.77    | 0.13   | 0.35                 | 0.79           | 0.43          | 0.24                |
| 4506146950676480 | 4507936016302080 | go         | 1267    | 0.9              | 0.56          | 0.99    | 0.62   | 0.78                 | 0.99           | 0.86          | 0.22                |
| 937001           | 4506358788718592 | go         | 1358    | 0.94             | 0.68          | 0.99    | 0.72   | 0.89                 | 1.0            | 0.95          | 0.21                |
| 494704           | 5566257          | go         | 2481    | 0.83             | 0.32          | 0.99    | 0.39   | 0.53                 | 0.98           | 0.62          | 0.2                 |
| 120871           | 6716242          | java       | 436     | 0.45             | 0.08          | 0.97    | 0.17   | 0.26                 | 0.81           | 0.46          | 0.18                |
| 1122625          | 6534409          | node       | 2825    | 0.89             | 0.6           | 1.0     | 0.67   | 0.77                 | 1.0            | 0.85          | 0.17                |
| 83388            | 4505567339675648 | native     | 1130    | 0.51             | 0.01          | 1.0     | 0.02   | 0.17                 | 0.95           | 0.32          | 0.17                |

### >= 10% group rate decrease

| org_id | project_id       | platform | n_pairs | label_GROUP_rate | v2_GROUP_rate | v2_prec | v2_rec | large-con_GROUP_rate | large-con_prec | large-con_rec | group_rate_decrease |
|--------|------------------|----------|---------|------------------|---------------|---------|--------|----------------------|----------------|---------------|---------------------|
| 213164 | 4504322135097344 | go       | 9407    | 1.0              | 0.99          | 1.0     | 0.99   | 0.32                 | 1.0            | 0.32          | 0.68                |
| 304550 | 5790930          | csharp   | 998     | 0.69             | 0.38          | 0.96    | 0.52   | 0.21                 | 0.98           | 0.3           | 0.16                |
