# gemini-cluster (dim=3072) vs large-no-prefix (dim=64), dataset: test_full2

Command to repro:

```bash
python eval/compare.py \
    --name_model1 gemini-cluster \
    --name_model2 large-no-prefix \
    --gcs_model1 gs://grouping-data/runs/gemini-embedding-2-cluster/similarities/test_full2 \
    --gcs_model2 gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full2 \
    --threshold_model1 0.99 \
    --threshold_model2 0.90 \
    --dim_model1 3072 \
    --dim_model2 64
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
| gemini-cluster  | 0.1             | 0.95            | 0.41               | 0.16         | 0.99            |
| large-no-prefix | 0.44            | 0.97            | 0.64               | 0.68         | 0.97            |

### Project-averaged metrics (210 projects)

| model           | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------------|-----------------|-----------------|--------------------|--------------|-----------------|
| gemini-cluster  | 0.1             | 0.91            | 0.5                | 0.16         | 0.98            |
| large-no-prefix | 0.33            | 0.95            | 0.64               | 0.52         | 0.95            |

### Conditional probabilities

P(large-no-prefix GROUP | gemini-cluster GROUP)    = 0.8910

P(large-no-prefix GROUP | gemini-cluster SEPARATE) = 0.3822

P(large-no-prefix GROUP | gemini-cluster GROUP, distance < 0.005) = 0.9365  (n=8989)

### Thresholds

```json
{
  "gemini-cluster": 0.99,
  "large-no-prefix": 0.9
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
| model           | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------------|-----------------|-----------------|--------------------|--------------|-----------------|
| gemini-cluster  | 0.12            | 0.95            | 0.2                | 0.14         | 0.97            |
| large-no-prefix | 0.52            | 0.98            | 0.36               | 0.63         | 0.95            |

### Long stacktraces (query_tokens >= p90 = 931 tokens, 23577 pairs)

label GROUP rate: 70.76%
| model           | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------------|-----------------|-----------------|--------------------|--------------|-----------------|
| gemini-cluster  | 0.25            | 0.99            | 0.39               | 0.35         | 0.99            |
| large-no-prefix | 0.56            | 0.98            | 0.64               | 0.78         | 0.96            |

## Threshold sweep


### Threshold sweep for large-no-prefix

| threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|-----------|-----------------|-----------------|--------------------|--------------|-----------------|
| 0.8       | 0.59            | 0.91            | 0.78               | 0.86         | 0.86            |
| 0.85      | 0.52            | 0.94            | 0.72               | 0.79         | 0.92            |
| 0.87      | 0.49            | 0.96            | 0.69               | 0.75         | 0.94            |
| 0.9       | 0.44            | 0.97            | 0.64               | 0.68         | 0.97            |

## Platform-level results


### Metrics by platform, avg over projects (gemini-cluster, threshold=0.99)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | 0.99          | 0.139           | 0.815           | 0.637              | 0.232        | 0.964           |
| csharp     | 25468   | 27         | 0.662            | 0.99          | 0.172           | 0.918           | 0.474              | 0.264        | 0.975           |
| go         | 23302   | 14         | 0.91             | 0.99          | 0.24            | 0.999           | 0.219              | 0.28         | 0.991           |
| java       | 25776   | 70         | 0.368            | 0.99          | 0.068           | 0.884           | 0.65               | 0.147        | 0.991           |
| javascript | 62545   | 82         | 0.781            | 0.99          | 0.106           | 0.962           | 0.31               | 0.139        | 0.987           |
| native     | 8834    | 49         | 0.272            | 0.99          | 0.132           | 0.888           | 0.773              | 0.307        | 0.98            |
| node       | 11573   | 20         | 0.825            | 0.99          | 0.124           | 0.99            | 0.265              | 0.148        | 0.998           |
| php        | 12019   | 14         | 0.722            | 0.99          | 0.064           | 0.953           | 0.456              | 0.105        | 0.998           |
| python     | 17559   | 15         | 0.535            | 0.99          | 0.035           | 0.975           | 0.461              | 0.057        | 0.999           |
| ruby       | 12943   | 15         | 0.61             | 0.99          | 0.057           | 0.921           | 0.431              | 0.087        | 0.985           |

### Metrics by platform, avg over projects (large-no-prefix, threshold=0.9)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | 0.9           | 0.177           | 0.931           | 0.682              | 0.319        | 0.975           |
| csharp     | 25468   | 27         | 0.662            | 0.9           | 0.415           | 0.943           | 0.64               | 0.629        | 0.944           |
| go         | 23302   | 14         | 0.91             | 0.9           | 0.625           | 0.993           | 0.435              | 0.744        | 0.956           |
| java       | 25776   | 70         | 0.368            | 0.9           | 0.153           | 0.951           | 0.716              | 0.335        | 0.985           |
| javascript | 62545   | 82         | 0.781            | 0.9           | 0.551           | 0.943           | 0.48               | 0.685        | 0.872           |
| native     | 8834    | 49         | 0.272            | 0.9           | 0.178           | 0.869           | 0.806              | 0.44         | 0.959           |
| node       | 11573   | 20         | 0.825            | 0.9           | 0.491           | 0.983           | 0.425              | 0.561        | 0.967           |
| php        | 12019   | 14         | 0.722            | 0.9           | 0.408           | 0.959           | 0.673              | 0.641        | 0.931           |
| python     | 17559   | 15         | 0.535            | 0.9           | 0.305           | 0.952           | 0.622              | 0.501        | 0.955           |
| ruby       | 12943   | 15         | 0.61             | 0.9           | 0.273           | 0.92            | 0.526              | 0.418        | 0.952           |

### Min threshold for >= 95% avg project precision_GROUP by platform (gemini-cluster)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | null          | null            | null            | null               | null         | null            |
| csharp     | 25468   | 27         | 0.662            | null          | null            | null            | null               | null         | null            |
| go         | 23302   | 14         | 0.91             | 0.97          | 0.526           | 0.972           | 0.321              | 0.616        | 0.824           |
| java       | 25776   | 70         | 0.368            | null          | null            | null            | null               | null         | null            |
| javascript | 62545   | 82         | 0.781            | 0.99          | 0.106           | 0.962           | 0.31               | 0.139        | 0.987           |
| native     | 8834    | 49         | 0.272            | null          | null            | null            | null               | null         | null            |
| node       | 11573   | 20         | 0.825            | 0.99          | 0.124           | 0.99            | 0.265              | 0.148        | 0.998           |
| php        | 12019   | 14         | 0.722            | 0.99          | 0.064           | 0.953           | 0.456              | 0.105        | 0.998           |
| python     | 17559   | 15         | 0.535            | 0.99          | 0.035           | 0.975           | 0.461              | 0.057        | 0.999           |
| ruby       | 12943   | 15         | 0.61             | null          | null            | null            | null               | null         | null            |

### Min threshold for >= 95% avg project precision_GROUP by platform (large-no-prefix)

| platform   | n_pairs | n_projects | label_GROUP_rate | min_threshold | pred_GROUP_rate | precision_GROUP | precision_SEPARATE | recall_GROUP | recall_SEPARATE |
|------------|---------|------------|------------------|---------------|-----------------|-----------------|--------------------|--------------|-----------------|
| cocoa      | 35279   | 66         | 0.365            | 0.94          | 0.12            | 0.955           | 0.643              | 0.202        | 0.985           |
| csharp     | 25468   | 27         | 0.662            | 0.91          | 0.397           | 0.955           | 0.627              | 0.604        | 0.955           |
| go         | 23302   | 14         | 0.91             | 0.77          | 0.788           | 0.95            | 0.555              | 0.903        | 0.709           |
| java       | 25776   | 70         | 0.368            | 0.9           | 0.153           | 0.951           | 0.716              | 0.335        | 0.985           |
| javascript | 62545   | 82         | 0.781            | 0.93          | 0.465           | 0.951           | 0.426              | 0.58         | 0.933           |
| native     | 8834    | 49         | 0.272            | 0.95          | 0.131           | 0.959           | 0.785              | 0.32         | 0.994           |
| node       | 11573   | 20         | 0.825            | 0.87          | 0.624           | 0.967           | 0.512              | 0.698        | 0.931           |
| php        | 12019   | 14         | 0.722            | 0.9           | 0.408           | 0.959           | 0.673              | 0.641        | 0.931           |
| python     | 17559   | 15         | 0.535            | 0.9           | 0.305           | 0.952           | 0.622              | 0.501        | 0.955           |
| ruby       | 12943   | 15         | 0.61             | 0.95          | 0.155           | 0.956           | 0.473              | 0.247        | 0.993           |

<details>
<summary>Metrics by platform</summary>

![Metrics by platform](metrics_by_platform.png)
</details>


<details>
<summary>Similarity distribution (gemini-cluster)</summary>

![Similarity distribution (gemini-cluster)](similarity_distribution_gemini-cluster.png)
</details>


<details>
<summary>Similarity distribution (large-no-prefix)</summary>

![Similarity distribution (large-no-prefix)](similarity_distribution_large-no-prefix.png)
</details>


## Project-level results

**Project win rate for large-no-prefix**: 83/210 (40%) projects where both precision_GROUP and recall_GROUP are higher


<details>
<summary>Dumbbell by project</summary>

![Dumbbell by project](dumbbell_by_project.png)
</details>


### >= 15% group rate increase (stratified by platform)

| org_id           | project_id       | platform   | n_pairs | label_GROUP_rate | gemini-cluster_GROUP_rate | gemini-cluster_prec | gemini-cluster_rec | large-no-prefix_GROUP_rate | large-no-prefix_prec | large-no-prefix_rec | group_rate_increase |
|------------------|------------------|------------|---------|------------------|---------------------------|---------------------|--------------------|----------------------------|----------------------|---------------------|---------------------|
| 36448            | 81737            | go         | 1248    | 1.0              | 0.01                      | 1.0                 | 0.01               | 1.0                        | 1.0                  | 1.0                 | 0.99                |
| 335354           | 6271291          | csharp     | 5539    | 1.0              | 0.04                      | 1.0                 | 0.04               | 1.0                        | 1.0                  | 1.0                 | 0.96                |
| 87425            | 5839818          | javascript | 2733    | 0.97             | 0.01                      | 0.97                | 0.01               | 0.96                       | 1.0                  | 0.99                | 0.95                |
| 93220            | 6276779          | javascript | 3649    | 0.99             | 0.01                      | 0.92                | 0.01               | 0.95                       | 1.0                  | 0.96                | 0.94                |
| 463977           | 4508760825462784 | javascript | 4758    | 0.98             | 0.03                      | 1.0                 | 0.03               | 0.93                       | 1.0                  | 0.95                | 0.91                |
| 157624           | 1824476          | node       | 1494    | 0.96             | 0.01                      | 1.0                 | 0.01               | 0.91                       | 1.0                  | 0.95                | 0.9                 |
| 18005            | 4508876123734016 | php        | 5687    | 0.96             | 0.04                      | 1.0                 | 0.05               | 0.94                       | 0.99                 | 0.98                | 0.9                 |
| 937001           | 4506358788718592 | go         | 1358    | 0.94             | 0.09                      | 1.0                 | 0.09               | 0.88                       | 1.0                  | 0.94                | 0.8                 |
| 956749           | 5906164          | python     | 960     | 0.61             | 0.05                      | 0.91                | 0.07               | 0.69                       | 0.83                 | 0.95                | 0.65                |
| 135626           | 5689183          | node       | 629     | 0.86             | 0.12                      | 1.0                 | 0.14               | 0.75                       | 0.99                 | 0.87                | 0.63                |
| 474806           | 4509124176052224 | node       | 1083    | 0.89             | 0.16                      | 0.99                | 0.18               | 0.74                       | 0.98                 | 0.81                | 0.58                |
| 10377            | 5323974          | java       | 7       | 0.57             | 0.0                       | NaN                 | 0.0                | 0.57                       | 1.0                  | 1.0                 | 0.57                |
| 494745           | 4508122107412480 | csharp     | 2923    | 0.99             | 0.43                      | 1.0                 | 0.44               | 0.98                       | 1.0                  | 0.99                | 0.55                |
| 4506146950676480 | 4507936016302080 | go         | 1267    | 0.9              | 0.25                      | 1.0                 | 0.28               | 0.79                       | 1.0                  | 0.87                | 0.54                |
| 1198943          | 6321418          | native     | 467     | 0.61             | 0.02                      | 1.0                 | 0.04               | 0.51                       | 0.97                 | 0.8                 | 0.49                |
| 448768           | 5430655          | php        | 1634    | 0.63             | 0.12                      | 0.99                | 0.19               | 0.58                       | 0.98                 | 0.9                 | 0.45                |
| 141073           | 5741739          | python     | 2380    | 0.74             | 0.12                      | 1.0                 | 0.17               | 0.54                       | 0.98                 | 0.72                | 0.42                |
| 433797           | 4504451302621184 | ruby       | 2078    | 0.71             | 0.06                      | 1.0                 | 0.09               | 0.45                       | 0.99                 | 0.62                | 0.39                |
| 482609           | 4508210482905088 | python     | 88      | 0.64             | 0.06                      | 1.0                 | 0.09               | 0.44                       | 0.95                 | 0.66                | 0.39                |
| 4505071687041024 | 4508807082278912 | java       | 416     | 0.59             | 0.07                      | 1.0                 | 0.12               | 0.45                       | 1.0                  | 0.77                | 0.38                |
| 194313           | 4506110352293888 | cocoa      | 202     | 0.65             | 0.06                      | 1.0                 | 0.09               | 0.43                       | 0.74                 | 0.49                | 0.37                |
| 131610           | 290653           | php        | 971     | 0.59             | 0.03                      | 0.89                | 0.04               | 0.39                       | 0.93                 | 0.62                | 0.37                |
| 4505624712839168 | 4505958273974272 | ruby       | 1831    | 0.62             | 0.07                      | 0.97                | 0.11               | 0.39                       | 0.96                 | 0.6                 | 0.32                |
| 304550           | 5790930          | csharp     | 998     | 0.69             | 0.02                      | 1.0                 | 0.02               | 0.33                       | 0.98                 | 0.47                | 0.32                |
| 166814           | 1278840          | ruby       | 1026    | 0.86             | 0.02                      | 1.0                 | 0.03               | 0.3                        | 0.93                 | 0.32                | 0.28                |
| 512760           | 5974150          | java       | 458     | 0.68             | 0.13                      | 0.9                 | 0.17               | 0.4                        | 0.98                 | 0.58                | 0.28                |
| 18924            | 180537           | native     | 1739    | 0.46             | 0.13                      | 0.96                | 0.27               | 0.36                       | 0.74                 | 0.59                | 0.23                |
| 83388            | 4505567339675648 | native     | 1130    | 0.51             | 0.03                      | 0.97                | 0.05               | 0.22                       | 0.96                 | 0.42                | 0.2                 |
| 26978            | 4508754982076416 | cocoa      | 715     | 0.58             | 0.02                      | 0.92                | 0.03               | 0.2                        | 0.95                 | 0.33                | 0.19                |
| 1129             | 4506378119806976 | cocoa      | 1494    | 0.72             | 0.12                      | 0.9                 | 0.15               | 0.29                       | 0.98                 | 0.39                | 0.17                |
