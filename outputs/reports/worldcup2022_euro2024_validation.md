# Tournament Validation Report: World Cup 2022 & Euro 2024

Replays real tournament fixtures through the trained pipeline and compares
predictions against actual results.

## FIFA World Cup 2022 (64 matches)

| Date         | Home                 | Away                 | Score | XGB Pick | Correct | Poisson | Exact |
|--------------|----------------------|----------------------|-------|----------|---------|---------|-------|
| 2022-11-20   | Qatar                | Ecuador              |   0-2 | home_win |      No | 1-    1 |    No |
| 2022-11-21   | England              | Iran                 |   6-2 | home_win |     Yes | 2-    1 |    No |
| 2022-11-21   | Senegal              | Netherlands          |   0-2 | away_win |     Yes | 1-    2 |    No |
| 2022-11-21   | United States        | Wales                |   1-1 | home_win |      No | 2-    1 |    No |
| 2022-11-22   | Argentina            | Saudi Arabia         |   1-2 | home_win |      No | 2-    1 |    No |
| 2022-11-22   | Denmark              | Tunisia              |   0-0 | home_win |      No | 2-    1 |    No |
| 2022-11-22   | France               | Australia            |   4-1 | home_win |     Yes | 2-    1 |    No |
| 2022-11-22   | Mexico               | Poland               |   0-0 | home_win |      No | 1-    1 |    No |
| 2022-11-23   | Belgium              | Canada               |   1-0 | home_win |     Yes | 2-    1 |    No |
| 2022-11-23   | Germany              | Japan                |   1-2 | away_win |     Yes | 2-    1 |    No |
| 2022-11-23   | Morocco              | Croatia              |   0-0 | away_win |      No | 1-    1 |    No |
| 2022-11-23   | Spain                | Costa Rica           |   7-0 | home_win |     Yes | 2-    1 |    No |
| 2022-11-24   | Brazil               | Serbia               |   2-0 | home_win |     Yes | 2-    1 |    No |
| 2022-11-24   | Portugal             | Ghana                |   3-2 | home_win |     Yes | 2-    1 |    No |
| 2022-11-24   | Switzerland          | Cameroon             |   1-0 | home_win |     Yes | 2-    1 |    No |
| 2022-11-24   | Uruguay              | South Korea          |   0-0 | home_win |      No | 2-    1 |    No |
| 2022-11-25   | England              | United States        |   0-0 | home_win |      No | 2-    1 |    No |
| 2022-11-25   | Netherlands          | Ecuador              |   1-1 | home_win |      No | 2-    1 |    No |
| 2022-11-25   | Qatar                | Senegal              |   1-3 | home_win |      No | 1-    1 |    No |
| 2022-11-25   | Wales                | Iran                 |   0-2 | home_win |      No | 1-    1 |    No |
| 2022-11-26   | Argentina            | Mexico               |   2-0 | home_win |     Yes | 2-    1 |    No |
| 2022-11-26   | France               | Denmark              |   2-1 | home_win |     Yes | 2-    1 |   Yes |
| 2022-11-26   | Poland               | Saudi Arabia         |   2-0 | home_win |     Yes | 1-    1 |    No |
| 2022-11-26   | Tunisia              | Australia            |   0-1 | home_win |      No | 1-    1 |    No |
| 2022-11-27   | Belgium              | Morocco              |   0-2 | home_win |      No | 2-    1 |    No |
| 2022-11-27   | Croatia              | Canada               |   4-1 | home_win |     Yes | 2-    1 |    No |
| 2022-11-27   | Japan                | Costa Rica           |   0-1 | home_win |      No | 2-    1 |    No |
| 2022-11-27   | Spain                | Germany              |   1-1 | home_win |      No | 2-    1 |    No |
| 2022-11-28   | Brazil               | Switzerland          |   1-0 | home_win |     Yes | 2-    1 |    No |
| 2022-11-28   | Cameroon             | Serbia               |   3-3 | away_win |      No | 1-    2 |    No |
| 2022-11-28   | Portugal             | Uruguay              |   2-0 | home_win |     Yes | 1-    1 |    No |
| 2022-11-28   | South Korea          | Ghana                |   2-3 | home_win |      No | 2-    1 |    No |
| 2022-11-29   | Ecuador              | Senegal              |   1-2 | home_win |      No | 1-    1 |    No |
| 2022-11-29   | Iran                 | United States        |   0-1 | away_win |     Yes | 1-    1 |    No |
| 2022-11-29   | Qatar                | Netherlands          |   0-2 | away_win |     Yes | 1-    2 |    No |
| 2022-11-29   | Wales                | England              |   0-3 | away_win |     Yes | 1-    2 |    No |
| 2022-11-30   | Australia            | Denmark              |   1-0 | away_win |      No | 1-    1 |    No |
| 2022-11-30   | Poland               | Argentina            |   0-2 | away_win |     Yes | 1-    2 |    No |
| 2022-11-30   | Saudi Arabia         | Mexico               |   1-2 | away_win |     Yes | 1-    1 |    No |
| 2022-11-30   | Tunisia              | France               |   1-0 | away_win |      No | 1-    2 |    No |
| 2022-12-01   | Canada               | Morocco              |   1-2 | away_win |     Yes | 1-    1 |    No |
| 2022-12-01   | Costa Rica           | Germany              |   2-4 | away_win |     Yes | 1-    1 |    No |
| 2022-12-01   | Croatia              | Belgium              |   0-0 | home_win |      No | 1-    1 |    No |
| 2022-12-01   | Japan                | Spain                |   2-1 | away_win |      No | 1-    2 |    No |
| 2022-12-02   | Cameroon             | Brazil               |   1-0 | away_win |      No | 1-    3 |    No |
| 2022-12-02   | Ghana                | Uruguay              |   0-2 | away_win |     Yes | 1-    2 |    No |
| 2022-12-02   | Serbia               | Switzerland          |   2-3 | away_win |     Yes | 1-    1 |    No |
| 2022-12-02   | South Korea          | Portugal             |   2-1 | away_win |      No | 1-    2 |    No |
| 2022-12-03   | Argentina            | Australia            |   2-1 | home_win |     Yes | 2-    1 |   Yes |
| 2022-12-03   | Netherlands          | United States        |   3-1 | home_win |     Yes | 2-    1 |    No |
| 2022-12-04   | England              | Senegal              |   3-0 | home_win |     Yes | 2-    1 |    No |
| 2022-12-04   | France               | Poland               |   3-1 | home_win |     Yes | 2-    1 |    No |
| 2022-12-05   | Brazil               | South Korea          |   4-1 | home_win |     Yes | 2-    1 |    No |
| 2022-12-05   | Japan                | Croatia              |   1-1 | away_win |      No | 1-    1 |   Yes |
| 2022-12-06   | Morocco              | Spain                |   0-0 | away_win |      No | 1-    1 |    No |
| 2022-12-06   | Portugal             | Switzerland          |   6-1 | home_win |     Yes | 2-    1 |    No |
| 2022-12-09   | Croatia              | Brazil               |   1-1 | away_win |      No | 1-    2 |    No |
| 2022-12-09   | Netherlands          | Argentina            |   2-2 | away_win |      No | 1-    1 |    No |
| 2022-12-10   | England              | France               |   1-2 | home_win |      No | 1-    1 |    No |
| 2022-12-10   | Morocco              | Portugal             |   1-0 | away_win |      No | 1-    1 |    No |
| 2022-12-13   | Argentina            | Croatia              |   3-0 | home_win |     Yes | 2-    1 |    No |
| 2022-12-14   | France               | Morocco              |   2-0 | home_win |     Yes | 1-    1 |    No |
| 2022-12-17   | Croatia              | Morocco              |   2-1 | home_win |     Yes | 1-    1 |    No |
| 2022-12-18   | Argentina            | France               |   3-3 | home_win |      No | 2-    1 |    No |

## UEFA Euro 2024 (51 matches)

| Date         | Home                 | Away                 | Score | XGB Pick | Correct | Poisson | Exact |
|--------------|----------------------|----------------------|-------|----------|---------|---------|-------|
| 2024-06-14   | Germany              | Scotland             |   5-1 | home_win |     Yes | 2-    1 |    No |
| 2024-06-15   | Hungary              | Switzerland          |   1-3 | home_win |      No | 1-    1 |    No |
| 2024-06-15   | Italy                | Albania              |   2-1 | home_win |     Yes | 2-    1 |   Yes |
| 2024-06-15   | Spain                | Croatia              |   3-0 | home_win |     Yes | 2-    1 |    No |
| 2024-06-16   | Poland               | Netherlands          |   1-2 | away_win |     Yes | 1-    2 |   Yes |
| 2024-06-16   | Serbia               | England              |   0-1 | away_win |     Yes | 1-    1 |    No |
| 2024-06-16   | Slovenia             | Denmark              |   1-1 | home_win |      No | 1-    1 |   Yes |
| 2024-06-17   | Austria              | France               |   0-1 | away_win |     Yes | 1-    2 |    No |
| 2024-06-17   | Belgium              | Slovakia             |   0-1 | home_win |      No | 2-    1 |    No |
| 2024-06-17   | Romania              | Ukraine              |   3-0 | away_win |      No | 1-    1 |    No |
| 2024-06-18   | Portugal             | Czech Republic       |   2-1 | home_win |     Yes | 2-    1 |   Yes |
| 2024-06-18   | Turkey               | Georgia              |   3-1 | home_win |     Yes | 2-    1 |    No |
| 2024-06-19   | Croatia              | Albania              |   2-2 | home_win |      No | 2-    1 |    No |
| 2024-06-19   | Germany              | Hungary              |   2-0 | home_win |     Yes | 2-    1 |    No |
| 2024-06-19   | Scotland             | Switzerland          |   1-1 | away_win |      No | 1-    2 |    No |
| 2024-06-20   | Denmark              | England              |   1-1 | away_win |      No | 1-    1 |   Yes |
| 2024-06-20   | Slovenia             | Serbia               |   1-1 | home_win |      No | 1-    1 |   Yes |
| 2024-06-20   | Spain                | Italy                |   1-0 | home_win |     Yes | 2-    1 |    No |
| 2024-06-21   | Netherlands          | France               |   0-0 | away_win |      No | 1-    1 |    No |
| 2024-06-21   | Poland               | Austria              |   1-3 | home_win |      No | 1-    1 |    No |
| 2024-06-21   | Slovakia             | Ukraine              |   1-2 | home_win |      No | 1-    1 |    No |
| 2024-06-22   | Belgium              | Romania              |   2-0 | home_win |     Yes | 2-    1 |    No |
| 2024-06-22   | Georgia              | Czech Republic       |   1-1 | away_win |      No | 1-    2 |    No |
| 2024-06-22   | Turkey               | Portugal             |   0-3 | away_win |     Yes | 1-    2 |    No |
| 2024-06-23   | Germany              | Switzerland          |   1-1 | home_win |      No | 2-    1 |    No |
| 2024-06-23   | Scotland             | Hungary              |   0-1 | home_win |      No | 1-    1 |    No |
| 2024-06-24   | Albania              | Spain                |   0-1 | away_win |     Yes | 1-    2 |    No |
| 2024-06-24   | Croatia              | Italy                |   1-1 | home_win |      No | 1-    1 |   Yes |
| 2024-06-25   | Denmark              | Serbia               |   0-0 | home_win |      No | 2-    1 |    No |
| 2024-06-25   | England              | Slovenia             |   0-0 | home_win |      No | 2-    1 |    No |
| 2024-06-25   | France               | Poland               |   1-1 | home_win |      No | 2-    1 |    No |
| 2024-06-25   | Netherlands          | Austria              |   2-3 | home_win |      No | 2-    1 |    No |
| 2024-06-26   | Czech Republic       | Turkey               |   1-2 | home_win |      No | 2-    1 |    No |
| 2024-06-26   | Georgia              | Portugal             |   2-0 | away_win |      No | 1-    2 |    No |
| 2024-06-26   | Slovakia             | Romania              |   1-1 | home_win |      No | 1-    1 |   Yes |
| 2024-06-26   | Ukraine              | Belgium              |   0-0 | away_win |      No | 1-    1 |    No |
| 2024-06-29   | Germany              | Denmark              |   2-0 | home_win |     Yes | 2-    1 |    No |
| 2024-06-29   | Switzerland          | Italy                |   2-0 | away_win |      No | 1-    1 |    No |
| 2024-06-30   | England              | Slovakia             |   2-1 | home_win |     Yes | 2-    1 |   Yes |
| 2024-06-30   | Spain                | Georgia              |   4-1 | home_win |     Yes | 3-    1 |    No |
| 2024-07-01   | France               | Belgium              |   1-0 | home_win |     Yes | 1-    1 |    No |
| 2024-07-01   | Portugal             | Slovenia             |   0-0 | home_win |      No | 2-    1 |    No |
| 2024-07-02   | Austria              | Turkey               |   1-2 | home_win |      No | 2-    1 |    No |
| 2024-07-02   | Romania              | Netherlands          |   0-3 | away_win |     Yes | 1-    2 |    No |
| 2024-07-05   | Germany              | Spain                |   1-2 | home_win |      No | 1-    1 |    No |
| 2024-07-05   | Portugal             | France               |   0-0 | away_win |      No | 1-    1 |    No |
| 2024-07-06   | England              | Switzerland          |   1-1 | home_win |      No | 1-    1 |   Yes |
| 2024-07-06   | Netherlands          | Turkey               |   2-1 | home_win |     Yes | 2-    1 |   Yes |
| 2024-07-09   | Spain                | France               |   2-1 | home_win |     Yes | 1-    1 |    No |
| 2024-07-10   | Netherlands          | England              |   1-2 | home_win |      No | 1-    1 |    No |
| 2024-07-14   | Spain                | England              |   2-1 | home_win |     Yes | 2-    1 |   Yes |

## Summary Statistics

- **Matches evaluated**: 115
- **Classifier accuracy**: 47.0%
- **Classifier log-loss**: 1.0431
- **Classifier Brier score**: 0.6222
- **Exact scoreline matches (Poisson)**: 15/115 (13.0%)

### Top 3 Most Confident Correct Predictions

| Match                                         | Actual | XGB Prob |     Pick |
|-----------------------------------------------|-------|----------|----------|
| Spain vs Georgia                              | 4-  1 |    82.7% | home_win |
| Portugal vs Ghana                             | 3-  2 |    75.7% | home_win |
| Poland vs Argentina                           | 0-  2 |    75.3% | away_win |

### Top 3 Most Confident Wrong Predictions (Biggest Upsets)

| Match                                         | Actual | XGB Prob |     Pick |
|-----------------------------------------------|-------|----------|----------|
| Cameroon vs Brazil                            | 1-  0 |    90.7% | away_win |
| Georgia vs Portugal                           | 2-  0 |    72.8% | away_win |
| Belgium vs Slovakia                           | 0-  1 |    72.6% | home_win |
