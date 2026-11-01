# 只读内部探针：全部36题标签审计

固定16帧无干预baseline为25/36，原时序规则下6/8；所有标签逐题对照官方sample.json。没有已识别的遗忘案例。

| query_id | 官方问题 | 官方答案 | baseline答案 | margin | 原时序子集 | 旧full64正确 |
| --- | --- | --- | --- | ---: | --- | --- |
| perception_test:video_1580:mc:0 | How many letters changed position in the word after shuffling? | 3 | 2 | -0.500 | True | 0 |
| perception_test:video_1580:mc:2 | What is the order of the letters at the end? | enm | men | -3.125 | True | 0 |
| perception_test:video_1580:mc:5 | What can you say about the temperature of the water being poured? | The water seems cold. | The water seems hot. | -4.875 | False | 0 |
| perception_test:video_2564:mc:3 | What action not related to making tea did the person do? | cutting a ham slice with knife and fork | moving a pink container around | -1.375 | False | 0 |
| perception_test:video_3501:mc:3 | What action not related to making tea did the person do? | hitting the kettle with a steel spoon | putting a salt shaker inside the salad box | -0.500 | False | 1 |
| perception_test:video_8241:mc:0 | Which of the following statements describes the scene better? | There is an egg in the cup of sugar. | There is an egg next to the cup of sugar. | -2.250 | False | 0 |
| perception_test:video_8241:mc:4 | What can you say about the temperature of the water being poured? | The water seems cold. | The water seems hot. | -4.125 | False | 0 |
| perception_test:video_874:mc:2 | What action not related to making tea did the person do? | hitting the fork against the jar | hitting a steel fork against the oil bottle | -0.250 | False | 0 |
| perception_test:video_874:mc:3 | What can you say about the temperature of the water being poured? | The water seems cold. | The water seems hot. | -6.000 | False | 0 |
| perception_test:video_9253:mc:0 | Which of the following statements describes the scene better? | There is an egg in the sugar container. | There is an egg in the black cup. | -3.750 | False | 0 |
| perception_test:video_9253:mc:4 | What can you say about the temperature of the water being poured? | The water seems cold. | The water seems hot. | -2.250 | False | 0 |

旧no-video→新固定16帧的跨run四格：错→对14、错→错9、对→错2、对→对11。仅为描述，不能把所有差异归因为视觉证据保存。
固定16与旧full64总分相同，但2564:mc:4和3501:mc:3的正确性相反，不是同一输入条件。

所有36题与正反例边界见report.json；独立确认40视频未访问。
