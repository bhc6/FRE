# Small-model MCQA pilot

Status: `complete`

Descriptive diagnostics only; each question's conditions are paired observations.

| Condition | Questions | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: | ---: |
| no_memory | 36 | 13 | 0.361 | 0 |
| first_half | 36 | 20 | 0.556 | 6 |
| last_half | 36 | 21 | 0.583 | 4 |
| uniform_full | 36 | 19 | 0.528 | 5 |

## Model and outputs

Model: `Qwen/Qwen3-VL-2B-Instruct`; revision: `89644892e4d85e24eaac8bacfd4f463576704203`.

Complete prompts, payload hashes and generation metadata are retained in report.json and records.jsonl.

| Query | Condition | Raw answer | Parse | Reward | Input/output tokens | Generation/request seconds | Peak bytes |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| perception_test:video_1580:mc:1 | no_memory | "C" | valid | 1 | 95/2 | 0.5193386999999348/0.5530131999985315 | 4283119104 |
| perception_test:video_1580:mc:1 | first_half | "C" | valid | 1 | 642/2 | 0.3187424000025203/0.3728958000028797 | 4456241664 |
| perception_test:video_1580:mc:1 | last_half | "C" | valid | 1 | 647/2 | 0.2379919000013615/0.27575660000002244 | 4455686144 |
| perception_test:video_1580:mc:1 | uniform_full | "C" | valid | 1 | 645/2 | 0.24017889999959152/0.26710929999899236 | 4455401984 |
| perception_test:video_10993:mc:0 | no_memory | "C" | valid | 0 | 87/2 | 0.08460239999840269/0.08584230000269599 | 4282543616 |
| perception_test:video_10993:mc:0 | first_half | "B" | valid | 1 | 634/2 | 0.263071299999865/0.2985444000005373 | 4450364928 |
| perception_test:video_10993:mc:0 | last_half | "B" | valid | 1 | 639/2 | 0.24053330000242568/0.26763329999812413 | 4451402752 |
| perception_test:video_10993:mc:0 | uniform_full | "B" | valid | 1 | 637/2 | 0.23498370000015711/0.26246639999953914 | 4450987008 |
| perception_test:video_10993:mc:1 | no_memory | "C" | valid | 0 | 76/2 | 0.07074570000258973/0.07181449999916367 | 4280329728 |
| perception_test:video_10993:mc:1 | first_half | "B. a tea" | invalid_format | 0 | 623/5 | 0.3491165999985242/0.37558530000023893 | 4448159232 |
| perception_test:video_10993:mc:1 | last_half | "B" | valid | 1 | 628/2 | 0.240704600000754/0.2781898000030196 | 4448838144 |
| perception_test:video_10993:mc:1 | uniform_full | "B" | valid | 1 | 626/2 | 0.23818649999884656/0.2649387999990722 | 4448521216 |
| perception_test:video_10993:mc:2 | no_memory | "B" | valid | 0 | 92/2 | 0.07343929999842658/0.0746984999968845 | 4283555840 |
| perception_test:video_10993:mc:2 | first_half | "B" | valid | 0 | 639/2 | 0.25484750000032363/0.2822861000022385 | 4451402752 |
| perception_test:video_10993:mc:2 | last_half | "B" | valid | 0 | 644/2 | 0.24440410000170232/0.2729144999975688 | 4455259648 |
| perception_test:video_10993:mc:2 | uniform_full | "B" | valid | 0 | 642/2 | 0.23890559999927063/0.26695870000185096 | 4454976000 |
| perception_test:video_10993:mc:3 | no_memory | "C" | valid | 1 | 89/2 | 0.08138729999700445/0.08257760000196868 | 4282948096 |
| perception_test:video_10993:mc:3 | first_half | "C" | valid | 1 | 636/2 | 0.23738700000103563/0.26972029999888036 | 4450778112 |
| perception_test:video_10993:mc:3 | last_half | "C" | valid | 1 | 641/2 | 0.24930859999949462/0.27875979999953415 | 4454834688 |
| perception_test:video_10993:mc:3 | uniform_full | "C" | valid | 1 | 639/2 | 0.24124969999684254/0.28846169999815174 | 4451402752 |
| perception_test:video_1580:mc:0 | no_memory | "B" | valid | 0 | 82/2 | 0.07155030000285478/0.07280860000173561 | 4281533952 |
| perception_test:video_1580:mc:0 | first_half | "C" | valid | 0 | 629/2 | 0.23967320000156178/0.2689128999991226 | 4449041408 |
| perception_test:video_1580:mc:0 | last_half | "B" | valid | 0 | 634/2 | 0.23821479999969597/0.26826010000149836 | 4450364928 |
| perception_test:video_1580:mc:0 | uniform_full | "B" | valid | 0 | 632/2 | 0.24792619999789167/0.2785788000001048 | 4449951232 |
| perception_test:video_1580:mc:2 | no_memory | "C" | valid | 0 | 80/2 | 0.07623070000045118/0.07739919999949052 | 4281131008 |
| perception_test:video_1580:mc:2 | first_half | "A" | valid | 0 | 627/2 | 0.24058729999887873/0.2694463999978325 | 4448643072 |
| perception_test:video_1580:mc:2 | last_half | "B" | valid | 1 | 632/2 | 0.24080659999890486/0.2684155000024475 | 4449951232 |
| perception_test:video_1580:mc:2 | uniform_full | "A" | valid | 0 | 630/2 | 0.2386847999987367/0.27024170000004233 | 4449243136 |
| perception_test:video_1580:mc:3 | no_memory | "B" | valid | 1 | 79/2 | 0.07466430000204127/0.07591229999889038 | 4280931840 |
| perception_test:video_1580:mc:3 | first_half | "B" | valid | 1 | 626/2 | 0.2408636000000115/0.27054080000016256 | 4448521216 |
| perception_test:video_1580:mc:3 | last_half | "C" | valid | 0 | 631/2 | 0.23693670000284328/0.26517149999926914 | 4449746944 |
| perception_test:video_1580:mc:3 | uniform_full | "B" | valid | 1 | 629/2 | 0.2397934999971767/0.2684745999977167 | 4449041408 |
| perception_test:video_1580:mc:4 | no_memory | "A" | valid | 1 | 76/2 | 0.08557669999936479/0.08687529999951948 | 4280329728 |
| perception_test:video_1580:mc:4 | first_half | "A. a tea" | invalid_format | 0 | 623/5 | 0.343831099999079/0.37900940000326955 | 4448159232 |
| perception_test:video_1580:mc:4 | last_half | "A" | valid | 1 | 628/2 | 0.2428682999998273/0.27155670000138343 | 4448838144 |
| perception_test:video_1580:mc:4 | uniform_full | "A. a tea" | invalid_format | 0 | 626/5 | 0.3369812999990245/0.3704485000016575 | 4448521216 |
| perception_test:video_1580:mc:5 | no_memory | "C" | valid | 0 | 89/2 | 0.07776509999894188/0.0789561000019603 | 4282948096 |
| perception_test:video_1580:mc:5 | first_half | "C" | valid | 0 | 636/2 | 0.24188809999759542/0.2690043999973568 | 4450778112 |
| perception_test:video_1580:mc:5 | last_half | "A" | valid | 1 | 641/2 | 0.25791670000035083/0.29230939999979455 | 4454834688 |
| perception_test:video_1580:mc:5 | uniform_full | "C" | valid | 0 | 639/2 | 0.246174300002167/0.2747741999992286 | 4451402752 |
| perception_test:video_2564:mc:0 | no_memory | "B" | valid | 0 | 79/2 | 0.08284079999793903/0.08420399999886286 | 4280931840 |
| perception_test:video_2564:mc:0 | first_half | "C. static or shaking" | invalid_format | 0 | 626/6 | 0.3889314000007289/0.4179963000024145 | 4448521216 |
| perception_test:video_2564:mc:0 | last_half | "C. static or shaking" | invalid_format | 0 | 631/6 | 0.37546370000200113/0.41173130000242963 | 4449746944 |
| perception_test:video_2564:mc:0 | uniform_full | "C. static or shaking" | invalid_format | 0 | 628/6 | 0.405108900002233/0.4371934000009787 | 4448838144 |
| perception_test:video_2564:mc:1 | no_memory | "C" | valid | 0 | 84/2 | 0.0910026999990805/0.09239529999831575 | 4281936384 |
| perception_test:video_2564:mc:1 | first_half | "A" | valid | 1 | 631/2 | 0.24654280000322615/0.2765180999995209 | 4449746944 |
| perception_test:video_2564:mc:1 | last_half | "A" | valid | 1 | 636/2 | 0.24941729999773088/0.27903080000032787 | 4450778112 |
| perception_test:video_2564:mc:1 | uniform_full | "A" | valid | 1 | 633/2 | 0.24559669999871403/0.27424429999882705 | 4450159104 |
| perception_test:video_2564:mc:2 | no_memory | "A" | valid | 0 | 75/2 | 0.082631000001129/0.0841963000020769 | 4280130560 |
| perception_test:video_2564:mc:2 | first_half | "C" | valid | 1 | 622/2 | 0.23123610000038752/0.2603725000008126 | 4448037888 |
| perception_test:video_2564:mc:2 | last_half | "C" | valid | 1 | 627/2 | 0.24084249999941676/0.26751789999980247 | 4448643072 |
| perception_test:video_2564:mc:2 | uniform_full | "C" | valid | 1 | 624/2 | 0.23621279999861144/0.26469250000081956 | 4448278528 |
| perception_test:video_2564:mc:3 | no_memory | "A" | valid | 1 | 95/2 | 0.07613980000314768/0.07755809999798657 | 4284167680 |
| perception_test:video_2564:mc:3 | first_half | "A" | valid | 1 | 642/2 | 0.24463959999775398/0.27288620000035735 | 4454976000 |
| perception_test:video_2564:mc:3 | last_half | "C" | valid | 0 | 647/2 | 0.24981800000023213/0.2865405999982613 | 4455686144 |
| perception_test:video_2564:mc:3 | uniform_full | "C" | valid | 0 | 644/2 | 0.2533728999987943/0.2805393999988155 | 4455259648 |
| perception_test:video_2564:mc:4 | no_memory | "B" | valid | 0 | 89/2 | 0.07257400000162306/0.07399309999891557 | 4282948096 |
| perception_test:video_2564:mc:4 | first_half | "A" | valid | 1 | 636/2 | 0.23912040000141133/0.2746469000012439 | 4450778112 |
| perception_test:video_2564:mc:4 | last_half | "B" | valid | 0 | 641/2 | 0.25012860000060755/0.2779882000031648 | 4454834688 |
| perception_test:video_2564:mc:4 | uniform_full | "A" | valid | 1 | 638/2 | 0.23889259999850765/0.27567289999933564 | 4451193856 |
| perception_test:video_3501:mc:0 | no_memory | "B" | valid | 0 | 79/2 | 0.07949919999737176/0.08072850000098697 | 4280931840 |
| perception_test:video_3501:mc:0 | first_half | "C. static or shaking" | invalid_format | 0 | 627/6 | 0.3701645000001008/0.39729480000096373 | 4448643072 |
| perception_test:video_3501:mc:0 | last_half | "C. static or shaking" | invalid_format | 0 | 631/6 | 0.3711914000014076/0.4075462000000698 | 4449746944 |
| perception_test:video_3501:mc:0 | uniform_full | "C. static or shaking" | invalid_format | 0 | 629/6 | 0.37844029999905615/0.40600260000064736 | 4449041408 |
| perception_test:video_3501:mc:1 | no_memory | "C" | valid | 0 | 87/2 | 0.09161740000126883/0.09324009999909322 | 4282543616 |
| perception_test:video_3501:mc:1 | first_half | "B" | valid | 1 | 635/2 | 0.24155989999781013/0.27283560000068974 | 4450572288 |
| perception_test:video_3501:mc:1 | last_half | "B" | valid | 1 | 639/2 | 0.24766059999819845/0.27423579999958747 | 4451402752 |
| perception_test:video_3501:mc:1 | uniform_full | "B" | valid | 1 | 637/2 | 0.2406900999994832/0.26815720000013243 | 4450987008 |
| perception_test:video_3501:mc:2 | no_memory | "A" | valid | 1 | 75/2 | 0.07132890000139014/0.07258119999823975 | 4280130560 |
| perception_test:video_3501:mc:2 | first_half | "A. a tea" | invalid_format | 0 | 623/5 | 0.3310978999979852/0.3577261999998882 | 4448159232 |
| perception_test:video_3501:mc:2 | last_half | "A. a tea" | invalid_format | 0 | 627/5 | 0.3495562999996764/0.3818055000010645 | 4448643072 |
| perception_test:video_3501:mc:2 | uniform_full | "A. a tea" | invalid_format | 0 | 625/5 | 0.3496392000015476/0.3803357999968284 | 4448400896 |
| perception_test:video_3501:mc:3 | no_memory | "C" | valid | 1 | 99/2 | 0.07787599999937811/0.07932359999904293 | 4284984320 |
| perception_test:video_3501:mc:3 | first_half | "C" | valid | 1 | 647/2 | 0.2509934000008798/0.27800070000012056 | 4455686144 |
| perception_test:video_3501:mc:3 | last_half | "C" | valid | 1 | 651/2 | 0.2602356999996118/0.29496669999934966 | 4459584000 |
| perception_test:video_3501:mc:3 | uniform_full | "A" | valid | 0 | 649/2 | 0.2505386999982875/0.28541609999956563 | 4459445248 |
| perception_test:video_3501:mc:4 | no_memory | "C" | valid | 1 | 89/2 | 0.07374569999956293/0.07491690000097151 | 4282948096 |
| perception_test:video_3501:mc:4 | first_half | "C" | valid | 1 | 637/2 | 0.24154119999730028/0.2711323000003176 | 4450987008 |
| perception_test:video_3501:mc:4 | last_half | "C" | valid | 1 | 641/2 | 0.24784590000126627/0.2754437000003236 | 4454834688 |
| perception_test:video_3501:mc:4 | uniform_full | "C" | valid | 1 | 639/2 | 0.24505560000034166/0.2793342999975721 | 4451402752 |
| perception_test:video_6860:mc:0 | no_memory | "C" | valid | 0 | 75/2 | 0.07327380000060657/0.07442729999820585 | 4280130560 |
| perception_test:video_6860:mc:0 | first_half | "B" | valid | 1 | 623/2 | 0.23719019999771263/0.26824760000090464 | 4448159232 |
| perception_test:video_6860:mc:0 | last_half | "B" | valid | 1 | 627/2 | 0.24151260000144248/0.27407769999990705 | 4448643072 |
| perception_test:video_6860:mc:0 | uniform_full | "B" | valid | 1 | 625/2 | 0.23870909999823198/0.2643900000002759 | 4448400896 |
| perception_test:video_6860:mc:1 | no_memory | "B" | valid | 1 | 95/2 | 0.07819869999730145/0.07940020000023651 | 4284167680 |
| perception_test:video_6860:mc:1 | first_half | "B" | valid | 1 | 643/2 | 0.24925269999948796/0.2760631999990437 | 4455118336 |
| perception_test:video_6860:mc:1 | last_half | "B" | valid | 1 | 647/2 | 0.2559809000013047/0.2819918000022881 | 4455686144 |
| perception_test:video_6860:mc:1 | uniform_full | "B" | valid | 1 | 645/2 | 0.24738849999994272/0.2743149000016274 | 4455401984 |
| perception_test:video_8241:mc:0 | no_memory | "B" | valid | 0 | 101/2 | 0.07085759999972652/0.07254079999984242 | 4285393920 |
| perception_test:video_8241:mc:0 | first_half | "B" | valid | 0 | 647/2 | 0.25722989999849233/0.2845697000011569 | 4455686144 |
| perception_test:video_8241:mc:0 | last_half | "B" | valid | 0 | 653/2 | 0.25328040000022156/0.2813425000022107 | 4459779072 |
| perception_test:video_8241:mc:0 | uniform_full | "B" | valid | 0 | 650/2 | 0.24716339999940828/0.273539100002381 | 4459514368 |
| perception_test:video_8241:mc:1 | no_memory | "C" | valid | 1 | 81/2 | 0.08499970000048052/0.0861652999992657 | 4281333760 |
| perception_test:video_8241:mc:1 | first_half | "C" | valid | 1 | 627/2 | 0.23732939999899827/0.2687483000008797 | 4448643072 |
| perception_test:video_8241:mc:1 | last_half | "C" | valid | 1 | 633/2 | 0.24051179999878514/0.2735154000001785 | 4450159104 |
| perception_test:video_8241:mc:1 | uniform_full | "C" | valid | 1 | 630/2 | 0.2379657000019506/0.26786379999975907 | 4449243136 |
| perception_test:video_8241:mc:2 | no_memory | "C" | valid | 1 | 80/2 | 0.07290819999980158/0.0742733999977645 | 4281131008 |
| perception_test:video_8241:mc:2 | first_half | "C" | valid | 1 | 626/2 | 0.24201939999693423/0.27197199999864097 | 4448521216 |
| perception_test:video_8241:mc:2 | last_half | "C" | valid | 1 | 632/2 | 0.2436121999999159/0.27182189999803086 | 4449951232 |
| perception_test:video_8241:mc:2 | uniform_full | "C" | valid | 1 | 629/2 | 0.24240399999689544/0.2695662000005541 | 4449041408 |
| perception_test:video_8241:mc:3 | no_memory | "A" | valid | 1 | 76/2 | 0.0772648999991361/0.07853189999877941 | 4280329728 |
| perception_test:video_8241:mc:3 | first_half | "A. a tea" | invalid_format | 0 | 622/5 | 0.34486919999835663/0.3724610000026587 | 4448037888 |
| perception_test:video_8241:mc:3 | last_half | "A. a tea" | invalid_format | 0 | 628/5 | 0.34642819999862695/0.3738613000023179 | 4448838144 |
| perception_test:video_8241:mc:3 | uniform_full | "A. a tea" | invalid_format | 0 | 625/5 | 0.33625340000071446/0.3664984999995795 | 4448400896 |
| perception_test:video_8241:mc:4 | no_memory | "C" | valid | 0 | 89/2 | 0.07962190000034752/0.08148960000107763 | 4282948096 |
| perception_test:video_8241:mc:4 | first_half | "C" | valid | 0 | 635/2 | 0.2447159000003012/0.2715811999987636 | 4450572288 |
| perception_test:video_8241:mc:4 | last_half | "C" | valid | 0 | 641/2 | 0.25799819999883766/0.2923158000012336 | 4454834688 |
| perception_test:video_8241:mc:4 | uniform_full | "C" | valid | 0 | 638/2 | 0.24008669999966514/0.266598400001385 | 4451193856 |
| perception_test:video_874:mc:0 | no_memory | "C" | valid | 0 | 87/2 | 0.07479499999681138/0.07597669999813661 | 4282543616 |
| perception_test:video_874:mc:0 | first_half | "B" | valid | 1 | 635/2 | 0.2386091000007582/0.2675192999995488 | 4450572288 |
| perception_test:video_874:mc:0 | last_half | "B" | valid | 1 | 639/2 | 0.24880389999816543/0.2752637000012328 | 4451402752 |
| perception_test:video_874:mc:0 | uniform_full | "B" | valid | 1 | 637/2 | 0.23991879999812227/0.26732260000062524 | 4450987008 |
| perception_test:video_874:mc:1 | no_memory | "A" | valid | 0 | 75/2 | 0.0726396999998542/0.07395480000195676 | 4280130560 |
| perception_test:video_874:mc:1 | first_half | "B" | valid | 1 | 623/2 | 0.23740929999985383/0.2629720000004454 | 4448159232 |
| perception_test:video_874:mc:1 | last_half | "B" | valid | 1 | 627/2 | 0.24434490000203368/0.2726203999991412 | 4448643072 |
| perception_test:video_874:mc:1 | uniform_full | "B" | valid | 1 | 625/2 | 0.24479339999743388/0.2721961999995983 | 4448400896 |
| perception_test:video_874:mc:2 | no_memory | "A" | valid | 0 | 99/2 | 0.07486269999935757/0.07627430000138702 | 4284984320 |
| perception_test:video_874:mc:2 | first_half | "A" | valid | 0 | 647/2 | 0.24587740000060876/0.2762494999988121 | 4455686144 |
| perception_test:video_874:mc:2 | last_half | "A" | valid | 0 | 651/2 | 0.25051840000014636/0.2767658999982814 | 4459584000 |
| perception_test:video_874:mc:2 | uniform_full | "A" | valid | 0 | 649/2 | 0.25529440000173054/0.28262200000244775 | 4459445248 |
| perception_test:video_874:mc:3 | no_memory | "A" | valid | 0 | 89/2 | 0.08132599999953527/0.08283210000081453 | 4282948096 |
| perception_test:video_874:mc:3 | first_half | "A" | valid | 0 | 637/2 | 0.23740840000027674/0.2675947999996424 | 4450987008 |
| perception_test:video_874:mc:3 | last_half | "A" | valid | 0 | 641/2 | 0.2510454999974172/0.2782296999976097 | 4454834688 |
| perception_test:video_874:mc:3 | uniform_full | "A" | valid | 0 | 639/2 | 0.2455789999985427/0.2822889999988547 | 4451402752 |
| perception_test:video_9253:mc:0 | no_memory | "C" | valid | 0 | 98/2 | 0.07953170000109822/0.0808802999999898 | 4284779520 |
| perception_test:video_9253:mc:0 | first_half | "C" | valid | 0 | 644/2 | 0.24629410000125063/0.2778900000012072 | 4455259648 |
| perception_test:video_9253:mc:0 | last_half | "C" | valid | 0 | 650/2 | 0.2539572000023327/0.28118090000134544 | 4459514368 |
| perception_test:video_9253:mc:0 | uniform_full | "C" | valid | 0 | 647/2 | 0.2538514000007126/0.2806268000022101 | 4455686144 |
| perception_test:video_9253:mc:1 | no_memory | "C" | valid | 0 | 80/2 | 0.07726519999778247/0.07861330000014277 | 4281131008 |
| perception_test:video_9253:mc:1 | first_half | "B" | valid | 1 | 626/2 | 0.23719250000067404/0.2736971000013 | 4448521216 |
| perception_test:video_9253:mc:1 | last_half | "B" | valid | 1 | 632/2 | 0.2419339999978547/0.2692089999982272 | 4449951232 |
| perception_test:video_9253:mc:1 | uniform_full | "B" | valid | 1 | 629/2 | 0.2482092999998713/0.27926100000331644 | 4449041408 |
| perception_test:video_9253:mc:2 | no_memory | "B" | valid | 1 | 80/2 | 0.11120350000055623/0.1128564999999071 | 4281131008 |
| perception_test:video_9253:mc:2 | first_half | "B" | valid | 1 | 626/2 | 0.2413096999989648/0.2977759999994305 | 4448521216 |
| perception_test:video_9253:mc:2 | last_half | "B" | valid | 1 | 632/2 | 0.2389882999996189/0.2704578000011679 | 4449951232 |
| perception_test:video_9253:mc:2 | uniform_full | "B" | valid | 1 | 629/2 | 0.23984960000234423/0.2685058000024583 | 4449041408 |
| perception_test:video_9253:mc:3 | no_memory | "C" | valid | 0 | 75/2 | 0.07849410000198986/0.07972399999925983 | 4280130560 |
| perception_test:video_9253:mc:3 | first_half | "B" | valid | 1 | 621/2 | 0.24904779999997118/0.2761505999988003 | 4447918080 |
| perception_test:video_9253:mc:3 | last_half | "B" | valid | 1 | 627/2 | 0.23894570000265958/0.26600909999979194 | 4448643072 |
| perception_test:video_9253:mc:3 | uniform_full | "B" | valid | 1 | 624/2 | 0.23746290000053705/0.26558360000126413 | 4448278528 |
| perception_test:video_9253:mc:4 | no_memory | "A" | valid | 0 | 89/2 | 0.07837799999833805/0.07977840000239667 | 4282948096 |
| perception_test:video_9253:mc:4 | first_half | "A" | valid | 0 | 635/2 | 0.2569952999983798/0.2843119999997725 | 4450572288 |
| perception_test:video_9253:mc:4 | last_half | "A" | valid | 0 | 641/2 | 0.25768150000294554/0.28461909999896307 | 4454834688 |
| perception_test:video_9253:mc:4 | uniform_full | "A" | valid | 0 | 638/2 | 0.23850229999879957/0.26692779999939376 | 4451193856 |

## Limitations

- Descriptive small-model mechanism pilot; repeated conditions are not independent samples.
- Equal frame count does not prove equal visual token count or information coverage.
- No-memory has zero visual budget; differences are coverage diagnostics, not fair method comparisons.
- Official QA answers score real generations; they are never injected as memory-utility labels.
- Utility is measured against empty memory for one frozen readout, not lifelong storage quality.
- No probe is trained and no hidden-state path is fabricated by this command.
- Payload descriptors bind frame bytes but do not measure total retained-memory storage cost.
- Uniform-full covers the supplied frame manifest; original full-video coverage requires its sampling provenance.
