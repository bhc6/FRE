# 端点内部路由：独立CPU审计

完整160条件、32条视觉QA；32次native对旧基线、32次passthrough对native的全词表核验通过。96个显式mask从NPZ独立重算通过。
固定5层共320组gate/sham视觉状态哈希一致；未保存这些原始状态张量，所以这是运行时哈希的一致性核验。

causal_sham对native：全词表完全相同0/32，答案变化0/32，最大logit差0.3125。所有主效果以causal_sham为参照。

| 条件 | 全部 | 静态 | 混合 | 平局数 |
| --- | ---: | ---: | ---: | ---: |
| native | 20/32 | 8/8 | 12/24 | 1 |
| passthrough | 20/32 | 8/8 | 12/24 | 1 |
| causal_sham | 20/32 | 8/8 | 12/24 | 0 |
| first_gate | 20/32 | 8/8 | 12/24 | 0 |
| last_gate | 16/32 | 5/8 | 11/24 | 1 |
| target_gate | 24/32 | 6/8 | 18/24 | 0 |
| opposite_gate | 12/32 | 7/8 | 5/24 | 1 |

协议静态门槛：False。混合24题中target gate相对sham修复7题、退化1题；另有静态2题退化。全部32题共修复7题、退化3题。完整解决本构造混合24题：False。
相反gate读出另一端点词19/24；两个固定gate均读出各自指定源词13/24；按源内容校正的gate词级margin差为正24/24。

## 全部题目

单元格为预测词和men−enm连续margin；正号支持men，负号支持enm，0为原始平局。passthrough与native逐元素相等。

| 序列 | 问题 | 选项顺序 | Gold | native | causal_sham | first_gate | last_gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A16 | first | men_enm | men | men (+7.875) | men (+7.875) | men (+6.625) | men (+1.125) |
| A16 | first | enm_men | men | men (+4.875) | men (+5.000) | men (+3.875) | men (+2.875) |
| A16 | last | men_enm | men | men (+7.375) | men (+7.125) | men (+6.375) | men (+1.750) |
| A16 | last | enm_men | men | men (+5.250) | men (+5.125) | men (+3.875) | men (+2.875) |
| B16 | first | men_enm | enm | enm (-3.375) | enm (-3.500) | enm (-1.125) | enm (-0.375) |
| B16 | first | enm_men | enm | enm (-3.625) | enm (-3.750) | enm (-1.125) | men (+2.500) |
| B16 | last | men_enm | enm | enm (-2.375) | enm (-2.500) | enm (-0.250) | men (+0.125) |
| B16 | last | enm_men | enm | enm (-2.875) | enm (-2.875) | enm (-0.625) | men (+2.250) |
| A8B8 | first | men_enm | men | men (+4.000) | men (+4.000) | men (+6.250) | enm (-0.125) |
| A8B8 | first | enm_men | men | men (+0.125) | men (+0.250) | men (+2.875) | men (+0.875) |
| A8B8 | last | men_enm | enm | men (+4.125) | men (+4.000) | men (+6.125) | men (+0.625) |
| A8B8 | last | enm_men | enm | men (+1.125) | men (+1.000) | men (+3.750) | men (+1.500) |
| B8A8 | first | men_enm | enm | men (+0.875) | men (+0.875) | enm (-1.250) | men (+1.625) |
| B8A8 | first | enm_men | enm | enm (-1.750) | enm (-1.875) | enm (-1.125) | men (+2.625) |
| B8A8 | last | men_enm | men | men (+2.625) | men (+2.875) | enm (-0.125) | men (+2.750) |
| B8A8 | last | enm_men | men | enm (+0.000) | enm (-0.125) | enm (-0.875) | men (+3.125) |
| A2B14 | first | men_enm | men | men (+1.625) | men (+1.750) | men (+6.750) | men (+0.000) |
| A2B14 | first | enm_men | men | enm (-1.375) | enm (-1.125) | men (+3.375) | men (+1.500) |
| A2B14 | last | men_enm | enm | men (+2.750) | men (+2.875) | men (+6.500) | men (+0.500) |
| A2B14 | last | enm_men | enm | enm (-0.250) | enm (-0.250) | men (+3.750) | men (+2.000) |
| B14A2 | first | men_enm | enm | enm (-2.375) | enm (-2.500) | enm (-1.125) | men (+4.375) |
| B14A2 | first | enm_men | enm | enm (-3.250) | enm (-3.250) | enm (-1.125) | men (+4.000) |
| B14A2 | last | men_enm | men | enm (-0.750) | enm (-0.875) | enm (-0.250) | men (+4.375) |
| B14A2 | last | enm_men | men | enm (-1.875) | enm (-1.875) | enm (-0.625) | men (+4.000) |
| B2A14 | first | men_enm | enm | men (+4.375) | men (+4.125) | enm (-1.250) | men (+0.875) |
| B2A14 | first | enm_men | enm | men (+0.375) | men (+0.375) | enm (-1.250) | men (+2.250) |
| B2A14 | last | men_enm | men | men (+5.375) | men (+5.125) | enm (-0.250) | men (+2.000) |
| B2A14 | last | enm_men | men | men (+2.125) | men (+2.000) | enm (-0.750) | men (+2.875) |
| A14B2 | first | men_enm | men | men (+6.125) | men (+6.250) | men (+6.625) | men (+0.250) |
| A14B2 | first | enm_men | men | men (+3.125) | men (+3.250) | men (+3.875) | men (+0.375) |
| A14B2 | last | men_enm | enm | men (+5.875) | men (+5.750) | men (+6.375) | men (+0.750) |
| A14B2 | last | enm_men | enm | men (+3.500) | men (+3.625) | men (+3.875) | men (+0.875) |

## 首末问题与选项反转

| 条件 | 混合首末问题同词 | 混合首末均正确 | 两选项顺序同词 |
| --- | ---: | ---: | ---: |
| native | 12/12 | 0/12 | 12/16 |
| passthrough | 12/12 | 0/12 | 12/16 |
| causal_sham | 12/12 | 0/12 | 12/16 |
| first_gate | 12/12 | 0/12 | 16/16 |
| last_gate | 11/12 | 0/12 | 14/16 |
| target_gate | 6/12 | 6/12 | 16/16 |
| opposite_gate | 5/12 | 0/12 | 14/16 |

全部连续margin差、反转序列、静态位置效果及逐case掩码检查均保存在JSON；没有按效果选择层、序列或选项顺序。

## 解释边界

- One previously seen source example, two source images, repeated constructed controls; no population significance or independent confirmation.
- The main numerical reference is causal_sham; all explicit-mask differences from native are retained.
- Visual hashes are compared as runtime records at five fixed layers; original visual tensors and restoration logits were not archived for independent reconstruction.
- Input token bytes and source image files are checked directly; other processed input tensors are verified through matching recorded hashes, not regenerated on CPU.
- Text/timestamp keys remain available as indirect visual-information paths; success is total intervention sufficiency and failure does not rule out routing.
- All-layer gates change downstream text states and text Q/K/V; they do not identify one layer, FFN damage, original attention insufficiency, or long-term forgetting.
- Scoring selects between option-letter logits, not unrestricted word generation; option-order effects and exact ties are retained.
- This is a non-spectral routing control with no trained parameters; no frequency or memory-method novelty is claimed.

分析源码SHA256：15936e557120d352c60af65135a2c2be38abd7c68477cd5ce7af6e9ddccc9a39
原始记录SHA256：67041fde122195cc002d7e87678b2e75f8abc59eacbc19f37fac5f48bcc3b933
