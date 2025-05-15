# digital-intelligence

Run the script with:
```bash
python main.py \
  --data_dir ./rps_data \
  --model simple --batchnorm --epochs 25 --optimizer adam --lr 2e-4 --scheduler
```

## Lab4
### ResNet model
```
Epocha 1/100: Mokymo paklaida 0.8506; Tikslumas 0.6102 | Validacijos paklaida 0.7040; Tikslumas 0.6835
Epocha 2/100: Mokymo paklaida 0.6407; Tikslumas 0.7603 | Validacijos paklaida 0.3621; Tikslumas 0.8807
Epocha 3/100: Mokymo paklaida 0.5197; Tikslumas 0.8339 | Validacijos paklaida 0.3271; Tikslumas 0.9037
Epocha 4/100: Mokymo paklaida 0.4286; Tikslumas 0.8619 | Validacijos paklaida 0.2938; Tikslumas 0.8716
Epocha 5/100: Mokymo paklaida 0.4035; Tikslumas 0.8590 | Validacijos paklaida 0.2137; Tikslumas 0.9587
Epocha 6/100: Mokymo paklaida 0.3995; Tikslumas 0.8613 | Validacijos paklaida 0.1846; Tikslumas 0.9771
Epocha 7/100: Mokymo paklaida 0.3774; Tikslumas 0.8727 | Validacijos paklaida 0.3343; Tikslumas 0.8945
Epocha 8/100: Mokymo paklaida 0.3629; Tikslumas 0.8864 | Validacijos paklaida 0.1174; Tikslumas 0.9679
Epocha 9/100: Mokymo paklaida 0.3375; Tikslumas 0.8887 | Validacijos paklaida 0.1520; Tikslumas 0.9450
Epocha 10/100: Mokymo paklaida 0.3274; Tikslumas 0.8876 | Validacijos paklaida 0.4121; Tikslumas 0.7844
Epocha 11/100: Mokymo paklaida 0.3447; Tikslumas 0.8813 | Validacijos paklaida 0.1242; Tikslumas 0.9725
Epocha 12/100: Mokymo paklaida 0.3026; Tikslumas 0.9007 | Validacijos paklaida 0.0663; Tikslumas 0.9908
Epocha 13/100: Mokymo paklaida 0.3271; Tikslumas 0.8847 | Validacijos paklaida 0.5471; Tikslumas 0.7798
Epocha 14/100: Mokymo paklaida 0.3120; Tikslumas 0.8921 | Validacijos paklaida 0.1290; Tikslumas 0.9725
Epocha 15/100: Mokymo paklaida 0.3067; Tikslumas 0.8973 | Validacijos paklaida 0.2268; Tikslumas 0.9312
Epocha 16/100: Mokymo paklaida 0.3150; Tikslumas 0.8910 | Validacijos paklaida 0.1395; Tikslumas 0.9725
Epocha 17/100: Mokymo paklaida 0.2863; Tikslumas 0.9075 | Validacijos paklaida 0.0580; Tikslumas 0.9908
Epocha 18/100: Mokymo paklaida 0.2452; Tikslumas 0.9218 | Validacijos paklaida 0.0549; Tikslumas 0.9908
Epocha 19/100: Mokymo paklaida 0.2459; Tikslumas 0.9195 | Validacijos paklaida 0.0502; Tikslumas 0.9908
Epocha 20/100: Mokymo paklaida 0.2587; Tikslumas 0.9155 | Validacijos paklaida 0.0524; Tikslumas 0.9862
Epocha 21/100: Mokymo paklaida 0.2619; Tikslumas 0.9132 | Validacijos paklaida 0.0453; Tikslumas 0.9908
Epocha 22/100: Mokymo paklaida 0.2398; Tikslumas 0.9212 | Validacijos paklaida 0.0486; Tikslumas 0.9908
Epocha 23/100: Mokymo paklaida 0.2402; Tikslumas 0.9121 | Validacijos paklaida 0.0496; Tikslumas 0.9908
Epocha 24/100: Mokymo paklaida 0.2276; Tikslumas 0.9241 | Validacijos paklaida 0.0477; Tikslumas 0.9908
Epocha 25/100: Mokymo paklaida 0.2314; Tikslumas 0.9212 | Validacijos paklaida 0.0374; Tikslumas 0.9908
Epocha 26/100: Mokymo paklaida 0.2362; Tikslumas 0.9218 | Validacijos paklaida 0.0563; Tikslumas 0.9908
Epocha 27/100: Mokymo paklaida 0.2335; Tikslumas 0.9218 | Validacijos paklaida 0.0386; Tikslumas 0.9908
Epocha 28/100: Mokymo paklaida 0.2352; Tikslumas 0.9207 | Validacijos paklaida 0.0398; Tikslumas 0.9862
Epocha 29/100: Mokymo paklaida 0.2167; Tikslumas 0.9292 | Validacijos paklaida 0.0511; Tikslumas 0.9908
Epocha 30/100: Mokymo paklaida 0.2475; Tikslumas 0.9115 | Validacijos paklaida 0.0389; Tikslumas 0.9908
Epocha 31/100: Mokymo paklaida 0.2472; Tikslumas 0.9081 | Validacijos paklaida 0.0398; Tikslumas 0.9908
Epocha 32/100: Mokymo paklaida 0.2370; Tikslumas 0.9207 | Validacijos paklaida 0.0386; Tikslumas 0.9908
Epocha 33/100: Mokymo paklaida 0.2281; Tikslumas 0.9241 | Validacijos paklaida 0.0421; Tikslumas 0.9862
Epocha 34/100: Mokymo paklaida 0.2253; Tikslumas 0.9247 | Validacijos paklaida 0.0394; Tikslumas 0.9908
Baigiama anksčiau, nes nebuvo pagerinimo per 10 epochų.
Testavimo paklaida 0.0414; Tikslumas 0.9908
```

### Simple CNN model
```
Epocha 1/100: Mokymo paklaida 1.9263; Tikslumas 0.3545 | Validacijos paklaida 1.0966; Tikslumas 0.3073
Epocha 2/100: Mokymo paklaida 1.0920; Tikslumas 0.3476 | Validacijos paklaida 1.0818; Tikslumas 0.4817
Epocha 3/100: Mokymo paklaida 1.0908; Tikslumas 0.3704 | Validacijos paklaida 1.0654; Tikslumas 0.6606
Epocha 4/100: Mokymo paklaida 1.0807; Tikslumas 0.3813 | Validacijos paklaida 1.0676; Tikslumas 0.5642
Epocha 5/100: Mokymo paklaida 1.0880; Tikslumas 0.3938 | Validacijos paklaida 1.0675; Tikslumas 0.6284
Epocha 6/100: Mokymo paklaida 1.0868; Tikslumas 0.3995 | Validacijos paklaida 1.0465; Tikslumas 0.6376
Epocha 7/100: Mokymo paklaida 1.0789; Tikslumas 0.4047 | Validacijos paklaida 1.0058; Tikslumas 0.7339
Epocha 8/100: Mokymo paklaida 1.0767; Tikslumas 0.4184 | Validacijos paklaida 1.0092; Tikslumas 0.7615
Epocha 9/100: Mokymo paklaida 1.0784; Tikslumas 0.3921 | Validacijos paklaida 1.0157; Tikslumas 0.8257
Epocha 10/100: Mokymo paklaida 1.0744; Tikslumas 0.4018 | Validacijos paklaida 0.9988; Tikslumas 0.7339
Epocha 11/100: Mokymo paklaida 1.0703; Tikslumas 0.4098 | Validacijos paklaida 0.9773; Tikslumas 0.7477
Epocha 12/100: Mokymo paklaida 1.0530; Tikslumas 0.4321 | Validacijos paklaida 0.9186; Tikslumas 0.7477
Epocha 13/100: Mokymo paklaida 1.0637; Tikslumas 0.4321 | Validacijos paklaida 0.9442; Tikslumas 0.8394
Epocha 14/100: Mokymo paklaida 1.0453; Tikslumas 0.4212 | Validacijos paklaida 0.9452; Tikslumas 0.8853
Epocha 15/100: Mokymo paklaida 1.0517; Tikslumas 0.4195 | Validacijos paklaida 0.9508; Tikslumas 0.8761
Epocha 16/100: Mokymo paklaida 1.0431; Tikslumas 0.4258 | Validacijos paklaida 0.9055; Tikslumas 0.8761
Epocha 17/100: Mokymo paklaida 1.0492; Tikslumas 0.4098 | Validacijos paklaida 0.9375; Tikslumas 0.8532
Epocha 18/100: Mokymo paklaida 1.0389; Tikslumas 0.4326 | Validacijos paklaida 0.8868; Tikslumas 0.9037
Epocha 19/100: Mokymo paklaida 1.0329; Tikslumas 0.4395 | Validacijos paklaida 0.8179; Tikslumas 0.8761
Epocha 20/100: Mokymo paklaida 1.0159; Tikslumas 0.4623 | Validacijos paklaida 0.7938; Tikslumas 0.8670
Epocha 21/100: Mokymo paklaida 1.0273; Tikslumas 0.4498 | Validacijos paklaida 0.8015; Tikslumas 0.8991
Epocha 22/100: Mokymo paklaida 1.0132; Tikslumas 0.4503 | Validacijos paklaida 0.7942; Tikslumas 0.8991
Epocha 23/100: Mokymo paklaida 1.0238; Tikslumas 0.4435 | Validacijos paklaida 0.7924; Tikslumas 0.9220
Epocha 24/100: Mokymo paklaida 1.0405; Tikslumas 0.4583 | Validacijos paklaida 0.7600; Tikslumas 0.9358
Epocha 25/100: Mokymo paklaida 0.9911; Tikslumas 0.4840 | Validacijos paklaida 0.7479; Tikslumas 0.9174
Epocha 26/100: Mokymo paklaida 1.0035; Tikslumas 0.4777 | Validacijos paklaida 0.6429; Tikslumas 0.9266
Epocha 27/100: Mokymo paklaida 0.9844; Tikslumas 0.4840 | Validacijos paklaida 0.7176; Tikslumas 0.9083
Epocha 28/100: Mokymo paklaida 0.9974; Tikslumas 0.4789 | Validacijos paklaida 0.6340; Tikslumas 0.9404
Epocha 29/100: Mokymo paklaida 0.9928; Tikslumas 0.4646 | Validacijos paklaida 0.6723; Tikslumas 0.9358
Epocha 30/100: Mokymo paklaida 0.9653; Tikslumas 0.4812 | Validacijos paklaida 0.6205; Tikslumas 0.9312
Epocha 31/100: Mokymo paklaida 0.9597; Tikslumas 0.4943 | Validacijos paklaida 0.5983; Tikslumas 0.9128
Epocha 32/100: Mokymo paklaida 0.9642; Tikslumas 0.4823 | Validacijos paklaida 0.5862; Tikslumas 0.9358
Epocha 33/100: Mokymo paklaida 0.9707; Tikslumas 0.4989 | Validacijos paklaida 0.5709; Tikslumas 0.9450
Epocha 34/100: Mokymo paklaida 0.9510; Tikslumas 0.5006 | Validacijos paklaida 0.5310; Tikslumas 0.9358
Epocha 35/100: Mokymo paklaida 0.9520; Tikslumas 0.5137 | Validacijos paklaida 0.4927; Tikslumas 0.9312
Epocha 36/100: Mokymo paklaida 0.9413; Tikslumas 0.5086 | Validacijos paklaida 0.4787; Tikslumas 0.9037
Epocha 37/100: Mokymo paklaida 0.8999; Tikslumas 0.5519 | Validacijos paklaida 0.4568; Tikslumas 0.9358
Epocha 38/100: Mokymo paklaida 0.9380; Tikslumas 0.5251 | Validacijos paklaida 0.4475; Tikslumas 0.9266
Epocha 39/100: Mokymo paklaida 0.9185; Tikslumas 0.5325 | Validacijos paklaida 0.4556; Tikslumas 0.9541
Epocha 40/100: Mokymo paklaida 0.9195; Tikslumas 0.5599 | Validacijos paklaida 0.4621; Tikslumas 0.9358
Epocha 41/100: Mokymo paklaida 0.9062; Tikslumas 0.5411 | Validacijos paklaida 0.3902; Tikslumas 0.9495
Epocha 42/100: Mokymo paklaida 0.8951; Tikslumas 0.5422 | Validacijos paklaida 0.3613; Tikslumas 0.9541
Epocha 43/100: Mokymo paklaida 0.9050; Tikslumas 0.5525 | Validacijos paklaida 0.4126; Tikslumas 0.9220
Epocha 44/100: Mokymo paklaida 0.8679; Tikslumas 0.5588 | Validacijos paklaida 0.3483; Tikslumas 0.9587
Epocha 45/100: Mokymo paklaida 0.8797; Tikslumas 0.5531 | Validacijos paklaida 0.3248; Tikslumas 0.9587
Epocha 46/100: Mokymo paklaida 0.8552; Tikslumas 0.5839 | Validacijos paklaida 0.3361; Tikslumas 0.9633
Epocha 47/100: Mokymo paklaida 0.8558; Tikslumas 0.5965 | Validacijos paklaida 0.3003; Tikslumas 0.9495
Epocha 48/100: Mokymo paklaida 0.8366; Tikslumas 0.5930 | Validacijos paklaida 0.2868; Tikslumas 0.9633
Epocha 49/100: Mokymo paklaida 0.8505; Tikslumas 0.5828 | Validacijos paklaida 0.2772; Tikslumas 0.9587
Epocha 50/100: Mokymo paklaida 0.8481; Tikslumas 0.5856 | Validacijos paklaida 0.2798; Tikslumas 0.9679
Epocha 51/100: Mokymo paklaida 0.8346; Tikslumas 0.6039 | Validacijos paklaida 0.2575; Tikslumas 0.9633
Epocha 52/100: Mokymo paklaida 0.8499; Tikslumas 0.5782 | Validacijos paklaida 0.2923; Tikslumas 0.9633
Epocha 53/100: Mokymo paklaida 0.8145; Tikslumas 0.5953 | Validacijos paklaida 0.2490; Tikslumas 0.9633
Epocha 54/100: Mokymo paklaida 0.8430; Tikslumas 0.5947 | Validacijos paklaida 0.2481; Tikslumas 0.9633
Epocha 55/100: Mokymo paklaida 0.8360; Tikslumas 0.5879 | Validacijos paklaida 0.2705; Tikslumas 0.9725
Epocha 56/100: Mokymo paklaida 0.8374; Tikslumas 0.5942 | Validacijos paklaida 0.2660; Tikslumas 0.9633
Epocha 57/100: Mokymo paklaida 0.8179; Tikslumas 0.6084 | Validacijos paklaida 0.2438; Tikslumas 0.9771
Epocha 58/100: Mokymo paklaida 0.8051; Tikslumas 0.6102 | Validacijos paklaida 0.2300; Tikslumas 0.9633
Epocha 59/100: Mokymo paklaida 0.8062; Tikslumas 0.5999 | Validacijos paklaida 0.2364; Tikslumas 0.9450
Epocha 60/100: Mokymo paklaida 0.7950; Tikslumas 0.6159 | Validacijos paklaida 0.2126; Tikslumas 0.9312
Epocha 61/100: Mokymo paklaida 0.7668; Tikslumas 0.6147 | Validacijos paklaida 0.1768; Tikslumas 0.9679
Epocha 62/100: Mokymo paklaida 0.7827; Tikslumas 0.6067 | Validacijos paklaida 0.1995; Tikslumas 0.9725
Epocha 63/100: Mokymo paklaida 0.8122; Tikslumas 0.6062 | Validacijos paklaida 0.2035; Tikslumas 0.9817
Epocha 64/100: Mokymo paklaida 0.7668; Tikslumas 0.6358 | Validacijos paklaida 0.1753; Tikslumas 0.9771
Epocha 65/100: Mokymo paklaida 0.7823; Tikslumas 0.6119 | Validacijos paklaida 0.1945; Tikslumas 0.9495
Epocha 66/100: Mokymo paklaida 0.7938; Tikslumas 0.6376 | Validacijos paklaida 0.2098; Tikslumas 0.9633
Epocha 67/100: Mokymo paklaida 0.7832; Tikslumas 0.6250 | Validacijos paklaida 0.1913; Tikslumas 0.9679
Epocha 68/100: Mokymo paklaida 0.7693; Tikslumas 0.6341 | Validacijos paklaida 0.1931; Tikslumas 0.9633
Epocha 69/100: Mokymo paklaida 0.7627; Tikslumas 0.6438 | Validacijos paklaida 0.1837; Tikslumas 0.9771
Epocha 70/100: Mokymo paklaida 0.7838; Tikslumas 0.6193 | Validacijos paklaida 0.1794; Tikslumas 0.9771
Epocha 71/100: Mokymo paklaida 0.7857; Tikslumas 0.6199 | Validacijos paklaida 0.1847; Tikslumas 0.9725
Epocha 72/100: Mokymo paklaida 0.7446; Tikslumas 0.6438 | Validacijos paklaida 0.1771; Tikslumas 0.9725
Epocha 73/100: Mokymo paklaida 0.7790; Tikslumas 0.6199 | Validacijos paklaida 0.1776; Tikslumas 0.9771
Baigiama anksčiau, nes nebuvo pagerinimo per 100 epochų.
Testavimo paklaida 0.1976; Tikslumas 0.9633
```

### VGG model
```
Epocha 1/100: Mokymo paklaida 0.9433; Tikslumas 0.5788 | Validacijos paklaida 0.2982; Tikslumas 0.9358
Epocha 2/100: Mokymo paklaida 0.5247; Tikslumas 0.8076 | Validacijos paklaida 0.1800; Tikslumas 0.9679
Epocha 3/100: Mokymo paklaida 0.4530; Tikslumas 0.8316 | Validacijos paklaida 0.1173; Tikslumas 0.9725
Epocha 4/100: Mokymo paklaida 0.3952; Tikslumas 0.8602 | Validacijos paklaida 0.1214; Tikslumas 0.9679
Epocha 5/100: Mokymo paklaida 0.3471; Tikslumas 0.8727 | Validacijos paklaida 0.0649; Tikslumas 0.9817
Epocha 6/100: Mokymo paklaida 0.3836; Tikslumas 0.8539 | Validacijos paklaida 0.0379; Tikslumas 0.9908
Epocha 7/100: Mokymo paklaida 0.3007; Tikslumas 0.8944 | Validacijos paklaida 0.0376; Tikslumas 0.9908
Epocha 8/100: Mokymo paklaida 0.3271; Tikslumas 0.8830 | Validacijos paklaida 0.0771; Tikslumas 0.9817
Epocha 9/100: Mokymo paklaida 0.2859; Tikslumas 0.8916 | Validacijos paklaida 0.1314; Tikslumas 0.9587
Epocha 10/100: Mokymo paklaida 0.2948; Tikslumas 0.8927 | Validacijos paklaida 0.0472; Tikslumas 0.9862
Epocha 11/100: Mokymo paklaida 0.2574; Tikslumas 0.9007 | Validacijos paklaida 0.0280; Tikslumas 0.9908
Epocha 12/100: Mokymo paklaida 0.2729; Tikslumas 0.8950 | Validacijos paklaida 0.0389; Tikslumas 0.9862
Epocha 13/100: Mokymo paklaida 0.2629; Tikslumas 0.9047 | Validacijos paklaida 0.0459; Tikslumas 0.9954
Epocha 14/100: Mokymo paklaida 0.2447; Tikslumas 0.9041 | Validacijos paklaida 0.0518; Tikslumas 0.9862
Epocha 15/100: Mokymo paklaida 0.2709; Tikslumas 0.8978 | Validacijos paklaida 0.0265; Tikslumas 0.9908
Epocha 16/100: Mokymo paklaida 0.2293; Tikslumas 0.9172 | Validacijos paklaida 0.0210; Tikslumas 0.9954
Epocha 17/100: Mokymo paklaida 0.2529; Tikslumas 0.9047 | Validacijos paklaida 0.0284; Tikslumas 0.9954
Epocha 18/100: Mokymo paklaida 0.2488; Tikslumas 0.9047 | Validacijos paklaida 0.0490; Tikslumas 0.9862
Epocha 19/100: Mokymo paklaida 0.2319; Tikslumas 0.9155 | Validacijos paklaida 0.1427; Tikslumas 0.9541
Epocha 20/100: Mokymo paklaida 0.2471; Tikslumas 0.9064 | Validacijos paklaida 0.0177; Tikslumas 0.9954
Epocha 21/100: Mokymo paklaida 0.2336; Tikslumas 0.9087 | Validacijos paklaida 0.0207; Tikslumas 0.9908
Epocha 22/100: Mokymo paklaida 0.2291; Tikslumas 0.9104 | Validacijos paklaida 0.0320; Tikslumas 0.9862
Epocha 23/100: Mokymo paklaida 0.2347; Tikslumas 0.9058 | Validacijos paklaida 0.0373; Tikslumas 0.9908
Epocha 24/100: Mokymo paklaida 0.2411; Tikslumas 0.9138 | Validacijos paklaida 0.0271; Tikslumas 0.9908
Epocha 25/100: Mokymo paklaida 0.2174; Tikslumas 0.9150 | Validacijos paklaida 0.0144; Tikslumas 0.9954
Epocha 26/100: Mokymo paklaida 0.2098; Tikslumas 0.9252 | Validacijos paklaida 0.0150; Tikslumas 0.9908
Epocha 27/100: Mokymo paklaida 0.1943; Tikslumas 0.9275 | Validacijos paklaida 0.0125; Tikslumas 0.9954
Epocha 28/100: Mokymo paklaida 0.1909; Tikslumas 0.9269 | Validacijos paklaida 0.0127; Tikslumas 0.9908
Epocha 29/100: Mokymo paklaida 0.1765; Tikslumas 0.9309 | Validacijos paklaida 0.0135; Tikslumas 0.9908
Epocha 30/100: Mokymo paklaida 0.1952; Tikslumas 0.9264 | Validacijos paklaida 0.0147; Tikslumas 0.9954
Epocha 31/100: Mokymo paklaida 0.1949; Tikslumas 0.9201 | Validacijos paklaida 0.0117; Tikslumas 0.9954
Epocha 32/100: Mokymo paklaida 0.1906; Tikslumas 0.9275 | Validacijos paklaida 0.0093; Tikslumas 0.9954
Epocha 33/100: Mokymo paklaida 0.1793; Tikslumas 0.9315 | Validacijos paklaida 0.0097; Tikslumas 0.9954
Epocha 34/100: Mokymo paklaida 0.1891; Tikslumas 0.9281 | Validacijos paklaida 0.0113; Tikslumas 0.9954
Epocha 35/100: Mokymo paklaida 0.1833; Tikslumas 0.9258 | Validacijos paklaida 0.0092; Tikslumas 0.9954
Epocha 36/100: Mokymo paklaida 0.1858; Tikslumas 0.9264 | Validacijos paklaida 0.0107; Tikslumas 1.0000
Epocha 37/100: Mokymo paklaida 0.1723; Tikslumas 0.9309 | Validacijos paklaida 0.0092; Tikslumas 0.9954
Epocha 38/100: Mokymo paklaida 0.1547; Tikslumas 0.9424 | Validacijos paklaida 0.0092; Tikslumas 0.9954
Epocha 39/100: Mokymo paklaida 0.1602; Tikslumas 0.9406 | Validacijos paklaida 0.0080; Tikslumas 0.9954
Epocha 40/100: Mokymo paklaida 0.1608; Tikslumas 0.9412 | Validacijos paklaida 0.0080; Tikslumas 0.9954
Epocha 41/100: Mokymo paklaida 0.1386; Tikslumas 0.9463 | Validacijos paklaida 0.0079; Tikslumas 0.9954
Epocha 42/100: Mokymo paklaida 0.1761; Tikslumas 0.9349 | Validacijos paklaida 0.0080; Tikslumas 1.0000
Epocha 43/100: Mokymo paklaida 0.1596; Tikslumas 0.9424 | Validacijos paklaida 0.0095; Tikslumas 0.9954
Epocha 44/100: Mokymo paklaida 0.1477; Tikslumas 0.9463 | Validacijos paklaida 0.0097; Tikslumas 0.9954
Epocha 45/100: Mokymo paklaida 0.1439; Tikslumas 0.9469 | Validacijos paklaida 0.0105; Tikslumas 0.9908
Epocha 46/100: Mokymo paklaida 0.1595; Tikslumas 0.9378 | Validacijos paklaida 0.0094; Tikslumas 0.9954
Epocha 47/100: Mokymo paklaida 0.1349; Tikslumas 0.9441 | Validacijos paklaida 0.0084; Tikslumas 0.9954
Epocha 48/100: Mokymo paklaida 0.1815; Tikslumas 0.9292 | Validacijos paklaida 0.0089; Tikslumas 0.9954
Epocha 49/100: Mokymo paklaida 0.1667; Tikslumas 0.9361 | Validacijos paklaida 0.0085; Tikslumas 0.9954
Epocha 50/100: Mokymo paklaida 0.1649; Tikslumas 0.9389 | Validacijos paklaida 0.0092; Tikslumas 0.9954
Baigiama anksčiau, nes nebuvo pagerinimo per 10 epochų.

(process:1145723): Gtk-WARNING **: 21:08:59.222: Theme parser error: gtk.css:120:11-12: Expected a filter
Testavimo paklaida 0.0050; Tikslumas 1.0000
```