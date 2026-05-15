# Layout heatmap difference analysis

## Settings

```json
{
  "objects_csv": "out/layout_distribution_analysis/all_workspace_plus_rtx3060_20260515/objects_all.csv",
  "out_dir": "out/layout_heatmap_diff_analysis/all_workspace_plus_rtx3060_20260515",
  "reference": "3dfront",
  "classes": [
    "bed",
    "double_bed",
    "nightstand",
    "wardrobe",
    "cabinet",
    "chair",
    "table",
    "sofa",
    "tv",
    "lamp",
    "decor"
  ],
  "class_mode": "canonical",
  "grouping": "both",
  "grids": [
    20,
    32
  ],
  "sigmas": [
    1.25
  ],
  "augmentation": "rot90_flip",
  "min_ref_objects": 50,
  "min_method_objects": 30,
  "plot_mode": "reference"
}
```

## Filtered data counts

Filtered object rows: **84765**

### Methods

| method | trackable_objects |
| --- | --- |
| 3dfront | 55939 |
| cube | 8767 |
| infinigen | 8341 |
| retrieval | 4641 |
| diffuscene | 3782 |
| relaxed | 1751 |
| ollama_llm | 1236 |
| random | 262 |
| m3dlayout | 46 |

### Classes

| class_name_norm | trackable_objects |
| --- | --- |
| nightstand | 15847 |
| chair | 14959 |
| cabinet | 10671 |
| table | 9689 |
| bed | 9397 |
| wardrobe | 8042 |
| lamp | 4187 |
| tv | 4039 |
| decor | 3772 |
| sofa | 3562 |
| double_bed | 600 |

### Rooms

| room_type_norm | trackable_objects |
| --- | --- |
| bedroom | 35924 |
| livingroom | 35384 |
| unknown_room | 8826 |
| office | 3242 |
| kitchen | 445 |
| hallway | 396 |
| bathroom | 386 |
| apartment | 162 |

## Method ranking against reference

| grouping | grid | sigma | augmentation | method | n_groups | coverage_ref_weight | weighted_tv_distance | weighted_js_distance | weighted_sliced_wasserstein | weighted_centroid_shift | weighted_max_abs_diff_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class | 20 | 1.2500 | rot90_flip | infinigen | 7 | 0.7011 | 0.3231 | 0.3320 | 0.0487 | 0.0000 | 0.6596 |
| class | 20 | 1.2500 | rot90_flip | retrieval | 9 | 1.0000 | 0.3387 | 0.3760 | 0.0451 | 0.0000 | 0.7624 |
| class | 20 | 1.2500 | rot90_flip | cube | 9 | 1.0000 | 0.3526 | 0.3620 | 0.0584 | 0.0000 | 0.6862 |
| class | 20 | 1.2500 | rot90_flip | ollama_llm | 6 | 0.6399 | 0.4433 | 0.4473 | 0.0721 | 0.0041 | 0.7543 |
| class | 20 | 1.2500 | rot90_flip | random | 3 | 0.4292 | 0.5081 | 0.5198 | 0.0839 | 0.0000 | 0.8661 |
| class | 20 | 1.2500 | rot90_flip | relaxed | 9 | 1.0000 | 0.5591 | 0.5632 | 0.0820 | 0.0015 | 1.3725 |
| class | 20 | 1.2500 | rot90_flip | diffuscene | 5 | 0.7329 | 0.5833 | 0.6006 | 0.0773 | 0.0000 | 1.8430 |
| class | 32 | 1.2500 | rot90_flip | infinigen | 7 | 0.7011 | 0.3944 | 0.4091 | 0.0514 | 0.0000 | 0.4164 |
| class | 32 | 1.2500 | rot90_flip | cube | 9 | 1.0000 | 0.4100 | 0.4215 | 0.0613 | 0.0000 | 0.4590 |
| class | 32 | 1.2500 | rot90_flip | retrieval | 9 | 1.0000 | 0.4657 | 0.5011 | 0.0490 | 0.0000 | 0.7078 |
| class | 32 | 1.2500 | rot90_flip | ollama_llm | 6 | 0.6399 | 0.5483 | 0.5548 | 0.0758 | 0.0024 | 0.6245 |
| class | 32 | 1.2500 | rot90_flip | random | 3 | 0.4292 | 0.5882 | 0.5983 | 0.0862 | 0.0000 | 0.5434 |
| class | 32 | 1.2500 | rot90_flip | relaxed | 9 | 1.0000 | 0.6549 | 0.6621 | 0.0878 | 0.0011 | 1.0834 |
| class | 32 | 1.2500 | rot90_flip | diffuscene | 5 | 0.7329 | 0.7351 | 0.7451 | 0.0875 | 0.0000 | 1.4646 |
| room_class | 20 | 1.2500 | rot90_flip | cube | 4 | 0.4704 | 0.2105 | 0.2265 | 0.0317 | 0.0000 | 0.3896 |
| room_class | 20 | 1.2500 | rot90_flip | retrieval | 9 | 0.9556 | 0.3682 | 0.3924 | 0.0464 | 0.0000 | 0.9218 |
| room_class | 20 | 1.2500 | rot90_flip | infinigen | 9 | 0.4716 | 0.3991 | 0.4118 | 0.0510 | 0.0000 | 0.7979 |
| room_class | 20 | 1.2500 | rot90_flip | ollama_llm | 4 | 0.4310 | 0.4901 | 0.5091 | 0.0608 | 0.0106 | 1.6410 |
| room_class | 20 | 1.2500 | rot90_flip | relaxed | 7 | 0.8779 | 0.5735 | 0.5871 | 0.0838 | 0.0028 | 1.3730 |
| room_class | 20 | 1.2500 | rot90_flip | diffuscene | 3 | 0.3045 | 0.6958 | 0.7065 | 0.1137 | 0.0000 | 2.5238 |
| room_class | 32 | 1.2500 | rot90_flip | cube | 4 | 0.4704 | 0.2596 | 0.2800 | 0.0332 | 0.0000 | 0.2898 |
| room_class | 32 | 1.2500 | rot90_flip | infinigen | 9 | 0.4716 | 0.4892 | 0.5064 | 0.0543 | 0.0000 | 0.5688 |
| room_class | 32 | 1.2500 | rot90_flip | retrieval | 9 | 0.9556 | 0.4956 | 0.5210 | 0.0500 | 0.0000 | 0.8204 |
| room_class | 32 | 1.2500 | rot90_flip | ollama_llm | 4 | 0.4310 | 0.6295 | 0.6480 | 0.0681 | 0.0064 | 1.4878 |
| room_class | 32 | 1.2500 | rot90_flip | relaxed | 7 | 0.8779 | 0.6906 | 0.6990 | 0.0897 | 0.0018 | 1.1353 |
| room_class | 32 | 1.2500 | rot90_flip | diffuscene | 3 | 0.3045 | 0.8399 | 0.8434 | 0.1263 | 0.0000 | 1.7342 |

## Strongest per-category deviations from reference

| grouping | grid | sigma | augmentation | room_type | class_name | method_b | n_a | n_b | tv_distance | js_distance | sliced_wasserstein | max_abs_diff_pp | centroid_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | cabinet | retrieval | 1169 | 252 | 0.8857 | 0.8959 | 0.2052 | 1.7946 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | wardrobe | diffuscene | 5191 | 860 | 0.8598 | 0.8748 | 0.2044 | 3.0196 | 0.0001 |
| class | 32 | 1.2500 | rot90_flip | __all__ | wardrobe | diffuscene | 5309 | 863 | 0.8585 | 0.8740 | 0.2045 | 3.0193 | 0.0001 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | cabinet | diffuscene | 1169 | 616 | 0.8487 | 0.8396 | 0.1364 | 0.5468 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | table | relaxed | 5707 | 54 | 0.8381 | 0.8366 | 0.1533 | 1.5386 | 0.0015 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | nightstand | diffuscene | 8806 | 1096 | 0.8270 | 0.8253 | 0.0790 | 1.1341 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | cabinet | retrieval | 1169 | 252 | 0.8223 | 0.8395 | 0.1910 | 2.9675 | 0.0001 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | wardrobe | diffuscene | 5191 | 860 | 0.8210 | 0.8285 | 0.1887 | 4.4894 | 0.0001 |
| class | 20 | 1.2500 | rot90_flip | __all__ | wardrobe | diffuscene | 5309 | 863 | 0.8200 | 0.8277 | 0.1887 | 4.4874 | 0.0001 |
| class | 32 | 1.2500 | rot90_flip | __all__ | nightstand | diffuscene | 11411 | 1100 | 0.8199 | 0.8186 | 0.0722 | 1.1341 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | tv | relaxed | 3578 | 54 | 0.7782 | 0.7629 | 0.1023 | 1.3321 | 0.0002 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | table | relaxed | 5707 | 54 | 0.7664 | 0.7712 | 0.1479 | 1.9435 | 0.0025 |
| class | 32 | 1.2500 | rot90_flip | __all__ | chair | relaxed | 13461 | 333 | 0.7593 | 0.7661 | 0.1291 | 1.5507 | 0.0012 |
| class | 32 | 1.2500 | rot90_flip | __all__ | decor | retrieval | 970 | 189 | 0.7423 | 0.7484 | 0.0667 | 1.2889 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | cabinet | diffuscene | 3868 | 674 | 0.7349 | 0.7303 | 0.1280 | 0.6474 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | chair | relaxed | 12317 | 100 | 0.7274 | 0.7424 | 0.1045 | 1.1592 | 0.0034 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | tv | relaxed | 2360 | 54 | 0.7252 | 0.7195 | 0.1017 | 1.2756 | 0.0002 |
| class | 32 | 1.2500 | rot90_flip | __all__ | chair | diffuscene | 13461 | 46 | 0.7165 | 0.7216 | 0.0609 | 1.7107 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | table | retrieval | 5707 | 252 | 0.7137 | 0.7247 | 0.1039 | 0.7560 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | bed | random | 7290 | 60 | 0.7131 | 0.7265 | 0.0969 | 0.6557 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | cabinet | diffuscene | 1169 | 616 | 0.7107 | 0.7152 | 0.1243 | 0.8608 | 0.0001 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | wardrobe | ollama_llm | 5191 | 96 | 0.7012 | 0.7102 | 0.0738 | 2.1446 | 0.0084 |
| class | 32 | 1.2500 | rot90_flip | __all__ | bed | cube | 7290 | 1085 | 0.6946 | 0.7011 | 0.0982 | 0.6939 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | tv | relaxed | 3578 | 54 | 0.6905 | 0.6864 | 0.0964 | 1.7229 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | bed | ollama_llm | 6300 | 96 | 0.6790 | 0.6887 | 0.0450 | 2.3915 | 0.0115 |
| class | 20 | 1.2500 | rot90_flip | __all__ | chair | relaxed | 13461 | 333 | 0.6727 | 0.6677 | 0.1209 | 2.0937 | 0.0018 |
| class | 32 | 1.2500 | rot90_flip | __all__ | bed | relaxed | 7290 | 91 | 0.6660 | 0.6620 | 0.0825 | 0.9770 | 0.0003 |
| class | 32 | 1.2500 | rot90_flip | __all__ | sofa | relaxed | 3104 | 54 | 0.6637 | 0.6685 | 0.0830 | 1.0654 | 0.0002 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | sofa | relaxed | 3043 | 54 | 0.6628 | 0.6676 | 0.0825 | 1.0649 | 0.0002 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | chair | retrieval | 12317 | 252 | 0.6616 | 0.6721 | 0.0676 | 1.0161 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | wardrobe | relaxed | 5191 | 122 | 0.6570 | 0.6414 | 0.0806 | 0.9893 | 0.0003 |
| class | 32 | 1.2500 | rot90_flip | __all__ | tv | cube | 3578 | 56 | 0.6504 | 0.6494 | 0.0864 | 1.1263 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | bed | ollama_llm | 7290 | 235 | 0.6502 | 0.6548 | 0.0714 | 0.8949 | 0.0052 |
| class | 32 | 1.2500 | rot90_flip | __all__ | nightstand | ollama_llm | 11411 | 440 | 0.6499 | 0.6468 | 0.1132 | 0.5410 | 0.0008 |
| class | 32 | 1.2500 | rot90_flip | __all__ | wardrobe | relaxed | 5309 | 148 | 0.6489 | 0.6349 | 0.0776 | 0.9110 | 0.0007 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | tv | relaxed | 2360 | 54 | 0.6384 | 0.6444 | 0.0967 | 1.6614 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | table | relaxed | 6948 | 274 | 0.6337 | 0.6575 | 0.1080 | 0.7560 | 0.0005 |
| class | 32 | 1.2500 | rot90_flip | __all__ | table | retrieval | 6948 | 504 | 0.6257 | 0.6519 | 0.0857 | 0.7118 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | cabinet | ollama_llm | 1169 | 62 | 0.6255 | 0.6387 | 0.0921 | 0.6444 | 0.0043 |
| class | 20 | 1.2500 | rot90_flip | __all__ | cabinet | diffuscene | 3868 | 674 | 0.6232 | 0.6278 | 0.1151 | 0.9580 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | nightstand | diffuscene | 8806 | 1096 | 0.6200 | 0.6334 | 0.0681 | 1.5858 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | nightstand | relaxed | 8806 | 128 | 0.6200 | 0.6478 | 0.0475 | 0.8472 | 0.0026 |
| class | 32 | 1.2500 | rot90_flip | __all__ | tv | retrieval | 3578 | 252 | 0.6190 | 0.6484 | 0.0370 | 0.7609 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | bed | random | 7290 | 60 | 0.6168 | 0.6292 | 0.0967 | 1.0650 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | nightstand | diffuscene | 11411 | 1100 | 0.6156 | 0.6276 | 0.0601 | 1.5938 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | table | retrieval | 5707 | 252 | 0.6145 | 0.6309 | 0.1055 | 0.8435 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | table | infinigen | 5707 | 173 | 0.6123 | 0.6047 | 0.0685 | 0.7895 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | bed | relaxed | 6300 | 61 | 0.6120 | 0.6159 | 0.0685 | 1.2277 | 0.0004 |
| class | 20 | 1.2500 | rot90_flip | __all__ | bed | cube | 7290 | 1085 | 0.6057 | 0.6128 | 0.0984 | 1.0424 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | wardrobe | ollama_llm | 5309 | 239 | 0.5965 | 0.5921 | 0.0646 | 0.8930 | 0.0036 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | chair | relaxed | 12317 | 100 | 0.5856 | 0.6097 | 0.0965 | 1.4262 | 0.0054 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | tv | cube | 2360 | 56 | 0.5836 | 0.6171 | 0.0866 | 1.0698 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | wardrobe | ollama_llm | 5191 | 96 | 0.5823 | 0.5864 | 0.0645 | 2.3398 | 0.0141 |
| class | 32 | 1.2500 | rot90_flip | __all__ | nightstand | random | 11411 | 53 | 0.5795 | 0.5801 | 0.0954 | 0.4902 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | chair | retrieval | 13461 | 504 | 0.5758 | 0.5994 | 0.0697 | 0.6093 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | table | relaxed | 6948 | 274 | 0.5736 | 0.5895 | 0.1029 | 0.9819 | 0.0005 |
| class | 20 | 1.2500 | rot90_flip | __all__ | decor | retrieval | 970 | 189 | 0.5711 | 0.5928 | 0.0564 | 1.2527 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | sofa | relaxed | 3104 | 54 | 0.5702 | 0.5892 | 0.0820 | 1.1692 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | sofa | relaxed | 3043 | 54 | 0.5687 | 0.5877 | 0.0815 | 1.1690 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | tv | cube | 3578 | 56 | 0.5680 | 0.5717 | 0.0809 | 1.4665 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | bed | relaxed | 7290 | 91 | 0.5668 | 0.5564 | 0.0778 | 1.2806 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | nightstand | ollama_llm | 11411 | 440 | 0.5668 | 0.5689 | 0.1108 | 0.8388 | 0.0013 |
| class | 32 | 1.2500 | rot90_flip | __all__ | nightstand | relaxed | 11411 | 134 | 0.5630 | 0.5839 | 0.0364 | 0.8927 | 0.0025 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | wardrobe | relaxed | 5191 | 122 | 0.5600 | 0.5576 | 0.0746 | 1.5007 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | nightstand | ollama_llm | 8806 | 192 | 0.5524 | 0.5836 | 0.0781 | 0.5660 | 0.0018 |
| class | 32 | 1.2500 | rot90_flip | __all__ | decor | relaxed | 970 | 92 | 0.5499 | 0.5740 | 0.0758 | 0.6730 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | wardrobe | relaxed | 5309 | 148 | 0.5461 | 0.5455 | 0.0721 | 1.4460 | 0.0008 |
| class | 32 | 1.2500 | rot90_flip | __all__ | table | diffuscene | 6948 | 56 | 0.5378 | 0.5796 | 0.0519 | 0.7977 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | chair | diffuscene | 13461 | 46 | 0.5372 | 0.5542 | 0.0528 | 1.7070 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | bed | infinigen | 6300 | 434 | 0.5349 | 0.5393 | 0.0728 | 0.5199 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | cabinet | retrieval | 3868 | 756 | 0.5295 | 0.5634 | 0.0656 | 0.5758 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | table | retrieval | 6948 | 504 | 0.5216 | 0.5542 | 0.0851 | 0.7521 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | nightstand | cube | 11411 | 2007 | 0.5197 | 0.5181 | 0.0914 | 0.4721 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | tv | retrieval | 2360 | 252 | 0.5180 | 0.5599 | 0.0291 | 0.7365 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | sofa | retrieval | 3104 | 252 | 0.5167 | 0.5428 | 0.0416 | 0.6581 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | sofa | retrieval | 3043 | 252 | 0.5161 | 0.5417 | 0.0411 | 0.6582 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | cabinet | retrieval | 2699 | 189 | 0.5153 | 0.5477 | 0.0343 | 0.5529 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | tv | infinigen | 2360 | 65 | 0.5117 | 0.5544 | 0.0490 | 0.6982 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | nightstand | random | 11411 | 53 | 0.5081 | 0.5086 | 0.0914 | 0.7840 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | tv | cube | 2360 | 56 | 0.5080 | 0.5439 | 0.0817 | 1.3924 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | bed | relaxed | 6300 | 61 | 0.5074 | 0.4999 | 0.0627 | 1.5132 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | bed | ollama_llm | 7290 | 235 | 0.5072 | 0.5074 | 0.0689 | 0.9651 | 0.0084 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | chair | retrieval | 12317 | 252 | 0.5036 | 0.5253 | 0.0643 | 1.1952 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | bed | ollama_llm | 6300 | 96 | 0.5025 | 0.5067 | 0.0402 | 2.3512 | 0.0186 |
| class | 32 | 1.2500 | rot90_flip | __all__ | bed | infinigen | 7290 | 450 | 0.5016 | 0.4921 | 0.0670 | 0.4804 | 0.0001 |
| class | 20 | 1.2500 | rot90_flip | __all__ | decor | relaxed | 970 | 92 | 0.4990 | 0.5194 | 0.0727 | 1.1526 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | table | infinigen | 5707 | 173 | 0.4968 | 0.4864 | 0.0626 | 1.0284 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | cabinet | relaxed | 3868 | 174 | 0.4935 | 0.5006 | 0.0762 | 0.9330 | 0.0010 |
| class | 32 | 1.2500 | rot90_flip | __all__ | cabinet | ollama_llm | 3868 | 76 | 0.4872 | 0.4941 | 0.0715 | 0.6122 | 0.0038 |
| class | 20 | 1.2500 | rot90_flip | __all__ | wardrobe | ollama_llm | 5309 | 239 | 0.4762 | 0.4724 | 0.0602 | 0.9110 | 0.0058 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | cabinet | ollama_llm | 1169 | 62 | 0.4709 | 0.4838 | 0.0816 | 0.8034 | 0.0086 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | nightstand | relaxed | 8806 | 128 | 0.4708 | 0.5004 | 0.0424 | 0.7466 | 0.0045 |
| class | 32 | 1.2500 | rot90_flip | __all__ | wardrobe | cube | 5309 | 1083 | 0.4671 | 0.4606 | 0.0511 | 0.5827 | 0.0001 |
| class | 20 | 1.2500 | rot90_flip | __all__ | cabinet | relaxed | 3868 | 174 | 0.4636 | 0.4681 | 0.0722 | 1.2852 | 0.0004 |
| class | 20 | 1.2500 | rot90_flip | __all__ | nightstand | cube | 11411 | 2007 | 0.4604 | 0.4624 | 0.0859 | 0.7168 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | bed | infinigen | 6300 | 434 | 0.4602 | 0.4614 | 0.0700 | 0.9223 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | sofa | cube | 3104 | 56 | 0.4588 | 0.5078 | 0.0624 | 0.6414 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | sofa | cube | 3043 | 56 | 0.4572 | 0.5061 | 0.0617 | 0.6402 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | livingroom | sofa | infinigen | 3043 | 81 | 0.4525 | 0.4929 | 0.0479 | 0.5773 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | sofa | infinigen | 3104 | 94 | 0.4445 | 0.4809 | 0.0465 | 0.4914 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | tv | retrieval | 3578 | 252 | 0.4406 | 0.4705 | 0.0276 | 0.8146 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | wardrobe | random | 5309 | 58 | 0.4355 | 0.4614 | 0.0516 | 0.5036 | 0.0001 |
| room_class | 20 | 1.2500 | rot90_flip | bedroom | nightstand | ollama_llm | 8806 | 192 | 0.4295 | 0.4686 | 0.0705 | 0.8322 | 0.0031 |
| class | 20 | 1.2500 | rot90_flip | __all__ | bed | infinigen | 7290 | 450 | 0.4290 | 0.4176 | 0.0646 | 0.8686 | 0.0001 |
| class | 20 | 1.2500 | rot90_flip | __all__ | chair | retrieval | 13461 | 504 | 0.4289 | 0.4674 | 0.0670 | 0.6422 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | chair | infinigen | 13461 | 109 | 0.4285 | 0.4423 | 0.0653 | 0.4777 | 0.0000 |
| class | 32 | 1.2500 | rot90_flip | __all__ | decor | ollama_llm | 970 | 49 | 0.4271 | 0.4638 | 0.0609 | 0.6243 | 0.0004 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | tv | infinigen | 2360 | 65 | 0.4240 | 0.4628 | 0.0500 | 0.8154 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | table | diffuscene | 6948 | 56 | 0.4166 | 0.4575 | 0.0467 | 0.9880 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | nightstand | relaxed | 11411 | 134 | 0.4105 | 0.4267 | 0.0302 | 0.7780 | 0.0043 |
| class | 32 | 1.2500 | rot90_flip | __all__ | decor | cube | 970 | 204 | 0.3885 | 0.4207 | 0.0365 | 0.4991 | 0.0004 |
| class | 32 | 1.2500 | rot90_flip | __all__ | tv | infinigen | 3578 | 97 | 0.3882 | 0.4579 | 0.0402 | 0.5927 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | sofa | infinigen | 3043 | 81 | 0.3791 | 0.4000 | 0.0431 | 0.6870 | 0.0000 |
| room_class | 32 | 1.2500 | rot90_flip | bedroom | table | infinigen | 1241 | 213 | 0.3775 | 0.4025 | 0.0374 | 0.3189 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | wardrobe | cube | 5309 | 1083 | 0.3763 | 0.3828 | 0.0487 | 0.8242 | 0.0001 |
| class | 20 | 1.2500 | rot90_flip | __all__ | sofa | cube | 3104 | 56 | 0.3746 | 0.4236 | 0.0603 | 0.7518 | 0.0000 |
| room_class | 20 | 1.2500 | rot90_flip | livingroom | sofa | cube | 3043 | 56 | 0.3729 | 0.4213 | 0.0596 | 0.7502 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | cabinet | retrieval | 3868 | 756 | 0.3695 | 0.4098 | 0.0589 | 0.9518 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | sofa | infinigen | 3104 | 94 | 0.3652 | 0.3833 | 0.0412 | 0.6792 | 0.0000 |
| class | 20 | 1.2500 | rot90_flip | __all__ | cabinet | ollama_llm | 3868 | 76 | 0.3645 | 0.3650 | 0.0629 | 0.7025 | 0.0076 |

_Shown 120 of 168 rows._

## Explanation for thesis text

The comparison is performed by normalized heatmaps of object centers. For each method and category, the sum of the heatmap equals 100%, so the analysis measures spatial distribution shape rather than absolute object count.

Gaussian smoothing is applied after histogram construction and followed by renormalization to 100%. Smoothing reduces sensitivity to grid-cell boundaries.

Orientation augmentation is applied equally to 3D-FRONT and to every compared method. The transforms include vertical and horizontal reflection and rotations by 90, 180 and 270 degrees. This is necessary because absolute room orientation is usually arbitrary; a valid layout may be rotated or mirrored without changing semantic quality.