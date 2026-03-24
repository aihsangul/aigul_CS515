| experiment_name | hidden_sizes | activation | dropout | use_batch_norm | learning_rate | scheduler | regularizer | best_val_loss | best_val_acc | test_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_relu | [256, 128] | relu | 0.2 | False | 0.001 | none | none | 0.07414034351271888 | 0.9793333333333333 | 0.9804 |
| reg_l1_1e5 | [256, 128] | relu | 0.2 | False | 0.001 | none | l1 | 0.07426526289557417 | 0.9815 | 0.9808 |
| reg_l2_1e5 | [256, 128] | relu | 0.2 | False | 0.001 | none | l2 | 0.07758179262280464 | 0.9771666666666666 | 0.9817 |
| reg_l2_1e4 | [256, 128] | relu | 0.2 | False | 0.001 | none | l2 | 0.07921056987345218 | 0.9768333333333333 | 0.9773 |