| experiment_name | hidden_sizes | activation | dropout | use_batch_norm | learning_rate | scheduler | regularizer | best_val_loss | best_val_acc | test_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| bn_on | [256, 128] | relu | 0.2 | True | 0.001 | none | none | 0.06093493517053624 | 0.9811666666666666 | 0.983 |
| baseline_relu | [256, 128] | relu | 0.2 | False | 0.001 | none | none | 0.07414034351271888 | 0.9793333333333333 | 0.9804 |