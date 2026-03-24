| experiment_name | hidden_sizes | activation | dropout | use_batch_norm | learning_rate | scheduler | regularizer | best_val_loss | best_val_acc | test_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| dropout_02 | [256, 128] | relu | 0.2 | False | 0.001 | none | none | 0.07414034351271888 | 0.9793333333333333 | 0.9804 |
| dropout_00 | [256, 128] | relu | 0.0 | False | 0.001 | none | none | 0.08292990322162708 | 0.976 | 0.9784 |
| dropout_05 | [256, 128] | relu | 0.5 | False | 0.001 | none | none | 0.08585439555098613 | 0.9748333333333333 | 0.9791 |