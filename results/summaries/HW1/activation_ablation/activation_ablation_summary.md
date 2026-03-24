| experiment_name | hidden_sizes | activation | dropout | use_batch_norm | learning_rate | scheduler | regularizer | best_val_loss | best_val_acc | test_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_relu | [256, 128] | relu | 0.2 | False | 0.001 | none | none | 0.07414034351271888 | 0.9793333333333333 | 0.9804 |
| act_gelu | [256, 128] | gelu | 0.2 | False | 0.001 | none | none | 0.08165380743642649 | 0.9758333333333333 | 0.9791 |
| act_gelu_deep | [512, 256, 128] | gelu | 0.2 | False | 0.001 | none | none | 0.08290308447678883 | 0.9761666666666666 | 0.9787 |