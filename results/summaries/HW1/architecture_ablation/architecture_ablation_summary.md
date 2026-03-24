| experiment_name | hidden_sizes | activation | dropout | use_batch_norm | learning_rate | scheduler | regularizer | best_val_loss | best_val_acc | test_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| arch_medium_256_128 | [256, 128] | relu | 0.2 | False | 0.001 | none | none | 0.07414034351271888 | 0.9793333333333333 | 0.9804 |
| arch_deep_512_256_128 | [512, 256, 128] | relu | 0.2 | False | 0.001 | none | none | 0.08108403160360952 | 0.979 | 0.9785 |
| arch_shallow_128 | [128] | relu | 0.2 | False | 0.001 | none | none | 0.0821263378560543 | 0.9768333333333333 | 0.9779 |