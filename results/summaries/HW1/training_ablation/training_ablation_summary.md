| experiment_name | hidden_sizes | activation | dropout | use_batch_norm | learning_rate | scheduler | regularizer | best_val_loss | best_val_acc | test_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| train_plateau | [256, 128] | relu | 0.2 | False | 0.001 | plateau | none | 0.06532134033677478 | 0.983 | 0.9827 |
| baseline_relu | [256, 128] | relu | 0.2 | False | 0.001 | none | none | 0.07414034351271888 | 0.9793333333333333 | 0.9804 |
| train_cosine | [256, 128] | relu | 0.2 | False | 0.001 | cosine | none | 0.07469713910203427 | 0.983 | 0.9832 |
| train_lr_1e4 | [256, 128] | relu | 0.2 | False | 0.0001 | none | none | 0.07847719851632913 | 0.977 | 0.979 |
| train_lr_3e3 | [256, 128] | relu | 0.2 | False | 0.003 | none | none | 0.1179951309065024 | 0.9671666666666666 | 0.9685 |