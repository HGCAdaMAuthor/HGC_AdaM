

AM (GRAPHSAGE): 

|               | SIDER   | ClinTox | BACE  | HIV     | BBBP    | Tox21   | ToxCast |
| ------------- | ------- | ------- | ----- | ------- | ------- | ------- | ------- |
| batch_size    | 64      | 64      | 64    | 64      | 64      | 32      | 32      |
| lr            | 0.00843 | 0.00652 | 0.001 | 0.00049 | 0.00549 | 0.00071 | 0.001   |
| dropout_ratio | 0.7     | 0.3     | 0.2   | 0.3     | 0.3     | 0.6     | 0.7     |
| graph_pooling | sum     | sum     | sum   | sum     | sum     | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0   | 1.0     | 1.0     | 1.0     | 1.0     |



CON_HGC (GRAPHSAGE): 

|               | SIDER | ClinTox | BACE    | HIV     | BBBP  | Tox21   | ToxCast |
| ------------- | ----- | ------- | ------- | ------- | ----- | ------- | ------- |
| batch_size    | 32    | 64      | 32      | 64      | 64    | 64      | 32      |
| lr            | 0.001 | 0.00462 | 0.00018 | 0.00489 | 0.001 | 0.00549 | 0.00054 |
| dropout_ratio | 0.7   | 0.2     | 0.6     | 0.2     | 0.5   | 0.5     | 0.6     |
| graph_pooling | sum   | sum     | sum     | sum     | sum   | sum     | sum     |
| lr_scale      | 1.0   | 1.0     | 1.0     | 1.0     | 1.0   | 1.0     | 1.0     |



CON_HGC_RW (GRAPHSAGE): 

|               | SIDER   | ClinTox | BACE    | HIV     | BBBP    | Tox21   | ToxCast |
| ------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| batch_size    | 128     | 64      | 32      | 32      | 256     | 64      | 64      |
| lr            | 0.00963 | 0.00186 | 0.00095 | 0.00059 | 0.00214 | 0.00068 | 0.00858 |
| dropout_ratio | 0.7     | 0.1     | 0.6     | 0.3     | 0.3     | 0.7     | 0.5     |
| graph_pooling | sum     | sum     | sum     | sum     | sum     | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     |



CON_HGCAM (GRAPHSAGE): 

|               | SIDER   | ClinTox | BACE    | HIV     | BBBP    | Tox21   | ToxCast |
| ------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| batch_size    | 32      | 32      | 64      | 32      | 64      | 32      | 32      |
| lr            | 0.00266 | 0.00131 | 0.00012 | 0.00095 | 0.00223 | 0.00859 | 0.00292 |
| dropout_ratio | 0.7     | 0.0     | 0.5     | 0.0     | 0.6     | 0.6     | 0.6     |
| graph_pooling | sum     | sum     | sum     | sum     | sum     | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     |



CON_HGCAM_RW (GRAPHSAGE): 

|               | SIDER   | ClinTox | BACE    | HIV     | BBBP    | Tox21   | ToxCast |
| ------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| batch_size    | 64      | 64      | 32      | 64      | 32      | 64      | 32      |
| lr            | 0.00663 | 0.00095 | 0.00036 | 0.00234 | 0.00153 | 0.00047 | 0.00445 |
| dropout_ratio | 0.7     | 0.1     | 0.3     | 0.3     | 0.5     | 0.5     | 0.7     |
| graph_pooling | sum     | sum     | sum     | sum     | sum     | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     |

