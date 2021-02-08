

AM (GIN): 

|               | SIDER   | ClinTox | BACE  | HIV     | BBBP    | Tox21 | ToxCast |
| ------------- | ------- | ------- | ----- | ------- | ------- | ----- | ------- |
| batch_size    | 256     | 64      | 32    | 32      | 64      | 64    | 32      |
| lr            | 0.00181 | 0.001   | 0.001 | 0.00379 | 0.00723 | 0.001 | 0.00027 |
| dropout_ratio | 0.7     | 0.6     | 0.5   | 0.4     | 0.5     | 0.6   | 0.4     |
| graph_pooling | sum     | sum     | sum   | sum     | sum     | sum   | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0   | 1.0     | 1.0     | 1.0   | 1.0     |



CON_HGC (GIN): 

|               | SIDER   | ClinTox | BACE    | HIV   | BBBP  | Tox21   | ToxCast |
| ------------- | ------- | ------- | ------- | ----- | ----- | ------- | ------- |
| batch_size    | 256     | 256     | 256     | 256   | 32    | 64      | 32      |
| lr            | 0.00462 | 0.00197 | 0.00175 | 0.001 | 0.001 | 0.00053 | 0.001   |
| dropout_ratio | 0.3     | 0.3     | 0.5     | 0.7   | 0.1   | 0.4     | 0.7     |
| graph_pooling | sum     | sum     | sum     | sum   | sum   | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0     | 1.0   | 1.0   | 1.0     | 1.0     |



CON_HGC_RW (GIN): 

|               | SIDER   | ClinTox  | BACE    | HIV   | BBBP   | Tox21   | ToxCast |
| ------------- | ------- | -------- | ------- | ----- | ------ | ------- | ------- |
| batch_size    | 256     | 64       | 128     | 64    | 64     | 256     | 64      |
| lr            | 0.00784 | 0.000555 | 0.00173 | 0.001 | 0.0006 | 0.00421 | 0.00858 |
| dropout_ratio | 0.7     | 0.3      | 0.5     | 0.6   | 0.6    | 0.7     | 0.5     |
| graph_pooling | sum     | sum      | sum     | sum   | sum    | sum     | sum     |
| lr_scale      | 1.0     | 1.0      | 1.0     | 1.0   | 1.0    | 1.0     | 1.0     |



CON_HGCAM (GIN): 

|               | SIDER | ClinTox | BACE    | HIV   | BBBP  | Tox21   | ToxCast |
| ------------- | ----- | ------- | ------- | ----- | ----- | ------- | ------- |
| batch_size    | 32    | 64      | 64      | 32    | 32    | 256     | 32      |
| lr            | 0.001 | 0.001   | 0.00018 | 0.001 | 0.001 | 0.00078 | 0.00519 |
| dropout_ratio | 0.5   | 0.4     | 0.6     | 0.7   | 0.5   | 0.6     | 0.6     |
| graph_pooling | sum   | sum     | sum     | sum   | sum   | sum     | sum     |
| lr_scale      | 1.0   | 1.0     | 1.0     | 1.0   | 1.0   | 1.0     | 1.0     |



CON_HGCAM_RW (GIN): 

|               | SIDER  | ClinTox  | BACE    | HIV     | BBBP  | Tox21 | ToxCast |
| ------------- | ------ | -------- | ------- | ------- | ----- | ----- | ------- |
| batch_size    | 128    | 256      | 32      | 32      | 256   | 64    | 256     |
| lr            | 0.0095 | 0.000555 | 0.00029 | 0.00239 | 0.001 | 0.001 | 0.001   |
| dropout_ratio | 0.7    | 0.6      | 0.2     | 0.4     | 0.1   | 0.6   | 0.5     |
| graph_pooling | sum    | sum      | sum     | sum     | sum   | sum   | sum     |
| lr_scale      | 1.0    | 1.0      | 1.0     | 1.1     | 1.0   | 1.0   | 1.0     |

