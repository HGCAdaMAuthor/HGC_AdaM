

AM (GCN): 

|               | SIDER   | ClinTox | BACE    | HIV   | BBBP    | Tox21   | ToxCast |
| ------------- | ------- | ------- | ------- | ----- | ------- | ------- | ------- |
| batch_size    | 64      | 32      | 64      | 64    | 32      | 64      | 64      |
| lr            | 0.00593 | 0.0008  | 0.00184 | 0.001 | 0.00769 | 0.00694 | 0.00086 |
| dropout_ratio | 0.7     | 0.1     | 0.6     | 0.4   | 0.4     | 0.5     | 0.5     |
| graph_pooling | sum     | sum     | sum     | sum   | sum     | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0     | 1.0   | 1.0     | 1.0     | 1.0     |



CON_HGC (GCN): 

|               | SIDER   | ClinTox | BACE  | HIV     | BBBP    | Tox21   | ToxCast |
| ------------- | ------- | ------- | ----- | ------- | ------- | ------- | ------- |
| batch_size    | 64      | 32      | 32    | 32      | 64      | 32      | 32      |
| lr            | 0.00221 | 0.00027 | 0.001 | 0.00227 | 0.00115 | 0.00542 | 0.001   |
| dropout_ratio | 0.4     | 0.0     | 0.3   | 0.4     | 0.1     | 0.6     | 0.1     |
| graph_pooling | sum     | sum     | sum   | sum     | sum     | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0   | 1.0     | 1.0     | 1.0     | 1.0     |



CON_HGC_RW (GCN): 

|               | SIDER  | ClinTox | BACE    | HIV     | BBBP    | Tox21   | ToxCast |
| ------------- | ------ | ------- | ------- | ------- | ------- | ------- | ------- |
| batch_size    | 128    | 32      | 32      | 32      | 64      | 64      | 32      |
| lr            | 0.0044 | 0.00044 | 0.00029 | 0.00502 | 0.00043 | 0.00706 | 0.00437 |
| dropout_ratio | 0.6    | 0.0     | 0.2     | 0.0     | 0.1     | 0.5     | 0.4     |
| graph_pooling | sum    | sum     | sum     | sum     | sum     | sum     | sum     |
| lr_scale      | 1.0    | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     |



CON_HGCAM (GCN): 

|               | SIDER   | ClinTox | BACE  | HIV     | BBBP    | Tox21   | ToxCast |
| ------------- | ------- | ------- | ----- | ------- | ------- | ------- | ------- |
| batch_size    | 32      | 64      | 32    | 32      | 32      | 64      | 32      |
| lr            | 0.00874 | 0.001   | 0.001 | 0.00164 | 0.00209 | 0.00101 | 0.001   |
| dropout_ratio | 0.6     | 0.1     | 0.5   | 0.3     | 0.0     | 0.6     | 0.6     |
| graph_pooling | sum     | sum     | sum   | sum     | sum     | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0   | 1.0     | 1.0     | 1.0     | 1.0     |



CON_HGCAM_RW (GCN): 

|               | SIDER   | ClinTox | BACE    | HIV     | BBBP   | Tox21   | ToxCast |
| ------------- | ------- | ------- | ------- | ------- | ------ | ------- | ------- |
| batch_size    | 64      | 32      | 256     | 256     | 256    | 256     | 128     |
| lr            | 0.00925 | 0.0021  | 0.00051 | 0.00086 | 0.0059 | 0.00034 | 0.0033  |
| dropout_ratio | 0.7     | 0.0     | 0.4     | 0.3     | 0.3    | 0.3     | 0.6     |
| graph_pooling | sum     | sum     | sum     | sum     | sum    | sum     | sum     |
| lr_scale      | 1.0     | 1.0     | 1.0     | 1.0     | 1.0    | 1.0     | 1.0     |

