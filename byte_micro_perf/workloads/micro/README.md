# micro architecture testing


## per_core_read
Take NVIDIA A100 as an example, it has 108 SMs, memory bandwidth is 2039 GB/s. 

**per_core_read** will test sequential read performance from 1 sm to 108 sms.

## per_core_write
Take NVIDIA A100 as an example, it has 108 SMs, memory bandwidth is 2039 GB/s.

**per_core_write** will test sequential write performance from 1 sm to 108 sms.



## vector_fma
Not considering memory reading and writing, **vector_fma** will test vector fma performance.

| tensor_dtype | formula | 
| --- | --- |
| float32 | a * b + c |
| float16 | a * b + c |
| bfloat16 | a * b + c |
| int8 | a * b + c |





