# Tuning the model servers

Status: working note. This document records how to measure an indexing build, and what the measurements mean. It applies to any client that sends embedding or rerank requests to these servers.

## Two instruments are necessary

A tuning decision needs a record from the client and a record from the server. A record from one side only cannot show which side sets the limit.

### The client: the duration of each phase

The client writes one line for each unit of work. The line gives the duration of the three phases: read, embed, and insert.

```text
index: page 12/97 indexed=500 failures=0 read=118ms embed=3810ms insert=942ms
```

Sum the mean of each phase. Divide that sum by the wall clock. The result is the number of pages that were in flight. A high number with a flat token rate shows a limit in another part of the system.

### The server: the vLLM statistics line

A vLLM server writes one line at each interval.

```text
Engine 000: Avg prompt throughput: 55319.7 tokens/s, Running: 123 reqs, Waiting: 5 reqs
```

- `Avg prompt throughput` is the token rate of the interval. It is an average, not a sample.
- `Running` is the number of sequences in the current step. One embedding request with 128 inputs becomes approximately 128 sequences. This number is not the number of client requests.
- `Waiting` is the number of sequences that cannot start. A queue is present when this number is more than zero.

## What the numbers tell you

1. `Waiting` is zero at each interval. The client does not offer more work than the servers can start. The servers are not the limit. Do not add concurrency: more requests only make each page slower.
2. `Waiting` is more than zero at some intervals. The servers cannot start all the work. The servers are the limit. More concurrency adds wait time only.
3. Two replicas of one model show the same token rate. The router divides the requests equally. A large difference between the replicas usually shows a registration defect, not a hardware defect. See "One container, one entry".
4. An idle server has no work, not more capacity. The GPU graph shows empty periods when the client has no request ready. More server instances cannot fill these periods. More client concurrency can fill them, until item 2 becomes true.

## Batch limits

- vLLM sets the token budget of one step. For a pooling model (an embedding model) it sets the budget to 32768 tokens. You do not have to set this value.
- `--max-batch-tokens` is a Text Embeddings Inference (TEI) option. vLLM does not have this option. Do not put it on a vLLM service.
- The step limit is a token count, not a text count. One client batch of 128 texts of 300 tokens is 38400 tokens, which is more than one step. A larger client batch therefore changes nothing when the token budget is already the limit.

## Health checks

- The vLLM `/health` endpoint answers 200 when the process listens. It does not show that the model is loaded. It is not a readiness test.
- The `/v1/models` endpoint answers 503 until the model is ready. Use this endpoint when a client must wait for a loaded model.
- Compose refuses a file that has `condition: service_healthy` on a service without a healthcheck. Add a healthcheck to that service, or use the default start order.

## One container, one entry

The router can receive backends from two sources: a static list and the Docker socket. This is dangerous.

- Compose names a container `<project>-<service>-<index>`. The network alias of the container is the service name.
- Two sources gave one container two names: `nano` and `inference-nano-1`. Both names resolved to the same container.
- The rotation used `turn % urls.length`. Three entries gave one container two of every three requests. The second container served at half rate, and the token rates showed a ratio of 1.9 to 1.
- The identity of a container is its compose service label, `com.docker.compose.service`. Name a backend by that label. A discovered container and a static entry then agree, and the list has one entry for each container.

Do not keep a static list beside discovery. Add the service to the compose file and start it again.

## Procedure

1. Measure the client phases and the server token rate with the current settings.
2. Change one item.
3. Measure again. Compare the mean of each phase, not the wall clock only. Two runs with the same settings can differ by 10 percent or more.
4. Stop when the mean of a phase stops to fall.
