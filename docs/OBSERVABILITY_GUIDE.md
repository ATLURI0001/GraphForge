# Observability & Tracing Integration Guidelines

## OpenTelemetry Spans
Every graph execution emits standard distributed trace spans:
- `graph.run`: Root span capturing overall execution time and terminal status.
- `node.invoke`: Span for individual node execution with input/output token metrics.
- `tool.call`: Sub-span tracking external tool invocation latency and parameter schemas.

## Redaction Policy
Customer PII and sensitive environment variables are masked via configurable regex filters prior to span export.
