# State Graph Checkpointing & Persistent Memory Guide

## State Persistence in LangGraph
GraphForge relies on transactional checkpointing to allow stateful multi-agent workflows to pause, resume, and inspect execution history.

## Checkpointer Implementations
- **MemorySaver**: Fast in-memory state tracking for local development and unit tests.
- **SqliteSaver**: File-backed thread checkpointing with zero cloud dependencies.
- **PostgresSaver**: Production connection pooling with ACID guarantees across distributed agent instances.
