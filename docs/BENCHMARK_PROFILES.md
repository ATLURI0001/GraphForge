# Agent Invocation Latency & Token Consumption Profiles

## Metrics Across Common Topologies
| Graph Topology | Avg Steps | Token Consumption | P95 Latency |
| :--- | :---: | :---: | :---: |
| Single-turn ReAct | 3 | 1,450 tokens | 1.8s |
| Plan-and-Execute | 8 | 4,800 tokens | 4.2s |
| Hierarchical Swarm (3 Workers) | 14 | 9,200 tokens | 7.9s |

## Optimization Techniques
- State filtering: Prune intermediate tool execution artifacts before forwarding state to user-facing synthesizers.
