# ⚙️ GraphForge

> **Forging intelligent agents, one graph at a time.**

GraphForge is my hands-on workspace for learning and building **AI agents with LangGraph**.

This repository contains experiments, notebooks, agent implementations, MCP integrations, and utilities that I use while exploring the world of **LLM-powered agents and stateful AI workflows**.

---

## 🚀 What is GraphForge?

GraphForge is built around one idea:

**Learn by building.**

Instead of only reading about agent architectures, I use this repository to experiment with:

* 🤖 AI Agents
* 🧠 Stateful workflows
* 🛠️ Tool calling
* 🔄 Agent loops
* 💾 Memory
* 🔌 MCP integrations
* 🧩 LangGraph
* 🦜 LangChain
* 🧪 Experiments and prototypes

The repository will evolve as I explore more advanced agentic patterns.

---

## 📂 Project Structure

```text
GraphForge/
│
├── agents/              # Agent implementations
│
├── images/              # Images and supporting assets
│
├── mcp/                 # MCP experiments and integrations
│
├── notebooks/           # Learning notebooks and experiments
│
├── utils/               # Shared utilities and helpers
│
├── langgraph.json       # LangGraph configuration
├── pyproject.toml       # Project configuration & dependencies
├── uv.lock              # Locked dependencies
├── .env.example         # Environment variable template
└── README.md
```

---

## 🧠 Learning Roadmap

### 01 • Foundations

Start with the fundamentals of building agents.

* LangChain basics
* LangGraph fundamentals
* Models
* Messages
* Tools
* State
* Graphs
* Nodes & edges

### 02 • Agent Workflows

Move from simple chains to stateful agent systems.

* Conditional routing
* Agent loops
* Tool execution
* State management
* Streaming
* Checkpoints
* Memory

### 03 • Advanced Agents

Explore more sophisticated architectures.

* Multi-agent systems
* Human-in-the-loop
* Middleware
* Agent orchestration
* Long-running workflows
* Error handling
* Dynamic workflows

### 04 • MCP

Experiment with the **Model Context Protocol** and connecting agents with external tools and resources.

### 05 • Production

Move experiments toward reliable real-world systems.

* Observability
* Persistence
* Evaluation
* Debugging
* Deployment
* Scalable agent architectures

---

## 🛠️ Tech Stack

| Technology              | Purpose                                 |
| ----------------------- | --------------------------------------- |
| 🐍 **Python**           | Primary language                        |
| 🦜 **LangChain**        | LLM application framework               |
| 🕸️ **LangGraph**       | Stateful agent workflows                |
| 🔌 **MCP**              | Tool and context integration            |
| ⚡ **uv**                | Python package & environment management |
| 🧪 **Jupyter**          | Experiments and learning                |
| 🎨 **LangGraph Studio** | Agent visualization and debugging       |

---

## ⚡ Getting Started

### Clone the repository

```bash
git clone https://github.com/ATLURI0001/GraphForge.git
cd GraphForge
```

### Install dependencies

This project uses `uv` for dependency management.

```bash
pip install uv
uv sync
```

Activate the virtual environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows
.venv\Scripts\activate
```

### Configure environment variables

Create your local environment file:

```bash
cp .env.example .env
```

Then add the required API keys and configuration values.

> ⚠️ Never commit `.env` or any API credentials to GitHub.

---

## 🧪 Running LangGraph

If an agent is configured through `langgraph.json`, start the development server with:

```bash
langgraph dev
```

This allows you to experiment with agents locally and use the LangGraph development interface for testing and debugging.

---

## 📓 Notebooks

The `notebooks/` directory contains experiments and learning material.

Each notebook focuses on a specific concept or agent-building technique.

The idea is to progress from:

```text
Concept
   ↓
Experiment
   ↓
Implementation
   ↓
Agent
   ↓
Real-world Workflow
```

---

## 🔌 MCP

The `mcp/` directory contains experiments around **Model Context Protocol**.

The goal is to understand how agents can interact with external tools, services, and context through standardized interfaces.

---

## 📈 Progress

* [x] Project setup
* [x] LangGraph environment
* [ ] LangGraph fundamentals
* [ ] Agent workflows
* [ ] Tool calling
* [ ] State & memory
* [ ] Human-in-the-loop
* [ ] Multi-agent systems
* [ ] MCP integrations
* [ ] Advanced agent architectures
* [ ] Production workflows

This checklist will evolve as GraphForge grows.

---

## 🎯 Philosophy

> **Don't just use agents. Understand how they work.**

GraphForge is a continuous learning project.

Every notebook, experiment, and agent is another piece of the puzzle toward understanding how reliable AI systems are designed and built.

```text
Learn → Build → Experiment → Break → Debug → Understand → Forge
```

---

## 📚 Resources

* [LangChain Documentation](https://docs.langchain.com/)
* [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
* [Model Context Protocol](https://modelcontextprotocol.io/)
* [LangGraph Studio](https://docs.langchain.com/langsmith/home)
* [LangChain Academy](https://academy.langchain.com/)

---

## ⭐ GraphForge

**A place to experiment with agents, graphs, tools, and ideas.**

Built while learning.
Improved through experimentation.
**Forged one graph at a time. ⚙️**
