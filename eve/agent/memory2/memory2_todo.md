
Core:
- inject 'assigned_user' field for user memories and create corresponding user memories in db accordingly
- Create "search_memories" agent tool that uses RAG to retrieve FACTS
- Create "get_old_memories" agent tool that retrieves old reflections from certain timewindow

Optional:
- use the users main LLM models for memory system (tied to subscription tier)
- auto-add word counts to consolidated memory blobs so agent is aware of context usage
- optimize extraction prompts to omit empty memory sections (less distractions, less tokens)

Final:
- completely remove old memory_v1 system