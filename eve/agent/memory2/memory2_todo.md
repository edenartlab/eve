
Urgent:


- disable agent memory when user loses pro tier (run check in ui when loading agent and update flag)

- verify agent creation flow (form and conversation --> user memory on? agent memory depends on subscription tier?) and make sure memory2 stuff all works: prompts no longer needed, but two booleans should be set depending on subscription tier.

- move consolidation word limits and other settings to db instead of hardcoded



Next:
- Create "search_memories" agent tool that uses RAG to retrieve FACTS
- Create "get_old_memories" agent tool that retrieves old reflections from certain timewindow



Final:
- completely remove old memory_v1 system in both eve and eden


Remove memory1 from UI:

 Current Architecture

  The memory UI has a toggle controlled by ?memory1=true query parameter:                                                                                                           
  - Default (memory2): Uses AgentMemoryView2 → UserMemory2 + CollectiveMemory2                                                                                                      
  - Legacy (memory1): Uses AgentMemoryView → embedded user memory + CollectiveMemory                                                                                                
                                                                                                                                                                                    
  ---                                                                                                                                                                               
  🔴 LEGACY Components (Can Be Deprecated)                                                                                                                                          
  ┌───────────────────────┬──────────────────┬─────────────────────────────────────────────────┐                                                                                    
  │         File          │    Component     │                     Status                      │                                                                                    
  ├───────────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                                    
  │ agent-memory-view.tsx │ AgentMemoryView  │ Legacy - only loaded via ?memory1=true          │                                                                                    
  ├───────────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                                    
  │ collective-memory.tsx │ CollectiveMemory │ Legacy - only imported by agent-memory-view.tsx │                                                                                    
  └───────────────────────┴──────────────────┴─────────────────────────────────────────────────┘                                                                                    
  These files contain:                                                                                                                                                              
  - Old user memory UI embedded directly in AgentMemoryView                                                                                                                         
  - Old collective memory component (CollectiveMemory)                                                                                                                              
  - Still functional via URL param, but not the default experience                                                                                                                  
                                                                                                                                                                                    
  Imported by (can be removed):                                                                                                                                                     
  - agent-settings-modal.tsx - conditional render behind useLegacyMemory                                                                                                            
  - agent-edit-view.tsx - still imports AgentMemoryView                                                                                                                             
  - sessions-sidebar.tsx - still imports AgentMemoryView                                                                                                                            
                                                                                                                                                                                    
  ---                                                                                                                                                                               
  🟢 ACTIVE Components (memory2 System)                                                                                                                                             
  ┌─────────────────────────────────────┬───────────────────┬───────────────────────────────────────────────┐                                                                       
  │                File                 │     Component     │                    Status                     │                                                                       
  ├─────────────────────────────────────┼───────────────────┼───────────────────────────────────────────────┤                                                                       
  │ agent-memory-view2.tsx              │ AgentMemoryView2  │ Active - default container with tabs          │                                                                       
  ├─────────────────────────────────────┼───────────────────┼───────────────────────────────────────────────┤                                                                       
  │ user-memory2.tsx                    │ UserMemory2       │ Active - user memory with toggle              │                                                                       
  ├─────────────────────────────────────┼───────────────────┼───────────────────────────────────────────────┤                                                                       
  │ collective-memory2.tsx              │ CollectiveMemory2 │ Active - collective memory with toggle        │                                                                       
  ├─────────────────────────────────────┼───────────────────┼───────────────────────────────────────────────┤                                                                       
  │ memory2-constants.ts                │ Utilities         │ Active - shared constants & formatRelativeAge │                                                                       
  ├─────────────────────────────────────┼───────────────────┼───────────────────────────────────────────────┤                                                                       
  │ memory-edit-confirmation-dialog.tsx │ Dialog            │ Shared by both systems                        │                                                                       
  └─────────────────────────────────────┴───────────────────┴───────────────────────────────────────────────┘                                                                       
  ---                                                                                                                                                                               
  Deprecation Actions                                                                                                                                                               
                                                                                                                                                                                    
  1. Remove legacy toggle: Delete useLegacyMemory logic and ?memory1=true support from agent-settings-modal.tsx                                                                     
  2. Delete legacy files:                                                                                                                                                           
    - agent-memory-view.tsx                                                                                                                                                         
    - collective-memory.tsx                                                                                                                                                         
  3. Update remaining imports:                                                                                                                                                      
    - agent-edit-view.tsx:11 - switch to AgentMemoryView2                                                                                                                           
    - sessions-sidebar.tsx:30 - switch to AgentMemoryView2                                                                                                                          
  4. Clean up unused imports:                                                                                                                                                       
    - Remove AgentMemoryView import from agent-settings-modal.tsx                                                                                                                   
                                                                                                                                                                
  ---                                                                                                                                                                         
  The backend endpoints (/user-memory-enabled, /agent-memory-enabled) are shared by both systems - no backend deprecation needed. The user_memory_enabled flag is correctly checked 
  in agentMemoryController.ts:917 before querying user memory.                                                                                                                      
