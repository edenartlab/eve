# Memory2 UI Implementation Plan

*Created: 2026-01-14*
*Purpose: Implementation plan for memory2 UI alongside new memory2 architecture*

---

## Overview

This plan creates a v2 version of the memory UI that works with the new memory2 architecture while keeping the old system intact for comparison during migration. A URL flag `memory2=true` enables toggling between views.

**Key Constraint**: Keep the same User / Collective (Agent) tab distinction as v1. No session memory UI needed.

---

## Architecture Comparison

| Aspect | Old System | New Memory2 System |
|--------|------------|-------------------|
| **User Memory** | `UserMemory` collection with `content` blob + `unabsorbed_memory_ids` (directives) | `ConsolidatedMemory` (scope_type="user") + `Reflection` records (scope="user") |
| **Collective Memory** | `AgentMemory` shards with `content`, `facts[]`, `unabsorbed_memory_ids` (suggestions), `extraction_prompt` | `ConsolidatedMemory` (scope_type="agent") + `Reflection` records (scope="agent") + `Fact` records |
| **Session Memory** | Episodes only (FIFO buffer, no UI) | `ConsolidatedMemory` (scope_type="session") + `Reflection` records - **NO UI** |
| **Shards** | Multiple per agent | Single consolidated blob per scope |
| **Extraction Prompts** | Per-shard configurable | Global system prompts (not user-editable) |

### MongoDB Collections

**Old System:**
- `memory_user` - UserMemory documents
- `memory_agent` - AgentMemory shard documents
- `memory_sessions` - SessionMemory (raw extracted items)

**New Memory2 System:**
- `memory2_consolidated` - ConsolidatedMemory documents (one per scope)
- `memory2_reflections` - Reflection documents (unabsorbed items)
- `memory2_facts` - Fact documents (for RAG retrieval)

---

## Phase 1: Eve Backend API Routes

### New File: `eve/api/memory2_routes.py`

Create FastAPI routes that proxy to memory2 data:

```python
# User Memory (scope_type="user")
GET  /memory2/agent/{agent_id}/user-memory
     Query: user_id (required)
     Returns: { consolidated_content, last_consolidated_at, unabsorbed_reflections[] }

POST /memory2/agent/{agent_id}/user-memory
     Query: user_id (required)
     Body: { content: string }
     Returns: { success: bool }

# Agent/Collective Memory (scope_type="agent")
GET  /memory2/agent/{agent_id}/agent-memory
     Returns: { consolidated_content, last_consolidated_at, unabsorbed_reflections[], facts[] }

POST /memory2/agent/{agent_id}/agent-memory
     Body: { content: string }
     Returns: { success: bool }

# Reflection CRUD
PATCH  /memory2/reflections/{reflection_id}
       Body: { content: string }
       Returns: { success: bool }

DELETE /memory2/reflections/{reflection_id}
       Returns: { success: bool }

# Fact CRUD
PATCH  /memory2/facts/{fact_id}
       Body: { content: string }
       Returns: { success: bool }

DELETE /memory2/facts/{fact_id}
       Returns: { success: bool }
```

### Response Schemas

**User Memory Response:**
```typescript
interface UserMemory2Response {
  consolidated_content: string
  last_consolidated_at: string | null
  unabsorbed_reflections: Array<{
    _id: string
    content: string
    formed_at: string
  }>
}
```

**Agent/Collective Memory Response:**
```typescript
interface AgentMemory2Response {
  consolidated_content: string
  last_consolidated_at: string | null
  unabsorbed_reflections: Array<{
    _id: string
    content: string
    formed_at: string
  }>
  facts: Array<{
    _id: string
    content: string
    formed_at: string
    scope: string[]
    access_count: number
  }>
}
```

---

## Phase 2: Eden API Backend Routes

### New File: `apps/api/src/routes/v2/memory2Routes.ts`

Create separate routes file for memory2 endpoints:

```typescript
import { FastifyInstance } from 'fastify'
import { Type } from '@sinclair/typebox'
import {
  getAgentMemory2,
  saveAgentMemory2,
  getCollectiveMemory2,
  saveCollectiveMemory2,
  updateReflection2,
  deleteReflection2,
  updateFact2,
  deleteFact2,
} from '../controllers/agentMemory2Controller'

export async function memory2Routes(server: FastifyInstance) {
  // User Memory
  server.get('/v2/agents/:agentId/memory2', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Get User Memory (Memory2)',
      description: 'Retrieve user-scoped consolidated memory and unabsorbed reflections.',
      params: { agentId: Type.String() },
      response: {
        200: Type.Object({
          consolidated_content: Type.String(),
          last_consolidated_at: Type.Union([Type.String(), Type.Null()]),
          unabsorbed_reflections: Type.Array(Type.Object({
            _id: Type.String(),
            content: Type.String(),
            formed_at: Type.String(),
          })),
        }),
      },
    },
    handler: getAgentMemory2,
  })

  server.post('/v2/agents/:agentId/memory2', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Save User Memory (Memory2)',
      params: { agentId: Type.String() },
      body: Type.Object({ content: Type.String() }),
      response: { 200: Type.Object({ success: Type.Boolean() }) },
    },
    handler: saveAgentMemory2,
  })

  // Collective/Agent Memory
  server.get('/v2/agents/:agentId/memory2-agent', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Get Collective Memory (Memory2)',
      params: { agentId: Type.String() },
      response: {
        200: Type.Object({
          consolidated_content: Type.String(),
          last_consolidated_at: Type.Union([Type.String(), Type.Null()]),
          unabsorbed_reflections: Type.Array(Type.Object({
            _id: Type.String(),
            content: Type.String(),
            formed_at: Type.String(),
          })),
          facts: Type.Array(Type.Object({
            _id: Type.String(),
            content: Type.String(),
            formed_at: Type.String(),
            scope: Type.Array(Type.String()),
            access_count: Type.Number(),
          })),
        }),
      },
    },
    handler: getCollectiveMemory2,
  })

  server.post('/v2/agents/:agentId/memory2-agent', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Save Collective Memory (Memory2)',
      params: { agentId: Type.String() },
      body: Type.Object({ content: Type.String() }),
      response: { 200: Type.Object({ success: Type.Boolean() }) },
    },
    handler: saveCollectiveMemory2,
  })

  // Reflection CRUD
  server.patch('/v2/memory2-reflections/:reflectionId', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Update Reflection',
      params: { reflectionId: Type.String() },
      body: Type.Object({ content: Type.String() }),
      response: { 200: Type.Object({ success: Type.Boolean() }) },
    },
    handler: updateReflection2,
  })

  server.delete('/v2/memory2-reflections/:reflectionId', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Delete Reflection',
      params: { reflectionId: Type.String() },
      response: { 200: Type.Object({ success: Type.Boolean() }) },
    },
    handler: deleteReflection2,
  })

  // Fact CRUD
  server.patch('/v2/memory2-facts/:factId', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Update Fact',
      params: { factId: Type.String() },
      body: Type.Object({ content: Type.String() }),
      response: { 200: Type.Object({ success: Type.Boolean() }) },
    },
    handler: updateFact2,
  })

  server.delete('/v2/memory2-facts/:factId', {
    schema: {
      tags: ['Agents - Memory2'],
      summary: 'Delete Fact',
      params: { factId: Type.String() },
      response: { 200: Type.Object({ success: Type.Boolean() }) },
    },
    handler: deleteFact2,
  })
}
```

### New File: `apps/api/src/controllers/agentMemory2Controller.ts`

Controller with handler implementations:

```typescript
import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify'

// Get user-scoped memory (ConsolidatedMemory + Reflections)
export const getAgentMemory2 = async (
  server: FastifyInstance,
  request: FastifyRequest,
  reply: FastifyReply,
) => {
  const { agentId } = request.params as { agentId: string }
  const userId = request.user?.userId

  // Query memory2_consolidated for scope_type="user", agent_id, user_id
  // Query memory2_reflections for scope="user", agent_id, user_id, absorbed=false
  // Return combined response
}

// Save user-scoped consolidated content
export const saveAgentMemory2 = async (...) => {
  // Update memory2_consolidated.consolidated_content
}

// Get agent-scoped memory (ConsolidatedMemory + Reflections + Facts)
export const getCollectiveMemory2 = async (...) => {
  // Query memory2_consolidated for scope_type="agent", agent_id
  // Query memory2_reflections for scope="agent", agent_id, absorbed=false
  // Query memory2_facts for agent_id
  // Return combined response
}

// Save agent-scoped consolidated content
export const saveCollectiveMemory2 = async (...) => {
  // Update memory2_consolidated.consolidated_content
}

// Update reflection content
export const updateReflection2 = async (...) => {
  // Update memory2_reflections document
}

// Delete reflection
export const deleteReflection2 = async (...) => {
  // Delete from memory2_reflections
  // Also remove from ConsolidatedMemory.unabsorbed_ids if present
}

// Update fact content
export const updateFact2 = async (...) => {
  // Update memory2_facts document
  // Re-generate embedding after content change
}

// Delete fact
export const deleteFact2 = async (...) => {
  // Delete from memory2_facts
}
```

---

## Phase 3: Frontend Components

### File Structure

```
apps/media-playground/features/sessions/views/
├── agent-memory-view.tsx          # Existing v1
├── agent-memory-view2.tsx         # NEW - v2 wrapper with toggle
├── collective-memory.tsx          # Existing v1
├── collective-memory2.tsx         # NEW - v2 collective memory
├── user-memory2.tsx               # NEW - v2 user memory component
├── memory-constants.ts            # Existing v1
├── memory2-constants.ts           # NEW - v2 constants
├── memory-edit-confirmation-dialog.tsx  # Shared
└── ...
```

### New File: `memory2-constants.ts`

```typescript
export const MEMORY2_LIMITS = {
  // Consolidated memory blobs
  AGENT_CONSOLIDATED_MAX_CHARS: 8000,   // ~1000 words
  USER_CONSOLIDATED_MAX_CHARS: 3200,    // ~400 words

  // Individual items
  REFLECTION_MAX_CHARS: 280,            // ~35 words
  FACT_MAX_CHARS: 240,                  // ~30 words
} as const

export const MEMORY2_ERROR_MESSAGES = {
  AGENT_CONSOLIDATED_MAX_CHARS: `Agent memory cannot exceed ${MEMORY2_LIMITS.AGENT_CONSOLIDATED_MAX_CHARS} characters`,
  USER_CONSOLIDATED_MAX_CHARS: `User memory cannot exceed ${MEMORY2_LIMITS.USER_CONSOLIDATED_MAX_CHARS} characters`,
  REFLECTION_MAX_CHARS: `Reflection cannot exceed ${MEMORY2_LIMITS.REFLECTION_MAX_CHARS} characters`,
  FACT_MAX_CHARS: `Fact cannot exceed ${MEMORY2_LIMITS.FACT_MAX_CHARS} characters`,
} as const
```

### New File: `agent-memory-view2.tsx`

Main wrapper component with User/Collective tab toggle (same as v1):

```tsx
'use client'

import { FC, useState, useMemo } from 'react'
import { Agent } from '@edenlabs/eden-sdk'
import UserMemory2 from './user-memory2'
import CollectiveMemory2 from './collective-memory2'
import { useAgentPermissions } from '@/hooks/use-agent-permissions'
import { useAuthState } from '@/hooks/use-auth-state'
import { Switch } from '@/components/ui/switch'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'

interface AgentMemoryView2Props {
  agent: Agent
  variant?: 'standalone' | 'panel'
}

const AgentMemoryView2: FC<AgentMemoryView2Props> = ({
  agent,
  variant = 'standalone',
}) => {
  const { user: authUser } = useAuthState()
  const { hasEditAccess } = useAgentPermissions(agent, authUser?._id)
  const [memoryTab, setMemoryTab] = useState<'user' | 'collective'>('user')

  // Same tab toggle UI as v1
  // Render UserMemory2 or CollectiveMemory2 based on tab

  return (
    <TooltipProvider>
      <div className={outerWrapperClass}>
        {/* Tab Toggle - Same as v1 */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setMemoryTab('user')}
            className={/* styling based on active tab */}
          >
            User
          </button>
          <button
            onClick={() => setMemoryTab('collective')}
            className={/* styling based on active tab */}
          >
            Collective
          </button>
        </div>

        {/* Tab Content */}
        {memoryTab === 'user' ? (
          <UserMemory2 agent={agent} hasEditAccess={hasEditAccess} />
        ) : (
          <CollectiveMemory2 agent={agent} hasEditAccess={hasEditAccess} />
        )}
      </div>
    </TooltipProvider>
  )
}

export default AgentMemoryView2
```

### New File: `user-memory2.tsx`

User-scoped memory display:

```tsx
'use client'

import { FC, useEffect, useState } from 'react'
import { Agent } from '@edenlabs/eden-sdk'
import axios from 'axios'
import { Edit, Loader2, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { MEMORY2_LIMITS } from './memory2-constants'

interface Reflection {
  _id: string
  content: string
  formed_at: string
}

interface UserMemory2Props {
  agent: Agent
  hasEditAccess: boolean
}

const UserMemory2: FC<UserMemory2Props> = ({ agent, hasEditAccess }) => {
  const [consolidatedContent, setConsolidatedContent] = useState('')
  const [reflections, setReflections] = useState<Reflection[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await axios.get(`/api/agents/${agent._id}/memory2`)
        setConsolidatedContent(response.data.consolidated_content || '')
        setReflections(response.data.unabsorbed_reflections || [])
      } catch (error) {
        console.error('Error loading memory2 data:', error)
      } finally {
        setIsLoading(false)
      }
    }
    loadData()
  }, [agent._id])

  // Handlers for edit/delete consolidated content and reflections
  // Similar pattern to v1 but hitting /memory2 endpoints

  return (
    <div className="space-y-8">
      {/* Consolidated Memory Section */}
      <div className="rounded-xl border border-border bg-card shadow-sm overflow-hidden">
        <div className="flex items-start justify-between px-5 py-4 border-b">
          <div>
            <p className="text-xs font-semibold uppercase text-muted-foreground">
              Personal Memory
            </p>
            <p className="text-sm text-muted-foreground">
              Context {agent.name} keeps about you
            </p>
          </div>
          {hasEditAccess && (
            <Button onClick={handleEditClick} size="sm" variant="secondary">
              <Edit className="w-4 h-4 mr-1.5" />
              Edit
            </Button>
          )}
        </div>
        <div className="px-5 py-5 whitespace-pre-wrap min-h-[170px] max-h-[312px] overflow-y-auto">
          {consolidatedContent || (
            <span className="text-muted-foreground italic">
              💭 No user memories stored yet.
            </span>
          )}
        </div>
      </div>

      {/* Recent Reflections Section */}
      {reflections.length > 0 && (
        <div className="rounded-xl border border-border bg-card p-6 space-y-6">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold">Recent Memory Context</h2>
            <span className="text-xs bg-muted px-2 py-1 rounded-full">
              Will be integrated soon
            </span>
          </div>
          <div className="grid gap-2">
            {reflections.map(reflection => (
              <div key={reflection._id} className="relative group">
                <div className="rounded-lg border bg-card hover:bg-muted/50 transition-colors">
                  <div className="px-4 py-3 text-sm pr-20">
                    {reflection.content}
                  </div>
                  <div className="absolute top-3 right-3 flex gap-1 opacity-0 group-hover:opacity-100">
                    <Button onClick={() => handleReflectionEdit(reflection)} variant="ghost" size="sm">
                      <Edit className="w-3 h-3" />
                    </Button>
                    <Button onClick={() => handleReflectionDelete(reflection._id)} variant="ghost" size="sm">
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Edit modals - similar to v1 */}
    </div>
  )
}

export default UserMemory2
```

### New File: `collective-memory2.tsx`

Agent-scoped memory display (simpler than v1 - no shards, no extraction prompts):

```tsx
'use client'

import { FC, useEffect, useState } from 'react'
import { Agent } from '@edenlabs/eden-sdk'
import axios from 'axios'
import { Edit, Loader2, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { MEMORY2_LIMITS } from './memory2-constants'

interface Reflection {
  _id: string
  content: string
  formed_at: string
}

interface Fact {
  _id: string
  content: string
  formed_at: string
  scope: string[]
  access_count: number
}

interface CollectiveMemory2Props {
  agent: Agent
  hasEditAccess: boolean
}

const CollectiveMemory2: FC<CollectiveMemory2Props> = ({ agent, hasEditAccess }) => {
  const [consolidatedContent, setConsolidatedContent] = useState('')
  const [reflections, setReflections] = useState<Reflection[]>([])
  const [facts, setFacts] = useState<Fact[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await axios.get(`/api/agents/${agent._id}/memory2-agent`)
        setConsolidatedContent(response.data.consolidated_content || '')
        setReflections(response.data.unabsorbed_reflections || [])
        setFacts(response.data.facts || [])
      } catch (error) {
        console.error('Error loading collective memory2:', error)
      } finally {
        setIsLoading(false)
      }
    }
    loadData()
  }, [agent._id])

  return (
    <div className="space-y-8">
      {/* Consolidated Memory Section */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-1 h-6 bg-blue-400/60 rounded-full"></div>
          <h3 className="text-md font-medium">Memory</h3>
        </div>
        <div className="group relative rounded-lg border bg-blue-50/20 dark:bg-blue-950/15">
          <div className="px-4 py-4 whitespace-pre-wrap min-h-[216px] max-h-[450px] overflow-y-auto">
            {consolidatedContent || (
              <span className="text-muted-foreground italic">
                💭 No collective memory content yet.
              </span>
            )}
          </div>
          {hasEditAccess && (
            <Button
              onClick={handleContentEdit}
              size="sm"
              className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100"
            >
              <Edit className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Recent Reflections (suggestions) Section */}
      {reflections.length > 0 && (
        <div className="space-y-6 mb-8">
          <div className="flex items-center gap-3">
            <div className="w-1 h-6 bg-yellow-400/60 rounded-full"></div>
            <h2 className="text-lg font-semibold">Recent Memory Context</h2>
            <span className="text-xs bg-yellow-100/40 text-yellow-700 px-2 py-1 rounded-full">
              Will be integrated soon
            </span>
          </div>
          <div className="grid gap-2">
            {reflections.map(reflection => (
              <div key={reflection._id} className="relative group">
                <div className="rounded-lg border bg-yellow-50/25 hover:bg-yellow-100/35 transition-colors">
                  <div className="px-4 py-3 text-sm pr-20">
                    {reflection.content}
                  </div>
                  <div className="absolute top-3 right-3 flex gap-1 opacity-0 group-hover:opacity-100">
                    <Button onClick={() => handleReflectionEdit(reflection)} variant="ghost" size="sm">
                      <Edit className="w-3 h-3" />
                    </Button>
                    <Button onClick={() => handleReflectionDelete(reflection._id)} variant="ghost" size="sm">
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Facts Section */}
      {facts.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <div className="w-1 h-6 bg-green-400/60 rounded-full"></div>
            <h2 className="text-lg font-semibold">Facts</h2>
            <span className="text-xs bg-green-100/40 text-green-700 px-2 py-1 rounded-full">
              {facts.length} facts
            </span>
          </div>
          <div className="grid gap-2">
            {facts.map(fact => (
              <div key={fact._id} className="relative group">
                <div className="rounded-lg border bg-green-50/20 hover:bg-green-100/30 transition-colors">
                  <div className="px-4 py-3 text-sm pr-20">
                    {fact.content}
                  </div>
                  <div className="absolute top-3 right-3 flex items-center gap-2 opacity-0 group-hover:opacity-100">
                    {fact.access_count > 0 && (
                      <span className="text-xs text-muted-foreground">
                        accessed {fact.access_count}x
                      </span>
                    )}
                    <Button onClick={() => handleFactEdit(fact)} variant="ghost" size="sm">
                      <Edit className="w-3 h-3" />
                    </Button>
                    <Button onClick={() => handleFactDelete(fact._id)} variant="ghost" size="sm">
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Edit modals - similar to v1 */}
    </div>
  )
}

export default CollectiveMemory2
```

---

## Phase 4: Toggle Integration

### Modify: `agent-settings-modal.tsx`

Add URL-based toggle between v1 and v2:

```tsx
import { useSearchParams } from 'next/navigation'
import AgentMemoryView from './agent-memory-view'
import AgentMemoryView2 from './agent-memory-view2'

// Inside component:
const searchParams = useSearchParams()
const useMemory2 = searchParams.get('memory2') === 'true'

// In render:
{activeTab === 'memories' && (
  useMemory2 ? (
    <AgentMemoryView2 agent={agent} variant="panel" />
  ) : (
    <AgentMemoryView agent={agent} variant="panel" />
  )
)}
```

### Optional: Add Toggle Button in UI

```tsx
// Add a visual toggle to switch between views
<div className="flex items-center gap-2 text-sm">
  <span className={!useMemory2 ? 'font-medium' : 'text-muted-foreground'}>v1</span>
  <Switch
    checked={useMemory2}
    onCheckedChange={(checked) => {
      const url = new URL(window.location.href)
      if (checked) {
        url.searchParams.set('memory2', 'true')
      } else {
        url.searchParams.delete('memory2')
      }
      window.history.pushState({}, '', url.toString())
    }}
  />
  <span className={useMemory2 ? 'font-medium' : 'text-muted-foreground'}>v2</span>
</div>
```

---

## Phase 5: Data Mapping

### Old → New Field Mapping

| Old System | New Memory2 System | UI Display |
|------------|-------------------|------------|
| `UserMemory.content` | `ConsolidatedMemory.consolidated_content` (scope_type="user") | User → Personal Memory |
| `UserMemory.unabsorbed_memory_ids` → directives | `Reflection` records (scope="user", absorbed=false) | User → Recent Memory Context |
| `AgentMemory.content` | `ConsolidatedMemory.consolidated_content` (scope_type="agent") | Collective → Memory |
| `AgentMemory.unabsorbed_memory_ids` → suggestions | `Reflection` records (scope="agent", absorbed=false) | Collective → Recent Memory Context |
| `AgentMemory.facts[]` → fact SessionMemory | `Fact` records | Collective → Facts |
| `AgentMemory.extraction_prompt` | N/A (system-managed) | **Not displayed in v2** |
| `AgentMemory.shard_name` | N/A (single blob per scope) | **Not displayed in v2** |

### Key Simplifications in v2

1. **No shards** - Single consolidated blob per scope instead of multiple shards
2. **No extraction prompts** - System-managed, not user-editable
3. **No shard management UI** - No create/toggle shard functionality
4. **Simpler data model** - ConsolidatedMemory + Reflections + Facts

---

## Phase 6: Implementation Steps

### Step 1: Eve Backend (Python)
```
eve/api/memory2_routes.py  # New file - FastAPI routes
```

### Step 2: Eden API Backend (TypeScript)
```
apps/api/src/routes/v2/memory2Routes.ts           # New file - Route definitions
apps/api/src/controllers/agentMemory2Controller.ts # New file - Handlers
apps/api/src/routes/v2/index.ts                    # Modify - Register routes
```

### Step 3: Frontend Components
```
apps/media-playground/features/sessions/views/
├── agent-memory-view2.tsx    # New - Main wrapper
├── user-memory2.tsx          # New - User tab
├── collective-memory2.tsx    # New - Collective tab
└── memory2-constants.ts      # New - Constants
```

### Step 4: Toggle Integration
```
apps/media-playground/features/agent/agent-settings-modal.tsx  # Modify
```

---

## Phase 7: Testing Checklist

### API Testing
- [ ] GET /memory2 returns user-scoped data correctly
- [ ] POST /memory2 updates consolidated_content
- [ ] GET /memory2-agent returns agent-scoped data + facts
- [ ] POST /memory2-agent updates consolidated_content
- [ ] PATCH/DELETE reflection endpoints work
- [ ] PATCH/DELETE fact endpoints work

### UI Testing
- [ ] URL flag `?memory2=true` switches to v2 view
- [ ] User tab displays consolidated content + reflections
- [ ] Collective tab displays consolidated content + reflections + facts
- [ ] Edit consolidated memory works with confirmation
- [ ] Edit/delete reflections works
- [ ] Edit/delete facts works
- [ ] Loading states display correctly
- [ ] Error handling works

### Migration Comparison
- [ ] Compare v1 and v2 data for same agent/user
- [ ] Verify migrated data displays correctly in v2
- [ ] Toggle between v1/v2 shows consistent information

---

## Summary

This implementation:

1. **Keeps same User/Collective tab structure** as v1
2. **Parallel files** - All v2 code in separate files (`*2.tsx`, `*2.ts`)
3. **URL toggle** - `?memory2=true` switches to new UI
4. **Simpler architecture** - No shards, no extraction prompts
5. **No session UI** - Session memory handled internally only
6. **Clean comparison** - Easy to toggle and compare v1 vs v2 data during migration
