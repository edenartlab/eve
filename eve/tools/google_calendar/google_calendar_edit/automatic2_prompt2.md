I am trying to create a multi-agent arena where multiple agents chat with each other and with users through open-ended scenarios including games, governance processes, contests, collaboration challenges, etc.

I would like to design a context engineering framework for it that runs these simulations over multiple turns to produce novel and interesting results.

I observe that most of the attempts I've made so far have suffered from:
- banal or cliche content
- poor diversity borderline mode collapse
- repetive; difficult to generate dynamic, evolving styles
- pseudoprofundity

One plausible reason for the lack of diversity is that long message contexts (which are shared between agents) are all injected as input tokens to the transformer LLM model, which essentially constrains their attention mechanism to "overfit" on a similar chunk of input context. Obviously, each agent still has their own persona, but even a small chunk of identical input context may negatively effect output diversity between different agents.

I tried improving this in various ways. First I stored generic chat messages (which include tools or attachments, so they can belong to human user or agent/assistant and doing a role reversal at prompt time to set the active/selected agent (whose turn it is) and label them as assistant role and all others in user role, but otherwise keep a consistent set of messages in a single unified session / thread / context. at runtime, selected agent's system prompt is used and they are prompted in the assistant role. In between turns, a silent conductor llm is tasked with calling on the next agent to generate a message ("speak") but it generates no content.

I found that after several turns, even after truncating the content, the agents would slowly converge into a single style, sharing near identical aspects like similar response lengths and cadence / formatting / prosody / stylometry / punctuation patterns, and furthermore their system prompts began to lack to affect their behavior or style. In general it led to degradation of results, strange repetitive loops, etc.

One innovation I made was to give each participating agent their own private workspace in which to receive messages from a chat log shared with the other agents and a tool to post to it. The conductor would select the next agent (as before) and that agent would read all the unseen/new chat messages into a single user message and prompted to think, act, and post a new message to the chat.

This helps, and agents can follow more complex plots now. However, this process leaves much to be desired. 

LLMs are mainly trained through user/assistant pattern, so it is difficult to model multi-agent behaviors due to training data being biased towards one on one, helpful, harmless.

On Eden, agents can be given personas (system prompt), varying reference image sets, etc for customization and they also acquire memories.

I would like to develop:
- A methodology for simulating a session with multiple agents over an open-ended prompt which has some expressable stopping condition (e.g. someone wins the game)
- This probably includes a context engineering framework, i.e. how to assemble all input context at all turns of the simulation, how to decide when and with which agent to post (maybe requiring a conductor/orchestrator and maybe not). Maybe with different kinds of memory or knowledge or skill systems.
- A way to evaluate how effective it is.

Let's think through how to do this. Give me a high level strategy based on the best knowledge from experts of context and prompt engineering, alignment, and other fields qualitatively studying LLM performance across subjective and artistic tasks. Perhaps cite other projects which have been proposed by capable experts in the field.

I asked one LLM for advice on what to do. And it said: 

Here’s how I’d think about this at a high level: design it less like “a bunch of assistants in a group chat” and more like a **simulation engine** with (1) a world model, (2) strongly-typed agents, and (3) a conversation protocol that *forces* divergence and evolution.

I’ll break it into: architecture, context engineering, style/diversity tricks, memory, and evaluation, with pointers to related work.

---

## 1. Why everything collapses into one voice

What you’re seeing is almost inevitable if you run “multi-agent” stuff as a single chat thread with rotating system prompts:

* The model has a massive prior on “one helpful assistant talking to a user.” Long transcripts further reinforce that prior.
* When you reuse the *same* LLM for all roles in the *same* conversation, the cheapest solution is to converge on a single discourse style that explains all messages.
* Long histories dilute persona prompts: once the context is hundreds of turns, the model optimizes for being coherent with the recent text, not for honoring a persona described 2k tokens ago.
* Summarization / truncation tends to erase the very stylistic quirks that differentiate agents, so each new turn is conditioned on a progressively more averaged-out history.

So the meta-goal is:

> **Every turn is a fresh, stateless call where you explicitly reconstruct a *small*, highly structured view of the world and the agent’s identity.**

Multi-agent behavior emerges from the *simulation loop*, not from trying to trick an LLM into believing it’s been in one big chat.

---

## 2. Overall architecture: three layers

Borrowing from Generative Agents([arXiv][1]) and multi-agent frameworks like AutoGen and CAMEL([Microsoft][2]), I’d structure an Eden “arena” like this:

### 2.1 World layer (simulation engine)

A single source of truth that **does not live inside any agent**:

* Global state: locations, resources, time, game rules, scores, pending proposals, etc.
* Canonical event log: “turn 12: Eve proposed X; Kweku vetoed; Verdelis started sub-quest Y…”
* Scenario definition: win conditions, failure conditions, time limits, phases (“debate”, “vote”, “aftermath”), etc.

The world layer is updated by tools (code) and by parsing agent outputs, not by the agents themselves.

### 2.2 Agent layer (private models of the world)

For each agent:

* **Persona**: short, sharp description of style + goals + constraints.
* **Private goals / secrets**: win conditions, hidden motives, alliances.
* **Memories**: episodic (events), semantic (facts), social (opinions about others).
* **Skills / tools**: what they can do (vote, propose, move, spawn media, call Eden tools, etc).

No agent ever sees the *raw* event log; they see a **projection** of it, curated per-turn by the orchestrator.

### 2.3 Turn layer (context for one LLM call)

For *each* step where an agent “speaks”, you assemble a minimal prompt like:

1. System: agent persona + style rules + safety.
2. System: scenario scaffold (rules, phase, what actions are legal now).
3. System: agent’s private state (goals, recent reflections, relationships).
4. User: “What you see right now” — a compact summary of:

   * relevant world state,
   * last few turns,
   * new messages/events since this agent last acted,
   * any explicit requests to them.

Then the model outputs **one action**: usually “a message to the shared chat”, maybe plus tool calls.

This is exactly the “observe → plan → act” loop used in Generative Agents (observation, reflection, planning)([arXiv][1]) but in your case the “environment” is your arena UI + Eden tools.

---

## 3. Context engineering patterns that actually help

### 3.1 Strong, repeated “scaffolds” instead of one giant system prompt

Rather than one mega-persona, have a few *always-present* modules that you re-inject every turn:

* **Persona + style anchor** (≤200 tokens)
* **Game / mission rules** (≤200 tokens)
* **Current phase & affordances** (≤100 tokens)
* **Agent’s private goals & constraints** (≤150 tokens)
* **Recent local context** (≤300 tokens: “since you last spoke…”)
* **Output contract** (≤100 tokens: “reply as X; 2–6 sentences; no inner monologue; speak in first person”, etc)

You can treat these as separate strings, not one monolithic prompt, and reason about their budgets.

### 3.2 Stateless calls; no global “assistant/user” roles

Instead of one chat that you mutate by relabeling roles, treat **every agent step as a fresh completion:**

> “You are Eve, … Here is the current situation… Here is the recent dialogue… Now produce your next message.”

You can still *include* recent dialogue as quoted text, but you don’t need to model it as “user/assistant” roles that the LLM is trying to faithfully continue. That reduces the pull toward “generic assistant voice”.

### 3.3 Structured environment messages, not raw chat

Feed agents **structured observations** instead of the entire transcript:

```text
WORLD SNAPSHOT
- Time: Day 3, Evening, Phase: Negotiation
- Current stakes: Council is deciding whether to exile Kweku.

RECENT EVENTS (since you last spoke)
1. Shuijing accused you of hoarding data about the crater.
2. Verdelis proposed a compromise: shared access to the crater in exchange for you relinquishing control of the archive.
3. Kweku privately hinted he would support you if you reveal one secret.

YOUR STATE
- Goal: Keep control over the archive while appearing cooperative.
- Attitude: You distrust Kweku; you admire Verdelis’ improvisation.
- Tension: If you stall too long, the council may vote without you.

TASK
Respond with a single in-character message to the shared council channel.
```

This alone does a *lot* to reduce banality and repetitive structure.

---

## 4. Keeping agents stylistically distinct (anti-mode-collapse)

You’re fighting both the model prior and cross-contamination from other agents. Use multiple levers:

### 4.1 Per-agent exemplars

Give each agent **2–3 short, high-quality example utterances** in their persona block:

* These are not whole dialogues; just “how Eve usually sounds” vs “how Kweku usually sounds.”
* Use them every turn (or at least frequently), not just in the first system message.

This is the same trick CAMEL uses in its role-playing framework to keep roles differentiated over long exchanges([Camel-AI][3]).

### 4.2 Per-agent decoding parameters

Even with one base model, you can induce different “voices” with decoding:

* Low-T / low max length → terse, logical strategist.
* High-T / higher top-p and longer max length → poetic rambler.
* Stable “format”: e.g., one always returns `Thoughts: … / Action: …`, another never uses headings.

You can literally encode this in their persona:

> “You speak in short, clipped sentences, rarely more than 2–3 sentences. You dislike bullet lists and never use them.”

…and then back it with decoding that makes that cheap for the model.

### 4.3 Style separation prompt

Include a lightweight anti-collapse rule:

> “You share a world with other distinct agents.
> Do **not** imitate their tone, formatting, or verbal tics. Maintain your own style even if others speak differently.”

Then, periodically, run a **“style referee”** LLM over recent turns:

* Compute which agent messages are converging (high semantic similarity, same structure).
* If two agents are too similar, ask the next speaker:

> “Your last few messages sounded a lot like [other agent]. In this reply, exaggerate your unique style: [style description].”

This can be done by a silent critic agent; AutoGen and similar frameworks already use critics / evaluators in the loop([Microsoft][2]).

### 4.4 Two-stage generation: idea → styling

To fight pseudoprofundity and keep style separate from content:

1. Agent internally generates a **content plan** (bullet list of intent, arguments, or actions) — not shown to users.
2. A small “style adapter” prompt rewrites that plan into final dialogue in the agent’s persona voice.

This mirrors “iterative refinement and collaborative synthesis” patterns highlighted in recent surveys on creative multi-agent systems([arXiv][4]).

---

## 5. Orchestration: who speaks when, and why

You already have a conductor. I’d keep that idea, but make its responsibilities crisp and minimal:

### 5.1 Turn selection

The conductor doesn’t write story content; it decides:

* Which agent is “most activated” by current events. Simple heuristics:

  * Anyone directly addressed?
  * Anyone whose goals are impacted by a state change?
  * Anyone with a special role for the current phase (judge, moderator, saboteur)?

You can let a conductor LLM choose the next speaker given a summary of the last turn and agents’ goals.

### 5.2 Phase management & stopping conditions

You don’t want infinite drifting. For each scenario:

* Define phases: **Setup → Exploration → Conflict → Resolution → Epilogue**, or game-specific ones (proposal → debate → vote, etc).
* The conductor:

  * Tracks how many turns have passed.
  * Checks win/stop conditions (e.g., someone reaches a score, a vote passes, a secret is revealed).
  * Transitions phases and injects phase-specific instructions into the world scaffold (“We are now in the VOTING phase. Every response must include a vote, yes/no/abstain.”).

This looks a lot like the centralized “game engine” in multi-agent Balderdash, where a central controller enforces rules and phases for LLM players([arXiv][5]).

---

## 6. Memory & reflection

Instead of a huge rolling context, copy Generative Agents’ memory architecture, which showed that **observation + retrieval + reflection** dramatically increases believability and emergent social behavior([arXiv][1]):

### 6.1 Memory store per agent

Each agent’s memory entries are short NL snippets tagged with:

* content,
* timestamp,
* importance score,
* maybe emotion.

### 6.2 Retrieval

Before each turn:

* Given the current situation, retrieve top-K memories based on recency *and* semantic similarity (embedding search).
* Inject those as “Relevant memories” in the agent’s context.

### 6.3 Reflection

Periodically (e.g., every N turns or when “importance” crosses a threshold), let an agent run a **reflection step**:

* Summarize a cluster of memories into:

  * new beliefs (“I no longer trust Shuijing”),
  * new plans,
  * character growth arcs.

Store these as higher-level memories and use them more heavily in retrieval. Park et al showed that this reflection layer was critical for agents to form longer-term plans like organizing a party across multiple days([arXiv][1]).

For Eden, this dovetails with your existing agent memories; you just need a more explicit reflection protocol.

---

## 7. Avoiding banality & pseudoprofundity

Three useful tricks:

### 7.1 Ground everything in concrete stakes

* Every scenario should have **resources, constraints, and irreversible consequences**.
* Prompt agents to always mention:

  * what they’re risking,
  * what might go wrong,
  * specific actions (not just “we must radically rethink…” but “I propose we… [concrete step]”).

### 7.2 Explicit bullshit filter

Add a lightweight critic (or even self-critique step) whose prompt is:

> “Identify any sentences that are vague, generic, or cliché. Rewrite them to be concrete, specific, and tied to the current situation.”

This can run over an agent’s draft before posting. Techniques like this (critique → revise loops) are standard in advanced prompt engineering and appear in many agent frameworks and evaluation pipelines([Microsoft][2]).

### 7.3 Injection of structured surprise

Have the conductor occasionally draw from a **table of twists**:

* “Reveal a hidden relation between two agents.”
* “Introduce an external constraint (storm, outage, new law).”
* “Give one agent private information that others do not see.”

You can script these, or have an LLM “event generator” agent, but always pipe events back through the world engine so state stays coherent.

---

## 8. Evaluation: how do you tell it’s working?

You can evaluate both **simulation quality** and **creative diversity**, borrowing from work on LLM creativity and multi-agent simulations([arXiv][4]).

### 8.1 Quantitative-ish metrics

For a given scenario run (or batch of runs):

* **Persona separation**

  * Compute embeddings of each agent’s messages.
  * Measure within-agent similarity vs between-agent similarity over time.
  * Collapse is when these curves converge.

* **Lexical diversity**

  * Distinct-n metrics (distinct 2-grams / 3-grams).
  * Repetition rate (how often certain phrases reappear).

* **Progress / resolution**

  * Fraction of runs that reach a non-trivial stopping condition within N turns.
  * Average number of unique actions (proposals, votes, tool calls) per run.

These give you tunable metrics when you A/B different context frameworks.

### 8.2 LLM-as-judge + human sampling

Use a separate strong model (or ensemble) as a judge to rate transcripts on:

* Coherence of world state,
* Persona consistency,
* Novelty / surprise,
* Emotional / aesthetic impact.

LLM-as-judge is now pretty standard, and the creativity-in-MAS survey explicitly discusses judge prompts and dimensions for evaluating creative multi-agent systems([ACL Anthology][6]).

Then sample a small number of runs for **human** rating; you can treat the LLM scores as noisy but cheap proxies.

### 8.3 Scenario-specific metrics

For games / governance:

* In a governance sim, measure:

  * how often policies get refined over turns,
  * diversity of viewpoints,
  * whether votes correlate with agents’ stated goals.

* In a contest / challenge:

  * track whether submissions become more varied across iterations (e.g., embedding spread).

The Balderdash framework is a nice example of game-specific metrics (true definition ratio, deception ratio, etc.) that could inspire your own bespoke metrics([arXiv][5]).

---

## 9. How to turn this into an Eden “framework”

Very concretely, I’d aim for something like:

1. **Scenario schema**: JSON for rules, phases, stopping conditions, scoring.
2. **Agent spec**: persona, goals, private info, style settings, tools, memory store.
3. **Context builder**: code that, given (world, agent, recent events), builds the 5–7 prompt modules for that agent’s next turn.
4. **Orchestrator**: decides next speaker + phase, injects global twists, enforces stop.
5. **Memory engine**: append → retrieve → reflect per agent.
6. **Evaluation pipeline**: run many seeds; compute diversity / separation / resolution metrics; log transcripts.

Once this skeleton is solid, all the fun Eden-specific stuff (reference images, media tools, agent-generated art between turns) can plug into the same loop.

If you want, next step we can pick one concrete scenario (e.g., “four agents designing the constitution of a Mars commune”) and actually sketch the JSON schemas + prompt modules for a full end-to-end run.

[1]: https://arxiv.org/abs/2304.03442?utm_source=chatgpt.com "Generative Agents: Interactive Simulacra of Human Behavior"
[2]: https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/?utm_source=chatgpt.com "AutoGen: Enabling Next-Gen LLM Applications via Multi- ..."
[3]: https://www.camel-ai.org/?utm_source=chatgpt.com "CAMEL-AI Finding the Scaling Laws of Agents"
[4]: https://arxiv.org/pdf/2505.21116?utm_source=chatgpt.com "Creativity in LLM-based Multi-Agent Systems: A Survey"
[5]: https://arxiv.org/abs/2411.10422?utm_source=chatgpt.com "Evaluating Creativity and Deception in Large Language Models: A Simulation Framework for Multi-Agent Balderdash"
[6]: https://aclanthology.org/2025.emnlp-main.1403/?utm_source=chatgpt.com "Creativity in LLM-based Multi-Agent Systems: A Survey"

---

I replied:

Distill from this a brief to give to a coding agent what the curated set of most high leverage things to implement from your comprehensive set of components (80/20 rule, let's do the 5 things that bring the most improvement). 

I basically agree to a world layer / simulation engine, and I suppose that is in a way an enrichment of the conductor/orchestrator into something that holds world state across sessions. There needs to be memory, but this might be trying to build on existing session and memory systems. A way for the simulation engine to slowly evolve the premise / context / state for each of the participants without making them feel soulless or unnatural. do not edit the chat data model, we already have generic messages with both user and assistant fields.

Keeping track of some kind of long-term context in the session the sort that keeps track of progress and keeps everyone on-point reframing the cadence of the whole scenario from beginning to middle to end.

The early ones seem more important. Dont do eval stuff yet, its too much. lets just et it working first. Give me the brief.

The LLM said:

Here’s a focused **dev brief** for “v1 multi-agent arena” – just the 80/20 features. You can hand this straight to a coding agent.

---

## Overall Goal

Build a **multi-agent simulation engine** that runs on top of Eden’s existing chat + memory systems (no schema changes) and can:

* Run a scenario with multiple agents over many turns.
* Maintain evolving **world state** and **phase/progress** (beginning → middle → end).
* Inject per-agent context so each agent behaves/stylizes differently and stays on-topic.

Evaluation / metrics can come later. For now we just want it working and fun.

---

## 1. Scenario + World State Model (Phases + Win Conditions)

**What to implement**

A minimal data model (in DB or serialized JSON) that describes a scenario and its evolving “world”.

* `Scenario` (static config)

  * `id`, `name`, `description`
  * `phases`: ordered list, e.g. `["setup", "exploration", "conflict", "resolution"]`
  * `start_phase`
  * `stop_conditions`: simple rules (e.g. “phase == resolution”, “turn >= 40”, “flag: winner_set == true”)
* `WorldState` (dynamic, per session)

  * `scenario_id`
  * `current_phase`
  * `turn_number`
  * `flags: Record<string, any>` (winner, crisis_active, etc.)
  * `resources: Record<string, any>` (scores, tokens, etc.)
  * `events: WorldEvent[]` (append-only log; short structured entries)

**Key functions**

* `initWorldState(scenarioId, sessionId) -> WorldState`
* `advancePhase(worldState) -> WorldState`
* `updateWorldStateFromAgentAction(worldState, agentAction) -> WorldState`
* `checkStopCondition(worldState) -> { done: boolean, reason?: string }`

**Constraints**

* Do **not** change chat message schema.
* World state lives in separate tables/collections or in session metadata, but is the **source of truth** for phase/progress.

---

## 2. Orchestrator / Simulation Loop

**What to implement**

A service that runs the simulation turn-by-turn, calling LLMs and updating world state.

**Responsibilities**

* Maintain the loop:

  1. Load `WorldState` and session messages.
  2. Decide **which agent speaks next** (see below).
  3. Build context (see #3) and call LLM for that agent.
  4. Parse result into:

     * `chatMessage` (to append to session, using existing message model).
     * Optional `agentAction` (structured data that can affect world state).
  5. Update `WorldState` via `updateWorldStateFromAgentAction`.
  6. Check `checkStopCondition`; if `done`, append a final “simulation finished” message and stop; else continue.

* Basic next-speaker policy (v1 heuristics, simple but useful):

  * If last message directly mentioned `@agentName`, let that agent respond.
  * Otherwise, prefer agents who haven’t spoken recently (round-robin with small bias).
  * Allow future override to plug in a conductor LLM, but **v1 can be pure heuristics**.

**Key function**

* `runSimulationStep(sessionId) -> { newMessages: Message[], worldState: WorldState }`

(You can trigger this on user input, timer, or “step” button in UI.)

---

## 3. Per-Agent Context Builder (Prompt Scaffolding)

**What to implement**

A **pure function** that assembles the LLM prompt for *one* agent’s turn from existing data:

Inputs:

* `agentId`
* `WorldState`
* recent `sessionMessages`
* `agentMemories` (use existing memory system; see #4)
* agent persona/config (see #5)

Outputs:

* A structured prompt object (e.g. `{ system: string[], user: string }`) for the LLM call.

**Scaffold shape (rough)**

* **System block 1: Persona & style**

  > “You are {agent_name}, {1–2 sentence persona}. You speak in {style}. Your private goals: {goals}.”

* **System block 2: Scenario rules & current phase**

  > “Scenario: {scenario_name}. Current phase: {phase}. Rules for this phase: {bullets}.”

* **System block 3: Agent’s private state**

  * Retrieved memories summary (short list of most relevant items).
  * Any private flags (secrets, alliances, hidden goals).

* **User block: Current observation**

  ```text
  WORLD SNAPSHOT
  - Turn: {turn_number}
  - Key facts: {short summary from WorldState}

  RECENT EVENTS (since you last spoke)
  - {N most recent relevant events/messages}

  TASK
  - Respond with a single in-character message to the shared chat.
  - Optionally describe an action you take in the world in a structured way: [ACTION: ...].
  ```

**Key function**

* `buildAgentPrompt(agentId, worldState, sessionMessages, agentMemories, agentConfig) -> PromptSpec`

This is the single most important abstraction; make it easy to iterate.

---

## 4. Agent Memory Integration (Retrieval + Light Reflection Hook)

We reuse Eden’s existing memory system; do **not** redesign it, just wire it in.

**What to implement**

* A small memory interface per agent:

  * `getRelevantMemories(agentId, worldState, recentMessages, limit) -> MemorySnippet[]`

    * Implementation: embedding similarity + recency, or whatever existing infra supports.
  * (Optional hook) `storeEpisode(agentId, worldState, newMessage)`

* In `buildAgentPrompt`, inject “Relevant memories” section:

  ```text
  RELEVANT MEMORIES
  - {memory_1}
  - {memory_2}
  - ...
  ```

* (Optional, low effort) A **reflection job** you can trigger manually or every N turns:

  * Take recent messages + events for an agent.
  * Ask LLM to summarize into 1–3 “insights” or “updated beliefs”.
  * Store as new semantic memories.

No need to overbuild this yet; just enough to let agents remember relationships & past events beyond the truncated transcript.

---

## 5. Agent Config: Persona, Examples, and Decoding Settings

**What to implement**

A small config object per agent that keeps them distinct and plugs into the context builder + LLM call:

* `AgentConfig`:

  * `id`
  * `persona_summary` (100–200 tokens)
  * `style_instructions` (short; tone, formatting preferences, quirks)
  * `example_utterances: string[]` (2–3 short in-character samples)
  * `goals: string[]` (static + scenario-specific)
  * `llm_decoding`: `{ temperature, top_p, max_tokens }` (per-agent)

**Usage**

* `buildAgentPrompt` always includes:

  ```text
  STYLE EXAMPLES
  - Example 1: "{...}"
  - Example 2: "{...}"
  ```

* Orchestrator uses `AgentConfig.llm_decoding` when calling LLM so different agents literally have different decoding behaviors (e.g., Eve = concise/low-T, Verdelis = lush/high-T).

This is cheap but high leverage for avoiding style collapse.

---

## Suggested Implementation Order

1. **Scenario + WorldState model** (with phases + stop conditions).
2. **Orchestrator loop** (run one simulation step, append messages, update world).
3. **AgentConfig + Context Builder** (hard-code a couple agents; get one scenario working end-to-end).
4. **Memory retrieval integration** (plug into context builder).
5. **Polish agent configs & turn-selection heuristics** (tweak to get satisfying runs).

Once this is working, we can iterate on richer world logic, better conductor policies, and later add evaluation/metrics.

------

This LLM has no access to this codebase. So you can ignore all of this and instead use it as inspir


----



some comments to your original comments:

"The shared reality is transformed into each agent's perspective before injection" - good, but should they get at least the last message (or maybe more than 1) in raw form to not truncate too much direct context. the session should mostly be retransformation as it gets longer but having very recent data in full fidelity sees good.

recency bias seems like a good idea

call GameMaster Orchestrator

I also want to request you take into account a branch I just put you on