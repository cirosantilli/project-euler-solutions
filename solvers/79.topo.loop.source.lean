import solvers.«79.topo.iteration.source»

/-! The source while loop terminates by removing one remaining vertex at each successful pop. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def loop : Stmt := .whileDo (.name "heap") loopBody []

/-- Removing a ready vertex leaves the outgoing neighbors of every other vertex unchanged. -/
theorem successors_remove_ready (removed source : String) (edges : List (String × String))
    (ready : Ready removed edges) (different : source ≠ removed) :
    successors source (removeNode removed edges) = successors source edges := by
  unfold successors removeNode
  rw [List.filter_filter]
  congr 1
  apply List.filter_congr
  intro edge member
  have notRemoved := ready edge member
  by_cases matching : edge.1 = source <;> simp [matching, different, notRemoved]

theorem enter_loop (choices : ChoiceSchedule) (base : Mutable.Heap) (env : HeapCalls.Env)
    (vertex : String) (rest output : List String) (degrees : List (Constant × Mutable.Value))
    (getQueue : env "heap" = some (.ref base.length))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (loop :: following) env, stack, workspace base (vertex :: rest) output degrees⟩,
        used⟩
      ⟨⟨.block (loopBody ++ loop :: following) env, stack,
        workspace base (vertex :: rest) output degrees⟩, used⟩ := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next rfl
  apply ((Evaluates.lookup (module := program) (heap := workspace base (vertex :: rest)
    output degrees) getQueue) anchors _).trans
  exact .next (by
    simp [HeapCalls.step, run, Mutable.truth, workspace, values, loop,
      loopBody, popBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt]) .refl

/-- The completed output agrees with the greedy model, including partial output before a cycle. -/
theorem run_loop (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (nodes : List String) (edges : List (String × String))
    (nodesUnique : nodes.Nodup) (edgesUnique : edges.Nodup)
    (endpoints : ∀ edge ∈ edges, edge.1 ∈ nodes ∧ edge.2 ∈ nodes)
    (vertices : ∀ vertex ∈ nodes, vertex ∈ state.vertices)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (queueContents : queue.Perm (ProjectEulerStatements.TopologicalOrder.eligible nodes edges))
    (ordered : Ordered queue)
    (adjacency : ∀ vertex ∈ nodes, (state.neighbors vertex).Perm (successors vertex edges))
    (counted : Counts degrees (fun target => (indegree target edges : Int)) state.vertices)
    (env : HeapCalls.Env) (getQueue : env "heap" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getAdjacency : env "adj" = some (.ref roots.adjacency))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ emitted nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (loop :: following) env, stack, workspace base queue output degrees⟩, used⟩
        ⟨⟨.block following finalEnv, stack, workspace base [] (output ++ emitted) nextDegrees⟩,
          finalUsed⟩ ∧
      emitted.length ≤ nodes.length ∧
      order nodes edges = (if emitted.length = nodes.length then some emitted else none) ∧
      (∀ name, name ≠ "u" → name ≠ "v" → finalEnv name = env name) := by
  cases queue with
  | nil =>
      have noEligible : ProjectEulerStatements.TopologicalOrder.eligible nodes edges = [] :=
        List.length_eq_zero_iff.mp queueContents.length_eq.symm
      have noChoice : choose nodes edges = none := by
        change (ProjectEulerStatements.TopologicalOrder.eligible nodes edges).min? = none
        rw [noEligible]
        rfl
      refine ⟨[], degrees, env, used, ?_, by simp, ?_, ?_⟩
      · apply SetCalls.Reaches.of_heap
        simpa only [List.append_nil] using
          (BlockContinues.while_loop (body := loopBody) (otherwise := [])
            (Evaluates.lookup (module := program) (heap := workspace base [] output degrees)
              getQueue)
            (show Mutable.truth (.ref base.length) (workspace base [] output degrees) =
              some (.ok false, workspace base [] output degrees) by
                simp [Mutable.truth, workspace, values]) BlockContinues.nil
            following anchors stack)
      · by_cases empty : nodes = []
        · simp [order, empty]
        · have positive : nodes.length ≠ 0 := by simpa using empty
          rw [order, if_neg empty, noChoice]
          simp [Ne.symm positive]
      · intros; rfl
  | cons vertex rest =>
      have chosen := head_eq_choose nodes edges vertex rest queueContents ordered
      have selected := choose_spec nodes edges vertex chosen
      have member := selected.1
      have shorter : (nodes.erase vertex).length < nodes.length := by
        rw [List.length_erase_of_mem member]
        have := List.length_pos_of_mem member
        omega
      have nextEndpoints : ∀ edge ∈ removeNode vertex edges,
          edge.1 ∈ nodes.erase vertex ∧ edge.2 ∈ nodes.erase vertex := by
        intro edge retained
        have filtered := List.mem_filter.mp retained
        have different := of_decide_eq_true filtered.2
        have previous := endpoints edge filtered.1
        exact ⟨nodesUnique.mem_erase_iff.mpr ⟨different.1, previous.1⟩,
          nodesUnique.mem_erase_iff.mpr ⟨different.2, previous.2⟩⟩
      have nextAdjacency : ∀ source ∈ nodes.erase vertex,
          (state.neighbors source).Perm (successors source (removeNode vertex edges)) := by
        intro source retained
        have original := nodesUnique.mem_erase_iff.mp retained
        rw [successors_remove_ready vertex source edges selected.2.1 original.1]
        exact adjacency source original.2
      obtain ⟨nextQueue, nextDegrees, nextEnv, nextUsed, stepped, _, nextCounts,
          nextContents, nextOrdered, bindings⟩ :=
        iteration choices base roots state represented nodes edges nodesUnique edgesUnique
          endpoints vertices vertex rest output degrees queueContents ordered
          (adjacency vertex member) counted env getQueue getOutput getDegrees getAdjacency
          (loop :: following) anchors stack used
      obtain ⟨emitted, finalDegrees, finalEnv, finalUsed, completed, lengthBound,
          result, preserved⟩ :=
        run_loop choices base roots state represented (nodes.erase vertex) (removeNode vertex edges)
          (nodesUnique.erase vertex) (edgesUnique.filter _)
          nextEndpoints (fun source retained => vertices source (List.mem_of_mem_erase retained))
          nextQueue (output ++ [vertex]) nextDegrees nextContents nextOrdered nextAdjacency
          nextCounts nextEnv
          (by rw [bindings "heap" (by decide) (by decide)]; exact getQueue)
          (by rw [bindings "out" (by decide) (by decide)]; exact getOutput)
          (by rw [bindings "indeg2" (by decide) (by decide)]; exact getDegrees)
          (by rw [bindings "adj" (by decide) (by decide)]; exact getAdjacency)
          following anchors stack nextUsed
      refine ⟨vertex :: emitted, finalDegrees, finalEnv, finalUsed, ?_, ?_, ?_, ?_⟩
      · have entered := enter_loop choices base env vertex rest output degrees getQueue
          following anchors stack used
        simpa only [List.append_assoc, List.singleton_append] using
          entered.trans (stepped.trans completed)
      · simp only [List.length_cons]
        omega
      · have nonempty : nodes ≠ [] := List.ne_nil_of_mem member
        have size : nodes.length = (nodes.erase vertex).length + 1 := by
          rw [List.length_erase_of_mem member]
          have := List.length_pos_of_mem member
          omega
        rw [order, if_neg nonempty, chosen]
        dsimp only
        rw [result]
        by_cases complete : emitted.length = (nodes.erase vertex).length
        · simp [complete, size]
        · have incomplete : emitted.length + 1 ≠ nodes.length := by omega
          simp [complete, incomplete]
      · intro name notU notV
        exact (preserved name notU notV).trans (bindings name notU notV)
termination_by nodes.length

end solvers.«79».topo
