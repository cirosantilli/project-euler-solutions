import solvers.«79.discharge.source»

/-! One complete source iteration refines removal of the least ready graph vertex. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

theorem visit_successors (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (vertex : String) (member : vertex ∈ state.vertices)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (counts : String → Int) (counted : Counts degrees counts state.vertices)
    (ordered : Ordered queue) (env : HeapCalls.Env)
    (getQueue : env "heap" = some (.ref base.length))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getAdjacency : env "adj" = some (.ref roots.adjacency))
    (getVertex : env "u" = some (text vertex))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ nextQueue nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.forIn (.name "v") (.subscript (.name "adj") (.name "u"))
            successorBody [] :: following) env, stack, workspace base queue output degrees⟩,
          used⟩
        ⟨⟨.block following finalEnv, stack, workspace base nextQueue output nextDegrees⟩,
          finalUsed⟩ ∧
      Counts nextDegrees (decrementAll counts (state.neighbors vertex)) state.vertices ∧
      nextQueue.Perm
        (queue ++ (state.neighbors vertex).filter (fun target => decide (counts target = 1))) ∧
      Ordered nextQueue ∧ (∀ name, name ≠ "v" → finalEnv name = env name) := by
  have neighbors := represented.contents vertex member
  have inside : ∀ target ∈ state.neighbors vertex, target ∈ state.vertices := by
    intro target neighbor
    exact (represented.endpoints (vertex, target)
      ((mem_successors vertex target state.edges).mp (neighbors.mem_iff.mp neighbor))).2
  have unique := neighbors.nodup_iff.mpr
    (successors_nodup vertex state.edges represented.edgesUnique)
  have adjacencyFound : (workspace base queue output degrees)[roots.adjacency]? =
      some (.dict state.adjacency) := by
    rw [workspace, List.getElem?_append_left
      (List.getElem?_eq_some_iff.mp represented.adjacencyFound).1]
    exact represented.adjacencyFound
  have setFound : (workspace base queue output degrees)[state.address vertex]? =
      some (.set (stringSet (state.neighbors vertex))) := by
    rw [workspace, List.getElem?_append_left
      (List.getElem?_eq_some_iff.mp (represented.neighborsFound vertex member)).1]
    exact represented.neighborsFound vertex member
  have read : Evaluates program env (.subscript (.name "adj") (.name "u"))
      (workspace base queue output degrees) (.ref (state.address vertex))
      (workspace base queue output degrees) := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getAdjacency) (Evaluates.lookup getVertex)
    exact Mutable.getItem_dictionary _ roots.adjacency state.adjacency adjacencyFound
      (.string vertex) (represented.neighborsLookup vertex member)
  obtain ⟨nextQueue, nextDegrees, finalEnv, finalUsed, completed, countedAfter,
      contents, orderedAfter, preserved⟩ :=
    discharge_loop choices base (state.address vertex) (state.neighbors vertex)
      (state.neighbors vertex) state.vertices (represented.neighborsFound vertex member)
      inside unique queue output degrees counts counted ordered env getQueue getDegrees
      following anchors stack used
  refine ⟨nextQueue, nextDegrees, finalEnv, finalUsed, .next rfl ?_,
    countedAfter, contents, orderedAfter, preserved⟩
  apply (SetCalls.Reaches.of_heap (read anchors
    (.sequence "v" successorBody [] following env :: stack)) choices used).trans
  exact .next (by
    simp [SetCalls.step, HeapCalls.step, run, Mutable.sequence, setFound,
      Mutable.unsupported, SetCalls.iteratorStep]) completed

/-- Queue contents and every stored indegree become those of the residual graph. -/
theorem iteration (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (nodes : List String) (edges : List (String × String))
    (nodesUnique : nodes.Nodup) (edgesUnique : edges.Nodup)
    (endpoints : ∀ edge ∈ edges, edge.1 ∈ nodes ∧ edge.2 ∈ nodes)
    (vertices : ∀ vertex ∈ nodes, vertex ∈ state.vertices)
    (vertex : String) (rest output : List String) (degrees : List (Constant × Mutable.Value))
    (queueContents : (vertex :: rest).Perm
      (ProjectEulerStatements.TopologicalOrder.eligible nodes edges))
    (ordered : Ordered (vertex :: rest))
    (adjacency : (state.neighbors vertex).Perm (successors vertex edges))
    (counted : Counts degrees (fun target => (indegree target edges : Int)) state.vertices)
    (env : HeapCalls.Env) (getQueue : env "heap" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getAdjacency : env "adj" = some (.ref roots.adjacency))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ nextQueue nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (loopBody ++ following) env, stack,
          workspace base (vertex :: rest) output degrees⟩, used⟩
        ⟨⟨.block following finalEnv, stack,
          workspace base nextQueue (output ++ [vertex]) nextDegrees⟩, finalUsed⟩ ∧
      choose nodes edges = some vertex ∧
      Counts nextDegrees (fun target => (indegree target (removeNode vertex edges) : Int))
        state.vertices ∧
      nextQueue.Perm (ProjectEulerStatements.TopologicalOrder.eligible
        (nodes.erase vertex) (removeNode vertex edges)) ∧ Ordered nextQueue ∧
      (∀ name, name ≠ "u" → name ≠ "v" → finalEnv name = env name) := by
  obtain ⟨poppedQueue, popped, chosen, permutation, poppedOrdered⟩ :=
    pop_emit base env nodes edges vertex rest output degrees
      getQueue getOutput queueContents ordered
  have chosenSpec := choose_spec nodes edges vertex chosen
  obtain ⟨nextQueue, nextDegrees, finalEnv, finalUsed, visited, countsAfter,
      contents, orderedAfter, preserved⟩ :=
    visit_successors choices base roots state represented vertex (vertices vertex chosenSpec.1)
      poppedQueue (output ++ [vertex]) degrees (fun target => (indegree target edges : Int))
      counted poppedOrdered (setLocal env "u" (text vertex))
      (by simpa [setLocal] using getQueue) (by simpa [setLocal] using getDegrees)
      (by simpa [setLocal] using getAdjacency) (by simp [setLocal])
      following anchors stack used
  have dischargeContents : nextQueue.Perm (poppedQueue ++
      (discharge (fun target => (indegree target edges : Int)) (state.neighbors vertex)).2) := by
    rw [discharge_new _ _ (adjacency.nodup_iff.mpr (successors_nodup vertex edges edgesUnique))]
    exact contents
  have residual := discharge_queue vertex nodes edges poppedQueue (state.neighbors vertex)
    nextQueue nodesUnique edgesUnique endpoints ((permutation.cons vertex).trans queueContents)
    adjacency dischargeContents
  refine ⟨nextQueue, nextDegrees, finalEnv, finalUsed, ?_, chosen, ?_, residual.2,
    orderedAfter, ?_⟩
  · exact (SetCalls.Reaches.of_heap (popped
      (.forIn (.name "v") (.subscript (.name "adj") (.name "u")) successorBody [] :: following)
      anchors stack) choices used).trans visited
  · rw [discharge_counts] at residual
    simpa only [residual.1] using countsAfter
  · intro name notU notV
    rw [preserved name notV]
    simp [setLocal, notU]

end solvers.«79».topo
