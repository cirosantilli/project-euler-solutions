import solvers.«79.topo.step.source»

/-! Every order of successor traversal decrements the same counts and enqueues the same vertices. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def Counts (degrees : List (Constant × Mutable.Value)) (counts : String → Int)
    (vertices : List String) : Prop :=
  ∀ vertex ∈ vertices, ScalarDict.lookup degrees (.string vertex) = some (number (counts vertex))

theorem Counts.decrease (degrees : List (Constant × Mutable.Value)) (counts : String → Int)
    (vertices : List String) (counted : Counts degrees counts vertices) (vertex : String) :
    Counts (decrease degrees vertex (counts vertex)) (decrement counts vertex) vertices := by
  intro target member
  by_cases same : vertex = target
  · subst target
    simp [topo.decrease, decrement, ScalarDict.lookup_insert_self]
  · simpa [topo.decrease, decrement, same,
      lookup_insert_other_string degrees vertex target _ same] using counted target member

/-- The original adjacency set stays untouched while the queue and copied dictionary change. -/
theorem discharge_loop (choices : ChoiceSchedule) (base : Mutable.Heap)
    (address : Nat) (original remaining vertices : List String)
    (found : base[address]? = some (.set (stringSet original)))
    (inside : ∀ vertex ∈ remaining, vertex ∈ vertices) (unique : remaining.Nodup)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (counts : String → Int) (counted : Counts degrees counts vertices) (ordered : Ordered queue)
    (env : HeapCalls.Env) (getQueue : env "heap" = some (.ref base.length))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ nextQueue nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.setStepLoop "v" address (stringSet original) (stringSet remaining)
            successorBody [] :: following) env, stack, workspace base queue output degrees⟩,
          used⟩
        ⟨⟨.block following finalEnv, stack, workspace base nextQueue output nextDegrees⟩,
          finalUsed⟩ ∧
      Counts nextDegrees (decrementAll counts remaining) vertices ∧
      nextQueue.Perm (queue ++ remaining.filter (fun vertex => decide (counts vertex = 1))) ∧
      Ordered nextQueue ∧ (∀ name, name ≠ "v" → finalEnv name = env name) := by
  have addressInside := (List.getElem?_eq_some_iff.mp found).1
  have setFound : (workspace base queue output degrees)[address]? =
      some (.set (stringSet original)) := by
    rw [workspace, List.getElem?_append_left addressInside]
    exact found
  cases remainingEq : remaining with
  | nil =>
      refine ⟨queue, degrees, env, used, .next ?_ .refl, counted, ?_, ordered, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, stringSet, setFound]
      · simp
      · intros; rfl
  | cons head tail =>
      let index := choices used % (head :: tail).length
      have bound : index < (head :: tail).length := Nat.mod_lt _ (by simp)
      let vertex := (head :: tail)[index]
      let rest := (head :: tail).eraseIdx index
      let boundEnv := setLocal env "v" (text vertex)
      have permutation : (vertex :: rest).Perm (head :: tail) :=
        List.getElem_cons_eraseIdx_perm bound
      have selectedUnique : (vertex :: rest).Nodup :=
        permutation.nodup_iff.mpr (by simpa only [remainingEq] using unique)
      have restUnique : rest.Nodup := (List.nodup_cons.mp selectedUnique).2
      have restInside : ∀ value ∈ rest, value ∈ vertices := by
        intro value member
        exact inside value (by rw [remainingEq]; exact List.mem_of_mem_eraseIdx member)
      have vertexInside : vertex ∈ vertices :=
        inside vertex (by rw [remainingEq]; exact List.getElem_mem bound)
      have selected : (stringSet (head :: tail))[choices used % (tail.length + 1)]? =
          some (.string vertex) := by
        simp only [stringSet, List.getElem?_map]
        rw [show choices used % (tail.length + 1) = index from rfl,
          List.getElem?_eq_getElem bound]
        rfl
      have selected' : (Constant.string head :: tail.map Constant.string)[
          choices used % (tail.length + 1)]? = some (.string vertex) := selected
      have erased : (Constant.string head :: tail.map Constant.string).eraseIdx
          (choices used % (tail.length + 1)) = rest.map Constant.string :=
        List.eraseIdx_map Constant.string (head :: tail) index
      obtain ⟨oneQueue, oneStep, oneContents, oneOrdered⟩ :=
        discharge_one base boundEnv queue output degrees vertex (counts vertex)
          (by simpa [boundEnv, setLocal] using getQueue)
          (by simpa [boundEnv, setLocal] using getDegrees)
          (by simp [boundEnv, setLocal]) (counted vertex vertexInside) ordered
      obtain ⟨nextQueue, nextDegrees, finalEnv, finalUsed, completed, finalCounts,
          contents, finalOrdered, preserved⟩ :=
        discharge_loop choices base address original rest vertices found restInside restUnique
          oneQueue output (decrease degrees vertex (counts vertex)) (decrement counts vertex)
          (counted.decrease degrees counts vertices vertex) oneOrdered boundEnv
          (by simpa [boundEnv, setLocal] using getQueue)
          (by simpa [boundEnv, setLocal] using getDegrees) following anchors stack (used + 1)
      refine ⟨nextQueue, nextDegrees, finalEnv, finalUsed, .next (following :=
        ⟨⟨.block (successorBody ++ [.setStepLoop "v" address (stringSet original)
            (stringSet rest) successorBody []] ++ following) boundEnv,
          stack, workspace base queue output degrees⟩, used + 1⟩) ?_ ?_, ?_, ?_,
        finalOrdered, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, setFound,
          selected', erased, stringSet, boundEnv,
          successorBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt]
      · exact (SetCalls.Reaches.of_heap (oneStep
          (.setStepLoop "v" address (stringSet original) (stringSet rest)
            successorBody [] :: following) anchors stack) choices (used + 1)).trans completed
      · have same := decrementAll_perm counts (vertex :: rest) (head :: tail) permutation
        change decrementAll (decrement counts vertex) rest =
          decrementAll counts (head :: tail) at same
        simpa only [same] using finalCounts
      · have unchanged : rest.filter (fun target => decide (decrement counts vertex target = 1)) =
            rest.filter (fun target => decide (counts target = 1)) := by
          apply List.filter_congr
          intro target member
          have different : vertex ≠ target := by
            intro same
            exact (List.nodup_cons.mp selectedUnique).1 (same ▸ member)
          simp [decrement, different]
        rw [unchanged] at contents
        apply contents.trans
        apply (oneContents.append_right _).trans
        have filtered := permutation.filter (fun target => decide (counts target = 1))
        have appended := filtered.append_left queue
        by_cases crossing : counts vertex = 1 <;>
          simpa [crossing, List.append_assoc] using appended
      · intro name different
        rw [preserved name different]
        simp [boundEnv, setLocal, different]
termination_by remaining.length
decreasing_by
  simp only [remainingEq]
  change ((head :: tail).eraseIdx index).length < (head :: tail).length
  rw [List.length_eraseIdx_of_lt bound]
  simp

end solvers.«79».topo
