import solvers.«84.py.lean.d».LandingMoves

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

theorem move_loop (remaining before : List Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (source root : Nat) (entries : List Entry)
    (table : heap[source]? = some (.tuple ((before ++ remaining).map fun value => .scalar (.int value))))
    (found : heap[root]? = some (.keyDict (encode entries))) (valid : Valid heap entries)
    (getWeights : env "weights" = some (.ref root))
    (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : ∀ (nextEnv : HeapCalls.Env) (nextEntries : List Entry) (extra : Mutable.Heap),
      nextEnv "weights" = some (.ref root) →
      Valid ((heap ++ extra).set root (.keyDict (encode nextEntries))) nextEntries →
      (∀ landing, (lookup nextEntries landing).getD 0 =
        (lookup entries landing).getD 0 + total (remaining.flatMap moveData) landing) →
      HeapMemoCalls.BlockReturnsRel program pool.spec nextEnv body
        ((heap ++ extra).set root (.keyDict (encode nextEntries))) post globals) :
    HeapMemoCalls.BlockReturnsRel program pool.spec env
      (.sequenceStepLoop "destination" (.ref source) before.length moveBody [] :: body) heap post globals := by
  induction remaining generalizing before env heap entries with
  | nil =>
      have unchanged : heap.set root (.keyDict (encode entries)) = heap := by
        obtain ⟨inside, same⟩ := List.getElem?_eq_some_iff.mp found
        rw [← same, List.set_getElem_self inside]
      apply HeapMemoCalls.BlockReturnsRel.sequence_done
        (items := before.map fun value => .scalar (.int value))
        (by simp [HeapCalls.sequenceValue, Mutable.sequence, table]) (by simp)
      simpa [unchanged] using rest env entries [] getWeights (by simpa [unchanged] using valid)
        (by intro landing; simp [total])
  | cons destination remaining ih =>
      apply HeapMemoCalls.BlockReturnsRel.sequence_step
        (items := (before ++ destination :: remaining).map fun value => .scalar (.int value))
        (value := .scalar (.int destination))
        (by simp [HeapCalls.sequenceValue, Mutable.sequence, table]) (by simp [List.map_append])
      apply move_continue destination (HeapCalls.setLocal env "destination" (.scalar (.int destination))) heap globals root entries
        found valid (by simpa [HeapCalls.setLocal] using getWeights) (by simp [HeapCalls.setLocal])
      intro nextEnv nextEntries extra nextWeights nextValid nextCounts
      let middle := (heap ++ extra).set root (.keyDict (encode nextEntries))
      have extended : (heap ++ extra)[root]? = some (.keyDict (encode entries)) :=
        (List.getElem?_append_left (l₂ := extra) (List.getElem?_eq_some_iff.mp found).1).trans found
      have nextFound : middle[root]? = some (.keyDict (encode nextEntries)) :=
        List.getElem?_set_self (List.getElem?_eq_some_iff.mp extended).1
      have nextTable : middle[source]? = some (.tuple (((before ++ [destination]) ++ remaining).map fun value => .scalar (.int value))) := by
        have present := (List.getElem?_append_left (l₂ := extra) (List.getElem?_eq_some_iff.mp table).1).trans table
        simpa [List.append_assoc] using TupleDictionary.tuple_frame (heap ++ extra) root _ _ extended source _ present
      have continued := ih (before ++ [destination]) nextEnv middle nextEntries nextTable nextFound nextValid nextWeights
      simp only [List.length_append, List.length_singleton] at continued
      change HeapMemoCalls.BlockReturnsRel program pool.spec nextEnv
        (.sequenceStepLoop "destination" (.ref source) (before.length + 1) moveBody [] :: body) middle post globals
      apply continued
      intro finalEnv finalEntries more finalWeights finalValid finalCounts
      have flattened : (middle ++ more).set root (.keyDict (encode finalEntries)) =
          (heap ++ (extra ++ more)).set root (.keyDict (encode finalEntries)) := by
        dsimp only [middle]
        rw [← List.set_append_left root _ (List.getElem?_eq_some_iff.mp extended).1]
        simp [List.append_assoc]
      rw [flattened] at finalValid ⊢
      apply rest finalEnv finalEntries (extra ++ more) finalWeights finalValid
      intro landing
      rw [finalCounts, nextCounts]
      simp [List.flatMap_cons, total, List.map_append, List.sum_append, Int.add_assoc]

end solvers.«84».resolve_landing.weights
