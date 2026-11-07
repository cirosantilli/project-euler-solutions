import solvers.«98.py.lean.d».GroupLoop

namespace solvers.«98».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open mapped_word (number)
set_option Elab.async false

/-- The whole optimized Python solver terminates and returns its exact grouped search maximum. -/
theorem result (words : List String) (heap : Mutable.Heap) (wordsAddress : Nat)
    (found : heap[wordsAddress]? = some (.list (words.map text))) :
    ∃ answer extra, ReturnsValue program "solve" [.ref wordsAddress] heap (number answer) (heap ++ extra) ∧
      ((answer = 0 ∧ AnagramSearch.solve words = 0) ∨
        ∃ buckets, StringValueBuckets.Partition keyOfValue (groupValues words) buckets ∧
          answer = group_search.scan words (groups.selected buckets) 0) := by
  rcases setup words heap wordsAddress found with early | prepared
  · obtain ⟨extra, returned, correct⟩ := early
    exact ⟨0, extra, returned, Or.inl ⟨rfl, correct⟩⟩
  · obtain ⟨setupExtra, env, setupRan, ⟨ready⟩⟩ := prepared
    obtain ⟨finalEnv, searchExtra, searched, getBest, _⟩ := group_search.result_loop (search_state.ofReady ready) words 0 (heap ++ setupExtra) env
      (search_state.ready_valid ready) (search_state.ready_locals ready) ready.getBest
    refine ⟨group_search.scan words (groups.selected ready.buckets) 0, setupExtra ++ searchExtra, ?_,
      Or.inr ⟨ready.buckets, ready.partition, rfl⟩⟩
    have returned : BlockReturns program finalEnv (source.drop 10) ((heap ++ setupExtra) ++ searchExtra)
        (number (group_search.scan words (groups.selected ready.buckets) 0)) ((heap ++ setupExtra) ++ searchExtra) :=
      BlockReturns.return_value (Evaluates.lookup getBest)
    apply BlockReturns.function rfl rfl
    simpa only [List.append_assoc] using setupRan.returns (searched.returns returned)

end solvers.«98».solve
