import solvers.«98.py.lean.d».Maximum

namespace solvers.«98».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open mapped_word (number)
set_option Elab.async false

/-- The complete optimized source returns the exhaustive reference maximum for every input word list. -/
theorem correct_result (words : List String) (heap : Mutable.Heap) (wordsAddress : Nat)
    (found : heap[wordsAddress]? = some (.list (words.map text))) :
    ∃ extra, ReturnsValue program "solve" [.ref wordsAddress] heap (number (AnagramSearch.solve words)) (heap ++ extra) := by
  obtain ⟨answer, extra, returned, early | grouped⟩ := result words heap wordsAddress found
  · obtain ⟨rfl, correct⟩ := early
    exact ⟨extra, by simpa only [correct] using returned⟩
  · obtain ⟨buckets, partition, rfl⟩ := grouped
    exact ⟨extra, by simpa only [maximum.grouped_eq words buckets partition] using returned⟩

end solvers.«98».solve
