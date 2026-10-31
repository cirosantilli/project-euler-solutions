import ProjectEulerSolutions.Lib.LazyDijkstraQueue
import Mathlib.Data.List.GetD

/-! Counting false entries supplies a structural measure for mutable visited buffers. -/
namespace ProjectEulerSolutions.VisitedListProgress

theorem mark_decreases (flags : List Bool) (index : Nat) (inside : index < flags.length)
    (waiting : flags[index] = false) :
    (flags.set index true).count false < flags.count false := by
  have positive : 0 < flags.count false := List.count_pos_iff.mpr
    (waiting ▸ List.getElem_mem inside)
  rw [List.count_set inside, waiting]
  simpa using Nat.sub_lt positive (by decide : 0 < 1)

end ProjectEulerSolutions.VisitedListProgress
