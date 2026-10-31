import ProjectEulerSolutions.Lib.LazyDijkstraQueue

/-! Queue coverage depends on distance and visited values only at valid vertices. -/
namespace ProjectEulerSolutions.LazyDijkstraQueue

theorem Invariant.congr {α : Type} {valid : α → Prop} {infinity : Int}
    {distance otherDistance : α → Int} {visited otherVisited : α → Bool} {queue : List (Int × α)}
    (invariant : Invariant valid infinity distance visited queue)
    (values : ∀ v, valid v → distance v = otherDistance v)
    (flags : ∀ v, valid v → visited v = otherVisited v) :
    Invariant valid infinity otherDistance otherVisited queue := by
  refine ⟨invariant.valid_entry, ?_, invariant.finite, ?_⟩
  · intro entry member
    rw [← values entry.2 (invariant.valid_entry entry member)]
    exact invariant.lower entry member
  · intro v inside waiting finite
    rw [← values v inside] at finite ⊢
    rw [← flags v inside] at waiting
    exact invariant.covered v inside waiting finite

end ProjectEulerSolutions.LazyDijkstraQueue
