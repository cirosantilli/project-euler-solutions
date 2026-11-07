import solvers.«98.py.lean.d».SquareBounds

namespace solvers.«98».pattern_signature
open FormalSW.PyAst HeapCalls

theorem «Returns» (text : String) :
    (HeapCalls.call program "pattern_signature" [.scalar (.string text)] []).Returns := by
  obtain ⟨_, _, returned, _⟩ := result text []
  exact returned.returns

theorem «Supported» (text : String) :
    (HeapCalls.call program "pattern_signature" [.scalar (.string text)] []).Supported :=
  («Returns» text).supported

/-- Two actual source calls yield equal tuple keys exactly when an injective substitution exists. -/
theorem pair_result (word digits : String) (heap : Mutable.Heap) :
    ∃ middle final left right,
      ReturnsValue program "pattern_signature" [.scalar (.string word)] heap (.ref left) middle ∧
      ReturnsValue program "pattern_signature" [.scalar (.string digits)] middle (.ref right) final ∧
      (final[left]? = final[right]? ↔
        ∃ f : Char → Char, word.toList.map f = digits.toList ∧
          Set.InjOn f {letter | letter ∈ word.toList}) := by
  obtain ⟨middle, left, first, firstFound, _⟩ := result word heap
  obtain ⟨final, right, second, secondFound, preserved⟩ := result digits middle
  obtain ⟨extra, rfl⟩ := preserved
  refine ⟨middle, middle ++ extra, left, right, first, second, ?_⟩
  have kept := ProjectEulerSolutions.HeapCollection.prefix_found middle extra left _ firstFound
  rw [kept, secondFound]
  have injective : Function.Injective number := by
    intro a b same
    have equal : (a : Int) = b := by simpa [number] using same
    omega
  simp only [Option.some.injEq, Mutable.Object.tuple.injEq]
  constructor
  · intro same
    have keys : ProjectEulerSolutions.AnagramSquares.labels [] word.toList =
        ProjectEulerSolutions.AnagramSquares.labels [] digits.toList :=
      (List.map_injective_iff.mpr injective) same
    exact (ProjectEulerSolutions.AnagramSquares.labels_eq_iff_substitution '0' _ _).mp keys
  · intro assignment
    have keys := (ProjectEulerSolutions.AnagramSquares.labels_eq_iff_substitution '0' _ _).mpr assignment
    simp [keys]

end solvers.«98».pattern_signature
