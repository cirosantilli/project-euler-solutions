import ProjectEulerSolutions.Lib.RationalResultSpec
import ProjectEulerSolutions.Lib.CanonicalIntegers

/-! Complete canonical reduction results have the same integers as exhaustive expression enumeration. -/
namespace ProjectEulerSolutions.RationalPair
open ArithmeticExpressions
set_option Elab.async false

theorem expected_values (state candidates : List Pair)
    (complete : ∀ pair, pair ∈ candidates ↔ Expected state pair) (q : Rat) :
    q ∈ candidates.map value ↔ Reduces (state.map value) q := by
  constructor
  · rintro member
    obtain ⟨pair, present, rfl⟩ := List.mem_map.mp member
    exact ((complete pair).mp present).2
  · intro derived
    let pair : Pair := (q.num, q.den)
    have represented : value pair = q := Rat.num_divInt_den q
    have canonical : Canonical pair := by
      unfold Canonical
      rw [represented]
    exact List.mem_map.mpr ⟨pair, (complete pair).mpr ⟨canonical, represented.symm ▸ derived⟩, represented⟩

theorem expected_integer_members (state candidates : List Pair)
    (complete : ∀ pair, pair ∈ candidates ↔ Expected state pair) (n : Int) :
    n ∈ canonicalIntegerResults candidates ↔ 0 < n ∧ (n : Rat) ∈ permuted (state.map value) := by
  rw [mem_canonicalIntegerResults candidates (fun pair member => ((complete pair).mp member).1),
    expected_values state candidates complete, reduces_iff_mem_permuted]

/-- The positive run depends only on expression coverage, not on normalization or enumeration order. -/
theorem expected_integer_run (state candidates : List Pair) (reference : List Int)
    (complete : ∀ pair, pair ∈ candidates ↔ Expected state pair)
    (referenceMembers : ∀ n, n ∈ reference ↔ 0 < n ∧ (n : Rat) ∈ permuted (state.map value)) :
    PositiveRun.length (canonicalIntegerResults candidates) = PositiveRun.length reference := by
  apply PositiveRun.congr
  intro n _
  exact (expected_integer_members state candidates complete n).trans (referenceMembers n).symm

end ProjectEulerSolutions.RationalPair
