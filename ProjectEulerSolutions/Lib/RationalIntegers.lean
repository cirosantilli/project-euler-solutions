import ProjectEulerSolutions.Lib.RationalEnumeration
import ProjectEulerSolutions.Lib.PositiveRun

/-! Positive integer extraction from exact rational pairs, including unreduced pairs. -/
namespace ProjectEulerSolutions.RationalPair
set_option Elab.async false

def integerResult (pair : Pair) : List Int :=
  if 0 < pair.1 ∧ pair.1 % pair.2 = 0 then [pair.1 / pair.2] else []

def integerResults (pairs : List Pair) : List Int := pairs.flatMap integerResult

/-- Exact divisibility accepts precisely the positive integers represented by a pair. -/
theorem mem_integerResult (pair : Pair) (positive : 0 < pair.2) (n : Int) :
    n ∈ integerResult pair ↔ 0 < n ∧ value pair = (n : Rat) := by
  have equal : value pair = (n : Rat) ↔ pair.1 = n * pair.2 := by
    rw [value, Rat.intCast_eq_divInt, Rat.divInt_eq_divInt_iff positive.ne' (by decide)]
    simp
  rw [equal]
  unfold integerResult
  split_ifs with accepted
  · simp only [List.mem_singleton]
    constructor
    · intro same
      subst n
      have exactDivision : pair.1 / pair.2 * pair.2 = pair.1 :=
        Int.ediv_mul_cancel (Int.dvd_of_emod_eq_zero accepted.2)
      constructor
      · nlinarith [accepted.1]
      · exact exactDivision.symm
    · rintro ⟨_, numerator⟩
      rw [numerator, Int.mul_ediv_cancel _ positive.ne']
  · simp only [List.not_mem_nil, false_iff, not_and]
    intro positiveN numerator
    apply accepted
    rw [numerator]
    exact ⟨mul_pos positiveN positive, by simp⟩

theorem mem_integerResults (pairs : List Pair) (positive : ∀ pair ∈ pairs, 0 < pair.2) (n : Int) :
    n ∈ integerResults pairs ↔ 0 < n ∧ (n : Rat) ∈ pairs.map value := by
  simp only [integerResults, List.mem_flatMap]
  constructor
  · rintro ⟨pair, member, accepted⟩
    obtain ⟨positiveN, same⟩ := (mem_integerResult pair (positive pair member) n).mp accepted
    exact ⟨positiveN, List.mem_map.mpr ⟨pair, member, same⟩⟩
  · rintro ⟨positiveN, member⟩
    obtain ⟨pair, member, same⟩ := List.mem_map.mp member
    exact ⟨pair, member, (mem_integerResult pair (positive pair member) n).mpr ⟨positiveN, same⟩⟩

theorem mem_enumerated_integers (xs : List Int) (n : Int) :
    n ∈ integerResults (enumeratePairs xs) ↔
      0 < n ∧ (n : Rat) ∈ ArithmeticExpressions.enumerate (xs.map (fun (a : Int) => (a : Rat))) := by
  rw [mem_integerResults _ (enumeratePairs_spec xs).1, (enumeratePairs_spec xs).2]

/-- Equivalent rational result sets give the same initial positive-integer run. -/
theorem integerResults_run_congr (left right : List Pair)
    (leftPositive : ∀ pair ∈ left, 0 < pair.2) (rightPositive : ∀ pair ∈ right, 0 < pair.2)
    (same : ∀ q : Rat, q ∈ left.map value ↔ q ∈ right.map value) :
    PositiveRun.length (integerResults left) = PositiveRun.length (integerResults right) := by
  apply PositiveRun.congr
  intro n _
  rw [mem_integerResults left leftPositive, mem_integerResults right rightPositive, same]

end ProjectEulerSolutions.RationalPair
