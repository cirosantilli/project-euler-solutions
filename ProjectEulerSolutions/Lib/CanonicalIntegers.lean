import ProjectEulerSolutions.Lib.RationalChoices
import ProjectEulerSolutions.Lib.RationalIntegers

/-! Integer extraction from canonical rational keys agrees with exact divisibility. -/
namespace ProjectEulerSolutions.RationalPair
set_option Elab.async false

def canonicalIntegerResult (pair : Pair) : List Int :=
  if pair.2 = 1 ∧ 0 < pair.1 then [pair.1] else []

def canonicalIntegerResults (pairs : List Pair) : List Int := pairs.flatMap canonicalIntegerResult

/-- Collect positive integer values while retaining the first occurrence. -/
def collectCanonicalInteger (before : List Int) (pair : Pair) : List Int :=
  if pair.2 = 1 ∧ 0 < pair.1 then
    if pair.1 ∈ before then before else before ++ [pair.1]
  else before

theorem mem_collectCanonicalInteger (before : List Int) (pair : Pair) (n : Int) :
    n ∈ collectCanonicalInteger before pair ↔ n ∈ before ∨ n ∈ canonicalIntegerResult pair := by
  unfold collectCanonicalInteger canonicalIntegerResult
  split_ifs <;> simp_all

theorem mem_fold_collectCanonicalInteger (pairs : List Pair) (before : List Int) (n : Int) :
    n ∈ pairs.foldl collectCanonicalInteger before ↔ n ∈ before ∨ n ∈ canonicalIntegerResults pairs := by
  induction pairs generalizing before with
  | nil => simp [canonicalIntegerResults]
  | cons pair rest ih =>
    simp only [List.foldl_cons, ih, mem_collectCanonicalInteger, canonicalIntegerResults,
      List.flatMap_cons, List.mem_append]
    tauto

/-- A normalized pair represents an integer precisely when it is that integer over one. -/
theorem Canonical.integer_iff {pair : Pair} (canonical : Canonical pair) (n : Int) :
    value pair = (n : Rat) ↔ pair = (n, 1) := by
  constructor
  · intro equal
    rw [Canonical, equal] at canonical
    simpa using canonical
  · rintro rfl
    simp [value]

theorem mem_canonicalIntegerResult (pair : Pair) (canonical : Canonical pair) (n : Int) :
    n ∈ canonicalIntegerResult pair ↔ 0 < n ∧ value pair = (n : Rat) := by
  rw [canonical.integer_iff n]
  unfold canonicalIntegerResult
  split_ifs with accepted
  · simp only [List.mem_singleton]
    constructor
    · rintro rfl
      exact ⟨accepted.2, Prod.ext rfl accepted.1⟩
    · rintro ⟨_, equal⟩
      exact (congrArg Prod.fst equal).symm
  · simp only [List.not_mem_nil, false_iff, not_and]
    intro positive equal
    apply accepted
    rw [equal]
    exact ⟨rfl, positive⟩

theorem mem_canonicalIntegerResults (pairs : List Pair)
    (canonical : ∀ pair ∈ pairs, Canonical pair) (n : Int) :
    n ∈ canonicalIntegerResults pairs ↔ 0 < n ∧ (n : Rat) ∈ pairs.map value := by
  simp only [canonicalIntegerResults, List.mem_flatMap]
  constructor
  · rintro ⟨pair, member, accepted⟩
    obtain ⟨positive, equal⟩ := (mem_canonicalIntegerResult pair (canonical pair member) n).mp accepted
    exact ⟨positive, List.mem_map.mpr ⟨pair, member, equal⟩⟩
  · rintro ⟨positive, member⟩
    obtain ⟨pair, member, equal⟩ := List.mem_map.mp member
    exact ⟨pair, member, (mem_canonicalIntegerResult pair (canonical pair member) n).mpr ⟨positive, equal⟩⟩

/-- Canonical-key filtering and unreduced reference fractions yield the same positive run. -/
theorem canonicalIntegerResults_run_congr (optimized reference : List Pair)
    (canonical : ∀ pair ∈ optimized, Canonical pair)
    (positive : ∀ pair ∈ reference, 0 < pair.2)
    (same : ∀ q : Rat, q ∈ optimized.map value ↔ q ∈ reference.map value) :
    PositiveRun.length (canonicalIntegerResults optimized) =
      PositiveRun.length (integerResults reference) := by
  apply PositiveRun.congr
  intro n _
  rw [mem_canonicalIntegerResults optimized canonical, mem_integerResults reference positive, same]

end ProjectEulerSolutions.RationalPair
