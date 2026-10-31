import ProjectEulerSolutions.Lib.RationalOperations

/-! Canonical results of choosing an unordered pair of expression fragments. -/
namespace ProjectEulerSolutions.RationalPair
set_option Elab.async false

def canonical (pair : Pair) : Pair := normalize pair.1 pair.2
def division (a b : Pair) : Pair := normalize (a.1 * b.2) (a.2 * b.1)
def initialChoices (a b : Pair) : List Pair :=
  [canonical (add a b), canonical (sub a b), canonical (sub b a), canonical (mul a b)]
def choices (a b : Pair) : List Pair :=
  initialChoices a b ++ (if b.1 = 0 then [] else [division a b]) ++
    (if a.1 = 0 then [] else [division b a])

def Canonical (pair : Pair) : Prop := pair = ((value pair).num, ((value pair).den : Int))

theorem canonical_spec (pair : Pair) (nonzero : pair.2 ≠ 0) :
    value (canonical pair) = value pair ∧ Canonical (canonical pair) := by
  have same := normalize_value pair.1 pair.2 nonzero
  refine ⟨same, ?_⟩
  unfold Canonical
  rw [show value (canonical pair) = value pair from same]
  exact normalize_canonical pair.1 pair.2 nonzero

theorem division_spec (a b : Pair) (positiveA : 0 < a.2) (nonzeroB : b.1 ≠ 0) :
    value (division a b) = value a / value b ∧ Canonical (division a b) := by
  have spec := canonical_spec (a.1 * b.2, a.2 * b.1) (mul_ne_zero positiveA.ne' nonzeroB)
  exact ⟨by simpa only [canonical, value, Rat.divInt_div_divInt] using spec.1, spec.2⟩

theorem initialChoices_values (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2) :
    (initialChoices a b).map value = [value a + value b, value a - value b, value b - value a, value a * value b] := by
  have ab := mul_ne_zero positiveA.ne' positiveB.ne'
  have ba := mul_ne_zero positiveB.ne' positiveA.ne'
  simp only [initialChoices, List.map_cons, List.map_nil,
    (canonical_spec (add a b) ab).1, (canonical_spec (sub a b) ab).1,
    (canonical_spec (sub b a) ba).1, (canonical_spec (mul a b) ab).1,
    add_value a b positiveA.ne' positiveB.ne', sub_value a b positiveA.ne' positiveB.ne',
    sub_value b a positiveB.ne' positiveA.ne', mul_value]

theorem choices_canonical (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2)
    (pair : Pair) (member : pair ∈ choices a b) : Canonical pair := by
  have ab := mul_ne_zero positiveA.ne' positiveB.ne'
  have ba := mul_ne_zero positiveB.ne' positiveA.ne'
  simp only [choices, initialChoices, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with ((rfl | rfl | rfl | rfl) | forward) | backward
  · exact (canonical_spec (add a b) ab).2
  · exact (canonical_spec (sub a b) ab).2
  · exact (canonical_spec (sub b a) ba).2
  · exact (canonical_spec (mul a b) ab).2
  · split_ifs at forward with zero
    · simp at forward
    · obtain rfl := List.mem_singleton.mp forward
      exact (division_spec a b positiveA zero).2
  · split_ifs at backward with zero
    · simp at backward
    · obtain rfl := List.mem_singleton.mp backward
      exact (division_spec b a positiveB zero).2

theorem Canonical.positive {pair : Pair} (canonical : Canonical pair) : 0 < pair.2 := by
  have denominator := congrArg Prod.snd canonical
  rw [denominator]
  change (0 : Int) < ((value pair).den : Int)
  exact_mod_cast (value pair).pos

/-- Both subtraction and quotient orders occur; division by zero contributes no result. -/
theorem choices_values (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2) :
    (choices a b).map value =
      [value a + value b, value a - value b, value b - value a, value a * value b] ++
      (if value b = 0 then [] else [value a / value b]) ++
      (if value a = 0 then [] else [value b / value a]) := by
  have zeroA : value a = 0 ↔ a.1 = 0 := Rat.divInt_eq_zero positiveA.ne'
  have zeroB : value b = 0 ↔ b.1 = 0 := Rat.divInt_eq_zero positiveB.ne'
  simp only [choices, List.map_append, initialChoices_values a b positiveA positiveB, zeroA, zeroB]
  by_cases aZero : a.1 = 0 <;> by_cases bZero : b.1 = 0 <;>
    simp [aZero, bZero, (division_spec a b positiveA), (division_spec b a positiveB)]

/-- Unordered-pair choice has exactly the values of ordered operations in either direction. -/
theorem mem_choices_values (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2) (q : Rat) :
    q ∈ (choices a b).map value ↔
      q ∈ ArithmeticExpressions.operations (value a) (value b) ∨
      q ∈ ArithmeticExpressions.operations (value b) (value a) := by
  rw [choices_values a b positiveA positiveB]
  simp only [ArithmeticExpressions.operations, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
  rw [add_comm (value b) (value a), mul_comm (value b) (value a)]
  tauto

end ProjectEulerSolutions.RationalPair
