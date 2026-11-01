import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Tactic

/-! Product-sum representations, removal of ones, and complete search bounds. -/
namespace ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

def Represents (size value : Nat) : Prop :=
  ∃ factors : List Nat, (∀ factor ∈ factors, 1 ≤ factor) ∧
    factors.length = size ∧ factors.sum = value ∧ factors.prod = value

def Admissible (factors : List Nat) : Prop := ∀ factor ∈ factors, 2 ≤ factor

/-- The number of terms after padding the factor list with ones. -/
def effectiveSize (factors : List Nat) : Nat :=
  factors.length + factors.prod - factors.sum

theorem product_positive (factors : List Nat) (valid : Admissible factors) :
    1 ≤ factors.prod :=
  List.one_le_prod_of_one_le (by intro factor member; have := valid factor member; omega)

theorem product_two (factors : List Nat) (valid : Admissible factors)
    (nonempty : factors ≠ []) : 2 ≤ factors.prod := by
  cases factors with
  | nil => contradiction
  | cons head tail =>
      have headBound := valid head (by simp)
      have tailBound := product_positive tail (by intro factor member; exact valid factor (by simp [member]))
      simp only [List.prod_cons]
      nlinarith

theorem sum_le_product (factors : List Nat) (valid : Admissible factors) :
    factors.sum ≤ factors.prod := by
  induction factors with
  | nil => simp
  | cons head tail ih =>
      have headBound := valid head (by simp)
      have tailValid : Admissible tail := by
        intro factor member
        exact valid factor (by simp [member])
      have tailBound := ih tailValid
      by_cases empty : tail = []
      · simp [empty]
      · have two := product_two tail tailValid empty
        simp only [List.sum_cons, List.prod_cons]
        nlinarith

theorem effectiveSize_positive (factors : List Nat) (valid : Admissible factors) :
    1 ≤ effectiveSize factors := by
  have bound := sum_le_product factors valid
  cases factors with
  | nil => simp [effectiveSize]
  | cons head tail =>
      simp only [effectiveSize, List.length_cons]
      omega

/-- Python subtracts in the integers; reachable factor states agree with the natural model. -/
theorem effectiveSize_int (factors : List Nat) (valid : Admissible factors) :
    (effectiveSize factors : Int) = (factors.length : Int) + (factors.prod : Int) - factors.sum := by
  have bound := sum_le_product factors valid
  simp only [effectiveSize, Nat.cast_sub (by omega : factors.sum ≤ factors.length + factors.prod),
    Nat.cast_add]

/-- The positive witness works for every size, not just the Euler instance. -/
theorem bound_witness (size : Nat) (atLeastTwo : 2 ≤ size) :
    Represents size (2 * size) := by
  refine ⟨2 :: size :: List.replicate (size - 2) 1, ?_, ?_, ?_, ?_⟩
  · intro factor member
    simp only [List.mem_cons, List.mem_replicate] at member
    rcases member with rfl | rfl | ⟨_, rfl⟩ <;> omega
  · simp only [List.length_cons, List.length_replicate]
    omega
  · simp only [List.sum_cons, List.sum_replicate, smul_eq_mul, mul_one]
    omega
  · simp

/-- Removing ones preserves the product and subtracts one from sum and length together. -/
theorem strip_ones_balance (factors : List Nat) :
    (factors.filter (· != 1)).sum + factors.length =
      factors.sum + (factors.filter (· != 1)).length := by
  induction factors with
  | nil => simp
  | cons head tail ih =>
      by_cases one : head = 1
      · simp [one] at *
        omega
      · simp [one] at *
        omega

theorem strip_ones_admissible (factors : List Nat)
    (positive : ∀ factor ∈ factors, 1 ≤ factor) :
    Admissible (factors.filter (· != 1)) := by
  intro factor member
  obtain ⟨member, notOne⟩ := List.mem_filter.mp member
  have bound := positive factor member
  simp only [bne_iff_ne] at notOne
  omega

theorem represents_effective (factors : List Nat) (valid : Admissible factors) :
    Represents (effectiveSize factors) factors.prod := by
  have bound := sum_le_product factors valid
  refine ⟨factors ++ List.replicate (factors.prod - factors.sum) 1, ?_, ?_, ?_, ?_⟩
  · intro factor member
    simp only [List.mem_append, List.mem_replicate] at member
    rcases member with member | ⟨_, rfl⟩
    · have := valid factor member
      omega
    · omega
  · simp only [List.length_append, List.length_replicate, effectiveSize]
    omega
  · simp only [List.sum_append, List.sum_replicate, smul_eq_mul, mul_one]
    omega
  · simp

/-- Searching factors at least two covers exactly the positive product-sum representations. -/
theorem represents_iff_factors (size value : Nat) :
    Represents size value ↔ ∃ factors : List Nat,
      Admissible factors ∧ factors.prod = value ∧ effectiveSize factors = size := by
  constructor
  · rintro ⟨factors, positive, lengthEq, sumEq, productEq⟩
    refine ⟨factors.filter (· != 1), strip_ones_admissible factors positive, ?_, ?_⟩
    · simpa only [List.prod_filter_bne_one] using productEq
    · have balance := strip_ones_balance factors
      simp only [effectiveSize, List.prod_filter_bne_one]
      omega
  · rintro ⟨factors, valid, rfl, rfl⟩
    exact represents_effective factors valid

/-- Appending a factor cannot decrease the effective size, justifying size pruning. -/
theorem effectiveSize_append (factors : List Nat) (factor : Nat)
    (valid : Admissible factors) (factorBound : 2 ≤ factor) :
    effectiveSize (factors ++ [factor]) =
      effectiveSize factors + (factors.prod - 1) * (factor - 1) := by
  have positive := product_positive factors valid
  have sumBound := sum_le_product factors valid
  have extendedValid : Admissible (factors ++ [factor]) := by
    intro x member
    simp only [List.mem_append, List.mem_singleton] at member
    rcases member with member | rfl
    · exact valid x member
    · exact factorBound
  have extendedBound := sum_le_product (factors ++ [factor]) extendedValid
  simp only [List.prod_append, List.prod_cons, List.prod_nil, mul_one,
    List.sum_append, List.sum_cons, List.sum_nil, add_zero] at extendedBound
  simp only [effectiveSize, List.length_append, List.length_cons, List.length_nil,
    List.prod_append, List.prod_cons, List.prod_nil, mul_one,
    List.sum_append, List.sum_cons, List.sum_nil, add_zero]
  have p : factors.prod - 1 + 1 = factors.prod := by omega
  have f : factor - 1 + 1 = factor := by omega
  have old : factors.length + factors.prod - factors.sum + factors.sum =
      factors.length + factors.prod := by omega
  have next : factors.length + 1 + factors.prod * factor - (factors.sum + factor) +
      (factors.sum + factor) = factors.length + 1 + factors.prod * factor := by omega
  nlinarith

theorem effectiveSize_mono (factors : List Nat) (factor : Nat)
    (valid : Admissible factors) (factorBound : 2 ≤ factor) :
    effectiveSize factors ≤ effectiveSize (factors ++ [factor]) := by
  rw [effectiveSize_append factors factor valid factorBound]
  omega

theorem admissible_append (before after : List Nat)
    (left : Admissible before) (right : Admissible after) : Admissible (before ++ after) := by
  intro factor member
  rcases List.mem_append.mp member with member | member
  · exact left factor member
  · exact right factor member

theorem effectiveSize_prefix (before after : List Nat)
    (left : Admissible before) (right : Admissible after) :
    effectiveSize before ≤ effectiveSize (before ++ after) := by
  induction after generalizing before with
  | nil => simp
  | cons head tail ih =>
      have headBound := right head (by simp)
      have single : Admissible [head] := by simpa [Admissible] using headBound
      have tailValid : Admissible tail := by
        intro factor member
        exact right factor (by simp [member])
      have first := effectiveSize_mono before head left headBound
      have rest := ih (before ++ [head]) (admissible_append before [head] left single) tailValid
      simpa only [List.append_assoc, List.singleton_append] using first.trans rest

/-- Each recursive descent strictly increases the product. -/
theorem descent (limit product factor : Nat) (positive : 1 ≤ product)
    (factorBound : 2 ≤ factor) (inside : product * factor ≤ limit) :
    limit + 1 - product * factor < limit + 1 - product := by
  have growth : product < product * factor := by nlinarith
  omega

end ProjectEulerSolutions.ProductSumNumbers
