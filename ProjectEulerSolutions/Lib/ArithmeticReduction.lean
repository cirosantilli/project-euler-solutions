import ProjectEulerSolutions.Lib.ArithmeticExpressions
import Mathlib.Data.List.Forall2

/-! Pair reduction and exhaustive expression-tree enumeration describe the same rational values. -/
namespace ProjectEulerSolutions.ArithmeticExpressions
set_option Elab.async false

/-- Replace every leaf by a complete expression with that value. -/
theorem Derives.substitute {xs : List Rat} {value : Rat} (derived : Derives xs value)
    (chunks : List (List Rat)) (replacements : List.Forall₂ Derives chunks xs) :
    Derives chunks.flatten value := by
  induction derived generalizing chunks with
  | leaf a =>
      cases replacements with
      | cons replacement remaining =>
          cases remaining
          simpa using replacement
  | @branch left right a b value leftDerived rightDerived operation leftIH rightIH =>
      have replacedLeft := leftIH _ (List.forall₂_take_append chunks left right replacements)
      have replacedRight := rightIH _ (List.forall₂_drop_append chunks left right replacements)
      have together := Derives.branch replacedLeft replacedRight operation
      simpa only [← List.flatten_append, List.take_append_drop] using together

theorem singleton_replacements (xs : List Rat) : List.Forall₂ Derives (xs.map (fun x => [x])) xs := by
  induction xs with
  | nil => exact .nil
  | cons x xs ih => exact .cons (Derives.leaf x) ih

/-- Replace one designated occurrence, even when other leaves have the same value. -/
theorem Derives.replace {before after : List Rat} {a value : Rat}
    (derived : Derives (before ++ a :: after) value) {replacement : List Rat}
    (expanded : Derives replacement a) : Derives (before ++ replacement ++ after) value := by
  have substitutions := List.rel_append (singleton_replacements before)
    (List.Forall₂.cons expanded (singleton_replacements after))
  have flattenSingletons (xs : List Rat) : (xs.map (fun x => [x])).flatten = xs := by
    induction xs with
    | nil => rfl
    | cons x xs ih => simpa using ih
  simpa [flattenSingletons] using derived.substitute _ substitutions

theorem pair_front (before after : List Rat) (a b : Rat) :
    (before ++ a :: b :: after).Perm (a :: b :: (before ++ after)) := by
  exact List.perm_middle.trans (List.perm_middle.cons a)

/-- Reinsert an operation into any expression using its result once. -/
theorem Unordered.expand {rest : List Rat} {a b combined value : Rat}
    (derived : Unordered (combined :: rest) value) (operation : combined ∈ operations a b) :
    Unordered (a :: b :: rest) value := by
  obtain ⟨ordered, permutation, tree⟩ := derived
  have member : combined ∈ ordered := permutation.mem_iff.mpr (by simp)
  obtain ⟨before, after, rfl⟩ := List.mem_iff_append.mp member
  have remainder : (before ++ after).Perm rest := (List.perm_middle.symm.trans permutation).cons_inv
  refine ⟨before ++ a :: b :: after, (pair_front before after a b).trans ((remainder.cons b).cons a), ?_⟩
  simpa using tree.replace (Derives.branch (Derives.leaf a) (Derives.leaf b) operation)

/-- Choose any two current fragments, replace them by a defined operation, and repeat. -/
inductive Reduces : List Rat → Rat → Prop where
  | leaf (a : Rat) : Reduces [a] a
  | step {xs rest : List Rat} {a b combined value : Rat} :
      xs.Perm (a :: b :: rest) → combined ∈ operations a b →
      Reduces (combined :: rest) value → Reduces xs value

theorem Reduces.perm {xs ys : List Rat} {value : Rat} (derived : Reduces xs value)
    (permutation : xs.Perm ys) : Reduces ys value := by
  cases derived with
  | leaf =>
      have equal := List.perm_singleton.mp permutation.symm
      subst ys
      exact Reduces.leaf _
  | step selected operation continued =>
      exact Reduces.step (permutation.symm.trans selected) operation continued

theorem Reduces.sound {xs : List Rat} {value : Rat} (derived : Reduces xs value) :
    Unordered xs value := by
  induction derived with
  | leaf a => exact ⟨[a], .refl _, Derives.leaf a⟩
  | step selected operation _ ih =>
      exact (unordered_perm selected _).mpr (ih.expand operation)

/-- Sibling contraction reduces the leaf count, so every finite tree admits a complete pair reduction. -/
theorem Derives.reduce {xs : List Rat} {value : Rat} (derived : Derives xs value) : Reduces xs value := by
  rcases derived.leaf_or_pair with equal | ⟨before, a, b, after, combined, equal, operation, contracted⟩
  · rw [equal]
    exact Reduces.leaf _
  · have shorter := contracted.reduce
    rw [equal]
    exact Reduces.step (pair_front before after a b) operation (shorter.perm List.perm_middle)
termination_by xs.length
decreasing_by simp only [equal, List.length_append, List.length_cons]; omega

/-- Memoized pair reduction and exhaustive permutation/split enumeration have identical possible values. -/
theorem reduces_iff_mem_permuted (xs : List Rat) (value : Rat) :
    Reduces xs value ↔ value ∈ permuted xs := by
  rw [mem_permuted]
  constructor
  · exact Reduces.sound
  · rintro ⟨ordered, permutation, tree⟩
    exact tree.reduce.perm permutation

end ProjectEulerSolutions.ArithmeticExpressions
