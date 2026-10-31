import Mathlib.Data.Rat.Lemmas
import Mathlib.Data.List.Range
import Mathlib.Data.List.Permutation
import Mathlib.Tactic

/-! Exact arithmetic expressions using every input once, without concatenation. -/
namespace ProjectEulerSolutions.ArithmeticExpressions
set_option Elab.async false

/-- The four ordered operations, with division admitted exactly when defined. -/
def operations (a b : Rat) : List Rat :=
  [a + b, a - b, a * b] ++ if b = 0 then [] else [a / b]

/-- A binary expression tree whose leaves, read left to right, are exactly the inputs. -/
inductive Derives : List Rat → Rat → Prop where
  | leaf (a : Rat) : Derives [a] a
  | branch {xs ys : List Rat} {a b result : Rat} :
      Derives xs a → Derives ys b → result ∈ operations a b → Derives (xs ++ ys) result

theorem Derives.nonempty {xs : List Rat} {value : Rat} (derived : Derives xs value) : xs ≠ [] := by
  induction derived with
  | leaf => simp
  | @branch xs ys _ _ _ _ _ _ left _ => cases xs <;> simp_all

theorem Derives.singleton {a value : Rat} (derived : Derives [a] value) : value = a := by
  generalize input : [a] = xs at derived
  cases derived with
  | leaf => simpa using input.symm
  | branch hleft hright operation =>
      have leftPos := List.length_pos_iff.mpr hleft.nonempty
      have rightPos := List.length_pos_iff.mpr hright.nonempty
      have lengths := congrArg List.length input
      simp only [List.length_append, List.length_cons, List.length_nil] at lengths
      omega

theorem Derives.decompose {xs : List Rat} {value : Rat} (derived : Derives xs value)
    (size : 2 ≤ xs.length) :
    ∃ left right a b, xs = left ++ right ∧ Derives left a ∧ Derives right b ∧ value ∈ operations a b := by
  cases derived with
  | leaf => simp at size
  | branch left right operation => exact ⟨_, _, _, _, rfl, left, right, operation⟩

/-- Recursively split at every interior position; no associativity shape is omitted. -/
def enumerate (xs : List Rat) : List Rat :=
  match xs with
  | [] => []
  | [a] => [a]
  | a :: b :: rest =>
      (List.range' 1 (rest.length + 1)).attach.flatMap fun (split : {i // i ∈ List.range' 1 (rest.length + 1)}) =>
        have lower : 0 < split.val := by
          obtain ⟨i, bound, equal⟩ := List.mem_range'.mp split.property
          omega
        have upper : split.val < (a :: b :: rest).length := by
          obtain ⟨i, bound, equal⟩ := List.mem_range'.mp split.property
          simp only [List.length_cons]
          omega
        (enumerate ((a :: b :: rest).take split.val)).flatMap fun left =>
          (enumerate ((a :: b :: rest).drop split.val)).flatMap fun right => operations left right
termination_by xs.length
decreasing_by
  all_goals simp_wf
  all_goals simp only [List.length_cons] at *
  all_goals omega

/-- Membership in the exhaustive list is exactly derivability by an ordered binary tree. -/
theorem mem_enumerate (xs : List Rat) (value : Rat) : value ∈ enumerate xs ↔ Derives xs value := by
  revert value
  refine (measure List.length).wf.induction (C := fun inputs => ∀ value, value ∈ enumerate inputs ↔ Derives inputs value) xs ?_
  intro xs ih value
  cases xs with
  | nil =>
      simp only [enumerate, List.not_mem_nil, false_iff]
      exact fun derived => derived.nonempty rfl
  | cons a tail =>
      cases tail with
      | nil =>
          simp only [enumerate, List.mem_singleton]
          exact ⟨fun h => h ▸ Derives.leaf a, fun h => h.singleton⟩
      | cons b rest =>
          rw [enumerate]
          constructor
          · intro member
            obtain ⟨split, _, member⟩ := List.mem_flatMap.mp member
            obtain ⟨left, leftMem, member⟩ := List.mem_flatMap.mp member
            obtain ⟨right, rightMem, operation⟩ := List.mem_flatMap.mp member
            obtain ⟨i, bound, equal⟩ := List.mem_range'.mp split.property
            have lower : 0 < split.val := by omega
            have upper : split.val < (a :: b :: rest).length := by simp only [List.length_cons]; omega
            have leftDerived := (ih _ (by change List.length _ < List.length _; simp only [List.length_take]; omega) left).mp leftMem
            have rightDerived := (ih _ (by change List.length _ < List.length _; simp only [List.length_drop]; omega) right).mp rightMem
            simpa using Derives.branch leftDerived rightDerived operation
          · intro derived
            obtain ⟨left, right, x, y, input, leftDerived, rightDerived, operation⟩ :=
              derived.decompose (by simp)
            have leftPos := List.length_pos_iff.mpr leftDerived.nonempty
            have rightPos := List.length_pos_iff.mpr rightDerived.nonempty
            have lengths := congrArg List.length input
            simp only [List.length_cons, List.length_append] at lengths
            have cutMem : left.length ∈ List.range' 1 (rest.length + 1) := by
              apply List.mem_range'.mpr
              exact ⟨left.length - 1, by omega, by omega⟩
            have take : (a :: b :: rest).take left.length = left := by rw [input]; simp
            have drop : (a :: b :: rest).drop left.length = right := by rw [input]; simp
            apply List.mem_flatMap.mpr
            refine ⟨⟨left.length, cutMem⟩, by simp, ?_⟩
            change value ∈ (enumerate ((a :: b :: rest).take left.length)).flatMap _
            rw [take, drop]
            apply List.mem_flatMap.mpr
            refine ⟨x, (ih left (by change left.length < (a :: b :: rest).length; simp only [List.length_cons]; omega) x).mpr leftDerived, ?_⟩
            apply List.mem_flatMap.mpr
            exact ⟨y, (ih right (by change right.length < (a :: b :: rest).length; simp only [List.length_cons]; omega) y).mpr rightDerived, operation⟩

/-- Arbitrary operand order, retaining multiplicity even when some operands are equal. -/
def Unordered (xs : List Rat) (value : Rat) : Prop :=
  ∃ ordered, ordered.Perm xs ∧ Derives ordered value

def permuted (xs : List Rat) : List Rat := xs.permutations.flatMap enumerate

theorem mem_permuted (xs : List Rat) (value : Rat) : value ∈ permuted xs ↔ Unordered xs value := by
  simp only [permuted, List.mem_flatMap, List.mem_permutations, mem_enumerate, Unordered]

theorem unordered_perm {xs ys : List Rat} (permutation : xs.Perm ys) (value : Rat) :
    Unordered xs value ↔ Unordered ys value := by
  constructor
  · rintro ⟨ordered, order, derived⟩
    exact ⟨ordered, order.trans permutation, derived⟩
  · rintro ⟨ordered, order, derived⟩
    exact ⟨ordered, order.trans permutation.symm, derived⟩

/-- A nontrivial tree has two sibling leaves that can be replaced by their operation result. -/
theorem Derives.leaf_or_pair {xs : List Rat} {value : Rat} (derived : Derives xs value) :
    xs = [value] ∨ ∃ before a b after combined,
      xs = before ++ a :: b :: after ∧ combined ∈ operations a b ∧
        Derives (before ++ combined :: after) value := by
  induction derived with
  | leaf => exact Or.inl rfl
  | @branch left right x y result leftDerived rightDerived operation leftIH rightIH =>
      rcases leftIH with rfl | ⟨before, a, b, after, combined, rfl, chosen, contracted⟩
      · rcases rightIH with rfl | ⟨before, a, b, after, combined, rfl, chosen, contracted⟩
        · exact Or.inr ⟨[], x, y, [], result, rfl, operation, Derives.leaf result⟩
        · refine Or.inr ⟨x :: before, a, b, after, combined, by simp, chosen, ?_⟩
          simpa using Derives.branch leftDerived contracted operation
      · refine Or.inr ⟨before, a, b, after ++ right, combined, by simp [List.append_assoc], chosen, ?_⟩
        simpa [List.append_assoc] using Derives.branch contracted rightDerived operation

end ProjectEulerSolutions.ArithmeticExpressions
