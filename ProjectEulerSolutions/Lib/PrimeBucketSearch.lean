import ProjectEulerSolutions.Lib.PrimePermutationGroups
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.ScalarSetLemmas
import Mathlib.Data.List.TakeDrop
import Mathlib.Data.List.Sort

/-! Finite ordered-pair searches inside a digit-permutation bucket. -/
namespace ProjectEulerSolutions.PrimeBucketSearch
open PrimePermutationGroups
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

def third (a b : Nat) : Nat := b + (b - a)
def candidate (a b : Nat) : Row := (a,b,third a b)
def accepted (values : List Nat) (a b : Nat) : Bool :=
  decide (third a b ∈ values ∧ candidate a b ≠ known)

def row (values : List Nat) (a : Nat) : List Nat → Option Row
  | [] => none
  | b :: rest => if accepted values a b then some (candidate a b) else row values a rest

def pairs (values : List Nat) : List Nat → Option Row
  | [] => none
  | a :: rest => (row values a rest).orElse (fun _ => pairs values rest)

def Classifies (values : List Nat) : Prop :=
  ∀ a ∈ values, ∀ b ∈ values, a < b → third a b ∈ values →
    candidate a b = known ∨ candidate a b = winner

theorem row_some (values : List Nat) (a : Nat) (remaining : List Nat) (answer : Row)
    (result : row values a remaining = some answer) :
    ∃ b ∈ remaining, accepted values a b = true ∧ answer = candidate a b := by
  induction remaining with
  | nil => simp [row] at result
  | cons first rest ih =>
      cases selected : accepted values a first with
      | true =>
          have same : candidate a first = answer := by simpa [row, selected] using result
          exact ⟨first, List.mem_cons_self, selected, same.symm⟩
      | false =>
          obtain ⟨b, member, accepted, same⟩ := ih (by simpa [row, selected] using result)
          exact ⟨b, List.mem_cons_of_mem _ member, accepted, same⟩

theorem pairs_some (values remaining : List Nat) (ordered : remaining.Pairwise (· < ·)) (answer : Row)
    (result : pairs values remaining = some answer) :
    ∃ a ∈ remaining, ∃ b ∈ remaining, a < b ∧ accepted values a b = true ∧ answer = candidate a b := by
  induction remaining with
  | nil => simp [pairs] at result
  | cons first rest ih =>
      have bounds := List.pairwise_cons.mp ordered
      cases selected : row values first rest with
      | some hit =>
          have same : hit = answer := by simpa [pairs, selected] using result
          obtain ⟨b, member, accepted, candidate⟩ := row_some values first rest hit selected
          exact ⟨first, List.mem_cons_self, b, List.mem_cons_of_mem _ member, bounds.1 b member,
            accepted, same.symm.trans candidate⟩
      | none =>
          obtain ⟨a, aMember, b, bMember, ab, accepted, candidate⟩ :=
            ih bounds.2 (by simpa [pairs, selected] using result)
          exact ⟨a, List.mem_cons_of_mem _ aMember, b, List.mem_cons_of_mem _ bMember, ab, accepted, candidate⟩

/-- No successful scan can return an unclassified row or the excluded example. -/
theorem pairs_sound (values remaining : List Nat) (inside : ∀ n ∈ remaining, n ∈ values)
    (ordered : remaining.Pairwise (· < ·)) (classifies : Classifies values) (answer : Row)
    (result : pairs values remaining = some answer) : answer = winner := by
  obtain ⟨a, aMember, b, bMember, ab, accepted, same⟩ := pairs_some values remaining ordered answer result
  have valid : third a b ∈ values ∧ candidate a b ≠ known := of_decide_eq_true accepted
  exact same.trans ((classifies a (inside a aMember) b (inside b bMember) ab valid.1).resolve_left valid.2)

theorem row_has_candidate (values : List Nat) (a b : Nat) (remaining : List Nat)
    (member : b ∈ remaining) (accepted : accepted values a b = true) :
    (row values a remaining).isSome = true := by
  induction remaining with
  | nil => simp at member
  | cons first rest ih =>
      cases selected : PrimeBucketSearch.accepted values a first with
      | true => simp [row, selected]
      | false =>
          have tail : b ∈ rest := by
            rcases List.mem_cons.mp member with rfl | tail
            · simp [selected] at accepted
            · exact tail
          simpa [row, selected] using ih tail

/-- An accepted increasing pair cannot be lost by either finite scan. -/
theorem pairs_has_candidate (values remaining : List Nat) (ordered : remaining.Pairwise (· < ·))
    (a b : Nat) (aMember : a ∈ remaining) (bMember : b ∈ remaining) (ab : a < b)
    (accepted : accepted values a b = true) : (pairs values remaining).isSome = true := by
  induction remaining with
  | nil => simp at aMember
  | cons first rest ih =>
      have bounds := List.pairwise_cons.mp ordered
      cases selected : row values first rest with
      | some hit => simp [pairs, selected]
      | none =>
          by_cases same : first = a
          · subst first
            have tail : b ∈ rest := by
              rcases List.mem_cons.mp bMember with equal | member
              · omega
              · exact member
            have success := row_has_candidate values a b rest tail accepted
            simp [selected] at success
          · have aTail : a ∈ rest := (List.mem_cons.mp aMember).resolve_left (Ne.symm same)
            have bTail : b ∈ rest := by
              rcases List.mem_cons.mp bMember with equal | member
              · have before := bounds.1 a aTail
                omega
              · exact member
            simpa [pairs, selected] using ih bounds.2 aTail bTail

/-- A bucket containing the required three values returns their row, regardless of earlier candidates. -/
theorem complete (values : List Nat) (ordered : values.Pairwise (· < ·)) (classifies : Classifies values)
    (first : 2969 ∈ values) (second : 6299 ∈ values) (last : 9629 ∈ values) :
    pairs values values = some winner := by
  have accepted : accepted values 2969 6299 = true := by
    apply decide_eq_true
    exact ⟨by simpa only [third] using last, by decide⟩
  have success := pairs_has_candidate values values ordered 2969 6299 first second (by decide) accepted
  cases result : pairs values values with
  | none => simp [result] at success
  | some answer =>
      have same := pairs_sound values values (fun _ member => member) ordered classifies answer result
      simp only [same]

def scalarKeys (items : List Nat) : List Constant := items.map (fun (n : Nat) => .int n)
def numbers (items : List Nat) : List Mutable.Value := (scalarKeys items).map Mutable.Value.scalar
def setKeys (items : List Nat) : List Constant := items.eraseDups.map (fun (n : Nat) => .int n)

/-- Sorting an already ordered bucket writes the same list back to its original address. -/
theorem sort_ordered (heap : Mutable.Heap) (address : Nat) (items : List Nat)
    (found : heap[address]? = some (.list (numbers items))) (ordered : items.Pairwise (· ≤ ·)) :
    primitive (.sort address) [] heap = some (.ok (.scalar .none), heap) := by
  let integers := items.map (fun (n : Nat) => (n : Int))
  have orderedInts : integers.Pairwise (· ≤ ·) :=
    ordered.map (fun (n : Nat) => (n : Int)) (fun _ _ lower => Int.ofNat_le.mpr lower)
  have sorted : ScalarSort.ints integers = integers :=
    List.Perm.eq_of_pairwise (fun _ _ _ _ lower upper => le_antisymm lower upper)
      (ScalarSort.ints_ordered _) orderedInts (ScalarSort.ints_perm _)
  have keys : ScalarSort.sort (scalarKeys items) = .ok (scalarKeys items) := by
    have result := ScalarSort.sort_ints integers
    rw [sorted] at result
    simpa only [integers, scalarKeys, List.map_map, Function.comp_def] using result
  have unchanged : heap.set address (.list (numbers items)) = heap := by
    have bound := (List.getElem?_eq_some_iff.mp found).1
    have same := (List.getElem?_eq_some_iff.mp found).2
    rw [← same, List.set_getElem_self bound]
  have operation := primitive_sort_list heap address (scalarKeys items) (scalarKeys items) found keys
  change primitive (.sort address) [] heap = some (.ok (.scalar .none), heap.set address (.list (numbers items))) at operation
  simpa only [unchanged] using operation

theorem set_from_numbers (heap : Mutable.Heap) (address : Nat) (items : List Nat)
    (found : heap[address]? = some (.list (numbers items))) :
    Mutable.setFrom (.ref address) heap = some (.ok (.ref heap.length), heap ++ [.set (setKeys items)]) := by
  have built := Mutable.setFrom_list heap address (scalarKeys items) found
  simpa only [scalarKeys, ScalarSet.fold_insert_nats, setKeys] using built

/-- Membership in the freshly built scalar set is exactly membership in the original bucket. -/
theorem set_contains (items : List Nat) (n : Nat) :
    ScalarSet.contains (setKeys items) (.int n) = decide (n ∈ items) := by
  calc
    _ = decide ((n : Int) ∈ items.eraseDups.map (fun (x : Nat) => (x : Int))) := by
      simpa only [setKeys, List.map_map, Function.comp_def] using
        ScalarSet.contains_ints (items.eraseDups.map (fun (x : Nat) => (x : Int))) n
    _ = _ := by
      apply Bool.eq_iff_iff.mpr
      simp

end ProjectEulerSolutions.PrimeBucketSearch
