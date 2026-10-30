import Mathlib.Data.List.Basic
import Mathlib.Data.Int.Basic

namespace ProjectEulerSolutions.SortedNatTree

/-- A closed balanced set can be reduced without scanning its whole row. -/
inductive Tree where
  | empty
  | node (key : Nat) (left right : Tree)

def values : Tree → List Nat
  | .empty => []
  | .node key left right => values left ++ key :: values right

def contains : Tree → Nat → Bool
  | .empty, _ => false
  | .node key left right, query =>
      if query < key then contains left query else
        if key < query then contains right query else true

def ordered : Tree → Bool
  | .empty => true
  | .node key left right =>
      (ordered left && ordered right) &&
        ((values left).all (fun q => decide (q < key)) && (values right).all (fun q => decide (key < q)))

theorem contains_iff (tree : Tree) (valid : ordered tree = true) (query : Nat) :
    contains tree query = true ↔ query ∈ values tree := by
  induction tree with
  | empty => simp [contains, values]
  | node key left right ihLeft ihRight =>
    simp only [ordered, Bool.and_eq_true] at valid
    have before : ∀ q ∈ values left, q < key := by
      simpa only [List.all_eq_true, decide_eq_true_eq] using valid.2.1
    have after : ∀ q ∈ values right, key < q := by
      simpa only [List.all_eq_true, decide_eq_true_eq] using valid.2.2
    simp only [contains, values, List.mem_append, List.mem_cons]
    by_cases lower : query < key
    · rw [if_pos lower, ihLeft valid.1.1]
      constructor
      · exact Or.inl
      · rintro (member | equal | member)
        · exact member
        · omega
        · have := after query member
          omega
    · rw [if_neg lower]
      by_cases higher : key < query
      · rw [if_pos higher, ihRight valid.1.2]
        constructor
        · exact fun member => Or.inr (Or.inr member)
        · rintro (member | equal | member)
          · have := before query member
            omega
          · omega
          · exact member
      · have equal : query = key := by omega
        simp [equal]

theorem contains_eq_list (tree : Tree) (valid : ordered tree = true) :
    contains tree = (values tree).contains := by
  funext query
  apply Bool.eq_iff_iff.mpr
  simpa only [List.contains_iff_mem] using contains_iff tree valid query

end ProjectEulerSolutions.SortedNatTree
