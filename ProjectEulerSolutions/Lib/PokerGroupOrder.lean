import ProjectEulerSolutions.Lib.TupleSorting
import ProjectEulerSolutions.Lib.PokerCountPatterns

/-! Frequency/rank groups sort lexicographically while preserving their total mass. -/
namespace ProjectEulerSolutions.PokerGroupOrder
open FormalSW.PyAst

set_option Elab.async false

def row (pair : Int × Int) : List Int := [pair.1, pair.2]

def pairLess (left right : Int × Int) : Bool := HeapSort.lexInt (row left) (row right)

def pairLE (left right : Int × Int) : Prop :=
  left.1 < right.1 ∨ left.1 = right.1 ∧ left.2 ≤ right.2

def sortedPairs (reverse : Bool) (pairs : List (Int × Int)) : List (Int × Int) :=
  if reverse then (HeapSort.sortBy pairLess pairs.reverse).reverse
  else HeapSort.sortBy pairLess pairs

theorem pairLess_eq (left right : Int × Int) :
    pairLess left right =
      if left.1 = right.1 then decide (left.2 < right.2) else decide (left.1 < right.1) := by
  by_cases same : left.2 = right.2
  · simp [pairLess, row, HeapSort.lexInt, same]
  · simp [pairLess, row, HeapSort.lexInt, same]

theorem pairLess_true_le (left right : Int × Int) (compared : pairLess left right = true) :
    pairLE left right := by
  by_cases same : left.1 = right.1
  · have smaller : left.2 < right.2 := by simpa [pairLess_eq, same] using compared
    exact .inr ⟨same, by omega⟩
  · have smaller : left.1 < right.1 := by simpa [pairLess_eq, same] using compared
    exact .inl smaller

theorem pairLess_false_le (left right : Int × Int) (compared : pairLess left right = false) :
    pairLE right left := by
  by_cases same : left.1 = right.1
  · have notSmaller : ¬ left.2 < right.2 := by simpa [pairLess_eq, same] using compared
    exact .inr ⟨same.symm, by omega⟩
  · have notSmaller : ¬ left.1 < right.1 := by simpa [pairLess_eq, same] using compared
    exact .inl (by omega)

theorem pairLE_trans (first middle last : Int × Int)
    (firstMiddle : pairLE first middle) (middleLast : pairLE middle last) : pairLE first last := by
  unfold pairLE at firstMiddle middleLast ⊢
  omega

theorem pairLE_primary (left right : Int × Int) (ordered : pairLE left right) : left.1 ≤ right.1 := by
  unfold pairLE at ordered
  omega

/-- Sorting decoded two-field rows agrees with sorting the original integer pairs. -/
theorem sortedPairs_rows (reverse : Bool) (pairs : List (Int × Int)) :
    (sortedPairs reverse pairs).map row = TupleSorting.sortedRows reverse (pairs.map row) := by
  have mapped (values : List (Int × Int)) :
      (HeapSort.sortBy pairLess values).map row = HeapSort.sortBy HeapSort.lexInt (values.map row) :=
    HeapSort.sortBy_map row HeapSort.lexInt values
  cases reverse with
  | false => exact mapped pairs
  | true =>
      simpa only [sortedPairs, TupleSorting.sortedRows, ↓reduceIte, List.map_reverse] using
        congrArg List.reverse (mapped pairs.reverse)

theorem sortedPairs_perm (reverse : Bool) (pairs : List (Int × Int)) :
    (sortedPairs reverse pairs).Perm pairs := by
  cases reverse with
  | false => exact HeapSort.sortBy_perm pairLess pairs
  | true =>
      exact (List.reverse_perm _).trans
        ((HeapSort.sortBy_perm pairLess pairs.reverse).trans (List.reverse_perm pairs))

private theorem insert_ordered (value : Int × Int) (pairs : List (Int × Int))
    (ordered : pairs.Pairwise pairLE) : (HeapSort.insertBy pairLess value pairs).Pairwise pairLE := by
  revert ordered
  induction pairs with
  | nil => intro _; simp [HeapSort.insertBy]
  | cons first rest ih =>
      intro ordered
      obtain ⟨firstLower, tailOrdered⟩ := List.pairwise_cons.mp ordered
      by_cases before : pairLess first value = true
      · rw [HeapSort.insertBy, before]
        apply List.pairwise_cons.mpr
        refine ⟨?_, ih tailOrdered⟩
        intro item member
        have original := (HeapSort.insertBy_perm pairLess value rest).mem_iff.mp member
        cases List.mem_cons.mp original with
        | inl same => subst item; exact pairLess_true_le first value before
        | inr remaining => exact firstLower item remaining
      · have stopped : pairLess first value = false := Bool.eq_false_iff.mpr before
        rw [HeapSort.insertBy, stopped]
        apply List.pairwise_cons.mpr
        refine ⟨?_, ordered⟩
        intro item member
        have valueLower := pairLess_false_le first value stopped
        cases List.mem_cons.mp member with
        | inl same => subst item; exact valueLower
        | inr remaining => exact pairLE_trans value first item valueLower (firstLower item remaining)

theorem sortBy_ordered (pairs : List (Int × Int)) : (HeapSort.sortBy pairLess pairs).Pairwise pairLE := by
  induction pairs with
  | nil => exact .nil
  | cons first rest ih => exact insert_ordered first (HeapSort.sortBy pairLess rest) ih

/-- Both frequency and rank descend; equal-frequency pairs retain descending rank order. -/
theorem sortedPairs_descending (pairs : List (Int × Int)) :
    (sortedPairs true pairs).Pairwise (fun left right => pairLE right left) := by
  change (HeapSort.sortBy pairLess pairs.reverse).reverse.Pairwise (fun left right => pairLE right left)
  exact List.pairwise_reverse.mpr (sortBy_ordered pairs.reverse)

theorem sortedPairs_frequencies_descending (pairs : List (Int × Int)) :
    ((sortedPairs true pairs).map Prod.fst).Pairwise (fun left right : Int => right ≤ left) := by
  apply List.pairwise_map.mpr
  exact (sortedPairs_descending pairs).imp (fun {left right} related => pairLE_primary right left related)

theorem sortedPairs_positive (reverse : Bool) (pairs : List (Int × Int))
    (positive : ∀ pair ∈ pairs, 0 < pair.1) :
    ∀ pair ∈ sortedPairs reverse pairs, 0 < pair.1 :=
  fun pair member => positive pair ((sortedPairs_perm reverse pairs).mem_iff.mp member)

private theorem sum_of_perm {left right : List Int} (same : left.Perm right) : left.sum = right.sum := by
  induction same with
  | nil => rfl
  | cons _ _ ih => simp only [List.sum_cons, ih]
  | swap => simp only [List.sum_cons]; omega
  | trans _ _ first second => exact first.trans second

theorem sortedPairs_sum (reverse : Bool) (pairs : List (Int × Int)) :
    ((sortedPairs reverse pairs).map Prod.fst).sum = (pairs.map Prod.fst).sum :=
  sum_of_perm ((sortedPairs_perm reverse pairs).map Prod.fst)

/-- Sorting a five-card frequency table yields one of the complete category profiles. -/
theorem sorted_profile (pairs : List (Int × Int)) (positive : ∀ pair ∈ pairs, 0 < pair.1)
    (total : (pairs.map Prod.fst).sum = 5) :
    (sortedPairs true pairs).map Prod.fst ∈ PokerCountPatterns.profiles := by
  apply PokerCountPatterns.classify
  · intro count member
    obtain ⟨pair, pairMember, rfl⟩ := List.mem_map.mp member
    exact sortedPairs_positive true pairs positive pair pairMember
  · exact sortedPairs_frequencies_descending pairs
  · rw [sortedPairs_sum]
    exact total

theorem row_frequencies (reverse : Bool) (pairs : List (Int × Int)) :
    (TupleSorting.sortedRows reverse (pairs.map row)).map (fun values => values.headD 0) =
      (sortedPairs reverse pairs).map Prod.fst := by
  rw [← sortedPairs_rows]
  simp only [List.map_map, Function.comp_def, row, List.headD_cons]

theorem rows_profile (pairs : List (Int × Int)) (positive : ∀ pair ∈ pairs, 0 < pair.1)
    (total : (pairs.map Prod.fst).sum = 5) :
    (TupleSorting.sortedRows true (pairs.map row)).map (fun values => values.headD 0) ∈ PokerCountPatterns.profiles := by
  rw [row_frequencies]
  exact sorted_profile pairs positive total

end ProjectEulerSolutions.PokerGroupOrder
