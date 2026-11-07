import lang.py.FormalSW.Permutations
import lang.py.FormalSW.HeapSort
import Mathlib.Data.List.Pairwise
import Mathlib.Data.List.InsertIdx

/-! Factoradic unranking and lexicographically ordered finite permutations.
These results concern lists and arithmetic only, independently of source execution. -/
open FormalSW.PyAst
namespace ProjectEulerSolutions.PermutationRank

private theorem choices_values (items : List α) :
    (Permutations.choices items).map Prod.fst = items := by
  induction items with
  | nil => rfl
  | cons first rest ih => simp [Permutations.choices, List.map_map, Function.comp_def, ih]

private theorem choice_sublist (selected : (value, remaining) ∈ Permutations.choices items) :
    remaining.Sublist items := by
  induction items generalizing remaining with
  | nil => simp [Permutations.choices] at selected
  | cons first rest ih =>
      simp only [Permutations.choices, List.mem_cons, List.mem_map] at selected
      rcases selected with equal | ⟨⟨other, tail⟩, member, equal⟩
      · cases equal
        exact List.sublist_cons_self _ _
      · cases equal
        exact (ih member).cons_cons first

/-- With a strictly increasing pool, positional permutations are already lexicographic. -/
theorem rows_sorted (length : Nat) (items : List Int) (sorted : items.Pairwise (· < ·)) :
    (Permutations.rows length items).Pairwise (fun a b => HeapSort.lexInt a b = true) := by
  induction length generalizing items with
  | zero => simp [Permutations.rows]
  | succ length ih =>
      rw [Permutations.rows, List.pairwise_flatMap]
      constructor
      · intro pair selected
        have shorter := sorted.sublist (choice_sublist selected)
        apply List.Pairwise.map
        · intro a b less
          simpa [HeapSort.lexInt] using less
        · exact ih pair.2 shorter
      · have choices : (Permutations.choices items).Pairwise (fun a b => a.1 < b.1) := by
          apply List.pairwise_map.mp
          rw [choices_values]
          exact sorted
        apply choices.imp
        intro left right less a aMember b bMember
        obtain ⟨aTail, _, rfl⟩ := List.mem_map.mp aMember
        obtain ⟨bTail, _, rfl⟩ := List.mem_map.mp bMember
        simp [HeapSort.lexInt, Int.ne_of_lt less, less]

private theorem lex_asymm (a b : List Int) (less : HeapSort.lexInt a b = true) :
    HeapSort.lexInt b a = false := by
  induction a generalizing b with
  | nil => cases b <;> simp_all [HeapSort.lexInt]
  | cons first rest ih =>
      cases b with
      | nil => simp [HeapSort.lexInt] at less
      | cons next tail =>
          by_cases same : first = next
          · subst next
            simpa [HeapSort.lexInt] using ih tail (by simpa [HeapSort.lexInt] using less)
          · have order : first < next := by simpa [HeapSort.lexInt, same] using less
            simp [HeapSort.lexInt, Ne.symm same]; omega

private theorem sort_sorted (items : List (List Int))
    (sorted : items.Pairwise (fun a b => HeapSort.lexInt a b = true)) :
    HeapSort.sortBy HeapSort.lexInt items = items := by
  induction items with
  | nil => rfl
  | cons first rest ih =>
      obtain ⟨before, ordered⟩ := List.pairwise_cons.mp sorted
      rw [HeapSort.sortBy, ih ordered]
      cases rest with
      | nil => rfl
      | cons next tail =>
          simp [HeapSort.insertBy, lex_asymm first next (before next List.mem_cons_self)]

theorem sort_rows (items : List Int) (sorted : items.Pairwise (· < ·)) :
    HeapSort.sortBy HeapSort.lexInt (Permutations.rows items.length items) =
      Permutations.rows items.length items := sort_sorted _ (rows_sorted _ _ sorted)


private theorem choices_at (items : List α) (position : Nat) (bound : position < items.length) :
    (Permutations.choices items)[position]? = some (items[position], items.eraseIdx position) := by
  induction items generalizing position with
  | nil => simp at bound
  | cons first rest ih =>
      cases position with
      | zero => rfl
      | succ position =>
          have inside : position < rest.length := by simpa using bound
          simpa [Permutations.choices, List.getElem?_map] using
            congrArg (Option.map (fun pair : α × List α => (pair.1, first :: pair.2))) (ih position inside)

private theorem flatten_at (blocks : List (List α)) (width block offset : Nat)
    (sizes : ∀ row ∈ blocks, row.length = width)
    (blockBound : block < blocks.length) (offsetBound : offset < width) :
    blocks.flatten[block * width + offset]? = blocks[block][offset]? := by
  induction blocks generalizing block with
  | nil => simp at blockBound
  | cons first rest ih =>
      have size := sizes first List.mem_cons_self
      cases block with
      | zero =>
          simpa only [List.flatten_cons, Nat.zero_mul, Nat.zero_add, List.getElem_cons_zero] using
            (List.getElem?_append_left (l₁ := first) (l₂ := rest.flatten) (by omega : offset < first.length))
      | succ block =>
          have bound : block < rest.length := by simpa using blockBound
          have later : first.length ≤ (block + 1) * width + offset := by rw [size, Nat.succ_mul]; omega
          have index : (block + 1) * width + offset - first.length = block * width + offset := by
            rw [size, Nat.succ_mul]; omega
          rw [List.flatten_cons, List.getElem?_append_right later, index]
          exact ih block (fun row member => sizes row (List.mem_cons_of_mem _ member)) bound

theorem factorial_positive (size : Nat) : 0 < Math.factorial size := by
  induction size with
  | zero => decide
  | succ size ih => exact Nat.mul_pos (by omega) ih

/-- Choose a factorial-sized block, then repeat within the remaining pool. -/
def unrank : Nat → List Int → Nat → List Int
  | 0, _, _ => []
  | size + 1, items, rank =>
      let position := rank / Math.factorial size
      items[position]?.getD 0 :: unrank size (items.eraseIdx position) (rank % Math.factorial size)

/-- The factoradic decomposition selects exactly the matching positional permutation. -/
theorem unrank_at (size : Nat) (items : List Int) (rank : Nat)
    (length : items.length = size) (bound : rank < Math.factorial size) :
    (Permutations.rows size items)[rank]? = some (unrank size items rank) := by
  induction size generalizing items rank with
  | zero =>
      have zero : rank = 0 := by simpa [Math.factorial] using bound
      subst rank
      rfl
  | succ size ih =>
      let width := Math.factorial size
      let position := rank / width
      have positive : 0 < width := factorial_positive size
      have positionBound : position < items.length := by
        apply (Nat.div_lt_iff_lt_mul positive).mpr
        simpa [length, width, Math.factorial] using bound
      have offsetBound : rank % width < width := Nat.mod_lt rank positive
      have choice := choices_at items position positionBound
      let blocks := (Permutations.choices items).map fun pair =>
        (Permutations.rows size pair.2).map (pair.1 :: ·)
      have blockSizes : ∀ row ∈ blocks, row.length = width := by
        intro row member
        obtain ⟨pair, selected, rfl⟩ := List.mem_map.mp member
        have shorter := Permutations.choice_length selected
        have remaining : pair.2.length = size := by omega
        simp [List.length_map, Permutations.rows_length, remaining, Permutations.count_full, width]
      have blockBound : position < blocks.length := by simpa [blocks] using positionBound
      have flat := flatten_at blocks width position (rank % width) blockSizes blockBound offsetBound
      have decomposition : position * width + rank % width = rank := by
        simpa [position, Nat.mul_comm] using Nat.div_add_mod rank width
      rw [decomposition] at flat
      have blockAt : blocks[position] =
          (Permutations.rows size (items.eraseIdx position)).map (items[position] :: ·) := by
        have mapped := congrArg (Option.map (fun pair : Int × List Int =>
          (Permutations.rows size pair.2).map (pair.1 :: ·))) choice
        simpa [blocks, ← List.getElem?_map, List.getElem?_eq_getElem blockBound] using mapped
      have shorter : (items.eraseIdx position).length = size := by
        rw [List.length_eraseIdx_of_lt positionBound, length]
        omega
      have recursive := ih (items.eraseIdx position) (rank % width) shorter offsetBound
      rw [blockAt, List.getElem?_map, recursive] at flat
      simpa [Permutations.rows, List.flatMap, blocks, unrank, position, width,
        List.getElem?_eq_getElem positionBound] using flat

def selectRow (rows : List (List Int)) (index : Int) : List Int :=
  if 0 ≤ index - 1 ∧ index - 1 < rows.length then rows[(index - 1).toNat]?.getD [] else []

def reference (digits : List Int) (index : Int) : List Int :=
  selectRow (HeapSort.sortBy HeapSort.lexInt (Permutations.rows digits.length digits)) index

/-- On the optimized solver's domain, sorting and one-based selection give the factoradic result. -/
theorem reference_unrank (items : List Int) (rank : Nat) (sorted : items.Pairwise (· < ·))
    (bound : rank < Math.factorial items.length) :
    reference items ((rank : Int) + 1) =
      unrank items.length items rank := by
  rw [reference, sort_rows items sorted]
  have length := Permutations.rows_full_length items
  have valid : 0 ≤ (rank : Int) + 1 - 1 ∧ (rank : Int) + 1 - 1 < (Permutations.rows items.length items).length := by
    rw [length]
    omega
  rw [selectRow, if_pos valid]
  simp only [Int.add_sub_cancel, Int.toNat_natCast, unrank_at items.length items rank rfl bound, Option.getD_some]

end ProjectEulerSolutions.PermutationRank
