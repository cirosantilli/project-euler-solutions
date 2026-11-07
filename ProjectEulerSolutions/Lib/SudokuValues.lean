import ProjectEulerSolutions.Lib.SudokuSearch
import ProjectEulerSolutions.Lib.ListEraseDups

/-! The reference solver represents an empty cell by zero and a digit by its value from one to nine. -/
namespace ProjectEulerSolutions.Sudoku.Values

set_option Elab.async false

def encode : Option Digit → Int
  | none => 0
  | some digit => digit.val + 1

@[simp] theorem zero (entry : Option Digit) : encode entry = 0 ↔ entry = none := by
  cases entry with
  | none => simp [encode]
  | some digit => simp only [encode, Option.some_ne_none, iff_false]; omega

theorem bounded (entry : Option Digit) : 0 ≤ encode entry ∧ encode entry ≤ 9 := by
  cases entry with
  | none => simp [encode]
  | some digit => have := digit.isLt; simp only [encode]; omega

theorem injective : Function.Injective encode := by
  intro left right same
  cases left with
  | none => exact ((zero right).mp same.symm).symm
  | some a =>
      cases right with
      | none => simp only [encode] at same; omega
      | some b =>
          congr 1
          apply Fin.ext
          simp only [encode] at same
          omega

def nonzero (items : List Int) : List Int := items.filter (· ≠ 0)

theorem nonzero_options (items : List (Option Digit)) :
    nonzero (items.map encode) = (items.filterMap id).map (fun digit => (digit.val + 1 : Int)) := by
  induction items with
  | nil => rfl
  | cons head tail ih =>
      cases head with
      | none => simpa [nonzero, encode] using ih
      | some digit =>
          have positive : (digit.val + 1 : Int) ≠ 0 := by omega
          simpa [nonzero, encode, positive] using congrArg (List.cons (digit.val + 1 : Int)) ih

/-- The reference's set-length comparison is exactly a no-repeated-digit check. -/
theorem distinct_iff (items : List Int) :
    (nonzero items).eraseDups.length = (nonzero items).length ↔ (nonzero items).Nodup := by
  rw [ListEraseDups.length_eq_card]
  exact Multiset.toFinset_card_eq_card_iff_nodup

def unit (board : Board) (positions : Fin 9 → Cell) : List Int :=
  (List.finRange 9).map fun i => encode (board (positions i))

theorem unit_length (board : Board) (positions : Fin 9 → Cell) : (unit board positions).length = 9 := by
  simp [unit]

theorem unit_valid (board : Board) (valid : Valid board) (positions : Fin 9 → Cell)
    (peers : ∀ i j, i ≠ j → Peer (positions i) (positions j)) :
    (nonzero (unit board positions)).Nodup := by
  have filled : ((List.finRange 9).filterMap fun i => board (positions i)).Nodup := by
    apply List.Pairwise.filterMap (fun i => board (positions i)) _ (List.nodup_finRange 9)
    intro i j different left atLeft right atRight equal
    subst right
    exact valid (positions i) (positions j) (peers i j different) left atLeft atRight
  have converted := nonzero_options ((List.finRange 9).map fun i => board (positions i))
  simp only [List.map_map, List.filterMap_map, Function.comp_def] at converted
  rw [show nonzero (unit board positions) = _ from converted]
  apply filled.map
  intro a b same
  apply Fin.ext
  change (a.val : Int) + 1 = (b.val : Int) + 1 at same
  omega

def row (board : Board) (r : Fin 9) : List Int := unit board (fun c => (r, c))
def column (board : Board) (c : Fin 9) : List Int := unit board (fun r => (r, c))
def box (board : Board) (r c : Fin 3) : List Int := unit board (boxCell r c)

theorem row_valid (board : Board) (valid : Valid board) (r : Fin 9) : (nonzero (row board r)).Nodup :=
  unit_valid board valid _ (fun i j different => ⟨by simpa using different, Or.inl rfl⟩)

theorem column_valid (board : Board) (valid : Valid board) (c : Fin 9) : (nonzero (column board c)).Nodup :=
  unit_valid board valid _ (fun i j different => ⟨by simpa using different, Or.inr (Or.inl rfl)⟩)

theorem box_valid (board : Board) (valid : Valid board) (r c : Fin 3) : (nonzero (box board r c)).Nodup :=
  unit_valid board valid _ (boxCell_peer r c)

theorem unit_get (board : Board) (positions : Fin 9 → Cell) (i : Fin 9) :
    (unit board positions)[i.val]? = some (encode (board (positions i))) := by
  rw [unit, ← List.ofFn_eq_map]
  simp only [List.getElem?_ofFn, dif_pos i.isLt]

theorem row_update (board : Board) (r c : Fin 9) (entry : Option Digit) :
    row (Function.update board (r, c) entry) r = (row board r).set c.val (encode entry) := by
  apply List.ext_getElem?
  intro index
  by_cases inside : index < 9
  · rw [List.getElem?_set]
    have before := unit_get board (fun j => (r, j)) ⟨index, inside⟩
    have after := unit_get (Function.update board (r, c) entry) (fun j => (r, j)) ⟨index, inside⟩
    change (row board r)[index]? = _ at before
    change (row (Function.update board (r, c) entry) r)[index]? = _ at after
    rw [before, after]
    by_cases same : c.val = index
    · have equal : (⟨index, inside⟩ : Fin 9) = c := Fin.ext same.symm
      rw [equal, Function.update_self, if_pos same]
      rw [if_pos (show c.val < (row board r).length by simpa only [row, unit_length] using c.isLt)]
    · have different : (r, (⟨index, inside⟩ : Fin 9)) ≠ (r, c) := by
        intro equal
        exact same (congrArg (fun p : Cell => p.2.val) equal).symm
      simp [same, Function.update_of_ne different]
  · rw [List.getElem?_eq_none (by simpa [row, unit_length] using Nat.le_of_not_lt inside),
      List.getElem?_eq_none (by simpa [row, unit_length] using Nat.le_of_not_lt inside)]

theorem row_update_other (board : Board) (r c other : Fin 9) (entry : Option Digit) (different : other ≠ r) :
    row (Function.update board (r, c) entry) other = row board other := by
  unfold row unit
  apply List.map_congr_left
  intro column _
  rw [Function.update_of_ne (show (other, column) ≠ (r, c) from fun same => different (congrArg Prod.fst same))]

end ProjectEulerSolutions.Sudoku.Values
