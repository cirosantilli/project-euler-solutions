import ProjectEulerSolutions.Lib.SudokuMasks

/-! Row-major heap lists and decimal text for the fixed nine-by-nine Sudoku board. -/
namespace ProjectEulerSolutions.Sudoku.Encoding

set_option Elab.async false
noncomputable section

def cell (index : Fin 81) : Cell :=
  (⟨index.val / 9, by have := index.isLt; omega⟩, ⟨index.val % 9, Nat.mod_lt _ (by decide)⟩)

def index (position : Cell) : Fin 81 :=
  ⟨position.1.val * 9 + position.2.val, by have := position.1.isLt; have := position.2.isLt; omega⟩

@[simp] theorem cell_index (position : Cell) : cell (index position) = position := by
  apply Prod.ext <;> apply Fin.ext <;> simp only [cell, index]
  · have := position.2.isLt; omega
  · have := position.2.isLt; omega

@[simp] theorem index_cell (slot : Fin 81) : index (cell slot) = slot := by
  apply Fin.ext
  simp only [index, cell]
  omega

def character : Option Digit → Char
  | none => '0'
  | some digit => Char.ofNat (49 + digit.val)

theorem digit_text (digit : Digit) : String.singleton (character (some digit)) = (digit.val + 1).repr := by
  revert digit
  decide

theorem digit_nonzero (digit : Digit) : character (some digit) ≠ '0' := by
  revert digit
  decide

def row (board : Board) (r : Fin 9) : String :=
  String.ofList (List.ofFn fun c => character (board (r, c)))

def rows (board : Board) : List String := List.ofFn (row board)

@[simp] theorem row_length (board : Board) (r : Fin 9) : (row board r).length = 9 := by
  simp [row]

theorem row_get (board : Board) (r c : Fin 9) :
    (row board r).toList[c.val]? = some (character (board (r, c))) := by
  simp only [row, String.toList_ofList, List.getElem?_ofFn, dif_pos c.isLt]

theorem rows_get (board : Board) (r : Fin 9) : (rows board)[r.val]? = some (row board r) := by
  simp only [rows, List.getElem?_ofFn, dif_pos r.isLt]

def mask (board : Board) (position : Cell) : Nat := Masks.encode (initialCandidates board position)

theorem mask_none (board : Board) (position : Cell) (empty : board position = none) :
    mask board position = 511 := by simp [mask, initialCandidates, empty, Masks.all]

theorem mask_some (board : Board) (position : Cell) (digit : Digit) (filled : board position = some digit) :
    mask board position = 2 ^ digit.val := by simp [mask, initialCandidates, filled]

/-- Cells before the cursor have been read; the remaining cells still contain the initial full mask. -/
def prepared (board : Board) (cursor : Nat) : List Nat :=
  List.ofFn fun slot : Fin 81 => if slot.val < cursor then mask board (cell slot) else 511

@[simp] theorem prepared_length (board : Board) (cursor : Nat) : (prepared board cursor).length = 81 := by
  simp [prepared]

theorem prepared_zero (board : Board) : prepared board 0 = List.replicate 81 511 := by
  simp [prepared]

theorem prepared_done (board : Board) :
    prepared board 81 = List.ofFn (fun slot => mask board (cell slot)) := by
  unfold prepared
  congr 1
  funext slot
  simp [slot.isLt]

theorem prepared_get (board : Board) (cursor : Nat) (slot : Fin 81) :
    (prepared board cursor)[slot.val]? = some (if slot.val < cursor then mask board (cell slot) else 511) := by
  simp only [prepared, List.getElem?_ofFn, dif_pos slot.isLt]

theorem prepared_step (board : Board) (slot : Fin 81) :
    (prepared board slot.val).set slot.val (mask board (cell slot)) = prepared board (slot.val + 1) := by
  apply List.ext_getElem?
  intro other
  by_cases inside : other < 81
  · rw [List.getElem?_set]
    by_cases same : slot.val = other
    · subst other
      simp only [prepared_length, if_pos slot.isLt, prepared_get, Nat.lt_succ_self, if_true]
    · have otherSlot : (⟨other, inside⟩ : Fin 81).val = other := rfl
      rw [show (prepared board slot.val)[other]? = _ from prepared_get board slot.val ⟨other, inside⟩,
        show (prepared board (slot.val + 1))[other]? = _ from prepared_get board (slot.val + 1) ⟨other, inside⟩]
      simp only [same, if_false]
      have order : other < slot.val + 1 ↔ other < slot.val := by omega
      simp only [order]
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by simp; omega)]

theorem prepared_skip (board : Board) (slot : Fin 81) (empty : board (cell slot) = none) :
    prepared board slot.val = prepared board (slot.val + 1) := by
  have step := prepared_step board slot
  rw [mask_none board (cell slot) empty] at step
  have current := prepared_get board slot.val slot
  simp only [lt_self_iff_false, if_false] at current
  have bound : slot.val < (prepared board slot.val).length := by simpa only [prepared_length] using slot.isLt
  have atSlot : (prepared board slot.val)[slot.val]'bound = 511 := by
    simpa only [List.getElem?_eq_getElem bound, Option.some.injEq] using current
  rw [← atSlot, List.set_getElem_self] at step
  exact step

end
end ProjectEulerSolutions.Sudoku.Encoding
