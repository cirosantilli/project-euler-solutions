import ProjectEulerSolutions.Lib.SudokuValues

/-! Row, column, and box membership tests characterize legal digits at an empty cell. -/
namespace ProjectEulerSolutions.Sudoku.Values
set_option Elab.async false

def block (coordinate : Fin 9) : Fin 3 := ⟨coordinate.val / 3, by omega⟩

def boxCoordinate (coordinate : Fin 9) (offset : Fin 3) : Fin 9 :=
  ⟨coordinate.val / 3 * 3 + offset.val, by omega⟩

theorem mem_unit (board : Board) (positions : Fin 9 → Cell) (digit : Digit) :
    encode (some digit) ∈ unit board positions ↔ ∃ i, board (positions i) = some digit := by
  simp only [unit, List.mem_map, List.mem_finRange, true_and, injective.eq_iff]

theorem any_unit (board : Board) (positions : Fin 9 → Cell) (digit : Digit) :
    (List.finRange 9).any (fun i => decide (board (positions i) = some digit)) =
      decide (encode (some digit) ∈ unit board positions) := by
  apply Bool.eq_iff_iff.mpr
  simp [mem_unit]

theorem any_box (board : Board) (r c : Fin 9) (digit : Digit) :
    (List.finRange 3).any (fun i => (List.finRange 3).any (fun j =>
      decide (board (boxCoordinate r i, boxCoordinate c j) = some digit))) =
      decide (encode (some digit) ∈ box board (block r) (block c)) := by
  apply Bool.eq_iff_iff.mpr
  simp only [List.any_eq_true, List.mem_finRange, true_and, decide_eq_true_eq, box, mem_unit]
  constructor
  · rintro ⟨i, j, found⟩
    let k : Fin 9 := ⟨3 * i.val + j.val, by omega⟩
    refine ⟨k, ?_⟩
    have located : boxCell (block r) (block c) k = (boxCoordinate r i, boxCoordinate c j) := by
      apply Prod.ext <;> apply Fin.ext <;> simp only [boxCell, boxCoordinate, block, k] <;> omega
    simpa only [located] using found
  · rintro ⟨k, found⟩
    let i : Fin 3 := ⟨k.val / 3, by omega⟩
    let j : Fin 3 := ⟨k.val % 3, by omega⟩
    refine ⟨i, j, ?_⟩
    have located : (boxCoordinate r i, boxCoordinate c j) = boxCell (block r) (block c) k := by
      apply Prod.ext <;> apply Fin.ext <;> simp only [boxCell, boxCoordinate, block, i, j] <;> omega
    simpa only [located] using found

theorem allowed_iff (board : Board) (position : Cell) (digit : Digit) (empty : board position = none) :
    Allowed board position digit ↔
      encode (some digit) ∉ row board position.1 ∧
      encode (some digit) ∉ column board position.2 ∧
      encode (some digit) ∉ box board (block position.1) (block position.2) := by
  simp only [row, column, box, mem_unit, not_exists]
  constructor
  · intro allowed
    have absent (other : Cell)
        (unit : position.1 = other.1 ∨ position.2 = other.2 ∨
          (position.1.val / 3 = other.1.val / 3 ∧ position.2.val / 3 = other.2.val / 3)) :
        board other ≠ some digit := by
      by_cases same : position = other
      · simp [← same, empty]
      · exact allowed other ⟨same, unit⟩
    refine ⟨fun c => absent (position.1, c) (Or.inl rfl),
      fun r => absent (r, position.2) (Or.inr (Or.inl rfl)), ?_⟩
    intro i
    apply absent
    right; right
    simp only [boxCell, block]
    constructor <;> omega
  · rintro ⟨rows, columns, boxes⟩ other ⟨_, sameRow | sameColumn | sameBox⟩
    · simpa [sameRow] using rows other.2
    · simpa [sameColumn] using columns other.1
    · let i : Fin 9 := ⟨3 * (other.1.val % 3) + other.2.val % 3, by omega⟩
      have located : boxCell (block position.1) (block position.2) i = other := by
        apply Prod.ext <;> apply Fin.ext
        · simp only [boxCell, block, i]
          omega
        · simp only [boxCell, block, i]
          omega
      simpa only [located] using boxes i

end ProjectEulerSolutions.Sudoku.Values
