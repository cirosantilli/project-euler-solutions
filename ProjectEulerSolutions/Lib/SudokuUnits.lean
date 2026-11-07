import ProjectEulerSolutions.Lib.SudokuCandidates

/-! The 27 units in the solver's row, column, and row-major box order. -/
namespace ProjectEulerSolutions.Sudoku.Units
set_option Elab.async false

def cells (unit : Fin 27) (index : Fin 9) : Cell :=
  if row : unit.val < 9 then (⟨unit.val, row⟩, index)
  else if column : unit.val < 18 then (index, ⟨unit.val - 9, by omega⟩)
  else boxCell ⟨(unit.val - 18) / 3, by omega⟩ ⟨(unit.val - 18) % 3, by omega⟩ index

theorem row (r : Fin 9) (index : Fin 9) : cells ⟨r.val, by omega⟩ index = (r, index) := by
  simp [cells, r.isLt]

theorem column (c : Fin 9) (index : Fin 9) : cells ⟨9 + c.val, by omega⟩ index = (index, c) := by
  simp [cells, show ¬ 9 + c.val < 9 by omega, show 9 + c.val < 18 by omega]

theorem box (r c : Fin 3) (index : Fin 9) : cells ⟨18 + 3 * r.val + c.val, by omega⟩ index = boxCell r c index := by
  have notRow : ¬ 18 + 3 * r.val + c.val < 9 := by omega
  have notColumn : ¬ 18 + 3 * r.val + c.val < 18 := by omega
  have rowEq : (18 + 3 * r.val + c.val - 18) / 3 = r.val := by omega
  have columnEq : (18 + 3 * r.val + c.val - 18) % 3 = c.val := by omega
  simp [cells, notRow, notColumn, rowEq, columnEq]

theorem related (unit : Fin 27) (i j : Fin 9) (different : i ≠ j) : Peer (cells unit i) (cells unit j) := by
  unfold cells
  split
  · exact ⟨by simpa using different, Or.inl rfl⟩
  · split
    · exact ⟨by simpa using different, Or.inr (Or.inl rfl)⟩
    · exact boxCell_peer _ _ i j different

/-- Every Sudoku peer pair shares one of exactly these rows, columns, or boxes. -/
theorem complete (a b : Cell) (peer : Peer a b) : ∃ unit : Fin 27, ∃ i j : Fin 9,
    cells unit i = a ∧ cells unit j = b := by
  rcases peer.2 with sameRow | sameColumn | ⟨boxRow, boxColumn⟩
  · refine ⟨⟨a.1.val, by omega⟩, a.2, b.2, ?_, ?_⟩
    · exact row a.1 a.2
    · simpa only [sameRow] using row a.1 b.2
  · refine ⟨⟨9 + a.2.val, by omega⟩, a.1, b.1, ?_, ?_⟩
    · exact column a.2 a.1
    · simpa only [sameColumn] using column a.2 b.1
  · let r : Fin 3 := ⟨a.1.val / 3, by omega⟩
    let c : Fin 3 := ⟨a.2.val / 3, by omega⟩
    let i : Fin 9 := ⟨3 * (a.1.val % 3) + a.2.val % 3, by omega⟩
    let j : Fin 9 := ⟨3 * (b.1.val % 3) + b.2.val % 3, by omega⟩
    refine ⟨⟨18 + 3 * r.val + c.val, by omega⟩, i, j, ?_, ?_⟩
    · rw [box]
      apply Prod.ext <;> apply Fin.ext <;> simp only [boxCell, r, c, i] <;> omega
    · rw [box]
      apply Prod.ext <;> apply Fin.ext <;> simp only [boxCell, r, c, j] <;> omega

end ProjectEulerSolutions.Sudoku.Units
