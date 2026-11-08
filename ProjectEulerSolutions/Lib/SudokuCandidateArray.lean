import ProjectEulerSolutions.Lib.SudokuEncoding

/-! Row-major arrays of candidate masks retain the cellwise Sudoku meaning after updates. -/
namespace ProjectEulerSolutions.Sudoku.CandidateArray
set_option Elab.async false
noncomputable section

def encode (candidates : Candidates) : List Nat :=
  List.ofFn fun slot : Fin 81 => Masks.encode (candidates (Encoding.cell slot))

@[simp] theorem length (candidates : Candidates) : (encode candidates).length = 81 := by simp [encode]

theorem get (candidates : Candidates) (slot : Fin 81) :
    (encode candidates)[slot.val]? = some (Masks.encode (candidates (Encoding.cell slot))) := by
  simp only [encode, List.getElem?_ofFn, dif_pos slot.isLt]

theorem update (candidates : Candidates) (cell : Cell) (replacement : Finset Digit) :
    (encode candidates).set (Encoding.index cell).val (Masks.encode replacement) =
      encode (Function.update candidates cell replacement) := by
  apply List.ext_getElem?
  intro index
  by_cases inside : index < 81
  · rw [List.getElem?_set]
    have before := get candidates ⟨index, inside⟩
    have after := get (Function.update candidates cell replacement) ⟨index, inside⟩
    rw [before, after]
    by_cases same : (Encoding.index cell).val = index
    · have equal : (⟨index, inside⟩ : Fin 81) = Encoding.index cell := Fin.ext same.symm
      have atCell : Encoding.cell ⟨index, inside⟩ = cell := (congrArg Encoding.cell equal).trans (Encoding.cell_index cell)
      rw [if_pos same, if_pos (show (Encoding.index cell).val < (encode candidates).length by simp)]
      have value := congrArg (Function.update candidates cell replacement) atCell
      exact congrArg (fun digits => some (Masks.encode digits)) (value.trans (Function.update_self cell replacement candidates)).symm
    · have different : Encoding.cell ⟨index, inside⟩ ≠ cell := by
        intro equal
        have slot := congrArg Encoding.index equal
        rw [Encoding.index_cell] at slot
        exact same (congrArg Fin.val slot).symm
      simp only [if_neg same, Function.update_of_ne different]
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by simp; omega)]

theorem initial (board : Board) : encode (initialCandidates board) = Encoding.prepared board 81 := by
  rw [Encoding.prepared_done]
  rfl

end
end ProjectEulerSolutions.Sudoku.CandidateArray
