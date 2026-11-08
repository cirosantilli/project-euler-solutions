import ProjectEulerSolutions.Lib.SudokuCandidates

/-! Connect singleton candidate solutions to the board-completion contract. -/
namespace ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

def assignment (board : Board) : Assignment := fun cell => (board cell).getD 0

theorem assignment_full (board : Board) (full : Full board) :
    (fun cell => some (assignment board cell)) = board := by
  funext cell
  cases found : board cell with
  | none => exact False.elim (full cell found)
  | some digit => simp [assignment, found]

theorem initial_nonempty (board : Board) : ∀ cell, initialCandidates board cell ≠ ∅ := by
  intro cell
  cases found : board cell with
  | none =>
      simp only [initialCandidates, found]
      exact Finset.nonempty_iff_ne_empty.mp ⟨0, Finset.mem_univ _⟩
  | some digit => simp [initialCandidates, found]

theorem completed_assignment (board solved : Board) (completed : Completion board solved) :
    Legal (assignment solved) ∧ Admits (initialCandidates board) (assignment solved) := by
  have full := assignment_full solved completed.2.1
  refine ⟨?_, (initial_admits_iff board (assignment solved)).mpr ?_⟩
  · intro a b peer same
    have first := congrFun full a
    have second := congrFun full b
    exact completed.1 a b peer (assignment solved a) first.symm (by simpa only [same] using second.symm)
  · simpa only [full] using completed.2.2

theorem admitted_completion (board : Board) (solution : Assignment) (legal : Legal solution)
    (admitted : Admits (initialCandidates board) solution) : Completion board (fun cell => some (solution cell)) :=
  ⟨(legal_full solution legal).1, (legal_full solution legal).2, (initial_admits_iff board solution).mp admitted⟩

theorem unique_assignment (board solved : Board) (unique : ∀ final, Completion board final → final = solved)
    (solution : Assignment) (legal : Legal solution) (admitted : Admits (initialCandidates board) solution) :
    solution = assignment solved := by
  have same := unique _ (admitted_completion board solution legal admitted)
  funext cell
  have atCell := congrFun same cell
  simp only [assignment, ← atCell, Option.getD_some]

end
end ProjectEulerSolutions.Sudoku
