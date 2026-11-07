import solvers.«96.py.lean.d».CandidateSolveEntry
import ProjectEulerSolutions.Lib.SudokuCandidateBits

namespace solvers.«96».solve_cands.bits
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev body : List Stmt := match solved.definition.2[4]? with
  | some (Stmt.whileDo _ body _) => body
  | _ => []
abbrev select : Stmt := .assign [.name "b"] (.binOp (.name "mask") .bitAnd (.unaryOp .uSub (.name "mask")))
abbrev advance : Stmt := .augAssign (.name "mask") .sub (.name "b")

theorem source_select : body.take 1 = [select] := rfl
theorem source_advance : body.drop 5 = [advance] := rfl
theorem source_initialize : solved.definition.2[3]? =
    some (.assign [.name "mask"] (.subscript (.name "cands") (.name "idx"))) := rfl

theorem start (address : Nat) (candidates : Candidates) (cell : Cell)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : View address heap candidates) (getCandidates : env "cands" = some (.ref address))
    (getIndex : env "idx" = some (number (Encoding.index cell).val)) :
    BlockContinues program env (solved.definition.2.drop 3 |>.take 1) heap
      (setLocal env "mask" (number (Masks.encode (candidates cell)))) heap globals := by
  apply BlockContinues.assign
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCandidates) (Evaluates.lookup getIndex)
  simpa only [Encoding.cell_index] using get_mask address heap candidates (Encoding.index cell) found

/-- The source signed conjunction chooses the smallest remaining candidate digit. -/
theorem choose (digits : Finset Digit) (nonempty : digits ≠ ∅) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (getMask : env "mask" = some (number (Masks.encode digits))) :
    ∃ digit ∈ digits, (∀ other ∈ digits, digit ≤ other) ∧
      BlockContinues program env (body.take 1) heap (setLocal env "b" (number (2 ^ digit.val))) heap globals := by
  obtain ⟨digit, member, extracted, least⟩ := Masks.least_bit digits nonempty
  refine ⟨digit, member, least, ?_⟩
  rw [source_select]
  apply BlockContinues.assign
  apply Evaluates.binary (middle := heap) (afterRight := heap)
    (av := number (Masks.encode digits)) (bv := number (-(Masks.encode digits : Int)))
    (Evaluates.lookup getMask) (Evaluates.unary (Evaluates.lookup getMask) rfl)
  simp [Mutable.binary, scalarBitAnd, scalarNumber, Masks.integerBitAnd_eq, extracted]

theorem next (digits : Finset Digit) (digit : Digit) (member : digit ∈ digits) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (getMask : env "mask" = some (number (Masks.encode digits)))
    (getBit : env "b" = some (number (2 ^ digit.val))) :
    BlockContinues program env (body.drop 5) heap
      (setLocal env "mask" (number (Masks.encode (digits.erase digit)))) heap globals := by
  rw [source_advance]
  apply BlockContinues.augAssign getMask (Evaluates.lookup getBit)
  simp [inplace, Mutable.binary, scalarBinary, scalarNumber, Masks.subtract_bit_int digits digit member]

end
end solvers.«96».solve_cands.bits
