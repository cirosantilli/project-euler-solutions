import solvers.«96.py.lean.d».CandidatePeerLoop
import ProjectEulerSolutions.Lib.SudokuOnlyPlace

namespace solvers.«96».reduce_cands.places
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev candidate : Expr := .subscript (.name "cands") (.name "i")
abbrev condition : Expr := .binOp candidate .bitAnd (.name "dbit")
abbrev collection : Expr := .listComp (.name "i") [("i", .name "unit", [condition])]
abbrev body (address : Nat) : List Stmt := comprehensionFilters [condition] [.comprehensionAppend address (.name "i")]
abbrev work (base : Mutable.Heap) (output : List Cell) : Mutable.Heap := HeapCollection.working base (peerLoop.encoded output) []
abbrev unitBody : List Stmt := match outerBody[3]? with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []
abbrev digitBody : List Stmt := match unitBody[0]? with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []

theorem source : digitBody[1]? = some (.assign [.name "places"] collection) := rfl
theorem source_bit : digitBody[0]? = some (.assign [.name "dbit"] (.binOp (.constant (.int 1)) .leftShift (.name "k"))) := rfl

theorem digit_result (digit : Digit) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getDigit : env "k" = some (number digit.val)) :
    BlockContinues program env (digitBody.take 1) heap (setLocal env "dbit" (number (2 ^ digit.val))) heap globals := by
  apply BlockContinues.assign
  apply Evaluates.binary (middle := heap) (afterRight := heap) Evaluates.constant (Evaluates.lookup getDigit)
  have nonnegative : ¬ (digit.val : Int) < 0 := by omega
  simp [Mutable.binary, scalarShiftLeft, scalarNumber, integerShiftLeft, nonnegative]

theorem test_result (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (cell : Cell) (digit : Digit)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getCell : env "i" = some (number (Encoding.index cell).val))
    (getDigit : env "dbit" = some (number (2 ^ digit.val))) :
    Evaluates program env condition heap (number (if digit ∈ candidates cell then 2 ^ digit.val else 0)) heap globals := by
  apply Evaluates.binary (middle := heap) (afterRight := heap)
    (av := number (Masks.encode (candidates cell))) (bv := number (2 ^ digit.val))
  · apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCandidates) (Evaluates.lookup getCell)
    simpa only [Encoding.cell_index] using get_mask address heap candidates (Encoding.index cell) found
  · exact Evaluates.lookup getDigit
  · simp [Mutable.binary, scalarBitAnd, scalarNumber, Masks.integerBitAnd_eq, Masks.and_bit, number]

theorem places_cons (candidates : Candidates) (cell : Cell) (rest : List Cell) (digit : Digit) :
    OnlyPlace.places candidates (cell :: rest) digit =
      (if digit ∈ candidates cell then [cell] else []) ++ OnlyPlace.places candidates rest digit := by
  by_cases present : digit ∈ candidates cell <;> simp [OnlyPlace.places, present]

theorem collect (address : Nat) (base : Mutable.Heap) (candidates : Candidates) (cell : Cell) (digit : Digit)
    (output : List Cell) (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address base candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getCell : env "i" = some (number (Encoding.index cell).val))
    (getDigit : env "dbit" = some (number (2 ^ digit.val))) :
    BlockContinues program env (body base.length) (work base output) env
      (work base (output ++ if digit ∈ candidates cell then [cell] else [])) globals := by
  have kept : View address (work base output) candidates := HeapCollection.prefix_found base _ address _ found
  have tested := test_result address (work base output) candidates cell digit env globals kept getCandidates getCell getDigit
  by_cases present : digit ∈ candidates cell
  · simp only [if_pos present]
    apply BlockContinues.branch (flag := true) tested
      (by simpa only [decide_eq_true present] using test_truth (work base output) (candidates cell) digit)
    have appended := BlockContinues.comprehension_append (module := program) (globals := globals)
      (Evaluates.lookup getCell) (HeapCollection.found base (peerLoop.encoded output) [])
    simpa only [work, HeapCollection.store, peerLoop.encoded, List.map_append, List.map_cons, List.map_nil] using appended
  · simp only [if_neg present, List.append_nil]
    exact BlockContinues.branch (flag := false) tested
      (by simpa only [decide_eq_false present] using test_truth (work base output) (candidates cell) digit) .nil

theorem loop (address unit : Nat) (base : Mutable.Heap) (candidates : Candidates) (digit : Digit)
    (before remaining output : List Cell) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address base candidates) (cells : base[unit]? = some (.list (peerLoop.encoded (before ++ remaining))))
    (getCandidates : env "cands" = some (.ref address)) (getDigit : env "dbit" = some (number (2 ^ digit.val))) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "i" (.ref unit) before.length (body base.length) []]
      (work base output) finalEnv (work base (output ++ OnlyPlace.places candidates remaining digit)) globals := by
  induction remaining generalizing before output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [OnlyPlace.places, List.filter_nil, List.append_nil]
      apply BlockContinues.sequence_done (items := peerLoop.encoded before)
      · have kept := HeapCollection.prefix_found base [.list (peerLoop.encoded output)] unit _ cells
        simp [Mutable.sequence, sequenceValue, work, HeapCollection.working, kept]
      · simp [peerLoop.encoded]
      · exact .nil
  | cons cell rest ih =>
      let ready := setLocal env "i" (number (Encoding.index cell).val)
      have candidatesKept : ready "cands" = some (.ref address) := by simpa [ready, setLocal] using getCandidates
      have digitKept : ready "dbit" = some (number (2 ^ digit.val)) := by simpa [ready, setLocal] using getDigit
      have collected := collect address base candidates cell digit output ready globals found candidatesKept (by simp [ready, setLocal]) digitKept
      obtain ⟨finalEnv, continued⟩ := ih (before ++ [cell]) (output ++ if digit ∈ candidates cell then [cell] else []) ready
        (by simpa only [List.append_assoc, List.singleton_append] using cells) candidatesKept digitKept
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (items := peerLoop.encoded (before ++ cell :: rest)) (value := number (Encoding.index cell).val)
      · have kept := HeapCollection.prefix_found base [.list (peerLoop.encoded output)] unit _ cells
        simp [Mutable.sequence, sequenceValue, work, HeapCollection.working, kept]
      · simp [peerLoop.encoded]
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body, comprehensionFilters,
          Bool.false_or, Bool.false_eq_true, if_false, places_cons, List.append_assoc,
          List.length_append, List.length_singleton] using collected.append continued

theorem result (address unit : Nat) (heap : Mutable.Heap) (candidates : Candidates) (digit : Digit) (cells : List Cell)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (unitFound : heap[unit]? = some (.list (peerLoop.encoded cells)))
    (getCandidates : env "cands" = some (.ref address)) (getUnit : env "unit" = some (.ref unit))
    (getDigit : env "dbit" = some (number (2 ^ digit.val))) :
    Evaluates program env collection heap (.ref heap.length) (work heap (OnlyPlace.places candidates cells digit)) globals := by
  let inner := comprehensionEnv env [("i", .name "unit", [condition])]
  obtain ⟨finalEnv, collected⟩ := loop address unit heap candidates digit [] cells [] inner globals found (by simpa using unitFound)
    (by simpa [inner, comprehensionEnv] using getCandidates) (by simpa [inner, comprehensionEnv] using getDigit)
  apply Evaluates.list_comp_sequence (finalEnv := finalEnv) (items := peerLoop.encoded cells)
    (by intros; simp) (Evaluates.lookup getUnit)
  · simp [Mutable.sequence, unitFound]
  · exact collected

end
end solvers.«96».reduce_cands.places
