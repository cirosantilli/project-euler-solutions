import solvers.«96.py.lean.d».CandidatePlaces

namespace solvers.«96».reduce_cands.onlyPlace
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev first : Expr := .subscript (.name "places") (.constant (.int 0))
abbrev candidate : Expr := .subscript (.name "cands") first
abbrev countOne : Expr := .compare (.builtinCall .len [.name "places"]) [(.eq, .constant (.int 1))]
abbrev different : Expr := .compare candidate [(.notEq, .name "dbit")]
abbrev condition : Expr := .boolOp .and [countOne, different]
abbrev force : List Stmt := [.assign [candidate] (.name "dbit"), .assign [.name "changed"] (.constant (.bool true))]
abbrev reject : Stmt := .ifThen (.unaryOp .not (.name "places")) [.return (.constant (.bool false))] []
abbrev choose : Stmt := .ifThen condition force []

theorem source : places.digitBody.drop 2 = [reject, choose] := rfl

theorem empty_result (address : Nat) (heap : Mutable.Heap) (cells : List Cell) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (found : heap[address]? = some (.list (peerLoop.encoded cells)))
    (getPlaces : env "places" = some (.ref address)) :
    Evaluates program env (.unaryOp .not (.name "places")) heap (.scalar (.bool cells.isEmpty)) heap globals := by
  apply Evaluates.unary (Evaluates.lookup getPlaces)
  cases cells <;> simp [HeapCalls.unary, Mutable.truth, found, peerLoop.encoded]

theorem first_result (address : Nat) (heap : Mutable.Heap) (cell : Cell) (rest : List Cell) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (found : heap[address]? = some (.list (peerLoop.encoded (cell :: rest))))
    (getPlaces : env "places" = some (.ref address)) :
    Evaluates program env first heap (number (Encoding.index cell).val) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getPlaces) Evaluates.constant
  simp [Mutable.getItem, Mutable.index, found, peerLoop.encoded, peerLoop.atom, Except.map]

theorem count_result (address : Nat) (heap : Mutable.Heap) (cells : List Cell) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (found : heap[address]? = some (.list (peerLoop.encoded cells)))
    (getPlaces : env "places" = some (.ref address)) :
    Evaluates program env countOne heap (.scalar (.bool (decide (cells.length = 1)))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number cells.length) (bv := number 1)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getPlaces) .nil)
    simp [primitive, Mutable.length, found, peerLoop.encoded]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega

theorem different_result (address placesAddress : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (cell : Cell) (rest : List Cell) (digit : Digit) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap candidates) (selected : heap[placesAddress]? = some (.list (peerLoop.encoded (cell :: rest))))
    (getCandidates : env "cands" = some (.ref address)) (getPlaces : env "places" = some (.ref placesAddress))
    (getDigit : env "dbit" = some (number (2 ^ digit.val))) :
    Evaluates program env different heap (.scalar (.bool (decide (candidates cell ≠ {digit})))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number (Masks.encode (candidates cell))) (bv := number (2 ^ digit.val))
  · apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCandidates)
      (first_result placesAddress heap cell rest env globals selected getPlaces)
    simpa only [Encoding.cell_index] using get_mask address heap candidates (Encoding.index cell) found
  · exact Evaluates.lookup getDigit
  · have value : (Masks.encode {digit} : Int) = (2 : Int) ^ digit.val := by simp
    have criterion : (Masks.encode (candidates cell) : Int) = (2 : Int) ^ digit.val ↔ candidates cell = {digit} := by
      rw [← value]
      rw [Int.natCast_inj]
      exact Masks.injective.eq_iff
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, criterion]

theorem condition_single (address placesAddress : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (cell : Cell) (digit : Digit) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap candidates) (selected : heap[placesAddress]? = some (.list (peerLoop.encoded [cell])))
    (getCandidates : env "cands" = some (.ref address)) (getPlaces : env "places" = some (.ref placesAddress))
    (getDigit : env "dbit" = some (number (2 ^ digit.val))) :
    Evaluates program env condition heap (.scalar (.bool (decide (candidates cell ≠ {digit})))) heap globals := by
  apply Evaluates.boolean_next (count_result placesAddress heap [cell] env globals selected getPlaces) rfl rfl
  exact Evaluates.boolean_last (different_result address placesAddress heap candidates cell [] digit env globals found selected getCandidates getPlaces getDigit) rfl

theorem condition_many (address : Nat) (heap : Mutable.Heap) (cells : List Cell) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (found : heap[address]? = some (.list (peerLoop.encoded cells)))
    (getPlaces : env "places" = some (.ref address)) (many : cells.length ≠ 1) :
    Evaluates program env condition heap (.scalar (.bool false)) heap globals := by
  have tested := count_result address heap cells env globals found getPlaces
  have result := Evaluates.boolean_stop (op := BoolOp.and) (rest := [different]) (flag := false) tested
    (by simp [Mutable.truth, scalarTruth, many]) rfl
  simpa only [decide_eq_false many] using result

theorem force_result (address placesAddress : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (cell : Cell) (digit : Digit) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap candidates) (selected : heap[placesAddress]? = some (.list (peerLoop.encoded [cell])))
    (getCandidates : env "cands" = some (.ref address)) (getPlaces : env "places" = some (.ref placesAddress))
    (getDigit : env "dbit" = some (number (2 ^ digit.val))) :
    BlockContinues program env force heap (setLocal env "changed" (.scalar (.bool true)))
      (write address heap candidates cell {digit}) globals := by
  have stored : BlockContinues program env [.assign [candidate] (.name "dbit")] heap env
      (write address heap candidates cell {digit}) globals := by
    apply BlockContinues.setItem (by intros; simp) (Evaluates.lookup getDigit)
      (.cons (Evaluates.lookup getCandidates) (.cons (first_result placesAddress heap cell [] env globals selected getPlaces) .nil))
    simpa [Masks.singleton] using set_mask address heap candidates cell {digit} found
  exact stored.append (BlockContinues.assign Evaluates.constant)

end
end solvers.«96».reduce_cands.onlyPlace
