import solvers.«96.py.lean.d».Projection
import ProjectEulerSolutions.Lib.SudokuCandidateArray

namespace solvers.«96».reduce_cands
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev values (candidates : Candidates) : List Mutable.Value := (CandidateArray.encode candidates).map fun n : Nat => number n
abbrev View (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) : Prop :=
  heap[address]? = some (.list (values candidates))
def write (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell)
    (replacement : Finset Digit) : Mutable.Heap :=
  heap.set address (.list (values (Function.update candidates position replacement)))

abbrev candidate : Expr := .subscript (.name "cands") (.name "p")
abbrev test : Expr := .binOp candidate .bitAnd (.name "val")
abbrev clear : Stmt := .augAssign candidate .bitAnd (.unaryOp .invert (.name "val"))
abbrev definition := (findFunctionIn program.body "reduce_cands").getD ([], [])
abbrev outerBody : List Stmt := match definition.2[2]? with
  | some (Stmt.whileDo _ body _) => body
  | _ => []
abbrev queueBody : List Stmt := match outerBody[2]? with
  | some (Stmt.whileDo _ body _) => body
  | _ => []
abbrev peerBody : List Stmt := match queueBody[2]? with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []

theorem source_clear : (match peerBody[0]? with
    | some (Stmt.ifThen _ body _) => body[0]?
    | _ => none) = some clear := rfl

theorem source_test : (match peerBody[0]? with
    | some (Stmt.ifThen condition _ _) => some condition
    | _ => none) = some test := rfl

theorem get_mask (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (slot : Fin 81)
    (found : View address heap candidates) :
    Mutable.getItem (.ref address) (number slot.val) heap =
      some (.ok (number (Masks.encode (candidates (Encoding.cell slot)))), heap) := by
  obtain ⟨_, selected⟩ := List.getElem_of_getElem? (CandidateArray.get candidates slot)
  simp [Mutable.getItem, found, values, Mutable.index, number, not_lt.mpr (Int.natCast_nonneg slot.val),
    not_le.mpr slot.isLt, Except.map, selected]

theorem test_result (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getPosition : env "p" = some (number (Encoding.index position).val))
    (getValue : env "val" = some (number (2 ^ digit.val))) :
    Evaluates program env test heap (number (if digit ∈ candidates position then 2 ^ digit.val else 0)) heap globals := by
  apply Evaluates.binary (middle := heap) (afterRight := heap)
    (av := number (Masks.encode (candidates position))) (bv := number (2 ^ digit.val))
  · apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCandidates) (Evaluates.lookup getPosition)
    simpa only [Encoding.cell_index] using get_mask address heap candidates (Encoding.index position) found
  · exact Evaluates.lookup getValue
  · simp [Mutable.binary, scalarBitAnd, scalarNumber, Masks.integerBitAnd_eq, Masks.and_bit, number]

theorem test_truth (heap : Mutable.Heap) (digits : Finset Digit) (digit : Digit) :
    Mutable.truth (number (if digit ∈ digits then 2 ^ digit.val else 0)) heap =
      some (.ok (decide (digit ∈ digits)), heap) := by
  have positive : (0 : Int) < 2 ^ digit.val := Int.pow_pos (by decide)
  by_cases present : digit ∈ digits <;> simp [Mutable.truth, scalarTruth, present, ne_of_gt positive]

theorem set_mask (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell)
    (replacement : Finset Digit) (found : View address heap candidates) :
    Mutable.setItem (.ref address) (number (Encoding.index position).val) (number (Masks.encode replacement)) heap =
      some (.ok (.scalar .none), write address heap candidates position replacement) := by
  have updated : (values candidates).set (Encoding.index position).val (number (Masks.encode replacement)) =
      values (Function.update candidates position replacement) := by
    simpa only [values, List.map_set] using congrArg (List.map fun n : Nat => number n) (CandidateArray.update candidates position replacement)
  have size : (values candidates).length = 81 := by simp [values]
  simp [Mutable.setItem, found, Mutable.index, number, size,
    not_lt.mpr (Int.natCast_nonneg (Encoding.index position).val), not_le.mpr (Encoding.index position).isLt,
    updated, write, Except.map]

theorem clear_result (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell) (digit : Digit)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getPosition : env "p" = some (number (Encoding.index position).val))
    (getValue : env "val" = some (number (2 ^ digit.val))) :
    BlockContinues program env [clear] heap env
      (write address heap candidates position ((candidates position).erase digit)) globals := by
  apply BlockContinues.augItem (afterOperation := heap)
    (updated := number (Masks.encode ((candidates position).erase digit))) (by intros; simp) (Evaluates.lookup getCandidates) (Evaluates.lookup getPosition)
    (get_mask address heap candidates (Encoding.index position) found)
    (Evaluates.unary (Evaluates.lookup getValue) rfl)
  · simp only [inplace, number, Mutable.binary, scalarBitAnd, scalarNumber,
      Masks.integerBitAnd_eq, Encoding.cell_index]
    change (some (Except.ok (number (FormalSW.Py.Builtin.Int.bitAnd (Masks.encode (candidates position))
      (FormalSW.Py.Builtin.Int.bitNot (2 ^ digit.val)))), heap) : Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
    rw [Masks.erase]
  · exact set_mask address heap candidates position ((candidates position).erase digit) found

theorem write_view (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell)
    (replacement : Finset Digit) (found : View address heap candidates) :
    View address (write address heap candidates position replacement) (Function.update candidates position replacement) :=
  List.getElem?_set_self (List.getElem?_eq_some_iff.mp found).1

theorem write_frame (address other : Nat) (heap : Mutable.Heap) (candidates : Candidates) (position : Cell)
    (replacement : Finset Digit) (different : address ≠ other) :
    (write address heap candidates position replacement)[other]? = heap[other]? :=
  List.getElem?_set_ne different

/-- Executing the source deletion preserves every legal completion and decreases the search measure. -/
theorem clear_peer (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (cell other : Cell) (digit : Digit)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address))
    (getPosition : env "p" = some (number (Encoding.index other).val))
    (getValue : env "val" = some (number (2 ^ digit.val)))
    (single : candidates cell = {digit}) (peer : Peer cell other) (present : digit ∈ candidates other) :
    let after := Function.update candidates other ((candidates other).erase digit)
    let final := write address heap candidates other ((candidates other).erase digit)
    BlockContinues program env [clear] heap env final globals ∧
      View address final after ∧
      (∀ assignment, Legal assignment → (Admits after assignment ↔ Admits candidates assignment)) ∧
      weight after < weight candidates ∧
      (∀ outside, address ≠ outside → final[outside]? = heap[outside]?) := by
  exact ⟨clear_result address heap candidates other digit env globals found getCandidates getPosition getValue,
    write_view address heap candidates other _ found,
    fun assignment legal => erase_peer_iff candidates assignment legal single peer,
    erase_weight_lt candidates other digit present,
    fun outside different => write_frame address outside heap candidates other _ different⟩

end
end solvers.«96».reduce_cands
