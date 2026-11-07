import solvers.«96.py.lean.d».CandidatePropagation
import lang.py.FormalSW.TruthCalls

namespace solvers.«96».solve_cands.solved
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev definition := (findFunctionIn program.body "solve_cands").getD ([], [])
abbrev element : Expr := .compare (.builtinCall .intBitCount [.name "x"]) [(.eq, .constant (.int 1))]
abbrev condition : Expr := .consumeGenerator (.builtinCall .all []) element [(.name "x", .name "cands", [])]
abbrev statement : Stmt := .ifThen condition [.return (.name "cands")] []

theorem source : definition.2[1]? = some statement := rfl

theorem element_result (digits : Finset Digit) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (lookup : env "x" = some (number (Masks.encode digits))) :
    Evaluates program env element heap (.scalar (.bool (decide (digits.card = 1)))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number digits.card) (bv := number 1)
  · apply Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil)
    change (some (Except.ok (number (FormalSW.Py.Builtin.Int.bitCount (Masks.encode digits))), heap) :
      Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
    rw [Masks.bitCount]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega

/-- The actual lazy all-generator tests precisely the cardinality of all 81 candidate sets. -/
theorem result (address : Nat) (candidates : Candidates) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (found : View address heap candidates) (lookup : env "cands" = some (.ref address)) :
    Evaluates program env condition heap (.scalar (.bool (decide (∀ cell, (candidates cell).card = 1)))) heap globals := by
  have tested := Evaluates.all_map (globals := globals) (module := program) (env := env)
    (name := "x") (values := List.ofFn (fun slot : Fin 81 => slot))
    (encode := fun slot => number (Masks.encode (candidates (Encoding.cell slot))))
    (predicate := fun slot => decide ((candidates (Encoding.cell slot)).card = 1)) (element := element)
    (by intros; simp) (by intros; simp) (by intros; simp) (Evaluates.lookup lookup)
    (by simp [Mutable.sequence, found, values, CandidateArray.encode])
    (fun inner slot value => element_result (candidates (Encoding.cell slot)) inner heap globals value)
  have allCells : (∀ slot : Fin 81, (candidates (Encoding.cell slot)).card = 1) ↔
      ∀ cell, (candidates cell).card = 1 := by
    constructor
    · intro single cell
      simpa only [Encoding.cell_index] using single (Encoding.index cell)
    · intro single slot
      exact single (Encoding.cell slot)
  have allSlots : (List.ofFn (fun slot : Fin 81 => slot)).all
      (fun slot => decide ((candidates (Encoding.cell slot)).card = 1)) =
      decide (∀ cell, (candidates cell).card = 1) := by
    apply Bool.eq_iff_iff.mpr
    simp only [List.all_eq_true, decide_eq_true_eq, List.forall_mem_ofFn_iff]
    exact allCells
  simpa only [allSlots] using tested

theorem accept (address : Nat) (candidates : Candidates) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (found : View address heap candidates) (lookup : env "cands" = some (.ref address))
    (single : ∀ cell, (candidates cell).card = 1) (following : List Stmt) :
    BlockReturns program env (statement :: following) heap (.ref address) heap globals := by
  apply BlockReturns.branch (flag := true) (result address candidates env heap globals found lookup)
    (by simp [Mutable.truth, scalarTruth, single])
  exact BlockReturns.return_value (Evaluates.lookup lookup)

theorem continue_search (address : Nat) (candidates : Candidates) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (found : View address heap candidates) (lookup : env "cands" = some (.ref address))
    (notSingle : ¬ ∀ cell, (candidates cell).card = 1) :
    BlockContinues program env [statement] heap env heap globals :=
  BlockContinues.branch (flag := false) (result address candidates env heap globals found lookup)
    (by simp only [decide_eq_false notSingle]; rfl) .nil

theorem accepted_solution (units : UnitTable) (complete : units.Complete) (before : Candidates)
    (after : Propagation.State) (finished : propagationLoop.Finished units before after)
    (address : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : View address heap after.candidates) (lookup : env "cands" = some (.ref address))
    (single : ∀ cell, (after.candidates cell).card = 1) (following : List Stmt) :
    (∃ assignment, Legal assignment ∧ Admits before assignment ∧
      (∀ cell, after.candidates cell = {assignment cell}) ∧
      (∀ other, Legal other → Admits before other → other = assignment)) ∧
    BlockReturns program env (statement :: following) heap (.ref address) heap globals :=
  ⟨finished.solved complete single, accept address after.candidates env heap globals found lookup single following⟩

end
end solvers.«96».solve_cands.solved
