import solvers.«96.py.lean.d».CandidateUnitTable
import ProjectEulerSolutions.Lib.SudokuUnits
import ProjectEulerSolutions.Lib.HeapFiniteRange

namespace solvers.«96».build_peers_units
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev definition := (findFunctionIn program.body "build_peers_units").getD ([], [])
abbrev cellExpression : Expr := .binOp (.binOp (.name "r") .mult (.constant (.int 9))) .add (.name "c")
def varying (column : Bool) : String := if column then "r" else "c"
def fixed (column : Bool) : String := if column then "c" else "r"
def position (column : Bool) (fixedIndex varyingIndex : Fin 9) : Cell :=
  if column then (varyingIndex, fixedIndex) else (fixedIndex, varyingIndex)
abbrev collection (column : Bool) : Expr :=
  .listComp cellExpression [(varying column, .call (.name "range") [.constant (.int 9)], [])]

theorem row_source : definition.2[1]? = some
    (.forIn (.name "r") (.call (.name "range") [.constant (.int 9)])
      [.expr (.call (.attribute (.name "units") "append") [collection false])] []) := rfl
theorem column_source : definition.2[2]? = some
    (.forIn (.name "c") (.call (.name "range") [.constant (.int 9)])
      [.expr (.call (.attribute (.name "units") "append") [collection true])] []) := rfl

theorem linear (r c : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getRow : env "r" = some (number r)) (getColumn : env "c" = some (number c)) :
    Evaluates program env cellExpression heap (number (r * 9 + c)) heap globals := by
  have multiplied : Evaluates program env (.binOp (.name "r") .mult (.constant (.int 9))) heap
      (number (r * 9)) heap globals := by
    apply Evaluates.binary (Evaluates.lookup getRow) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  apply Evaluates.binary multiplied (Evaluates.lookup getColumn)
  simp [Mutable.binary, scalarBinary, number]

/-- The original row and column comprehensions allocate the nine cell indices in source order. -/
theorem collect (column : Bool) (index : Fin 9) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getFixed : env (fixed column) = some (number index.val)) :
    Evaluates program env (collection column) heap (.ref heap.length)
      (heap ++ [.list (peerLoop.encoded ((List.finRange 9).map (position column index)))]) globals := by
  let storage := fun output => HeapCollection.working heap output []
  let invariant := fun scope : HeapCalls.Env => scope (fixed column) = some (number index.val)
  let update := fun output (other : Fin 9) => output ++ [number (Encoding.index (position column index other)).val]
  let inner := comprehensionEnv env [(varying column, .call (.name "range") [.constant (.int 9)], [])]
  have callback (output : List Mutable.Value) (scope : HeapCalls.Env) (other : Fin 9) (ready : invariant scope)
      (getOther : scope (varying column) = some (.scalar (.int (0 + 1 * other.val)))) :
      ∃ afterEnv, BlockContinues program scope [.comprehensionAppend heap.length cellExpression]
        (storage output) afterEnv (storage (update output other)) globals ∧ invariant afterEnv := by
    have getRow : scope "r" = some (number (position column index other).1.val) := by
      cases column with
      | false => exact ready
      | true => simpa [varying, position] using getOther
    have getColumn : scope "c" = some (number (position column index other).2.val) := by
      cases column with
      | false => simpa [varying, position] using getOther
      | true => exact ready
    have read := linear (position column index other).1.val (position column index other).2.val scope
      (storage output) globals getRow getColumn
    refine ⟨scope, ?_, ready⟩
    have emitted := BlockContinues.comprehension_append read (HeapCollection.found heap output [])
    simpa only [storage, update, HeapCollection.store, Encoding.index] using emitted
  obtain ⟨afterEnv, scanned, _⟩ := HeapFiniteRange.loop program (varying column)
    [.comprehensionAppend heap.length cellExpression] rfl 9 0 1 (by decide) storage update invariant globals
    (by intro scope other ready; cases column <;> simpa [invariant, fixed, varying, setLocal] using ready)
    callback (List.finRange 9) 0 rfl (by omega) [] inner
    (by cases column <;> simpa [invariant, inner, fixed, varying, comprehensionEnv] using getFixed)
  apply Evaluates.list_comp_range (finalEnv := afterEnv) (.cons Evaluates.constant .nil) rfl
  exact scanned

def table (base : Nat) : UnitTable :=
  ⟨base, List.ofFn fun unit : Fin 27 => ⟨base + 1 + unit.val, Units.cells unit, Units.related unit⟩⟩

theorem table_complete (base : Nat) : (table base).Complete := by
  intro a b peer
  obtain ⟨unit, i, j, first, second⟩ := Units.complete a b peer
  refine ⟨⟨base + 1 + unit.val, Units.cells unit, Units.related unit⟩, ?_, i, j, first, second⟩
  simp only [table, List.mem_ofFn]
  exact ⟨unit, rfl⟩

end
end solvers.«96».build_peers_units
