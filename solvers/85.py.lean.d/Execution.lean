import solvers.«85.py.lean.d».Loop

/-! Function entry, returned tuple, and area projection for the optimized solver. -/
namespace solvers.«85».Optimized
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangleApproximation ProjectEulerSolutions.RectangleSearch
set_option Elab.async false

theorem best_grid_result_of_run (target : Int) (final : Grid)
    (run : RowsRun target (optimizedRow target) 1 (1, 1) final)
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ address finalHeap, ReturnsValue program "best_grid_area_near" [integer target] heap
      (.ref address) finalHeap globals ∧
      finalHeap[address]? = some (.tuple (resultItems final)) ∧ heap <+: finalHeap := by
  let args := ((findFunctionIn program.body "best_grid_area_near").getD ([], [])).1
  let body := ((findFunctionIn program.body "best_grid_area_near").getD ([], [])).2
  let initial := functionEnv args body [integer target] globals
  let items := [number 1, number 1]
  let extended := heap ++ [.tuple items]
  let pairSet := assignNames initial ["best_m", "best_n"] items
  let differenceSet := setLocal pairSet "best_diff" (integer |1 - target|)
  let readyEnv := setLocal differenceSet "m" (number 1)
  have ready : Locals readyEnv target 1 (1, 1) := by
    constructor
    · cases globals <;> rfl
    · rfl
    · rfl
    · rfl
    · simp [readyEnv, differenceSet, setLocal, error, distance, count,
        ProjectEulerSolutions.TriPentHex.triangular]
  obtain ⟨address, finalHeap, executed, found, preserved⟩ :=
    loop_result run (by omega) readyEnv extended globals ready
  refine ⟨address, finalHeap, ?_, found, (List.prefix_append _ _).trans preserved⟩
  apply BlockReturns.function (args := args) (body := body) rfl rfl
  apply BlockReturns.unpack (names := ["best_m", "best_n"]) rfl
    (value := .ref heap.length) (items := items) (middle := extended) (afterUnpack := extended)
  · exact Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp [items])
  · simp [Mutable.sequence, extended, items]
  · rfl
  apply BlockReturns.assign (middle := extended) (value := integer |1 - target|)
  · apply Evaluates.builtin (middle := extended) (values := [integer (1 - target)])
    · exact .cons (Evaluates.binary Evaluates.constant
        (Evaluates.lookup (by cases globals <;> rfl)) rfl) .nil
    · change (some (.ok (integer ((1 - target).natAbs : Int)), extended) :
        Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
      simp [integer]
  apply BlockReturns.assign (middle := extended) (value := number 1) Evaluates.constant
  exact executed

theorem solve_result_of_run (target : Int) (final : Grid)
    (run : RowsRun target (optimizedRow target) 1 (1, 1) final)
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ finalHeap, ReturnsValue program "solve" [integer target] heap
      (number (final.1 * final.2)) finalHeap globals ∧ heap <+: finalHeap := by
  obtain ⟨address, finalHeap, returned, found, preserved⟩ := best_grid_result_of_run target final run heap globals
  refine ⟨finalHeap, ?_, preserved⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.subscript (value := .ref address) (position := number 2)
    (middle := finalHeap) (afterIndex := finalHeap) (by intro a b c; simp)
  · exact Evaluates.call (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil) returned
  · exact Evaluates.constant
  · simp [Mutable.getItem, found, Mutable.index, resultItems, number, Except.map]

/-- The complete Python solver terminates and returns the area of the uniquely least grid. -/
theorem solve_result (target : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ final finalHeap, Optimal target final ∧ ReturnsValue program "solve" [integer target] heap
      (number (final.1 * final.2)) finalHeap globals ∧ heap <+: finalHeap := by
  obtain ⟨final, run⟩ := rows_exists target _ (optimizedRow_contract target) 1 (by omega) (1, 1)
  obtain ⟨finalHeap, returned, preserved⟩ := solve_result_of_run target final run heap globals
  exact ⟨final, finalHeap, run.optimal (optimizedRow_contract target), returned, preserved⟩

end solvers.«85».Optimized
