import solvers.«85.py.lean.d».Arithmetic
import ProjectEulerSolutions.Lib.RectangleSearch

/-! The reference's actual nested loops, including tuple-allocation effects. -/
namespace solvers.«85».Reference
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangleApproximation ProjectEulerSolutions.RectangleSearch
set_option Elab.async false

abbrev program := P85Ast.module
abbrev integer (value : Int) : Mutable.Value := .scalar (.int value)

structure Locals (env : HeapCalls.Env) (target : Int) (m : Nat) (best : Grid) : Prop where
  targetValue : env "target" = some (integer target)
  rowValue : env "m" = some (number m)
  areaValue : env "best_area" = some (number (best.1 * best.2))
  differenceValue : env "best_diff" = some (integer (error target best))

abbrev countExpression (column : String) : Expr := .call (.name "rect_count") [.name "m", .name column]
abbrev condition (column : String) : Expr := .compare (countExpression column)
  [(.ltE, .binOp (.name "target") .add (.name "best_diff"))]
abbrev differenceExpression : Expr := .builtinCall .abs
  [.binOp (countExpression "n") .sub (.name "target")]
abbrev updateBody : List Stmt :=
  [.assign [.tuple [.name "best_area", .name "best_diff"]]
    (.tuple [.binOp (.name "m") .mult (.name "n"), .name "diff"])]
abbrev innerBody : List Stmt :=
  [.assign [.name "diff"] differenceExpression,
   .ifThen (.compare (.name "diff") [(.lt, .name "best_diff")]) updateBody [],
   .augAssign (.name "n") .add (.constant (.int 1))]
abbrev innerLoop : Stmt := .whileDo (condition "n") innerBody []
abbrev outerBody : List Stmt :=
  [.assign [.name "n"] (.name "m"), innerLoop,
   .augAssign (.name "m") .add (.constant (.int 1))]
abbrev outerLoop : Stmt := .whileDo (condition "m") outerBody []

theorem count_evaluate (m n : Nat) (column : String) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getM : env "m" = some (number m)) (getN : env column = some (number n)) :
    Evaluates program env (countExpression column) heap (number (count m n)) heap globals :=
  Evaluates.call (.cons (Evaluates.lookup getM) (.cons (Evaluates.lookup getN) .nil))
    (rect_count.result m n heap globals)

theorem condition_evaluate (target : Int) (m n : Nat) (best : Grid) (column : String)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (ready : Locals env target m best) (getN : env column = some (number n)) :
    Evaluates program env (condition column) heap
      (.scalar (.bool (decide ((count m n : Int) ≤ target + error target best)))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (count_evaluate m n column env heap globals ready.rowValue getN)
    (Evaluates.binary (Evaluates.lookup ready.targetValue) (Evaluates.lookup ready.differenceValue) rfl)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem difference_evaluate (target : Int) (m n : Nat) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n))
    (getTarget : env "target" = some (integer target)) :
    Evaluates program env differenceExpression heap (integer (distance target m n)) heap globals := by
  apply Evaluates.builtin (middle := heap)
    (values := [integer ((count m n : Int) - target)])
  · exact .cons (Evaluates.binary (count_evaluate m n "n" env heap globals getM getN)
      (Evaluates.lookup getTarget) rfl) .nil
  · change (some (.ok (integer (((count m n : Int) - target).natAbs : Int)), heap) :
      Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
    simp [integer, distance]

/-- Execute one candidate comparison and column increment, retaining the caller heap prefix. -/
theorem iteration (target : Int) (m n : Nat) (best : Grid) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (ready : Locals env target m best) (getN : env "n" = some (number n)) :
    ∃ nextEnv nextHeap,
      BlockContinues program env innerBody heap nextEnv nextHeap globals ∧
      Locals nextEnv target m (choose target best (m, n)) ∧
      nextEnv "n" = some (number (n + 1)) ∧ heap <+: nextHeap := by
  let assigned := setLocal env "diff" (integer (distance target m n))
  have assignment : BlockContinues program env [.assign [.name "diff"] differenceExpression]
      heap assigned heap globals :=
    BlockContinues.assign (difference_evaluate target m n env heap globals ready.rowValue getN ready.targetValue)
  have tested : Evaluates program assigned (.compare (.name "diff") [(.lt, .name "best_diff")])
      heap (.scalar (.bool (decide (error target (m, n) < error target best)))) heap globals := by
    apply Evaluates.compare (middle := heap) (afterRight := heap)
      (av := integer (distance target m n)) (bv := integer (error target best))
      (Evaluates.lookup (by simp [assigned, setLocal]))
      (Evaluates.lookup (by simpa [assigned, setLocal] using ready.differenceValue))
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, error]
  by_cases improves : error target (m, n) < error target best
  · let items := [number (m * n), integer (error target (m, n))]
    let extended := heap ++ [.tuple items]
    let updated := assignNames assigned ["best_area", "best_diff"] items
    let advanced := setLocal updated "n" (number (n + 1))
    have updatedBody : BlockContinues program assigned updateBody heap updated extended globals := by
      apply BlockContinues.unpack (names := ["best_area", "best_diff"]) rfl
        (value := .ref heap.length) (items := items) (middle := extended)
      · apply Evaluates.tuple (values := items)
        · exact .cons (Evaluates.binary
            (Evaluates.lookup (by simpa [assigned, setLocal] using ready.rowValue))
            (Evaluates.lookup (by simpa [assigned, setLocal] using getN))
            (by simp [Mutable.binary, scalarBinary, number]))
            (.cons (Evaluates.lookup (by simp [assigned, setLocal, error, integer])) .nil)
        · simp [items]
      · simp [Mutable.sequence, extended, items]
      · rfl
    have branch : BlockContinues program assigned
        [.ifThen (.compare (.name "diff") [(.lt, .name "best_diff")]) updateBody []]
        heap updated extended globals :=
      BlockContinues.branch (flag := true) (by simpa only [improves, decide_true] using tested) rfl updatedBody
    have increment : BlockContinues program updated [.augAssign (.name "n") .add (.constant (.int 1))]
        extended advanced extended globals := by
      apply BlockContinues.augAssign (left := number n) (right := number 1)
      · simpa [updated, assigned, assignNames, setLocal] using getN
      · exact Evaluates.constant
      · simp [inplace, Mutable.binary, scalarBinary, number]
    refine ⟨advanced, extended, assignment.append (branch.append increment), ?_, ?_, ?_⟩
    · rw [choose, if_pos improves]
      constructor
      · simpa [advanced, updated, assigned, assignNames, setLocal, items] using ready.targetValue
      · simpa [advanced, updated, assigned, assignNames, setLocal, items] using ready.rowValue
      · simp [advanced, updated, assigned, assignNames, setLocal, items]
      · simp [advanced, updated, assigned, assignNames, setLocal, items]
    · simp [advanced, setLocal]
    · exact List.prefix_append _ _
  · let advanced := setLocal assigned "n" (number (n + 1))
    have branch : BlockContinues program assigned
        [.ifThen (.compare (.name "diff") [(.lt, .name "best_diff")]) updateBody []]
        heap assigned heap globals :=
      BlockContinues.branch (flag := false) (by simpa only [improves, decide_false] using tested) rfl BlockContinues.nil
    have increment : BlockContinues program assigned [.augAssign (.name "n") .add (.constant (.int 1))]
        heap advanced heap globals := by
      apply BlockContinues.augAssign (left := number n) (right := number 1)
      · simpa [assigned, setLocal] using getN
      · exact Evaluates.constant
      · simp [inplace, Mutable.binary, scalarBinary, number]
    refine ⟨advanced, heap, assignment.append (branch.append increment), ?_, ?_, by simp⟩
    · rw [choose, if_neg improves]
      constructor
      · simpa [advanced, assigned, setLocal] using ready.targetValue
      · simpa [advanced, assigned, setLocal] using ready.rowValue
      · simpa [advanced, assigned, setLocal] using ready.areaValue
      · simpa [advanced, assigned, setLocal] using ready.differenceValue
    · simp [advanced, setLocal]

/-- Follow every step of a complete reference row execution. -/
theorem row_result {target : Int} {m n : Nat} {best final : Grid}
    (run : RowRun target m n best final) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (ready : Locals env target m best) (getN : env "n" = some (number n)) :
    ∃ finalEnv finalHeap,
      BlockContinues program env [innerLoop] heap finalEnv finalHeap globals ∧
      Locals finalEnv target m final ∧ heap <+: finalHeap := by
  induction run generalizing env heap with
  | stop n best finished =>
      have tested := condition_evaluate target m n best "n" env heap globals ready getN
      have inactive : ¬(count m n : Int) ≤ target + error target best := by omega
      refine ⟨env, heap, ?_, ready, by simp⟩
      exact BlockContinues.while_loop (flag := false)
        (by simpa only [inactive, decide_false] using tested) rfl BlockContinues.nil
  | step n best final active tail ih =>
      have tested := condition_evaluate target m n best "n" env heap globals ready getN
      obtain ⟨nextEnv, nextHeap, advanced, nextReady, nextN, preserved⟩ :=
        iteration target m n best env heap globals ready getN
      obtain ⟨finalEnv, finalHeap, continued, finalReady, restPrefix⟩ :=
        ih nextEnv nextHeap nextReady nextN
      refine ⟨finalEnv, finalHeap, ?_, finalReady, preserved.trans restPrefix⟩
      apply BlockContinues.while_loop (flag := true)
        (by simpa only [active, decide_true] using tested) rfl
      change BlockContinues program env (innerBody ++ [innerLoop]) heap finalEnv finalHeap globals
      exact advanced.append continued

/-- The nested reference loops realize the terminating mathematical scan. -/
theorem rows_result {target : Int} {m : Nat} {best final : Grid}
    (run : RowsRun target (referenceRow target) m best final) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (ready : Locals env target m best) :
    ∃ finalEnv finalHeap,
      BlockContinues program env [outerLoop] heap finalEnv finalHeap globals ∧
      finalEnv "best_area" = some (number (final.1 * final.2)) ∧ heap <+: finalHeap := by
  induction run generalizing env heap with
  | stop m best finished =>
      have tested := condition_evaluate target m m best "m" env heap globals ready ready.rowValue
      have inactive : ¬(count m m : Int) ≤ target + error target best := by omega
      refine ⟨env, heap, ?_, ready.areaValue, by simp⟩
      exact BlockContinues.while_loop (flag := false)
        (by simpa only [inactive, decide_false] using tested) rfl BlockContinues.nil
  | step m best next final active searched tail ih =>
      have tested := condition_evaluate target m m best "m" env heap globals ready ready.rowValue
      let started := setLocal env "n" (number m)
      have start : BlockContinues program env [.assign [.name "n"] (.name "m")]
          heap started heap globals := BlockContinues.assign (Evaluates.lookup ready.rowValue)
      have startReady : Locals started target m best := by
        constructor
        · simpa [started, setLocal] using ready.targetValue
        · simpa [started, setLocal] using ready.rowValue
        · simpa [started, setLocal] using ready.areaValue
        · simpa [started, setLocal] using ready.differenceValue
      obtain ⟨rowEnv, rowHeap, rowDone, rowReady, rowPrefix⟩ :=
        row_result searched started heap globals startReady (by simp [started, setLocal])
      let advanced := setLocal rowEnv "m" (number (m + 1))
      have increment : BlockContinues program rowEnv [.augAssign (.name "m") .add (.constant (.int 1))]
          rowHeap advanced rowHeap globals := by
        apply BlockContinues.augAssign rowReady.rowValue (right := number 1) Evaluates.constant
        simp [inplace, Mutable.binary, scalarBinary, number]
      have nextReady : Locals advanced target (m + 1) next := by
        constructor
        · simpa [advanced, setLocal] using rowReady.targetValue
        · simp [advanced, setLocal]
        · simpa [advanced, setLocal] using rowReady.areaValue
        · simpa [advanced, setLocal] using rowReady.differenceValue
      obtain ⟨finalEnv, finalHeap, continued, found, preserved⟩ := ih advanced rowHeap nextReady
      refine ⟨finalEnv, finalHeap, ?_, found, rowPrefix.trans preserved⟩
      apply BlockContinues.while_loop (flag := true)
        (by simpa only [active, decide_true] using tested) rfl
      change BlockContinues program env
        ([.assign [.name "n"] (.name "m")] ++ [innerLoop] ++
          [.augAssign (.name "m") .add (.constant (.int 1))] ++ [outerLoop])
        heap finalEnv finalHeap globals
      exact ((start.append rowDone).append increment).append continued

/-- The actual reference returns the scan's area for every integer target. -/
theorem naive_result_of_run (target : Int) (final : Grid)
    (run : RowsRun target (referenceRow target) 1 (1, 1) final)
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ finalHeap, ReturnsValue program "naive" [integer target] heap
      (number (final.1 * final.2)) finalHeap globals ∧ heap <+: finalHeap := by
  let args : List Arg := [⟨"target", some (.name "int"), none⟩]
  let code : List Stmt :=
    [.assign [.name "best_area"] (.constant (.int 1)),
     .assign [.name "best_diff"] (.builtinCall .abs [.binOp (.constant (.int 1)) .sub (.name "target")]),
     .assign [.name "m"] (.constant (.int 1)), outerLoop, .return (.name "best_area")]
  let initial := functionEnv args code [integer target] globals
  let areaSet := setLocal initial "best_area" (number 1)
  let differenceSet := setLocal areaSet "best_diff" (integer |1 - target|)
  let readyEnv := setLocal differenceSet "m" (number 1)
  have ready : Locals readyEnv target 1 (1, 1) := by
    constructor
    · cases globals <;> rfl
    · rfl
    · rfl
    · simp [readyEnv, differenceSet, setLocal, error, distance, count,
        ProjectEulerSolutions.TriPentHex.triangular]
  obtain ⟨finalEnv, finalHeap, executed, found, preserved⟩ := rows_result run readyEnv heap globals ready
  refine ⟨finalHeap, ?_, preserved⟩
  apply BlockReturns.function (args := args) (body := code) rfl rfl
  apply BlockReturns.assign (middle := heap) (value := number 1) Evaluates.constant
  apply BlockReturns.assign (middle := heap) (value := integer |1 - target|)
  · apply Evaluates.builtin (middle := heap) (values := [integer (1 - target)])
    · exact .cons (Evaluates.binary Evaluates.constant
        (Evaluates.lookup (by cases globals <;> rfl)) rfl) .nil
    · change (some (.ok (integer ((1 - target).natAbs : Int)), heap) :
        Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
      simp [integer]
  apply BlockReturns.assign (middle := heap) (value := number 1) Evaluates.constant
  exact executed.returns (BlockReturns.return_value (Evaluates.lookup found))

/-- Termination and the least-grid specification hold for the actual Python reference. -/
theorem naive_result (target : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ final finalHeap, Optimal target final ∧
      ReturnsValue program "naive" [integer target] heap
        (number (final.1 * final.2)) finalHeap globals ∧ heap <+: finalHeap := by
  obtain ⟨final, run⟩ := rows_exists target _ (referenceRow_contract target) 1 (by omega) (1, 1)
  obtain ⟨finalHeap, returned, preserved⟩ := naive_result_of_run target final run heap globals
  exact ⟨final, finalHeap, run.optimal (referenceRow_contract target), returned, preserved⟩

end solvers.«85».Reference
