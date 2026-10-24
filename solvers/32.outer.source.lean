import solvers.«32.collect.source»
import Mathlib.Data.Nat.Sqrt

set_option Elab.async false

namespace solvers.«32».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.PandigitalProductSearch
private abbrev program := Solvers.P32Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

def firstBExpr : Expr := .builtinCall .max
  [.name "a", .binOp (.binOp (.binOp (.name "lower") .add (.name "a")) .sub (.constant (.int 1)))
    .floorDiv (.name "a")]

def outerBody : List Stmt :=
  [.assign [.name "first_b"] firstBExpr,
   .forIn (.name "b") (.call (.name "range")
     [.name "first_b", .binOp (.binOp (.name "upper") .floorDiv (.name "a")) .add (.constant (.int 1))])
     candidateBody []]

theorem outer_source :
    ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[6]?) =
      some (.forIn (.name "a") (.call (.name "range")
        [.constant (.int 1), .binOp (.call (.name "isqrt") [.name "upper"]) .add (.constant (.int 1))])
        outerBody []) := by rfl

theorem firstB_result (width a : Nat) (positive : 0 < a) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getA : env "a" = some (number a))
    (getLower : env "lower" = some (number (lowerProduct width))) :
    Evaluates program env firstBExpr heap (number (firstB width a)) heap := by
  let quotient := (lowerProduct width + a - 1) / a
  have divided : Evaluates program env
      (.binOp (.binOp (.binOp (.name "lower") .add (.name "a")) .sub (.constant (.int 1)))
        .floorDiv (.name "a")) heap (number quotient) heap := by
    apply Evaluates.binary (av := number (lowerProduct width + a - 1)) (bv := number a)
      (middle := heap) (afterRight := heap)
    · apply Evaluates.binary (av := number (lowerProduct width + a)) (bv := number 1)
        (middle := heap) (afterRight := heap) (finalHeap := heap)
      · exact Evaluates.binary (av := number (lowerProduct width)) (bv := number a)
          (middle := heap) (afterRight := heap) (finalHeap := heap)
          (Evaluates.lookup getLower) (Evaluates.lookup getA) (by simp [Mutable.binary, scalarBinary])
      · exact Evaluates.constant
      · simp [Mutable.binary, scalarBinary, Int.natCast_sub (show 1 ≤ lowerProduct width + a by omega)]
    · exact Evaluates.lookup getA
    · simp [Mutable.binary, scalarBinary, quotient, positive, positive.ne']
  apply Evaluates.builtin (.cons (Evaluates.lookup getA) (.cons divided .nil))
  change scalarResult (Calls.positionalMaximum [.int a, .int quotient]) heap =
    some (.ok (number (max a quotient)), heap)
  by_cases better : a < quotient
  · simp [Calls.positionalMaximum, Calls.maximumFrom, scalarCompare, evalCmpOp,
      scalarResult, better, Nat.max_eq_right (Nat.le_of_lt better)]
  · simp [Calls.positionalMaximum, Calls.maximumFrom, scalarCompare, evalCmpOp,
      scalarResult, better, Nat.max_eq_left (Nat.le_of_not_gt better)]

theorem row_result (width a : Nat) (positive : 0 < a) (base : Mutable.Heap)
    (before : List Constant) (env : HeapCalls.Env)
    (getA : env "a" = some (number a)) (getWidth : env "width" = some (number width))
    (getLower : env "lower" = some (number (lowerProduct width)))
    (getUpper : env "upper" = some (number (upperProduct width)))
    (getProducts : env "products" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env outerBody (working base before) finalEnv
      (working base (inserted before (row width a))) ∧ Stable env finalEnv := by
  let ready := setLocal env "first_b" (number (firstB width a))
  have firstPositive : 0 < firstB width a := lt_of_lt_of_le positive (Nat.le_max_left ..)
  obtain ⟨finalEnv, scanned, stable⟩ := inner_loop width a (firstB width a)
    (lastB width a + 1 - firstB width a) (lastB width a + 1) positive firstPositive rfl base before ready
    (by simpa [ready, setLocal] using getA) (by simpa [ready, setLocal] using getWidth)
    (by simpa [ready, setLocal] using getProducts)
  rw [rowSlice_full] at scanned
  have upperExpr : Evaluates program ready
      (.binOp (.binOp (.name "upper") .floorDiv (.name "a")) .add (.constant (.int 1)))
      (working base before) (number (lastB width a + 1)) (working base before) := by
    apply Evaluates.binary (av := number (lastB width a)) (bv := number 1)
      (middle := working base before) (afterRight := working base before)
    · exact Evaluates.binary (av := number (upperProduct width)) (bv := number a)
        (middle := working base before) (afterRight := working base before)
        (Evaluates.lookup (by simpa [ready, setLocal] using getUpper))
        (Evaluates.lookup (by simpa [ready, setLocal] using getA))
        (by simp [Mutable.binary, scalarBinary, lastB, positive, positive.ne'])
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary]
  have iterated : BlockContinues program ready [outerBody[1]'(by decide)]
      (working base before) finalEnv (working base (inserted before (row width a))) := by
    exact BlockContinues.for_range
      (.cons (Evaluates.lookup (value := number (firstB width a))
        (by simp [ready, setLocal])) (.cons upperExpr .nil)) rfl scanned
  refine ⟨finalEnv, (BlockContinues.assign
    (firstB_result width a positive (working base before) env getA getLower)).append iterated, ?_⟩
  intro query unchanged
  have different : query ≠ "first_b" := by
    simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
    exact unchanged.2.2.1
  exact (stable query unchanged).trans (by simp [ready, setLocal, different])

theorem outer_loop (width first count stop : Nat) (positive : 0 < first) (endpoint : first + count = stop)
    (base : Mutable.Heap) (before : List Constant) (env : HeapCalls.Env)
    (getWidth : env "width" = some (number width))
    (getLower : env "lower" = some (number (lowerProduct width)))
    (getUpper : env "upper" = some (number (upperProduct width)))
    (getProducts : env "products" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "a" first stop 1 outerBody []]
      (working base before) finalEnv
      (working base (inserted before ((List.range' first count).flatMap (row width)))) ∧ Stable env finalEnv := by
  induction count generalizing first before env with
  | zero =>
      refine ⟨env, ?_, fun _ _ => rfl⟩
      simpa only [List.range'_zero, List.flatMap_nil, inserted, List.map_nil, List.foldl_nil] using
        (BlockContinues.range_done (module := program) (env := env) (name := "a")
          (next := first) (stop := stop) (stride := 1) (body := outerBody) (otherwise := [])
          (heap := working base before) (by simp [Calls.rangeContinues]; omega) .nil)
  | succ count ih =>
      let bound := setLocal env "a" (number first)
      obtain ⟨middle, executed, stable⟩ := row_result width first positive base before bound
        (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getWidth)
        (by simpa [bound, setLocal] using getLower) (by simpa [bound, setLocal] using getUpper)
        (by simpa [bound, setLocal] using getProducts)
      obtain ⟨finalEnv, completed, finalStable⟩ := ih (first + 1) (by omega) (by omega)
        (inserted before (row width first)) middle
        ((stable "width" (by decide)).trans (by simpa [bound, setLocal] using getWidth))
        ((stable "lower" (by decide)).trans (by simpa [bound, setLocal] using getLower))
        ((stable "upper" (by decide)).trans (by simpa [bound, setLocal] using getUpper))
        ((stable "products" (by decide)).trans (by simpa [bound, setLocal] using getProducts))
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [List.range'_succ, inserted_append, outerBody, candidateBody,
          Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using executed.append completed
      · intro query unchanged
        have different : query ≠ "a" := by
          simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
          exact unchanged.2.2.2
        exact (finalStable query unchanged).trans ((stable query unchanged).trans
          (by simp [bound, setLocal, different]))

end solvers.«32».solve
