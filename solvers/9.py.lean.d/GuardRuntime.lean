import solvers.«9.py.lean.d».ScalarRuntime

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

theorem triangleCondition_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total a b c : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) :
    ∃ flag cost, cost ≤ 57 ∧ EvaluatesIn .word program env triangleCondition heap cost (.scalar (.bool flag)) heap := by
  have first : EvaluatesIn .word program env (.compare (.name "a") [(.gt, .constant (.int 0))])
      heap 7 (.scalar (.bool (decide (0 < a)))) heap := by
    apply integer_compare (EvaluatesIn.lookup ha) EvaluatesIn.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have second : EvaluatesIn .word program env
      (.compare (.binOp (square "a") .add (square "b")) [(.eq, square "c")])
      heap 27 (.scalar (.bool (decide (a * a + b * b = c * c)))) heap := by
    apply integer_compare
      (integer_binary
        (integer_binary (EvaluatesIn.lookup ha) (EvaluatesIn.lookup ha) rfl)
        (integer_binary (EvaluatesIn.lookup hb) (EvaluatesIn.lookup hb) rfl) rfl)
      (integer_binary (EvaluatesIn.lookup hc) (EvaluatesIn.lookup hc) rfl)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have third : EvaluatesIn .word program env
      (.compare (.binOp (.binOp (.name "a") .add (.name "b")) .add (.name "c")) [(.eq, .name "total")])
      heap 17 (.scalar (.bool (decide (a + b + c = total)))) heap := by
    apply integer_compare
      (integer_binary
        (integer_binary (EvaluatesIn.lookup ha) (EvaluatesIn.lookup hb) rfl)
        (EvaluatesIn.lookup hc) rfl) (EvaluatesIn.lookup ht)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases positive : 0 < a
  · have firstTrue : EvaluatesIn .word program env (.compare (.name "a") [(.gt, .constant (.int 0))]) heap 7 (.scalar (.bool true)) heap := by
      simpa [positive] using first
    by_cases squares : a * a + b * b = c * c
    · refine ⟨decide (a + b + c = total), 57, by omega, ?_⟩
      apply EvaluatesIn.boolean_next (cb := 48) (flag := true) firstTrue rfl rfl
      apply EvaluatesIn.boolean_next (ca := 27) (cb := 19) (flag := true)
      · simpa [squares] using second
      · rfl
      · rfl
      · exact EvaluatesIn.boolean_last (flag := decide (a + b + c = total)) third rfl
    · refine ⟨false, 38, by omega, ?_⟩
      apply EvaluatesIn.boolean_next (cb := 29) (flag := true) firstTrue rfl rfl
      apply EvaluatesIn.boolean_stop (cost := 27) (flag := false)
      · simpa [squares] using second
      · rfl
      · rfl
  · refine ⟨false, 9, by omega, ?_⟩
    apply EvaluatesIn.boolean_stop (cost := 7) (flag := false)
    · simpa [positive] using first
    · rfl
    · rfl


theorem updateBest_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total a b c best : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (hbest : env "best" = some (.scalar (.int best))) :
    ∃ newBest finalEnv cost, cost ≤ 90 ∧ BlockContinuesIn .word program env updateBest heap cost finalEnv heap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "m" = env "m" ∧
      finalEnv "best" = some (.scalar (.int newBest)) := by
  obtain ⟨flag, cc, bounded, condition⟩ := triangleCondition_cost env heap total a b c ht ha hb hc
  cases flag with
  | false =>
      refine ⟨best, env, cc + 0 + 2, by omega, ?_, ht, rfl, hbest⟩
      exact BlockContinuesIn.branch condition rfl BlockContinuesIn.nil
  | true =>
      have maximum : EvaluatesIn .word program env (.builtinCall .max [.name "best", productExpression]) heap 26
          (.scalar (.int (max best (a * b * c)))) heap := by
        apply EvaluatesIn.maximum_pair (work := 9) (EvaluatesIn.lookup hbest)
          (integer_binary (integer_binary (EvaluatesIn.lookup ha) (EvaluatesIn.lookup hb) rfl)
            (EvaluatesIn.lookup hc) rfl)
        · by_cases larger : best < a * b * c
          · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom,
              HeapCalls.scalarResult, scalarCompare, evalCmpOp, larger, Int.max_def]
            omega
          · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom,
              HeapCalls.scalarResult, scalarCompare, evalCmpOp, larger, Int.max_def]
            omega
        · rfl
      refine ⟨max best (a * b * c), HeapCalls.setLocal env "best" (.scalar (.int (max best (a * b * c)))),
        cc + 28 + 2, by omega, BlockContinuesIn.branch condition rfl (BlockContinuesIn.assign maximum),
        ?_, ?_, rfl⟩
      · simpa [HeapCalls.setLocal] using ht
      · simp [HeapCalls.setLocal]

end solvers.«9».runtimeCost
