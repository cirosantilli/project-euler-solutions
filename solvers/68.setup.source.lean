import solvers.«68.scan.source»

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

abbrev integer (value : Int) : Mutable.Value := .scalar (.int value)
abbrev guardExpression : Expr := .boolOp .or
  [.compare (.name "n") [(.lt, .constant (.int 3))],
   .compare (.name "digits") [(.lt, .binOp (.constant (.int 3)) .mult (.name "n"))]]
abbrev upperExpression : Expr :=
  .binOp (.binOp (.constant (.int 2)) .mult (.name "n")) .add (.constant (.int 1))
abbrev labelBounds : List Expr := [.constant (.int 1), upperExpression]
abbrev labelsExpression : Expr := .listComp (.name "$formalsw.range.item")
  [("$formalsw.range.item", .call (.name "range") labelBounds, [])]
abbrev totalWidthExpression : Expr := .sumRange "value" widthElement labelBounds []
abbrev labelTotalExpression : Expr := .binOp (.name "n") .mult upperExpression
abbrev initializationCode : List Stmt :=
  [.assign [.name "labels"] labelsExpression,
   .assign [.name "nums"] (.builtinCall .set [.name "labels"]),
   .assign [.name "total_digits"] totalWidthExpression,
   .assign [.name "total"] labelTotalExpression,
   .assign [.name "best"] (.constant (.string ""))]

theorem solver_function_source : findFunctionIn program.body "solve" =
    some (solverArgs, solverBody) := rfl

theorem solver_code_source : solverCode =
    [.ifThen guardExpression [.return (.constant (.string ""))] []] ++
      initializationCode ++ [selectionLoop, .return (.name "best")] := rfl

theorem guard_result (n digits : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (integer n)) (getDigits : env "digits" = some (integer digits)) :
    Evaluates program env guardExpression heap
      (.scalar (.bool (decide (n < 3 ∨ digits < 3 * n)))) heap := by
  have low : Evaluates program env (.compare (.name "n") [(.lt, .constant (.int 3))]) heap
      (.scalar (.bool (decide (n < 3)))) heap :=
    Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getN) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  have short : Evaluates program env
      (.compare (.name "digits") [(.lt, .binOp (.constant (.int 3)) .mult (.name "n"))]) heap
      (.scalar (.bool (decide (digits < 3 * n)))) heap :=
    Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getDigits)
      (Evaluates.binary (middle := heap) (afterRight := heap) (value := integer (3 * n))
        Evaluates.constant (Evaluates.lookup getN) rfl)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases small : n < 3
  · simpa only [small, true_or, decide_true] using
      (Evaluates.boolean_stop (op := .or)
        (rest := [.compare (.name "digits")
          [(.lt, .binOp (.constant (.int 3)) .mult (.name "n"))]])
        (by simpa only [small, decide_true] using low) rfl rfl)
  · change Evaluates program env (.boolOp .or _) heap _ heap
    apply Evaluates.boolean_next (flag := false)
      (by simpa only [small, decide_false] using low) rfl rfl
    simpa only [small, false_or] using Evaluates.boolean_last short rfl

theorem upper_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    Evaluates program env upperExpression heap (number (2 * n + 1)) heap :=
  Evaluates.binary (av := number (2 * n)) (bv := number 1) (middle := heap) (afterRight := heap)
    (Evaluates.binary (middle := heap) (afterRight := heap)
      Evaluates.constant (Evaluates.lookup getN) (by simp [Mutable.binary, scalarBinary]))
    Evaluates.constant (by simp [Mutable.binary, scalarBinary])

theorem label_bounds_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    Evaluates.Arguments program env labelBounds heap [number 1, number (2 * n + 1)] heap :=
  .cons Evaluates.constant (.cons (upper_result n heap env getN) .nil)

theorem labels_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    ∃ extra, Evaluates program env labelsExpression heap (.ref heap.length)
      (heap ++ (.list (encode (labels n)) :: extra)) := by
  obtain ⟨extra, collected⟩ := ProjectEulerSolutions.HeapCollection.range_map (globals := [])
    program "$formalsw.range.item" (.name "$formalsw.range.item") number
    (fun heap env value found => ⟨[], by simpa using
      (Evaluates.lookup (module := program) (heap := heap) found)⟩)
    heap env labelBounds [number 1, number (2 * n + 1)] 1 (2 * n)
    (label_bounds_result n heap env getN)
    (by change Except.ok (ε := ExceptionObject) ((1 : Int), ((2 * n + 1 : Nat) : Int), (1 : Int)) =
          .ok (1, (1 : Int) + ((2 * n : Nat) : Int), 1)
        simp only [Int.natCast_add, Int.natCast_one, Int.add_comm])
  exact ⟨extra, by simpa only [labelsExpression, labels, encode,
    ProjectEulerSolutions.HeapCollection.working] using collected⟩

theorem range_width_loop (first count stop : Nat) (length : first + count = stop)
    (initial : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (accumulator : env sumAccumulator = some (integer initial)) :
    BlockReturns program env
      [.rangeStepLoop "value" first stop 1 (sumBody widthElement []) [],
       .return (.name sumAccumulator)] heap
      (integer (initial + (((List.range' first count).map digitWidth).sum : Int))) heap := by
  induction count generalizing first initial env with
  | zero =>
      simp only [List.range'_zero, List.map_nil, List.sum_nil, Nat.cast_zero, Int.add_zero]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup accumulator)
  | succ count ih =>
      let indexed := setLocal env "value" (number first)
      let updated := setLocal indexed sumAccumulator (integer (initial + digitWidth first))
      have nextRun := ih (first + 1) (by omega) (initial + digitWidth first) updated rfl
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturns.assign (value := integer (initial + digitWidth first))
      · exact Evaluates.binary (middle := heap) (afterRight := heap)
          (Evaluates.lookup (by simpa [indexed, setLocal] using accumulator))
          (width_element_result first heap indexed rfl) rfl
      · simpa only [indexed, updated, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true, if_false,
          List.range'_succ, List.map_cons, List.sum_cons, Nat.cast_add, Int.add_assoc,
          Int.natCast_add, Int.natCast_one] using nextRun

theorem total_width_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    Evaluates program env totalWidthExpression heap (number ((labels n).map digitWidth).sum)
      heap := by
  apply Evaluates.sum_range
  apply Evaluates.sum_generator_range (label_bounds_result n heap env getN)
    (first := 1) (stop := (1 : Int) + 2 * n) (stride := 1)
  · change Except.ok (ε := ExceptionObject) ((1 : Int), ((2 * n + 1 : Nat) : Int), (1 : Int)) =
      .ok (1, (1 : Int) + 2 * n, 1)
    simp only [Int.natCast_add, Int.natCast_one, Int.natCast_mul, Nat.cast_ofNat, Int.add_comm]
  · exact Evaluates.constant
  · rfl
  · have summed := range_width_loop 1 (2 * n) (2 * n + 1) (by omega) 0 heap
      (setLocal (comprehensionEnv env [("value", .call (.name "range") labelBounds, [])])
        sumAccumulator (integer 0)) rfl
    simpa only [labels, Int.zero_add, Int.natCast_add, Int.natCast_one,
      Int.natCast_mul, Nat.cast_ofNat, Int.add_comm] using summed

theorem label_total_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    Evaluates program env labelTotalExpression heap (number (n * (2 * n + 1))) heap :=
  Evaluates.binary (middle := heap) (afterRight := heap)
    (Evaluates.lookup getN) (upper_result n heap env getN)
    (by simp [Mutable.binary, scalarBinary])

end solvers.«68»
