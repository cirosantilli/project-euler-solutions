import solvers.«38.py.lean.d».Predicate
import lang.py.FormalSW.IntegerParsing

namespace solvers.«38».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerStatements.P38
open ProjectEulerStatements.ConcatenatedProducts ProjectEulerSolutions.PandigitalConcat
set_option Elab.async false
private abbrev program := Solvers.P38Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
variable {globals : Mutable.Globals}
noncomputable section
private abbrev code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[3]? : Option Stmt) with | some (.forIn _ _ statements _) => statements | _ => []
private abbrev candidate (width x : Nat) := concatProduct x (stop width x)
private abbrev candidateText (width x : Nat) : Mutable.Value := .scalar (.string (toString (candidate width x : Int)))
private abbrev returnedValues (width x : Nat) : List Mutable.Value := [candidateText width x, number (stop width x)]
private def extra (width x : Nat) : Mutable.Heap :=
  [.list ((parts x (stop width x)).map fun piece => .scalar (.string piece)), .tuple (returnedValues width x)]
private abbrev guard : Expr := .boolOp .and
  [.compare (.name "n") [(.gt, .constant (.int 1))], .call (.name "is_pandigital") [.name "s", .name "width"]]
private abbrev acceptedBody : List Stmt :=
  [.assign [.name "val"] (.builtinCall .int [.name "s"]),
   .ifThen (.compare (.name "val") [(.gt, .name "best")]) [.assign [.name "best"] (.name "val")] []]

private theorem pandigital_result (width x : Nat) (nonempty : 0 < width) (positive : 0 < x) (heap : Mutable.Heap) :
    ReturnsValue (globals := globals) program "is_pandigital" [candidateText width x, number width] heap
      (.scalar (.bool (ProjectEulerSolutions.PandigitalDigits.check width (digitsBE (candidate width x))))) heap := by
  have valuePositive := concatProduct_positive x (stop width x) positive
    (by have := (stop_bounds width x nonempty positive).1; omega)
  have printed : String.ofList ((digitsBE (candidate width x)).map Nat.digitChar) = toString (candidate width x : Int) := by
    apply String.toList_injective
    rw [String.toList_ofList, decimal_characters _ valuePositive]
  have checked := is_pandigital.result (globals := globals) width (digitsBE (candidate width x))
    (by intro digit member; exact Nat.digits_lt_base (by decide) (List.mem_reverse.mp member)) heap
  change ReturnsValue (globals := globals) program "is_pandigital"
    [.scalar (.string (String.ofList ((digitsBE (candidate width x)).map Nat.digitChar))), number width] heap
    (.scalar (.bool (ProjectEulerSolutions.PandigitalDigits.check width (digitsBE (candidate width x))))) heap at checked
  simpa only [printed] using checked

private theorem guard_result (width x : Nat) (nonempty : 0 < width) (positive : 0 < x)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getCount : env "n" = some (number (stop width x)))
    (getText : env "s" = some (candidateText width x)) (getWidth : env "width" = some (number width)) :
    Evaluates (globals := globals) program env guard heap (.scalar (.bool (accepted width x (stop width x)))) heap := by
  have tested : Evaluates (globals := globals) program env (.compare (.name "n") [(.gt, .constant (.int 1))])
      heap (.scalar (.bool (decide (1 < stop width x)))) heap :=
    Evaluates.compare (Evaluates.lookup getCount) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  rw [accepted]
  by_cases several : 1 < stop width x
  · simp only [several, decide_true, Bool.true_and]
    apply Evaluates.boolean_next (flag := true) (by simpa [several] using tested) rfl rfl
    exact Evaluates.boolean_last (Evaluates.call
      (.cons (Evaluates.lookup getText) (.cons (Evaluates.lookup getWidth) .nil)) (pandigital_result width x nonempty positive heap)) rfl
  · simp only [several, decide_false, Bool.false_and]
    exact Evaluates.boolean_stop (flag := false) (by simpa [several] using tested) rfl rfl

private theorem accepted_result (width x best : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getText : env "s" = some (candidateText width x)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env acceptedBody heap finalEnv heap ∧
      finalEnv "best" = some (number (max best (candidate width x))) ∧ finalEnv "width" = env "width" := by
  let e1 := setLocal env "val" (number (candidate width x))
  have parsed : Evaluates (globals := globals) program env (.builtinCall .int [.name "s"])
      heap (number (candidate width x)) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getText) .nil)
    have convert (n : Int) : primitive (.builtin .int) [.scalar (.string (toString n))] heap =
        some (.ok (.scalar (.int n)), heap) := by
      change scalarResult (Calls.integerConversion [.string (toString n)]) heap = _
      rw [Calls.integerConversion_int]
      rfl
    exact convert (candidate width x)
  have first : BlockContinues (globals := globals) program env [.assign [.name "val"] (.builtinCall .int [.name "s"])]
      heap e1 heap := BlockContinues.assign parsed
  have compared : Evaluates (globals := globals) program e1 (.compare (.name "val") [(.gt, .name "best")]) heap
      (.scalar (.bool (decide (best < candidate width x)))) heap :=
    Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup (by simpa [e1, setLocal] using getBest))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases better : best < candidate width x
  · let finalEnv := setLocal e1 "best" (number (candidate width x))
    refine ⟨finalEnv, first.append ?_, ?_, by simp [finalEnv, e1, setLocal]⟩
    · exact BlockContinues.branch (flag := true) (by simpa [better] using compared) rfl
        (BlockContinues.assign (Evaluates.lookup rfl))
    · simp [finalEnv, setLocal, max_eq_right (Nat.le_of_lt better)]
  · refine ⟨e1, first.append ?_, ?_, by simp [e1, setLocal]⟩
    · exact BlockContinues.branch (flag := false) (by simpa [better] using compared) rfl .nil
    · simpa [e1, setLocal, max_eq_left (Nat.le_of_not_gt better)] using getBest

private theorem iteration (width x best : Nat) (nonempty : 0 < width) (positive : 0 < x)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getX : env "x" = some (number x))
    (getBest : env "best" = some (number best)) (getWidth : env "width" = some (number width)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env body heap finalEnv (heap ++ extra width x) ∧
      finalEnv "best" = some (number (max best (seedScore width x))) ∧ finalEnv "width" = some (number width) := by
  let unpacked := assignNames env ["s", "n"] (returnedValues width x)
  let after := heap ++ extra width x
  have first : BlockContinues (globals := globals) program env [body[0]'(by decide)] heap unpacked after := by
    apply BlockContinues.unpack (names := ["s", "n"]) (items := returnedValues width x) rfl
      (Evaluates.call (.cons (Evaluates.lookup getX) (.cons (Evaluates.lookup getWidth) .nil))
        (concatenated_product.result width x nonempty positive heap))
    · simp [Mutable.sequence, after, extra]
      rfl
    · rfl
  have getText : unpacked "s" = some (candidateText width x) := by simp [unpacked, returnedValues, assignNames, setLocal]
  have getCount : unpacked "n" = some (number (stop width x)) := by simp [unpacked, returnedValues, assignNames, setLocal]
  have sameBest : unpacked "best" = some (number best) := by simpa [unpacked, returnedValues, assignNames, setLocal] using getBest
  have sameWidth : unpacked "width" = some (number width) := by simpa [unpacked, returnedValues, assignNames, setLocal] using getWidth
  have tested := guard_result (globals := globals) width x nonempty positive after unpacked getCount getText sameWidth
  cases yes : accepted width x (stop width x) with
  | false =>
      refine ⟨unpacked, first.append ?_, ?_, sameWidth⟩
      · exact BlockContinues.branch (flag := false) (by simpa [yes] using tested) rfl .nil
      · simpa only [seedScore, yes, Bool.false_eq_true, if_false, Nat.max_zero] using sameBest
  | true =>
      obtain ⟨finalEnv, executed, finalBest, finalWidth⟩ := accepted_result (globals := globals) width x best after unpacked getText sameBest
      refine ⟨finalEnv, first.append ?_, ?_, finalWidth.trans sameWidth⟩
      · exact BlockContinues.branch (flag := true) (by simpa [yes] using tested) rfl executed
      · simpa only [seedScore, yes, if_true] using finalBest

private theorem loop (width first count best : Nat) (nonempty : 0 < width) (positive : 0 < first)
    (endpoint : first + count = seedLimit width) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getBest : env "best" = some (number best)) (getWidth : env "width" = some (number width)) :
    ∃ finalHeap, BlockReturns (globals := globals) program env
      [.rangeStepLoop "x" first (seedLimit width) 1 body [], .return (.name "best")]
      heap (number (scan width first count best)) finalHeap := by
  induction count generalizing first best heap env with
  | zero =>
      refine ⟨heap, ?_⟩
      rw [scan_zero]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getBest)
  | succ count ih =>
      let bound := setLocal env "x" (number first)
      obtain ⟨middleEnv, iterated, middleBest, middleWidth⟩ := iteration (globals := globals) width first best nonempty positive heap bound
        (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getBest) (by simpa [bound, setLocal] using getWidth)
      obtain ⟨finalHeap, rest⟩ := ih (first + 1) (max best (seedScore width first)) (by omega) (by omega)
        (heap ++ extra width first) middleEnv middleBest middleWidth
      refine ⟨finalHeap, ?_⟩
      rw [scan_succ]
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.returns iterated
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

/-- Empty target widths return before allocating a candidate collector. -/
theorem empty (width : Int) (small : width < 2) (heap : Mutable.Heap) :
    ReturnsValue (globals := globals) program "solve" [.scalar (.int width)] heap (.scalar (.int 0)) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := true)
    (Evaluates.compare (av := .scalar (.int width)) (bv := number 2)
      (value := .scalar (.bool true)) (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])) rfl
  exact BlockReturns.return_value Evaluates.constant

/-- Widths requiring nondecimal digits also return before search. -/
theorem large (width : Int) (above : 9 < width) (heap : Mutable.Heap) :
    ReturnsValue (globals := globals) program "solve" [.scalar (.int width)] heap (.scalar (.int 0)) heap := by
  apply BlockReturns.function rfl rfl
  have input : (functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
      ((findFunctionIn program.body "solve").getD ([], [])).2 [.scalar (.int width)] globals) "width" =
      some (.scalar (.int width)) := by cases globals <;> rfl
  apply BlockReturns.branch (flag := false)
    (Evaluates.compare (av := .scalar (.int width)) (bv := number 2)
      (value := .scalar (.bool false)) (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup input) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
  apply BlockReturns.branch (flag := true)
    (Evaluates.compare (av := .scalar (.int width)) (bv := number 9)
      (value := .scalar (.bool true)) (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup input) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, above])) rfl
  exact BlockReturns.return_value Evaluates.constant

private theorem active (width : Nat) (two : 2 ≤ width) (decimal : width ≤ 9) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue (globals := globals) program "solve" [number width] heap
      (number (reference width)) finalHeap := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number width] globals
  have input : initial "width" = some (number width) := by cases globals <;> rfl
  let ready := setLocal initial "best" (number 0)
  have getWidth : ready "width" = some (number width) := by simpa [ready, setLocal] using input
  have positive : 0 < seedLimit width := by unfold seedLimit; positivity
  obtain ⟨finalHeap, searched⟩ := loop (globals := globals) width 1 (seedLimit width - 1) 0
    (by omega) (by decide) (by omega) heap ready (by simp [ready, setLocal]) getWidth
  refine ⟨finalHeap, ?_⟩
  rw [reference_scan width (by omega)]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false)
    (Evaluates.compare (av := number width) (bv := number 2) (value := .scalar (.bool false))
      (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup input) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
  apply BlockReturns.branch (flag := false)
    (Evaluates.compare (av := number width) (bv := number 9) (value := .scalar (.bool false))
      (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup input) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range (values := [number 1, number (seedLimit width)]) (middle := heap)
  · refine .cons Evaluates.constant (.cons ?_ .nil)
    apply Evaluates.binary (av := number 10) (bv := number (width / 2)) (middle := heap) (afterRight := heap) Evaluates.constant
    · exact Evaluates.binary (av := number width) (bv := number 2) (middle := heap) (afterRight := heap)
        (Evaluates.lookup getWidth) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · have nonnegative : ¬ (width : Int) / 2 < 0 := not_lt_of_ge (Int.ediv_nonneg (by omega) (by omega))
      simp [Mutable.binary, scalarBinary, number, seedLimit, nonnegative]
      rfl
  · rfl
  · exact searched

/-- The cached-width search returns the complete mathematical maximum for every natural width. -/
theorem result (width : Nat) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue (globals := globals) program "solve" [number width] heap
      (number (reference width)) finalHeap := by
  by_cases small : width < 2
  · exact ⟨heap, by simpa only [reference_small width small] using empty (globals := globals) width (by omega) heap⟩
  · by_cases above : 9 < width
    · exact ⟨heap, by simpa only [reference_large width above] using large (globals := globals) width (by omega) heap⟩
    · exact active width (by omega) (by omega) heap

end
end solvers.«38».solve
