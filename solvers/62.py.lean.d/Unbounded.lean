import solvers.«62.py.lean.d».Windows
import ProjectEulerSolutions.Lib.CubicWidthBounds
import lang.py.FormalSW.LoopProgress

/-! The two source searches agree across arbitrarily many completed digit widths. -/
set_option Elab.async false
namespace solvers.«62».unbounded
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.CubicSearch ProjectEulerSolutions.CubicWidthBounds
open data.«project-euler-statements».data.py.project_euler_statements.p62
private abbrev program := P62Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

noncomputable def widthAnswer (target : Int) (digits : Nat) : Int := window.answer (stop digits - 1 : Nat) target

theorem width_choice (target : Int) (digits : Nat) :
    ProjectEulerSolutions.CubicGroupCounts.choose target
      (ProjectEulerSolutions.CubicGroupCounts.collect (ProjectEulerSolutions.CubicWindows.prior (stop digits))) =
      widthAnswer target digits := by
  have boundary : stop digits - 1 + 1 = stop digits := by have positive := stop_positive digits; omega
  simpa only [boundary, widthAnswer] using closed_window.reference_choice (stop digits - 1) target

private def innerTest : Expr := .compare (window_count.cubeExpression "stop")
  [(.lt, .binOp (.constant (.int 10)) .pow (.name "digits"))]
private def innerBody : List Stmt := [.augAssign (.name "stop") .add (.constant (.int 1))]
private def innerLoop : Stmt := .whileDo innerTest innerBody []
private def searchExpression : Expr := .call (.name "_search_window")
  [.binOp (.name "stop") .sub (.constant (.int 1)), .name "target"]
private def resultTest : Expr := .compare (.name "result") [(.notEq, .constant (.int 0))]
private def body : List Stmt :=
  [innerLoop, .assign [.name "result"] searchExpression,
    .ifThen resultTest [.return (.name "result")] [],
    .augAssign (.name "digits") .add (.constant (.int 1))]

private theorem inner_test (digits root : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getDigits : env "digits" = some (number digits)) (getRoot : env "stop" = some (number root)) :
    Evaluates program env innerTest heap (flag (decide (cube root < 10 ^ digits))) heap := by
  apply Evaluates.compare (av := number (root ^ 3 : Nat)) (bv := number (10 ^ digits : Nat))
    (middle := heap) (afterRight := heap)
  · exact window_count.cube_evaluate root "stop" heap env getRoot
  · apply Evaluates.binary (av := number 10) (bv := number digits) (middle := heap) (afterRight := heap)
      Evaluates.constant (Evaluates.lookup getDigits)
    simp [Mutable.binary, scalarBinary, number, show ¬ (digits : Int) < 0 by omega]
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, cube, pow_succ]
    norm_cast

/-- The inner loop reaches the boundary derived from the requested digit width. -/
private theorem close_width (digits root : Nat) (bounded : root ≤ stop digits)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getDigits : env "digits" = some (number digits)) (getRoot : env "stop" = some (number root)) :
    ∃ ready, BlockContinues program env [innerLoop] heap ready heap ∧
      ready "stop" = some (number (stop digits)) ∧ ∀ name, name ≠ "stop" → ready name = env name := by
  have tested := inner_test digits root env heap getDigits getRoot
  by_cases before : root < stop digits
  · have small := before_stop digits root before
    let advanced := setLocal env "stop" (number (root + 1 : Nat))
    have incremented : BlockContinues program env innerBody heap advanced heap := by
      apply BlockContinues.augAssign (left := number root) (right := number 1) getRoot Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, number]
    obtain ⟨ready, continued, reached, stable⟩ := close_width digits (root + 1) (by omega) advanced heap
      (by simpa [advanced, setLocal] using getDigits) (by simp [advanced, setLocal])
    refine ⟨ready, ?_, reached, ?_⟩
    · apply BlockContinues.while_loop (flag := true) (by simpa only [small, decide_true] using tested) rfl
      simpa [innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        List.append_assoc] using incremented.append continued
    · intro name different
      exact (stable name different).trans (by simp [advanced, setLocal, different])
  · have same : root = stop digits := by omega
    have large : ¬ cube root < 10 ^ digits := by rw [same]; exact Nat.not_lt_of_ge (stop_reaches digits)
    refine ⟨env, ?_, by simpa only [same] using getRoot, by intros; rfl⟩
    apply BlockContinues.while_loop (flag := false) (by simpa only [large, decide_false] using tested) rfl
    exact BlockContinues.nil
termination_by stop digits - root
decreasing_by omega

private theorem prepare (target : Int) (digits root : Nat) (bounded : root ≤ stop digits)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) (getDigits : env "digits" = some (number digits))
    (getRoot : env "stop" = some (number root)) :
    ∃ ready extra, BlockContinues program env (body.take 2) heap ready (heap ++ extra) ∧
      ready "target" = some (number target) ∧ ready "digits" = some (number digits) ∧
      ready "stop" = some (number (stop digits)) ∧ ready "result" = some (number (widthAnswer target digits)) := by
  obtain ⟨closedEnv, closed, reached, stable⟩ := close_width digits root bounded env heap getDigits getRoot
  obtain ⟨extra, returned⟩ := window.result (stop digits - 1 : Nat) target heap
  have cast : ((stop digits - 1 : Nat) : Int) = (stop digits : Int) - 1 := by
    have positive := stop_positive digits
    omega
  have evaluated : Evaluates program closedEnv searchExpression heap
      (number (widthAnswer target digits)) (heap ++ extra) := by
    apply Evaluates.call (middle := heap) (values := [number (stop digits - 1 : Nat), number target])
    · refine .cons ?_ (.cons (Evaluates.lookup (by simpa only [stable "target" (by decide)] using getTarget)) .nil)
      exact Evaluates.binary (av := number (stop digits)) (bv := number 1) (middle := heap) (afterRight := heap)
        (Evaluates.lookup reached) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number, cast])
    · exact returned
  let ready := setLocal closedEnv "result" (number (widthAnswer target digits))
  have assigned : BlockContinues program closedEnv [body[1]'(by decide)] heap ready (heap ++ extra) :=
    BlockContinues.assign evaluated
  refine ⟨ready, extra, closed.append assigned, ?_, ?_, ?_, ?_⟩
  · simpa [ready, setLocal, stable "target" (by decide)] using getTarget
  · simpa [ready, setLocal, stable "digits" (by decide)] using getDigits
  · simpa [ready, setLocal] using reached
  · simp [ready, setLocal]

private theorem result_test (value : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (found : env "result" = some (number value)) :
    Evaluates program env resultTest heap (flag (decide (value ≠ 0))) heap := by
  exact Evaluates.compare (Evaluates.lookup found) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

private theorem loop_result (target : Int) (winner digits root : Nat)
    (positive : 1 ≤ digits) (before : digits ≤ winner) (bounded : root ≤ stop digits)
    (success : widthAnswer target winner ≠ 0)
    (earlier : ∀ width, 1 ≤ width → width < winner → widthAnswer target width = 0)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) (getDigits : env "digits" = some (number digits))
    (getRoot : env "stop" = some (number root)) :
    ∃ extra, BlockReturns program env [.whileDo (.constant (.bool true)) body []] heap
      (number (widthAnswer target winner)) (heap ++ extra) := by
  obtain ⟨ready, extra, prepared, finalTarget, finalDigits, finalRoot, finalResult⟩ :=
    prepare target digits root bounded env heap getTarget getDigits getRoot
  have tested := result_test (widthAnswer target digits) ready (heap ++ extra) finalResult
  by_cases selected : widthAnswer target digits ≠ 0
  · have same : digits = winner := by
      by_contra different
      exact selected (earlier digits positive (by omega))
    refine ⟨extra, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply prepared.returns
    apply BlockReturns.branch (flag := true) (value := flag true) (by simpa [selected] using tested) rfl
    exact BlockReturns.return_value (Evaluates.lookup (by simpa only [same] using finalResult))
  · have zero : widthAnswer target digits = 0 := by simpa using selected
    have earlierWidth : digits < winner := by
      by_contra late
      have same : digits = winner := by omega
      exact success (same ▸ zero)
    let advanced := setLocal ready "digits" (number (digits + 1 : Nat))
    have incremented : BlockContinues program ready [body[3]'(by decide)] (heap ++ extra) advanced (heap ++ extra) := by
      apply BlockContinues.augAssign (left := number digits) (right := number 1) finalDigits Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, number]
    obtain ⟨tail, returned⟩ := loop_result target winner (digits + 1) (stop digits) (by omega) (by omega)
      (stop_mono (by omega)) success earlier advanced (heap ++ extra)
      (by simpa [advanced, setLocal] using finalTarget) (by simp [advanced, setLocal])
      (by simpa [advanced, setLocal] using finalRoot)
    refine ⟨extra ++ tail, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply prepared.returns
    apply BlockReturns.branch (flag := false) (value := flag false) (by simpa only [zero, ne_self_iff_false, decide_false] using tested) rfl
    simpa only [List.append_assoc, body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
      Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using incremented.returns returned
termination_by winner - digits
decreasing_by omega

/-- A first successful complete width executes the entire unbounded reference call. -/
theorem reference_result (target : Int) (winner : Nat) (positive : 1 ≤ winner)
    (success : widthAnswer target winner ≠ 0)
    (earlier : ∀ width, 1 ≤ width → width < winner → widthAnswer target width = 0)
    (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "naive" [number target] heap
      (number (widthAnswer target winner)) (heap ++ extra) := by
  let entry := functionEnv ((findFunctionIn program.body "naive").getD ([], [])).1
    ((findFunctionIn program.body "naive").getD ([], [])).2 [number target] []
  let ready := setLocal (setLocal entry "digits" (number 1)) "stop" (number 1)
  obtain ⟨extra, returned⟩ := loop_result target winner 1 1 (by decide) positive
    (stop_positive 1) success earlier ready heap (by simp [ready, setLocal, entry]; rfl)
    (by simp [ready, setLocal]) (by simp [ready, setLocal])
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  exact returned

/-- If no complete width succeeds, every reference iteration advances to the next width. -/
theorem reference_diverges (target : Int) (absent : ∀ digits, 1 ≤ digits → widthAnswer target digits = 0) :
    (HeapCalls.scalarFunction program "naive" [.int target]).Diverges := by
  let args := ((findFunctionIn program.body "naive").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "naive").getD ([], [])).2
  let entry := functionEnv args sourceBody [number target] []
  let ready := setLocal (setLocal entry "digits" (number 1)) "stop" (number 1)
  let invariant : HeapCalls.Env → Mutable.Heap → Prop := fun env _ =>
    ∃ digits root : Nat, 1 ≤ digits ∧ root ≤ stop digits ∧
      env "target" = some (number target) ∧ env "digits" = some (number digits) ∧ env "stop" = some (number root)
  apply scalar_while_diverges program "naive" [.int target] args sourceBody
    [.assign [.name "digits"] (.constant (.int 1)), .assign [.name "stop"] (.constant (.int 1))]
    body [number target] rfl rfl rfl rfl invariant ready []
  · exact (BlockContinues.assign Evaluates.constant).append (BlockContinues.assign Evaluates.constant)
  · exact ⟨1, 1, by decide, stop_positive 1, by simp [ready, entry, setLocal]; rfl,
      by simp [ready, setLocal], by simp [ready, setLocal]⟩
  · intro env heap current
    obtain ⟨digits, root, positive, bounded, getTarget, getDigits, getRoot⟩ := current
    obtain ⟨ready, extra, prepared, finalTarget, finalDigits, finalRoot, finalResult⟩ :=
      prepare target digits root bounded env heap getTarget getDigits getRoot
    have tested := result_test 0 ready (heap ++ extra) (by simpa only [absent digits positive] using finalResult)
    have skipped : BlockContinues program ready [body[2]'(by decide)] (heap ++ extra) ready (heap ++ extra) :=
      BlockContinues.branch (flag := false) tested rfl BlockContinues.nil
    let advanced := setLocal ready "digits" (number (digits + 1 : Nat))
    have incremented : BlockContinues program ready [body[3]'(by decide)] (heap ++ extra) advanced (heap ++ extra) := by
      apply BlockContinues.augAssign (left := number digits) (right := number 1) finalDigits Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, number]
    refine ⟨advanced, heap ++ extra, prepared.append (skipped.append incremented), digits + 1, stop digits,
      by omega, stop_mono (by omega), ?_, ?_, ?_⟩
    · simpa [advanced, setLocal] using finalTarget
    · simp [advanced, setLocal]
    · simpa [advanced, setLocal] using finalRoot

end solvers.«62».unbounded
