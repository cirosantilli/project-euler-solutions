import solvers.«51.windows.source»
import lang.py.FormalSW.LoopProgress

/-! Unbounded width search for the least prime replacement family. -/
namespace solvers.«51».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.PrimeReplacementMinimum
set_option Elab.async false
private abbrev program := Solvers.P51Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private def increment : List Stmt := [.augAssign (.name "ndigits") .add (.constant (.int 1))]
private def body : List Stmt := digitBody ++ increment
private def searchLoop : Stmt := .whileDo (.constant (.bool true)) body []

private theorem loop (target : Int) (success : Success target) (first : Nat) (positive : 1 ≤ first)
    (before : first ≤ width target success) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) (getWidth : env "ndigits" = some (number first)) :
    ∃ finalHeap, BlockReturns program env [searchLoop] heap (number (answer target success)) finalHeap := by
  have scanned := digitSearch.result first positive target env heap getWidth getTarget
  rw [digitSearch.answer_eq_window, window_answer target success first (lower_before target success first before)] at scanned
  by_cases inside : answer target success ≤ upper first
  · rw [if_pos inside] at scanned
    obtain ⟨finalHeap, returned, _⟩ := scanned
    refine ⟨finalHeap, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    have finished := returned (increment ++ [searchLoop])
    simpa only [body, Calls.loopContinuation, show Calls.loopControl body = false from rfl,
      Bool.false_eq_true, if_false, List.append_assoc] using finished
  · rw [if_neg inside] at scanned
    obtain ⟨finalEnv, middleHeap, continued, stable, _⟩ := scanned
    have earlier : first < width target success := by
      have reached := inside_width target success
      by_contra late
      have same : first = width target success := by omega
      exact inside (same ▸ reached)
    have finalTarget : finalEnv "target" = some (number target) := (stable "target" (by decide)).trans getTarget
    have finalWidth : finalEnv "ndigits" = some (number first) := (stable "ndigits" (by decide)).trans getWidth
    let advanced := setLocal finalEnv "ndigits" (number (first + 1 : Nat))
    have incremented : BlockContinues program finalEnv increment middleHeap advanced middleHeap := by
      apply BlockContinues.augAssign (left := number first) (right := number 1) finalWidth Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, number]
    obtain ⟨finalHeap, rest⟩ := loop target success (first + 1) (by omega) (by omega) advanced middleHeap
      (by simpa [advanced, setLocal] using finalTarget) (by simp [advanced, setLocal])
    refine ⟨finalHeap, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    have finished := continued.returns (incremented.returns rest)
    simpa only [body, Calls.loopContinuation, show Calls.loopControl body = false from rfl,
      Bool.false_eq_true, if_false, List.append_assoc] using finished
termination_by width target success - first
decreasing_by omega

/-- The outer source loop reaches the actual least family after finitely many complete widths. -/
theorem result (target : Int) (success : Success target) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number target] heap (number (answer target success)) finalHeap := by
  let entry := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number target] []
  let ready := setLocal entry "ndigits" (number 1)
  obtain ⟨finalHeap, returned⟩ := loop target success 1 (by decide) (width_positive target success) ready heap
    (by simp [ready, entry, setLocal]; rfl) (by simp [ready, setLocal])
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  exact returned

/-- If no family qualifies, every source width completes and the next width begins. -/
theorem diverges (target : Int) (absent : ¬ Success target) : (HeapCalls.scalarFunction program "solve" [.int target]).Diverges := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "solve").getD ([], [])).2
  let entry := functionEnv args sourceBody [number target] []
  let ready := setLocal entry "ndigits" (number 1)
  let invariant : HeapCalls.Env → Mutable.Heap → Prop := fun env _ =>
    ∃ digits : Nat, 1 ≤ digits ∧ env "target" = some (number target) ∧ env "ndigits" = some (number digits)
  apply scalar_while_diverges program "solve" [.int target] args sourceBody
    [.assign [.name "ndigits"] (.constant (.int 1))] body [number target] rfl rfl rfl rfl invariant ready []
  · exact BlockContinues.assign Evaluates.constant
  · exact ⟨1, by decide, by simp [ready, entry, setLocal]; rfl, by simp [ready, setLocal]⟩
  · intro env heap current
    obtain ⟨digits, positive, getTarget, getWidth⟩ := current
    have scanned := digitSearch.result digits positive target env heap getWidth getTarget
    rw [digitSearch.answer_eq_window, window_absent target absent digits] at scanned
    obtain ⟨finalEnv, finalHeap, continued, stable, _⟩ := scanned
    have finalTarget : finalEnv "target" = some (number target) := (stable "target" (by decide)).trans getTarget
    have finalWidth : finalEnv "ndigits" = some (number digits) := (stable "ndigits" (by decide)).trans getWidth
    let advanced := setLocal finalEnv "ndigits" (number (digits + 1 : Nat))
    have incremented : BlockContinues program finalEnv increment finalHeap advanced finalHeap := by
      apply BlockContinues.augAssign (left := number digits) (right := number 1) finalWidth Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, number]
    refine ⟨advanced, finalHeap, continued.append incremented, digits + 1, by omega, ?_, ?_⟩
    · simpa [advanced, setLocal] using finalTarget
    · simp [advanced, setLocal]

end solvers.«51».solve
