import «.formalsw».solvers.«48.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p48.py»
import ProjectEulerSolutions.Lib.SelfPowers

set_option Elab.async false

namespace solvers.«48»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.SelfPowers
open data.«project-euler-statements».data.py.project_euler_statements.p48 (decimalModulus modulus_positive digits)
private abbrev program := Solvers.P48Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private def body : List Stmt := [.assign [.name "s"]
  (.binOp (.binOp (.name "s") .add (.builtinCall .pow [.name "i", .name "i", .name "mod"])) .mod (.name "mod"))]
private def rangeExpression : Expr := .call (.name "range")
  [.constant (.int 1), .binOp (.name "n") .add (.constant (.int 1))]

private structure Locals (env : HeapCalls.Env) (width : Nat) (total : Int) : Prop where
  modulus : env "mod"=some (number (decimalModulus width))
  width : env "k"=some (number width)
  total : env "s"=some (number total)

private theorem remainder_result (a : Int) (m : Nat) (positive : 0<m) (heap : Mutable.Heap) :
    Mutable.binary .mod (number a) (number m) heap=some (.ok (number (remainder a m)), heap) := by
  simp [Mutable.binary, scalarBinary, number, remainder, positive, positive.ne']

/-- The real three-argument builtin computes the same term as repeated multiplication. -/
private theorem term_result (first m : Nat) (positive : 0<m) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getI : env "i"=some (number ((first+1 : Nat) : Int))) (getMod : env "mod"=some (number m)) :
    Evaluates program env (.builtinCall .pow [.name "i", .name "i", .name "mod"])
      heap (number (selfPower first m)) heap := by
  rw [selfPower_exact first (m : Int) (by omega)]
  apply Evaluates.builtin (.cons (Evaluates.lookup getI) (.cons (Evaluates.lookup getI) (.cons (Evaluates.lookup getMod) .nil)))
  simp only [primitive, Calls.modularPower, scalarNumber,
    FormalSW.Py.Builtin.Int.powMod_nonnegative ((first+1 : Nat) : Int) (first+1) m positive]
  rfl

/-- Source iterations increment the index and reduce the running total after each modular power. -/
private theorem loop (width first count stop : Nat) (length : first+count=stop) (initial : Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env width initial) :
    ∃ finalEnv, Locals finalEnv width (series first count (decimalModulus width) initial) ∧
      BlockContinues program env [.rangeStepLoop "i" (first+1) (stop+1) 1 body []] heap finalEnv heap := by
  induction count generalizing first initial env with
  | zero =>
      refine ⟨env, locals, ?_⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      rw [series_succ]
      let bound := setLocal env "i" (number ((first+1 : Nat) : Int))
      have boundLocals : Locals bound width initial :=
        ⟨by simpa [bound, setLocal] using locals.modulus,
          by simpa [bound, setLocal] using locals.width,
          by simpa [bound, setLocal] using locals.total⟩
      let total := remainder (initial+selfPower first (decimalModulus width)) (decimalModulus width)
      let updated := setLocal bound "s" (number total)
      have updatedLocals : Locals updated width total :=
        ⟨by simpa [updated, setLocal] using boundLocals.modulus,
          by simpa [updated, setLocal] using boundLocals.width, rfl⟩
      have updatedValue : Evaluates program bound
          (.binOp (.binOp (.name "s") .add (.builtinCall .pow [.name "i", .name "i", .name "mod"])) .mod (.name "mod"))
          heap (number total) heap := by
        apply Evaluates.binary (av := number (initial+selfPower first (decimalModulus width)))
          (bv := number (decimalModulus width)) (middle := heap) (afterRight := heap)
        · exact Evaluates.binary (av := number initial) (bv := number (selfPower first (decimalModulus width)))
            (middle := heap) (afterRight := heap) (Evaluates.lookup boundLocals.total)
            (term_result first (decimalModulus width) (modulus_positive width) heap bound rfl boundLocals.modulus) rfl
        · exact Evaluates.lookup boundLocals.modulus
        · exact remainder_result _ (decimalModulus width) (modulus_positive width) heap
      obtain ⟨finalEnv, finalLocals, rest⟩ := ih (first+1) (by omega) total updated updatedLocals
      refine ⟨finalEnv, finalLocals, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply (BlockContinues.assign updatedValue).append
      simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Int.natCast_add, Int.natCast_one] using rest

private theorem format_result (width : Nat) (total : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getTotal : env "s"=some (number total)) (getWidth : env "k"=some (number width)) :
    Evaluates program env (.call (.attribute (.builtinCall .str [.name "s"]) "zfill") [.name "k"])
      heap (.scalar (.string (Strings.zfill (toString total) width))) heap := by
  apply Evaluates.zfill (text := toString total)
  · exact Evaluates.builtin (.cons (Evaluates.lookup getTotal) .nil) rfl
  · exact .cons (Evaluates.lookup getWidth) .nil
  · rfl
  · intro name impossible
    cases impossible

namespace solve
/-- All integer bounds and nonnegative widths preserve the heap and return the formatted reference series. -/
theorem result (n : Int) (width : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [number n, number width] heap (.scalar (.string (digits n width))) heap := by
  let entry := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number n, number width] []
  let modded := setLocal entry "mod" (number (decimalModulus width))
  let ready := setLocal modded "s" (number 0)
  have locals : Locals ready width 0 := ⟨rfl, rfl, rfl⟩
  have setModulus : Evaluates program entry (.binOp (.constant (.int 10)) .pow (.name "k"))
      heap (number (decimalModulus width)) heap := by
    exact Evaluates.binary (av := number 10) (bv := number width) Evaluates.constant (Evaluates.lookup rfl)
      (by simp [Mutable.binary, scalarBinary, number, decimalModulus, show ¬ (width : Int)<0 by omega])
  obtain ⟨finalEnv, finalLocals, scanned⟩ :
      ∃ finalEnv, Locals finalEnv width (series 0 n.toNat (decimalModulus width) 0) ∧
        BlockContinues program ready [.forIn (.name "i") rangeExpression body []] heap finalEnv heap := by
    by_cases below : n<0
    · have empty : n.toNat=0 := Int.toNat_eq_zero.mpr (by omega)
      refine ⟨ready, by simpa only [empty, series_zero] using locals, ?_⟩
      apply BlockContinues.for_range (values := [number 1, number (n+1)])
        (.cons Evaluates.constant (.cons
          (Evaluates.binary (av := number n) (bv := number 1) (middle := heap) (afterRight := heap) (finalHeap := heap)
            (Evaluates.lookup rfl) Evaluates.constant rfl) .nil)) rfl
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
    · lift n to Nat using (by omega : 0≤n)
      obtain ⟨finalEnv, finalLocals, run⟩ := loop width 0 n n (by omega) 0 heap ready locals
      refine ⟨finalEnv, by simpa only [Int.toNat_natCast] using finalLocals, ?_⟩
      apply BlockContinues.for_range (values := [number 1, number ((n+1 : Nat) : Int)])
        (.cons Evaluates.constant (.cons
          (Evaluates.binary (av := number n) (bv := number 1) (middle := heap) (afterRight := heap) (finalHeap := heap)
            (Evaluates.lookup rfl) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
      exact run
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign setModulus
  apply BlockReturns.assign Evaluates.constant
  apply scanned.returns
  exact BlockReturns.return_value (format_result width _ heap finalEnv finalLocals.total finalLocals.width)
end solve

private abbrev widthEnv (width : Int) : ScalarEnv :=
  fun name => if name="k" then some (.int width) else none

theorem solve.«Equiv(naive_digits)» (n width : Int)
    (condition : Calls.Condition program (widthEnv width) Solvers.P48Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int n, .int width]).Equiv
      (HeapCalls.scalarFunction P48Ast.module "naive_digits" [.int n, .int width]) := by
  have nonnegative : 0≤width := by simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
  lift width to Nat using nonnegative
  have returned := solve.result n width []
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p48.naive_digits.result n width []
  have optimized := returned.scalar_finishes (values := [.int n, .int width])
  have original := reference.scalar_finishes (values := [.int n, .int width])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem solve.«Supported» :
    (∀ n width : Int, Calls.Condition program (widthEnv width) Solvers.P48Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int n, .int width]).Supported) ∧
    (∀ n width : Int, Calls.Condition program (widthEnv width) Solvers.P48Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P48Ast.module "naive_digits" [.int n, .int width]).Supported) := by
  constructor
  · intro n width condition
    have nonnegative : 0≤width := by simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift width to Nat using nonnegative
    exact ((solve.result n width []).scalar_returns (values := [.int n, .int width])).supported
  · intro n width condition
    have nonnegative : 0≤width := by simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift width to Nat using nonnegative
    exact ((data.«project-euler-statements».data.py.project_euler_statements.p48.naive_digits.result n width []).scalar_returns
      (values := [.int n, .int width])).supported

end solvers.«48»
