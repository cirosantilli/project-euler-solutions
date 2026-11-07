import «.formalsw».solvers.«15.py.ast»
import lang.py.FormalSW.CallsRuntimeCostFunctions

namespace solvers.«15».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.Calls
open FormalSW.PyAst.Calls.Runtime

private abbrev program := Solvers.P15Ast.module
private def body : List Stmt :=
  match executableBody ((findFunctionIn program.body "binom").getD ([], [])).2 with
  | [_, _, _, .forIn _ _ body _, _] => body
  | _ => []

private def loop (start stop : Int) : List Stmt :=
  [.rangeStepLoop "i" start stop 1 body [], .return (.name "res")]

private theorem update_cost (n k i res : Int) (positive : 0 < i)
    (env : ScalarEnv) (hn : env "n" = some (.int n)) (hk : env "k" = some (.int k))
    (hi : env "i" = some (.int i)) (hr : env "res" = some (.int res)) :
    Runtime.EvaluatesIn .word program env
      (.binOp (.binOp (.name "res") .mult
        (.binOp (.binOp (.name "n") .sub (.name "k")) .add (.name "i"))) .floorDiv (.name "i"))
      21 (.int (res * (n - k + i) / i)) := by
  apply Runtime.EvaluatesIn.binary (work := 1) (ca := 16) (cb := 1) (av := .int (res * (n-k+i))) (bv := .int i) (op := .floorDiv)
  · apply Runtime.EvaluatesIn.binary (work := 1) (ca := 1) (cb := 11) (av := .int res) (bv := .int (n-k+i)) (op := .mult)
    · exact Runtime.EvaluatesIn.lookup hr
    · apply Runtime.EvaluatesIn.binary (work := 1) (ca := 6) (cb := 1) (av := .int (n-k)) (bv := .int i) (op := .add)
      · exact Runtime.EvaluatesIn.binary (op := .sub) (value := .int (n-k)) (work := 1) (Runtime.EvaluatesIn.lookup hn)
          (Runtime.EvaluatesIn.lookup hk) (by simp [scalarBinary]) rfl
      · exact Runtime.EvaluatesIn.lookup hi
      · simp [scalarBinary]
      · rfl
    · simp [scalarBinary]
    · rfl
  · exact Runtime.EvaluatesIn.lookup hi
  · simp [scalarBinary, positive, show i ≠ 0 by omega]
  · rfl

private theorem loop_cost (remaining : Nat) (start stop n k res : Int)
    (positive : 0 < start) (endpoint : stop = start + remaining)
    (env : ScalarEnv) (hn : env "n" = some (.int n)) (hk : env "k" = some (.int k))
    (hr : env "res" = some (.int res)) :
    ∃ answer, ∀ stack, RuntimeCost.Reaches (step program) (Runtime.charge .word program)
      ⟨.block (loop start stop) env, stack⟩ (27 * remaining + 6) ⟨.value (.int answer), stack⟩ := by
  induction remaining generalizing start env res with
  | zero =>
      have eq : stop = start := by simpa using endpoint
      refine ⟨res, fun stack => ?_⟩
      apply RuntimeCost.Reaches.next (amount := 4) (cost := 2) (following := ⟨.block [.return (.name "res")] env, stack⟩)
      · simp [step, loop, rangeContinues, eq]
      · rfl
      · exact RuntimeCost.Reaches.next (step := step program) (charge := Runtime.charge .word program)
          (state := ⟨.block [.return (.name "res")] env, stack⟩) rfl rfl (Runtime.EvaluatesIn.lookup hr stack)
  | succ remaining ih =>
      let env' := env.set "i" (.int start)
      let answer := res * (n - k + start) / start
      have updated := update_cost n k start res positive env'
        (by simpa [env', ScalarEnv.set] using hn)
        (by simpa [env', ScalarEnv.set] using hk) (by simp [env', ScalarEnv.set])
        (by simpa [env', ScalarEnv.set] using hr)
      obtain ⟨final, tail⟩ := ih (start + 1) answer (by omega) (by omega)
        (env'.set "res" (.int answer))
        (by simpa [env', ScalarEnv.set] using hn)
        (by simpa [env', ScalarEnv.set] using hk) (by simp [ScalarEnv.set])
      refine ⟨final, fun stack => ?_⟩
      have head := Runtime.assign updated "res" (loop (start + 1) stop) stack
      have active : rangeContinues start stop 1 = true := by simp [rangeContinues]; omega
      have run := RuntimeCost.Reaches.next (step := step program) (charge := Runtime.charge .word program) (amount := 4)
        (state := ⟨.block (loop start stop) env, stack⟩)
        (by simp only [step, loop, active, ↓reduceIte]; rfl) rfl (head.trans (tail stack))
      have eq : 4 + (21 + 2 + (27 * remaining + 6)) = 27 * (remaining + 1) + 6 := by omega
      rw [eq] at run
      exact run

private def args (n : Nat) : ScalarEnv :=
  argumentEnv [⟨"n", some (.name "int"), none⟩, ⟨"k", some (.name "int"), none⟩]
    [.int (2 * n), .int n]

private theorem guard_cost (n : Nat) : Runtime.EvaluatesIn .word program (args n)
    (.boolOp .or [.compare (.name "k") [(.lt, .constant (.int 0))],
      .compare (.name "k") [(.gt, .name "n")]]) 17 (.bool false) := by
  apply Runtime.EvaluatesIn.boolean_pair .or (ca := 7) (cb := 7) (av := .bool false)
  · exact Runtime.EvaluatesIn.compare (op := .lt) (value := false)
      (Runtime.EvaluatesIn.lookup (env := args n) (name := "k") (value := .int n) rfl)
      (Runtime.EvaluatesIn.constant (value := .int 0)) (by simp [scalarCompare, evalCmpOp])
  · exact Runtime.EvaluatesIn.compare (op := .gt) (value := false)
      (Runtime.EvaluatesIn.lookup (env := args n) (name := "k") (value := .int n) rfl)
      (Runtime.EvaluatesIn.lookup (name := "n") (value := .int (2*n)) rfl) (by simp [scalarCompare, evalCmpOp]; omega)
  · rfl

private theorem min_cost (n : Nat) : Runtime.EvaluatesIn .word program (args n)
    (.builtinCall .min [.name "k", .binOp (.name "n") .sub (.name "k")]) 21 (.int n) := by
  apply Runtime.EvaluatesIn.builtin_pair (work := 9)
    (Runtime.EvaluatesIn.lookup (value := .int n) rfl)
    (Runtime.EvaluatesIn.binary (value := .int n) (work := 1)
      (Runtime.EvaluatesIn.lookup (value := .int (2 * n)) rfl)
      (Runtime.EvaluatesIn.lookup (value := .int n) rfl)
      (by simp [scalarBinary]; congr 3; omega) rfl)
  · simp [Primitive.apply, positionalMinimum, minimumFrom, scalarCompare, evalCmpOp]
  · rfl

/-- The actual binomial loop uses n iterations for C(2n,n). -/
theorem binom_cost (n : Nat) : ∃ answer,
    Runtime.ReturnsIn .word program "binom" [.int (2*n), .int n] (27*n+68) (.int answer) := by
  let initialized := ((args n).set "k" (.int n)).set "res" (.int 1)
  obtain ⟨answer, tail⟩ := loop_cost n 1 (n+1) (2*n) n 1 (by decide) (by omega) initialized
    (by simp [initialized, args, ScalarEnv.set, argumentEnv])
    (by simp [initialized, ScalarEnv.set]) (by simp [initialized, ScalarEnv.set])
  refine ⟨answer, fun stack => ?_⟩
  have setup : RuntimeCost.Reaches (step program) (Runtime.charge .word program)
      ⟨.block [.forIn (.name "i") (.call (.name "range")
          [.constant (.int 1), .binOp (.name "k") .add (.constant (.int 1))]) body [],
        .return (.name "res")] initialized, stack⟩ 14
      ⟨.block (loop 1 (n+1)) initialized, stack⟩ := by
    repeat first | exact RuntimeCost.Reaches.refl | runtime_step
  have init : RuntimeCost.Reaches (step program) (Runtime.charge .word program)
      ⟨.invoke "binom" [.int (2*n), .int n], stack⟩ 62
      ⟨.block (loop 1 (n+1)) initialized, stack⟩ := by
    refine RuntimeCost.Reaches.next (amount := 3) (cost := 59)
      (following := ⟨.block (executableBody ((findFunctionIn program.body "binom").getD ([], [])).2) (args n), stack⟩) ?_ ?_ ?_
    · rfl
    · rfl
    · exact (Runtime.branch (guard_cost n) [.return (.constant (.int 0))] [] _ stack).trans
        ((Runtime.assign (min_cost n) "k" _ stack).trans
          ((Runtime.assign Runtime.EvaluatesIn.constant "res" _ stack).trans setup))
  simpa only [show 62 + (27*n+6) = 27*n+68 by omega] using init.trans (tail stack)

private def argsSolve (n : Nat) : ScalarEnv := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]

theorem source_bound (n : Nat) :
    Runtime.TimeBound .word program "solve" [.int n] (27*n+93) := by
  obtain ⟨answer, binom⟩ := binom_cost n
  have lattice : Runtime.ReturnsIn .word program "lattice_paths" [.int n, .int n]
      (27*n+83) (.int answer) := by
    intro stack
    have sum : Runtime.EvaluatesIn .word program
        (argumentEnv [⟨"grid_n", some (.name "int"), none⟩, ⟨"grid_m", some (.name "int"), none⟩]
          [.int n, .int n]) (.binOp (.name "grid_n") .add (.name "grid_m")) 6 (.int (2*n)) :=
      Runtime.EvaluatesIn.binary (work := 1) (Runtime.EvaluatesIn.lookup rfl)
        (Runtime.EvaluatesIn.lookup rfl) (by simp [scalarBinary]; congr 3; omega) rfl
    have called := Runtime.EvaluatesIn.call_pair sum (b := .name "grid_n") (Runtime.EvaluatesIn.lookup rfl) binom
    have run := RuntimeCost.Reaches.next (step := step program) (charge := Runtime.charge .word program) (amount := 3) (state := ⟨.invoke "lattice_paths" [.int n, .int n], stack⟩)
      rfl rfl (RuntimeCost.Reaches.next (step := step program) (charge := Runtime.charge .word program)
        (amount := 1) (following := ⟨.expr _ _, stack⟩) rfl rfl (called stack))
    simpa only [show 3 + (1 + (6 + 1 + (27*n+68) + 4)) = 27*n+83 by omega] using run
  have solve : Runtime.ReturnsIn .word program "solve" [.int n] (27*n+92) (.int answer) := by
    intro stack
    have called := Runtime.EvaluatesIn.call_pair (env := argsSolve n)
      (a := .name "n") (b := .name "n") (Runtime.EvaluatesIn.lookup rfl) (Runtime.EvaluatesIn.lookup rfl) lattice
    have run := RuntimeCost.Reaches.next (step := step program) (charge := Runtime.charge .word program) (amount := 2) (state := ⟨.invoke "solve" [.int n], stack⟩)
      rfl rfl (RuntimeCost.Reaches.next (step := step program) (charge := Runtime.charge .word program)
        (amount := 1) (following := ⟨.expr _ _, stack⟩) rfl rfl (called stack))
    simpa only [show 2 + (1 + (1 + 1 + (27*n+83) + 4)) = 27*n+92 by omega] using run
  exact solve.timeBound

end solvers.«15».runtimeCost

theorem solvers.«15».solve.«BigO(n)» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .word Solvers.P15Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P15Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P15Ast.requires_solve_0)
      id id := by
  refine ⟨120, 1, fun n _ large => ?_⟩
  exact (solvers.«15».runtimeCost.source_bound n).mono (by simp only [id_eq] at *; omega)
