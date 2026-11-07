import solvers.«17.py.lean.d».NumberRuntime
import lang.py.FormalSW.HeapRuntimeCostRangeSums
import lang.py.FormalSW.HeapRuntimeCostSetup

namespace solvers.«17».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P17Ast.module

/-- Each source iteration preserves the word tables while appending private tuples. -/
theorem source_run (n : Nat) (bound : n ≤ 1000)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (tables : Tables heap globals) :
    ∃ extra, ∃ answer : Int, ReturnsIn .word program "solve" [.scalar (.int n)] heap
      (617*n+40) (.scalar (.int answer)) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv args body [.scalar (.int n)] globals
  let stopExpression := Expr.binOp (.name "n") .add (.constant (.int 1))
  let bounds := [Expr.constant (.int 1), stopExpression]
  let element := Expr.call (.name "letters_in_number") [.name "i"]
  let summed := setLocal (comprehensionEnv env [("i", .call (.name "range") bounds, [])])
    sumAccumulator (.scalar (.int 0))
  obtain ⟨extra, answer, loop⟩ := range_sum_loop_invariant (globals := globals) (module := program)
    element "i" 606 (fun _ => True) (by decide) (by intros; trivial)
    (fun heap => Tables heap globals) (fun _ extra saved => saved.append extra) 1 (n+1)
    (by
      intro heap env i saved lo hi _ getI
      obtain ⟨extra, run⟩ := number_cost heap globals saved i.toNat (by omega) (by omega)
      have nonnegative : (i.toNat : Int) = i := Int.toNat_of_nonneg (by omega)
      rw [nonnegative] at run
      exact ⟨extra, _, EvaluatesIn.call_single (EvaluatesIn.lookup getI) run⟩)
    n 1 0 (by omega) heap summed tables (by omega) trivial rfl
  have stop : EvaluatesIn .word program env stopExpression heap 6
      (.scalar (.int (n+1))) heap globals :=
    EvaluatesIn.binary (op := .add) (work := 1) (av := .int n) (bv := .int 1)
      (EvaluatesIn.lookup (name := "n") (by cases globals <;> rfl)) EvaluatesIn.constant rfl rfl
  have evaluated := EvaluatesIn.sum_range
    (EvaluatesIn.sum_generator_range (work := 4)
      (ArgumentsIn.cons EvaluatesIn.constant (ArgumentsIn.cons stop ArgumentsIn.nil))
      rfl EvaluatesIn.constant rfl loop)
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := args) (body := body) (values := [.scalar (.int n)]) rfl rfl
    (BlockReturnsIn.return_value evaluated)
  have arity : args.length = 1 := rfl
  refine ⟨extra, answer, called.mono ?_⟩
  simp only [arity, List.length_cons, List.length_nil, PrimitiveWork.integer]
  omega

theorem source_bound (n : Nat) (bound : n ≤ 1000)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (tables : Tables heap globals) :
    TimeBound .word program "solve" [.scalar (.int n)] heap (1000*(n+1))
      (globals := globals) := by
  obtain ⟨extra, answer, run⟩ := source_run n bound heap globals tables
  exact run.timeBound_append.mono (by omega)

end solvers.«17».runtimeCost
