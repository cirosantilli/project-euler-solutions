import «.formalsw».solvers.«16.py.ast»
import lang.py.FormalSW.DecimalSumRuntimeCost

namespace solvers.«16».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P16Ast.module

/-- The helper uses the shared bound for decimal conversion and mapped summation. -/
theorem digit_sum_cost (m k : Nat) (positive : 0 < k) (bounded : m < 10^k) (heap : Heap) :
    ∃ answer, ReturnsIn .word program "digit_sum" [.scalar (.int m)] heap
      (24*k+30) (.scalar (.int answer)) heap := by
  let args : List Arg := [⟨"n", some (.name "int"), none⟩]
  let body := ((findFunctionIn program.body "digit_sum").getD ([], [])).2
  let env := functionEnv args body [.scalar (.int m)] []
  obtain ⟨answer, evaluated⟩ := DecimalSum.evaluates (module := program) (env := env)
    (source := .name "n") m k positive bounded (EvaluatesIn.lookup rfl)
  refine ⟨answer, ?_⟩
  have called := BlockReturnsIn.function (module := program) (name := "digit_sum")
    (args := args) (values := [.scalar (.int m)]) rfl rfl (BlockReturnsIn.return_value evaluated)
  intro anchors stack
  exact (called anchors stack).mono (by simp [args]; omega)

theorem source_bound (n : Nat) :
    TimeBound .word program "solve" [.scalar (.int n)] [] (26*n+75) := by
  obtain ⟨answer, sum⟩ := digit_sum_cost (2^n) (n+1) (by omega) (PrimitiveWork.power_two_decimal n) []
  let args : List Arg := [⟨"n", some (.name "int"), some (.int 1000)⟩]
  let env := functionEnv args ((findFunctionIn program.body "solve").getD ([], [])).2 [.scalar (.int n)] []
  have power : EvaluatesIn .word program env
      (.binOp (.constant (.int 2)) .pow (.name "n")) []
      (Nat.log2 (n+1)+6) (.scalar (.int ((2^n : Nat) : Int))) [] := by
    rw [show Nat.log2 (n+1)+6 = 1+1+(Nat.log2 (n+1)+1)+3 by omega]
    apply EvaluatesIn.binary (op := .pow) (work := Nat.log2 (n+1)+1) (av := .int 2) (bv := .int n)
      EvaluatesIn.constant (EvaluatesIn.lookup (env := env) (name := "n") rfl)
    · simp [Mutable.binary, scalarBinary, show ¬(n : Int) < 0 by omega]
    · simp [PrimitiveWork.binary, PrimitiveWork.integerBinary, scalarNumber, show ¬(n : Int) < 0 by omega]
  have evaluated := EvaluatesIn.call (ArgumentsIn.cons power ArgumentsIn.nil) sum
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := args) (values := [.scalar (.int n)]) rfl rfl (BlockReturnsIn.return_value evaluated)
  obtain ⟨cost, run, paid⟩ := called 0 []
  have terminal : RuntimeCost.Finishes (HeapCalls.step program 0) (charge .word program)
      ⟨.value (.scalar (.int answer)), [], []⟩ 1 (.ok ⟨.ok (.scalar (.int answer)), [], [], 0⟩) := .done rfl rfl
  refine ⟨cost+1, _, run.finishes terminal, ?_⟩
  have logarithm := Nat.log2_le_self (n+1)
  simp only [potential, args, List.length_cons, List.length_nil,
    show heapPotential ([] : Heap) = 0 from rfl] at paid
  omega

end solvers.«16».runtimeCost

theorem solvers.«16».solve.«BigO(n)» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P16Ast.module "solve"
      (fun n : Nat => [.scalar (.int n)]) (fun _ => [])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P16Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P16Ast.requires_solve_0)
      id id := by
  refine ⟨101, 1, fun n _ large => ?_⟩
  exact (solvers.«16».runtimeCost.source_bound n).mono (by simp only [id_eq] at *; omega)
