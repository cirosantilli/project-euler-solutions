import «.formalsw».solvers.«20.py.ast»
import lang.py.FormalSW.DecimalSumRuntimeCost
import lang.py.FormalSW.FactorialWorkBounds

namespace solvers.«20».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P20Ast.module
private def body : List Stmt :=
  match executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 with
  | [_, .forIn _ _ statements _, _] => statements
  | _ => []

/-- The loop follows the source multiplications and bounds the accumulator's width. -/
theorem loop_cost (remaining start n acc : Nat) (startBound : 2 ≤ start)
    (endpoint : start + remaining = max (n+1) 2)
    (bounded : acc * (n+1)^remaining ≤ (n+1)^n)
    (env : HeapCalls.Env) (heap : Heap)
    (getFact : env "fact" = some (.scalar (.int acc))) :
    ∃ finalEnv, ∃ final : Nat,
      BlockContinuesIn .word program env [.rangeStepLoop "k" start (n+1) 1 body []]
        heap (11*remaining+4) finalEnv heap ∧
      finalEnv "fact" = some (.scalar (.int final)) ∧ final ≤ (n+1)^n := by
  induction remaining generalizing start acc env with
  | zero =>
      refine ⟨env, acc, ?_, getFact, by simpa using bounded⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | succ remaining ih =>
      let envK := setLocal env "k" (.scalar (.int start))
      let envF := setLocal envK "fact" (.scalar (.int ((acc*start : Nat) : Int)))
      have nextBound : (acc*start) * (n+1)^remaining ≤ (n+1)^n := by
        calc
          (acc*start) * (n+1)^remaining ≤ (acc*(n+1)) * (n+1)^remaining :=
            Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (by omega))
          _ = acc * (n+1)^(remaining+1) := by rw [Nat.pow_succ]; ac_rfl
          _ ≤ (n+1)^n := bounded
      have updated : BlockContinuesIn .word program envK body heap 7 envF heap := by
        apply BlockContinuesIn.augAssign (left := .int acc) (right := .int start) (work := 1)
          (by simpa [envK, setLocal] using getFact) (EvaluatesIn.lookup rfl)
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · rfl
      obtain ⟨finalEnv, final, tail, found, size⟩ := ih (start+1) (acc*start) (by omega)
        (by omega) nextBound envF rfl
      refine ⟨finalEnv, final, ?_, found, size⟩
      have active : Calls.rangeContinues start (n+1) 1 = true := by simp [Calls.rangeContinues]; omega
      have combined := BlockContinuesIn.range_step (env := env) (name := "k") active
        (by simpa [body, Calls.loopContinuation, Int.natCast_add] using updated.append tail)
      exact combined.mono (by omega)

theorem source_bound (n : Nat) :
    TimeBound .word program "solve" [.scalar (.int n)] []
      (11*n+24*PrimitiveWork.factorialDigits n+60) := by
  let args : List Arg := [⟨"n", some (.name "int"), none⟩]
  let statements := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv args statements [.scalar (.int n)] []
  let initial := setLocal env "fact" (.scalar (.int 1))
  have bounded : 1 * (n+1)^(n-1) ≤ (n+1)^n := by
    simpa using Nat.pow_le_pow_right (by omega : 0 < n+1) (Nat.sub_le n 1)
  obtain ⟨finalEnv, final, loop, found, size⟩ := loop_cost (n-1) 2 n 1 (by decide)
    (by omega) bounded initial [] rfl
  have forLoop := BlockContinuesIn.for_range_pair
    (EvaluatesIn.constant (module := program) (env := initial) (heap := []) (value := .int 2))
    (EvaluatesIn.binary (op := .add) (work := 1) (av := .int n) (bv := .int 1)
      (EvaluatesIn.lookup (name := "n") rfl) EvaluatesIn.constant
      (by simp [Mutable.binary, scalarBinary]) rfl) loop
  obtain ⟨answer, sum⟩ := DecimalSum.evaluates (module := program) (env := finalEnv) (heap := [])
    (source := .name "fact") final (PrimitiveWork.factorialDigits n)
    (by simp [PrimitiveWork.factorialDigits])
    (Nat.lt_of_le_of_lt size (PrimitiveWork.power_factorial_decimal n)) (EvaluatesIn.lookup found)
  have initialized := BlockContinuesIn.assign (arithmetic := .word) (module := program) (env := env) (heap := [])
    (name := "fact") (EvaluatesIn.constant (value := .int 1))
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := args) (values := [.scalar (.int n)]) rfl rfl
    ((initialized.append forLoop).returns (BlockReturnsIn.return_value sum))
  obtain ⟨cost, run, paid⟩ := called 0 []
  have terminal : RuntimeCost.Finishes (HeapCalls.step program 0) (charge .word program)
      ⟨.value (.scalar (.int answer)), [], []⟩ 1 (.ok ⟨.ok (.scalar (.int answer)), [], [], 0⟩) := .done rfl rfl
  refine ⟨cost+1, _, run.finishes terminal, ?_⟩
  simp only [potential, args, List.length_cons, List.length_nil,
    show heapPotential ([] : Heap) = 0 from rfl] at paid
  omega

end solvers.«20».runtimeCost

theorem solvers.«20».solve.«BigO(n * (Nat.log2 (n + 1) + 1))» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P20Ast.module "solve"
      (fun n : Nat => [.scalar (.int n)]) (fun _ => [])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P20Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P20Ast.requires_solve_0)
      id (fun n => n*((n+1).log2+1)) := by
  refine ⟨120, 1, fun n _ large => ?_⟩
  apply (solvers.«20».runtimeCost.source_bound n).mono
  have size : n ≤ n*((n+1).log2+1) := by rw [Nat.mul_add, Nat.mul_one]; omega
  simp only [id_eq] at large ⊢
  simp only [FormalSW.PyAst.PrimitiveWork.factorialDigits]
  omega
