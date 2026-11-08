import solvers.«23.py.lean.d».SetupRuntime

/-! Compose the original solver's generated AST: divisor sieve, filtered list,
bytearray allocation, triangular pair marking and final unmarked-value scan. -/
namespace solvers.«23».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
abbrev program := Solvers.P23Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

theorem function_body_eq : executableBody definition.2 =
    [.ifThen (.compare (.name "limit") [(.lt, .constant (.int 0))])
      [.return (.constant (.int 0))] [],
     .assign [.name "sums"] (.call (.name "sum_proper_divisors_sieve") [.name "limit"]),
     .assign [.name "abundant"] Abundant.expression] ++
    Setup.statements ++ Marking.statements ++ Scan.statements := rfl

/-- Every natural input executes within a quadratic fixed-arithmetic budget.
The table contents and marking answers are obtained from actual executions;
no functional-correctness hypothesis is needed for this runtime theorem. -/
theorem source_bound (limit : Nat) :
    TimeBound .word program "solve" [number limit] [] (1000 * (limit + 1)^2) := by
  let entry := functionEnv definition.1 definition.2 [number limit] []
  obtain ⟨table, tableSize, sieve⟩ := Sieve.function_bound limit []
  let sieveHeap : Mutable.Heap := [.list [Sieve.number 0], .list (Sieve.encode table)]
  let sieveEnv := setLocal entry "sums" (.ref 1)
  let items := Abundant.values table limit
  let base : Mutable.Heap := sieveHeap ++ [.list (Abundant.encode items)]
  let abundantEnv := setLocal sieveEnv "abundant" (.ref 2)
  let allEnv := Setup.finalEnv abundantEnv limit items.length 2 base.length
  let bytes : List UInt8 := List.replicate (limit + 1) 0
  have allocateSieve : BlockContinuesIn .word program entry
      [.assign [.name "sums"] (.call (.name "sum_proper_divisors_sieve") [.name "limit"])] []
      (64 * ((limit + 1) * ((limit + 1).log2 + 2)) + 5) sieveEnv sieveHeap := by
    have run := EvaluatesIn.call_single
      (EvaluatesIn.lookup (arithmetic := .word) (module := program) (env := entry)
        (heap := []) (name := "limit") rfl) sieve
    exact (BlockContinuesIn.assign run).mono (by omega)
  have collect : BlockContinuesIn .word program sieveEnv
      [.assign [.name "abundant"] Abundant.expression] sieveHeap (30 * limit + 26) abundantEnv base :=
    BlockContinuesIn.assign (Abundant.expression_run table limit 1 tableSize sieveHeap rfl sieveEnv rfl rfl)
  have setup : BlockContinuesIn .word program abundantEnv Setup.statements base (limit + 27)
      allEnv (base ++ [.bytearray bytes]) :=
    Setup.source_run items limit 2 base rfl abundantEnv rfl rfl
  have locals : Marking.Locals allEnv limit items.length 2 base.length := ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨finalBytes, finalEnv, finalSize, finalLocals, marked⟩ :=
    Marking.source_run items limit 2 base rfl bytes (by simp [bytes]) allEnv locals
  obtain ⟨total, summed⟩ := Scan.source_run finalBytes limit base.length finalSize
    (base ++ [.bytearray finalBytes]) (by simp) finalEnv finalLocals.buffer finalLocals.bound
  have rest := allocateSieve.returns (collect.returns (setup.returns (marked.returns summed)))
  have negativeTest : EvaluatesIn .word program entry
      (.compare (.name "limit") [(.lt, .constant (.int 0))]) [] 7 (.scalar (.bool false)) [] := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup (name := "limit") rfl) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  have body := BlockReturnsIn.branch (yes := [.return (.constant (.int 0))]) (no := [])
    (flag := false) negativeTest rfl (by simpa only [List.nil_append] using rest)
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := definition.1) (body := definition.2) (values := [number limit]) rfl rfl
    (by simpa only [function_body_eq, List.cons_append, List.nil_append, List.append_assoc] using body)
  have elapsed : TimeBound .word program "solve" [number limit] []
      (3 + (7 + ((64 * ((limit + 1) * ((limit + 1).log2 + 2)) + 5) +
        ((30 * limit + 26) + ((limit + 27) +
          ((60 * (items.length + 1)^2 + 9) + (25 * limit + 23))))) + 2) + 1) := by
    exact called.timeBound_append
  apply elapsed.mono
  have count : items.length ≤ limit := Abundant.length_bound table limit
  have square := Nat.pow_le_pow_left (Nat.add_le_add_right count 1) 2
  have logarithm := Nat.log2_le_self (limit + 1)
  have harmonic : (limit + 1) * ((limit + 1).log2 + 2) ≤
      (limit + 1) * ((limit + 1) + 2) :=
    Nat.mul_le_mul_left (limit + 1) (Nat.add_le_add_right logarithm 2)
  nlinarith

end solvers.«23».runtimeCost
